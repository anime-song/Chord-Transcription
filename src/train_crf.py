import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import yaml
import time
from torch.utils.tensorboard import SummaryWriter

from .utils import (
    load_config,
    set_global_seed,
    build_label_processor,
    build_dataloaders,
    build_model_from_config,
    load_base_model_checkpoint,
)
from .models.crf_wrapper import TranscriptionCRFModel


def main():
    parser = argparse.ArgumentParser(description="凍結したTranscriptionModel上でCRF層(Root, Bass, Key)を学習します")
    parser.add_argument("--config", required=True, help="学習設定YAMLファイルへのパス")
    parser.add_argument("--checkpoint", required=True, help="学習済みBaseTranscriptionModelチェックポイントへのパス")
    parser.add_argument("--output_dir", default="checkpoints_crf", help="CRFチェックポイントの保存先ディレクトリ")

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    set_global_seed(config["experiment"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # DataLoaderの構築
    # 'crf_training' がなければ 'base_model_training' の設定を使用します
    train_cfg_key = "crf_training"
    if train_cfg_key not in config:
        print(f"警告: 設定に '{train_cfg_key}' が見つかりません。デフォルト値を使用します。")
        crf_config = {
            "epochs": 10,
            "lr": 1e-2,
            "batch_size": config["base_model_training"]["batch_size"],
            "segment_seconds": config["base_model_training"]["segment_seconds"],
        }
    else:
        crf_config = config[train_cfg_key]

    # DataLoaderのパラメータを一貫させる
    phase_config = {
        "batch_size": crf_config.get("batch_size", 4),
        "segment_seconds": crf_config.get("segment_seconds", 60.0),
    }

    label_processor = build_label_processor(config)
    train_loader, valid_loader = build_dataloaders(config, label_processor, phase_config=phase_config)

    # train_loaderとvalid_loaderの両方を学習に使用する
    import itertools

    combined_loader = itertools.chain(train_loader, valid_loader)
    total_batches = len(train_loader) + len(valid_loader)

    # Base Modelの構築と読み込み
    base_model = build_model_from_config(config, use_segment_model=True).to(device)
    load_base_model_checkpoint(base_model, args.checkpoint, device)

    # CRFモデルでラップ
    model = TranscriptionCRFModel(base_model).to(device)

    # Optimizer (CRF Root, Bass, Key 全てのパラメータ)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=float(crf_config.get("lr", 1e-2))
    )

    # ロギング設定
    log_dir = Path(config["data"]["log_dir"]) / f"crf_{config['experiment']['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = crf_config.get("epochs", 10)

    print("CRF (Root, Bass, Key) の学習を開始します...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        # itertools.chainは再利用できないため、エポックごとに再生成する必要があります
        combined_loader = itertools.chain(train_loader, valid_loader)
        progress = tqdm(combined_loader, total=total_batches, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            # batch keys: audio, root_chord_index, bass_index, key_index etc.
            waveform = batch["audio"].to(device)

            root_labels = batch["root_chord_index"].to(device)  # (B, T)
            bass_labels = batch["bass_index"].to(device)  # (B, T)
            key_labels = batch["key_index"].to(device)  # (B, T)

            # マスクの生成 (Rootラベルに-100が含まれている場合、パディングとみなす)
            # 全てのラベルでパディング位置は同じはずなので、Rootで代表させます
            mask = (root_labels != -100).byte()

            # パディング部分のラベルは学習に使われないようにする
            # 安全のため、マスクされる部分のラベルを0に置換
            safe_root = root_labels.clone()
            safe_root[~mask.bool()] = 0

            safe_bass = bass_labels.clone()
            safe_bass[~mask.bool()] = 0

            safe_key = key_labels.clone()
            safe_key[~mask.bool()] = 0

            optimizer.zero_grad()

            loss = model(waveform, root_labels=safe_root, bass_labels=safe_bass, key_labels=safe_key, mask=mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / count
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        writer.add_scalar("Train/loss", avg_loss, epoch)

        # チェックポイントの保存
        save_path = output_dir / f"crf_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
        print(f"チェックポイントを保存しました: {save_path}")

    print("学習が完了しました。")


if __name__ == "__main__":
    main()
