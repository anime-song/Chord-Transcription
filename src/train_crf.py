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


import numpy as np


def calculate_accuracy(preds_list, labels, mask):
    correct = 0
    total = 0
    mask = mask.bool().cpu()
    labels = labels.cpu()

    for b, pred_seq in enumerate(preds_list):
        valid_len = int(mask[b].sum().item())

        p = np.array(pred_seq[:valid_len])
        l = labels[b][:valid_len].numpy()

        correct += np.sum(p == l)
        total += valid_len
    return correct, total


def val_loop(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    metrics = {"root_correct": 0, "root_total": 0, "bass_correct": 0, "bass_total": 0, "key_correct": 0, "key_total": 0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            waveform = batch["audio"].to(device)
            root_labels = batch["root_chord_index"].to(device)
            bass_labels = batch["bass_index"].to(device)
            key_labels = batch["key_index"].to(device)

            mask = (root_labels != -100).byte()

            loss, preds = model.get_loss_and_preds(
                waveform,
                root_labels=root_labels.clone(),
                bass_labels=bass_labels.clone(),
                key_labels=key_labels.clone(),
                mask=mask,
            )

            total_loss += loss.item()
            count += 1

            # Accuracy
            r_corr, r_tot = calculate_accuracy(preds["root"], root_labels, mask)
            b_corr, b_tot = calculate_accuracy(preds["bass"], bass_labels, mask)
            k_corr, k_tot = calculate_accuracy(preds["key"], key_labels, mask)

            metrics["root_correct"] += r_corr
            metrics["root_total"] += r_tot
            metrics["bass_correct"] += b_corr
            metrics["bass_total"] += b_tot
            metrics["key_correct"] += k_corr
            metrics["key_total"] += k_tot

    avg_loss = total_loss / count if count > 0 else 0.0

    out_metrics = {"loss": avg_loss}
    out_metrics["root_acc"] = metrics["root_correct"] / metrics["root_total"] if metrics["root_total"] > 0 else 0.0
    out_metrics["bass_acc"] = metrics["bass_correct"] / metrics["bass_total"] if metrics["bass_total"] > 0 else 0.0
    out_metrics["key_acc"] = metrics["key_correct"] / metrics["key_total"] if metrics["key_total"] > 0 else 0.0

    return out_metrics


def main():
    parser = argparse.ArgumentParser(description="凍結したTranscriptionModel上でCRF層(Root, Bass, Key)を学習します")
    parser.add_argument("--config", required=True, help="学習設定YAMLファイルへのパス")
    parser.add_argument("--checkpoint", required=True, help="学習済みBaseTranscriptionModelチェックポイントへのパス")
    parser.add_argument("--output_dir", default="checkpoints_crf", help="CRFチェックポイントの保存先ディレクトリ")
    parser.add_argument(
        "--use_segment_model",
        action="store_true",
        help="SegmentTranscriptionModelを使用するかどうか (デフォルト: False)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    set_global_seed(config["experiment"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # DataLoaderの構築
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

    phase_config = {
        "batch_size": crf_config.get("batch_size", 4),
        "segment_seconds": crf_config.get("segment_seconds", 60.0),
    }

    label_processor = build_label_processor(config)
    train_loader, valid_loader = build_dataloaders(config, label_processor, phase_config=phase_config)

    # Base Modelの構築と読み込み
    base_model = build_model_from_config(config, use_segment_model=args.use_segment_model).to(device)
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
        # Training Loop
        model.train()
        total_loss = 0.0
        count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in progress:
            waveform = batch["audio"].to(device)
            root_labels = batch["root_chord_index"].to(device)
            bass_labels = batch["bass_index"].to(device)
            key_labels = batch["key_index"].to(device)

            mask = (root_labels != -100).byte()

            optimizer.zero_grad()
            loss = model(waveform, root_labels=root_labels, bass_labels=bass_labels, key_labels=key_labels, mask=mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / count
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Train/loss", avg_train_loss, epoch)

        # Validation Loop
        val_metrics = val_loop(model, valid_loader, device)
        print(f"Epoch {epoch} Valid Loss: {val_metrics['loss']:.4f}")
        print(f"  Root Acc: {val_metrics['root_acc']:.4f}")
        print(f"  Bass Acc: {val_metrics['bass_acc']:.4f}")
        print(f"  Key Acc:  {val_metrics['key_acc']:.4f}")

        writer.add_scalar("Valid/loss", val_metrics["loss"], epoch)
        writer.add_scalar("Valid/root_acc", val_metrics["root_acc"], epoch)
        writer.add_scalar("Valid/bass_acc", val_metrics["bass_acc"], epoch)
        writer.add_scalar("Valid/key_acc", val_metrics["key_acc"], epoch)

        # チェックポイントの保存
        save_path = output_dir / f"crf_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            },
            save_path,
        )
        print(f"チェックポイントを保存しました: {save_path}")

    print("学習が完了しました。")


if __name__ == "__main__":
    main()
