import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.dataset import ChordDataset
from .data.processing import LabelProcessor
from .chord_transcription.models.factory import (
    build_model_from_config,
    extract_label_vocab,
    extract_model_build_config,
    load_model_build_config_from_checkpoint,
    merge_model_build_config,
)
from .chord_transcription.models.crf_model import CRFTranscriptionModel
from .training.crf_trainer import CRFTrainer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a CRF transcription model.")
    parser.add_argument("--config", required=True, help="Path to the training configuration YAML file.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume CRF training from.")
    parser.add_argument("--base_model", required=True, help="Path to the pretrained base model checkpoint.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    # CRF 学習でも、土台となる base model の構成は checkpoint 側に合わせる。
    checkpoint_for_model_config = args.resume if args.resume else args.base_model
    saved_model_config = load_model_build_config_from_checkpoint(checkpoint_for_model_config, map_location="cpu")
    if saved_model_config is not None:
        config = merge_model_build_config(config, saved_model_config)
        print(f"Using model config embedded in checkpoint: {checkpoint_for_model_config}")

    set_global_seed(config["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ここから先は CRF 学習専用の実行環境を組み立てる。
    phase_config = config["crf_training"]
    loader_cfg = config["data_loader"]
    data_cfg = config["data"]
    backbone_cfg = config["model"]["backbone"]

    # ラベル系列の時間解像度は base model と一致している必要がある。
    label_processor = LabelProcessor(
        sample_rate=loader_cfg["sample_rate"],
        hop_length=backbone_cfg["hop_length"],
        n_fft=backbone_cfg["n_fft"],
    )

    # CRF 学習でも segment 長は train config 側の値を使う。
    dataset_config = copy.deepcopy(config)
    dataset_config["data_loader"]["segment_seconds"] = phase_config["segment_seconds"]
    train_dataset = ChordDataset(
        jsonl_path=Path(data_cfg["train_jsonl_path"]),
        label_processor=label_processor,
        config=dataset_config,
        random_crop=True,
    )
    valid_dataset = ChordDataset(
        jsonl_path=Path(data_cfg["valid_jsonl_path"]),
        label_processor=label_processor,
        config=dataset_config,
        random_crop=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_config["batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        prefetch_factor=loader_cfg["prefetch_factor"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=phase_config["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        prefetch_factor=loader_cfg["prefetch_factor"],
    )

    # まずは checkpoint と同じ base model を構築する。
    base_model = build_model_from_config(config)

    # CRF は base model の出力を前提にするので、先に base weights を読む。
    base_model_path = args.base_model
    print(f"Loading base model checkpoint from: {base_model_path}")
    checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Successfully loaded base model checkpoint.")

    # 読み込んだ base model を包む形で CRF モデルを作る。
    model_cfg = config["model"]
    crf_model = CRFTranscriptionModel(
        base_model=base_model,
        hidden_size=int(model_cfg["classifier"]["hidden_size"]),
        num_quality_classes=int(model_cfg["num_quality_classes"]),
        num_bass_classes=int(model_cfg["num_bass_classes"]),
        num_key_classes=int(model_cfg["num_key_classes"]),
        pitch_chroma_dim=12,
    ).to(device)

    # Note: build_optimizer_and_scheduler filters for requires_grad=True
    # so it naturally only optimizes the linear projections and CRFs here.
    #
    # パラメータグループ設定 (weight_decay を層ごとに分ける):
    #   - root_chord_proj: 重みノルムが異常に大きいため強い正則化
    #   - bass_proj:       正常範囲だが予防的に弱い正則化
    #   - その他 (CRF遷移行列, key_proj等): 設定ファイルのデフォルト値
    opt_cfg = config["crf_training"]["optimizer"]
    base_lr = float(opt_cfg["lr"])
    base_wd = float(opt_cfg.get("weight_decay", 1e-4))

    root_chord_proj_params = list(crf_model.root_chord_proj.parameters())
    bass_proj_params = list(crf_model.bass_proj.parameters())
    root_chord_proj_ids = {id(p) for p in root_chord_proj_params}
    bass_proj_ids = {id(p) for p in bass_proj_params}

    other_params = [
        p
        for p in crf_model.parameters()
        if p.requires_grad and id(p) not in root_chord_proj_ids and id(p) not in bass_proj_ids
    ]

    param_groups = [
        {
            "params": root_chord_proj_params,
            "lr": base_lr,
            "weight_decay": base_wd * 100,  # 1e-2: 強い正則化
            "name": "root_chord_proj",
        },
        {
            "params": bass_proj_params,
            "lr": base_lr,
            "weight_decay": base_wd * 10,  # 1e-3: 弱い正則化
            "name": "bass_proj",
        },
        {
            "params": other_params,
            "lr": base_lr,
            "weight_decay": base_wd,  # 1e-4: デフォルト
            "name": "others",
        },
    ]

    print(f"[CRF Optimizer] root_chord_proj weight_decay={base_wd * 100:.1e}")
    print(f"[CRF Optimizer] bass_proj        weight_decay={base_wd * 10:.1e}")
    print(f"[CRF Optimizer] others           weight_decay={base_wd:.1e}")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    # スケジューラーの構築
    sch_cfg = config["crf_training"].get("scheduler", {})
    scheduler_type = sch_cfg.get("type", "none").lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sch_cfg["cosine_tmax"]))
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sch_cfg["step_size"]), gamma=float(sch_cfg["gamma"])
        )
    else:
        scheduler = None

    resume_path = args.resume
    # 保存時に必要な metadata は train script 側でまとめて作る。
    checkpoint_extras = {
        "model_build_config": extract_model_build_config(config, use_crf_model=True),
        "label_vocab": extract_label_vocab(config),
    }

    # TensorBoard の出力先も実験入口で決める。
    log_dir = Path(config["data"]["log_dir"]) / f"{config['experiment']['name']}_crf_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Trainer には学習実行に必要なオブジェクトだけを渡す。
    trainer = CRFTrainer(
        model=crf_model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        config=config,
        writer=writer,
        scheduler=scheduler,
        checkpoint_extras=checkpoint_extras,
    )

    # CRF 側の resume は CRF trainer 自体の状態をそのまま復元する。
    if resume_path:
        trainer.load_checkpoint(resume_path, strict=True)

    print("Starting CRF training...")
    trainer.train()
    print("CRF Training finished successfully!")


if __name__ == "__main__":
    main()
