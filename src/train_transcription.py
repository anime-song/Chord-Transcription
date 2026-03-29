import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
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
from .training.losses import BalancedSoftmaxLoss
from .training.trainer import Trainer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_root_chord_class_counts(
    quality_class_counts: list[int],
    quality_labels: list[str],
    num_root_classes: int,
    num_quality_classes: int,
) -> list[int]:
    # root-chord のクラス頻度は、
    # 「No Chord 以外の quality 頻度」を全 root に複製し、
    # 最後に No Chord 1クラス分を足す形で作る。
    if len(quality_class_counts) != num_quality_classes:
        raise ValueError("quality_class_counts length does not match num_quality_classes")

    try:
        non_chord_index = next(idx for idx, label in enumerate(quality_labels) if label == "N")
    except StopIteration as exc:
        raise ValueError("quality vocabulary must contain 'N' for non-chord handling") from exc

    counts_without_non_chord = [count for idx, count in enumerate(quality_class_counts) if idx != non_chord_index]
    if len(counts_without_non_chord) != num_quality_classes - 1:
        raise ValueError("Failed to filter out non-chord count from quality_class_counts")

    root_chord_counts = counts_without_non_chord * (num_root_classes - 1)
    root_chord_counts.append(quality_class_counts[non_chord_index])
    return root_chord_counts


def main():
    parser = argparse.ArgumentParser(description="Train a transcription model.")
    parser.add_argument("--config", required=True, help="Path to the training configuration YAML file.")
    parser.add_argument(
        "--init-from",
        default=None,
        help="Checkpoint to initialize the backbone from (loaded backbone-only).",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint to resume training from.")
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="When resuming, override the starting epoch (global_step and optimizer states are reset).",
    )
    args = parser.parse_args()

    if args.resume_epoch is not None and args.resume_epoch < 1:
        parser.error("--resume_epoch must be >= 1")
    if args.resume_epoch is not None and args.resume is None:
        parser.error("--resume_epoch requires --resume")
    if args.resume is not None and args.init_from is not None:
        parser.error("--resume and --init-from cannot be used together")

    config_path = Path(args.config)
    config = load_config(config_path)

    # resume / init-from では checkpoint 側の model config を優先して、
    # 外側の train config とモデル実体が食い違わないようにする。
    if args.resume:
        saved_model_config = load_model_build_config_from_checkpoint(args.resume, map_location="cpu")
        if saved_model_config is not None:
            config = merge_model_build_config(config, saved_model_config)
            print(f"Using model config embedded in resume checkpoint: {args.resume}")
    elif args.init_from:
        saved_model_config = load_model_build_config_from_checkpoint(args.init_from, map_location="cpu")
        if saved_model_config is not None:
            config = merge_model_build_config(config, saved_model_config)
            print(f"Using model config embedded in init checkpoint: {args.init_from}")

    set_global_seed(config["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ここから先は「学習に必要な実行時オブジェクト」を順に組み立てる。
    phase_config = config["base_model_training"]
    loader_cfg = config["data_loader"]
    data_cfg = config["data"]
    backbone_cfg = config["model"]["backbone"]

    # ラベルのフレーム化は backbone の時間解像度と一致している必要がある。
    label_processor = LabelProcessor(
        sample_rate=loader_cfg["sample_rate"],
        hop_length=backbone_cfg["hop_length"],
        n_fft=backbone_cfg["n_fft"],
    )

    # Dataset には、この実験で使う segment 長だけを上書きして渡す。
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

    # モデル本体は config から組み立てる。checkpoint のロードは後段で行う。
    model = build_model_from_config(config).to(device)

    # optimizer / scheduler も train script 側で確定させてから Trainer に渡す。
    opt_cfg = phase_config["optimizer"]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    if opt_cfg["type"].lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_params,
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    elif opt_cfg["type"].lower() == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=float(opt_cfg["lr"]),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_cfg['type']}")

    sch_cfg = phase_config.get("scheduler", {})
    scheduler_type = sch_cfg.get("type", "none").lower()
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sch_cfg["cosine_tmax"]))
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg["step_size"]),
            gamma=float(sch_cfg["gamma"]),
        )
    else:
        scheduler = None

    with open(config["data"]["quality_class_count_path"], "r", encoding="utf-8") as f:
        quality_class_counts = json.load(f)

    # checkpoint に埋める metadata も入口で作っておく。
    checkpoint_extras = {
        "model_build_config": extract_model_build_config(config, use_crf_model=False),
        "label_vocab": extract_label_vocab(config),
    }

    # TensorBoard の出力先命名は実験入口で決める。
    log_dir = Path(config["data"]["log_dir"]) / f'{config["experiment"]["name"]}_{time.strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # BalancedSoftmaxLoss は Trainer 内で生成せず、学習設定からここで確定させる。
    root_chord_class_counts = build_root_chord_class_counts(
        quality_class_counts=quality_class_counts,
        quality_labels=checkpoint_extras["label_vocab"]["quality"],
        num_root_classes=int(config["model"]["num_root_classes"]),
        num_quality_classes=int(config["model"]["num_quality_classes"]),
    )
    tau = float(phase_config.get("loss", {}).get("balanced_softmax_tau", 1.0))
    root_chord_loss_fn = BalancedSoftmaxLoss(class_counts=root_chord_class_counts, tau=tau).to(device)

    resume_path = args.resume
    resume_epoch = args.resume_epoch
    init_from_path = args.init_from

    # Trainer には「すでに組み上がった学習実行環境」を渡すだけにする。
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        config=config,
        root_chord_loss_fn=root_chord_loss_fn,
        writer=writer,
        scheduler=scheduler,
        checkpoint_extras=checkpoint_extras,
    )

    # resume は optimizer なども含めた完全復元、
    # init-from は backbone prefix だけを読み込む初期化として扱う。
    if resume_path:
        trainer.load_checkpoint(resume_path, strict=False, resume_epoch=resume_epoch)
    elif init_from_path:
        trainer.load_checkpoint(
            init_from_path,
            strict=False,
            resume_epoch=1,
            load_prefixes=("backbone.",),
        )

    print("Starting training...")
    trainer.train()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
