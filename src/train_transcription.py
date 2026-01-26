import argparse
import json
from pathlib import Path

import torch

from .utils import (
    load_config,
    set_global_seed,
    build_label_processor,
    build_dataloaders,
    build_optimizer_and_scheduler,
    build_model_from_config,
)
from .training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a transcription model.")
    parser.add_argument("--config", required=True, help="Path to the training configuration YAML file.")
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

    config_path = Path(args.config)
    config = load_config(config_path)

    set_global_seed(config["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_processor = build_label_processor(config)
    train_loader, valid_loader = build_dataloaders(config, label_processor, phase_config=config["base_model_training"])

    model = build_model_from_config(config).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["base_model_training"])

    with open(config["data"]["quality_class_count_path"], "r") as f:
        quality_class_counts = json.load(f)

    resume_path = args.resume
    resume_epoch = args.resume_epoch

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        config=config,
        scheduler=scheduler,
        quality_class_counts=quality_class_counts,
    )

    if resume_path:
        trainer.load_checkpoint(resume_path, strict=False, resume_epoch=resume_epoch)

    print("Starting training...")
    trainer.train()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
