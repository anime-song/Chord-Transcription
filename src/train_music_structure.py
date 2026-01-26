import argparse
import json
from pathlib import Path
import torch

from .utils import (
    load_config,
    set_global_seed,
    build_label_processor,
    build_optimizer_and_scheduler,
    build_model_from_config,
)
from .training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train a MusicStructureTranscriptionModel initialized from a base model."
    )
    parser.add_argument("--config", required=True, help="Path to the training configuration YAML file.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pretrained base model checkpoint to initialize the backbone (loaded backbone-only).",
    )
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume structure training from.")
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="When resuming, override the starting epoch (global_step and optimizer states are reset).",
    )
    # Backbone is always frozen for structure training as per plan
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

    import copy
    from torch.utils.data import DataLoader
    from .data.dataset import ChordDataset

    data_cfg = config["data"]
    loader_cfg = config["data_loader"]
    phase_config = config["base_model_training"]

    dataset_config = copy.deepcopy(config)
    if "data_loader" not in dataset_config:
        dataset_config["data_loader"] = {}
    dataset_config["data_loader"]["segment_seconds"] = phase_config["segment_seconds"]

    print("Building datasets with structure label requirement...")
    train_dataset = ChordDataset(
        jsonl_path=Path(data_cfg["train_jsonl_path"]),
        label_processor=label_processor,
        config=dataset_config,
        random_crop=True,
        require_structure_label=True,
    )
    valid_dataset = ChordDataset(
        jsonl_path=Path(data_cfg["valid_jsonl_path"]),
        label_processor=label_processor,
        config=dataset_config,
        random_crop=False,
        require_structure_label=True,
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

    model = build_model_from_config(config, use_structure_model=True).to(device)

    # Freeze backbone
    model.backbone.requires_grad_(False)
    print("Backbone parameters are frozen (no gradients will be updated).")

    optimizer, scheduler = build_optimizer_and_scheduler(model, config["base_model_training"])

    with open(config["data"]["quality_class_count_path"], "r") as f:
        quality_class_counts = json.load(f)

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

    if args.resume:
        trainer.load_checkpoint(args.resume, strict=False, resume_epoch=args.resume_epoch)
    else:
        # Initial training: load backbone from base model checkpoint
        trainer.load_checkpoint(args.checkpoint, strict=False, resume_epoch=1, load_backbone_only=True)

    print("Starting music structure model training...")
    trainer.train()
    print("Music structure model training finished successfully!")


if __name__ == "__main__":
    main()
