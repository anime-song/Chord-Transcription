import random
import copy
import json
import numpy as np
import torch
import torch.optim as optim
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from torch.utils.data import DataLoader

from .data.processing import LabelProcessor
from .data.dataset import ChordDataset, UnlabeledChordDataset
from .models.transcription_model import (
    Backbone,
    BaseTranscriptionModel,
    AudioFeatureExtractor,
    MusicStructureTranscriptionModel,
)
from .models.segment_model import SegmentTranscriptionModel


def load_config(config_path: Path) -> Dict[str, Any]:
    """YAML設定ファイルを読み込みます。"""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int):
    """再現性のために乱数シードを固定します。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_label_processor(config: Dict[str, Any]) -> LabelProcessor:
    """設定からLabelProcessorを構築します。"""
    model_cfg = config["model"]["backbone"]  # STFTパラメータはモデル設定にある
    return LabelProcessor(
        sample_rate=config["data_loader"]["sample_rate"],
        hop_length=model_cfg["hop_length"],
        n_fft=model_cfg["n_fft"],
    )


def build_dataloaders(
    config: Dict[str, Any], label_processor: LabelProcessor, phase_config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """設定とLabelProcessorから学習用・検証用のDataLoaderを構築します。"""
    data_cfg = config["data"]
    loader_cfg = config["data_loader"]

    dataset_config = copy.deepcopy(config)
    if "data_loader" not in dataset_config:
        dataset_config["data_loader"] = {}
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
    return train_loader, valid_loader


def build_optimizer_and_scheduler(
    model: torch.nn.Module, train_config: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    指定された学習設定からOptimizerとSchedulerを構築します。

    Args:
        model (torch.nn.Module): 最適化対象のモデル
        train_config (Dict[str, Any]): 'optimizer'と'scheduler'のキーを含む設定辞書
                                       (例: config['base_model_training'])
    """
    opt_cfg = train_config["optimizer"]

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
            trainable_params, lr=float(opt_cfg["lr"]), betas=tuple(opt_cfg.get("betas", [0.9, 0.999]))
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_cfg['type']}")

    sch_cfg = train_config.get("scheduler", {})
    scheduler_type = sch_cfg.get("type", "none").lower()

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sch_cfg["cosine_tmax"]))
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sch_cfg["step_size"]), gamma=float(sch_cfg["gamma"])
        )
    else:
        scheduler = None

    return optimizer, scheduler


def build_model_from_config(
    cfg: Dict[str, Any], use_segment_model: bool = False, use_structure_model: bool = False
) -> torch.nn.Module:
    model_cfg = cfg["model"]
    classifier_cfg = model_cfg["classifier"]

    # Backbone を構築
    num_stems = len(cfg["data_loader"]["stem_order"])
    channels_per_stem = 1 if cfg["data_loader"].get("mixdown_to_mono", False) else 2
    backbone_params = model_cfg["backbone"]

    feature_extractor = AudioFeatureExtractor(
        sampling_rate=int(backbone_params["sampling_rate"]),
        n_fft=int(backbone_params["n_fft"]),
        hop_length=int(backbone_params["hop_length"]),
        num_audio_channels=num_stems * channels_per_stem,
        num_stems=num_stems,
        spec_augment_params=backbone_params["spec_augment_params"],
    )

    backbone = Backbone(
        feature_extractor=feature_extractor,
        hidden_size=int(backbone_params["hidden_size"]),
        output_dim=int(classifier_cfg["hidden_size"]),
        num_layers=int(backbone_params.get("num_layers", 1)),
        dropout=float(backbone_params.get("dropout", 0.0)),
    )

    # クラス数を決定
    num_root = int(model_cfg.get("num_root_classes", 13))
    num_bass = int(model_cfg.get("num_bass_classes", 13))
    num_quality = int(model_cfg["num_quality_classes"])
    num_key = int(model_cfg["num_key_classes"])

    common_kwargs = dict(
        backbone=backbone,
        hidden_size=int(classifier_cfg["hidden_size"]),
        num_quality_classes=num_quality,
        num_root_classes=num_root,
        num_bass_classes=num_bass,
        num_key_classes=num_key,
        num_tempo_classes=classifier_cfg.get("num_tempo_classes", None),
        dropout_probability=float(classifier_cfg.get("dropout_probability", 0.0)),
        use_layer_norm=bool(classifier_cfg.get("use_layer_norm", True)),
    )

    if use_structure_model:
        # MusicStructureTranscriptionModel
        structure_to_index = {}
        if "structure_json_path" in cfg["data"]:
            with open(cfg["data"]["structure_json_path"], "r", encoding="utf-8") as f:
                structure_to_index = {v: int(k) for k, v in json.load(f).items()}
        elif Path("data/music_structures.json").exists():
            with open("data/music_structures.json", "r", encoding="utf-8") as f:
                structure_to_index = {v: int(k) for k, v in json.load(f).items()}

        num_structure = len(structure_to_index)
        if num_structure == 0:
            num_structure = 11

        return MusicStructureTranscriptionModel(
            backbone=backbone,
            hidden_size=int(classifier_cfg["hidden_size"]),
            num_structure_classes=num_structure,
            dropout_probability=float(classifier_cfg.get("dropout_probability", 0.0)),
            use_layer_norm=bool(classifier_cfg.get("use_layer_norm", True)),
        )

    if use_segment_model:
        seg_conf = model_cfg["segment"]

        return SegmentTranscriptionModel(
            **common_kwargs,
            transformer_hidden_size=seg_conf["hidden_size"],
            transformer_num_heads=seg_conf["num_heads"],
            transformer_num_layers=seg_conf["num_layers"],
            segment_augment_params=seg_conf.get("segment_augment_params", {}),
        )
    return BaseTranscriptionModel(**common_kwargs)


def load_base_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """学習済みのBaseTranscriptionModelの重みを読み込みます。"""
    try:
        print(f"Loading base model checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Successfully loaded base model checkpoint.")
    except FileNotFoundError:
        print(f"ERROR: Base model checkpoint not found at {checkpoint_path}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load base model checkpoint: {e}")
        raise
