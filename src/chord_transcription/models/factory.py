from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..hub import resolve_pretrained_checkpoint_path
from .transcription_model import AudioFeatureExtractor, Backbone, BaseTranscriptionModel


MODEL_BUILD_CONFIG_VERSION = 1


def extract_model_build_config(cfg: Dict[str, Any], use_crf_model: bool = False) -> Dict[str, Any]:
    """
    モデル再構築に必要な最小設定だけを切り出す。

    学習用の data/log/checkpoint パスや optimizer 設定は含めず、
    重みと一緒に配布しても困らない情報だけを保存する。
    """
    return {
        "version": MODEL_BUILD_CONFIG_VERSION,
        "model_kind": "crf" if use_crf_model else "base",
        "model": copy.deepcopy(cfg["model"]),
        "data_loader": {
            "stem_order": list(cfg["data_loader"]["stem_order"]),
            "sample_rate": int(cfg["data_loader"]["sample_rate"]),
            "hop_length": int(cfg["data_loader"].get("hop_length", cfg["model"]["backbone"]["hop_length"])),
        },
    }


def merge_model_build_config(base_cfg: Dict[str, Any], model_build_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    学習/推論の外側設定を残したまま、モデル構築に必要な部分だけ checkpoint 側で上書きする。
    """
    merged = copy.deepcopy(base_cfg)
    if not model_build_config:
        return merged

    merged.setdefault("model", {})
    merged["model"] = copy.deepcopy(model_build_config["model"])

    merged.setdefault("data_loader", {})
    data_loader_cfg = model_build_config["data_loader"]
    merged["data_loader"]["stem_order"] = list(data_loader_cfg["stem_order"])
    if "sample_rate" in data_loader_cfg:
        merged["data_loader"]["sample_rate"] = int(data_loader_cfg["sample_rate"])
    if "hop_length" in data_loader_cfg:
        merged["data_loader"]["hop_length"] = int(data_loader_cfg["hop_length"])
    return merged


def load_model_build_config_from_checkpoint(
    checkpoint_or_path: Dict[str, Any] | str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Optional[Dict[str, Any]]:
    """
    checkpoint 内の model_build_config を読む。
    埋め込みが無い場合は sidecar JSON を探す。
    """
    checkpoint_path: Optional[Path] = None
    checkpoint: Dict[str, Any]

    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint_path = Path(checkpoint_or_path).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    else:
        checkpoint = checkpoint_or_path

    model_build_config = checkpoint.get("model_build_config")
    if model_build_config is not None:
        return model_build_config

    if checkpoint_path is None:
        return None

    sidecar_path = checkpoint_path.with_suffix(".model_config.json")
    if not sidecar_path.exists():
        return None
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def save_model_build_config_sidecar(checkpoint_path: str | Path, model_build_config: Dict[str, Any]) -> Path:
    sidecar_path = Path(checkpoint_path).with_suffix(".model_config.json")
    sidecar_path.write_text(json.dumps(model_build_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def load_quality_labels_from_json(quality_json_path: str | Path, num_quality_classes: int) -> List[str]:
    """
    quality.json から index 順の quality 語彙を復元する。
    checkpoint に埋め込む前の学習時だけ使用する。
    """
    quality_json_path = Path(quality_json_path)
    with quality_json_path.open("r", encoding="utf-8") as f:
        index_to_quality = json.load(f)

    quality_labels: List[str] = []
    for idx in range(num_quality_classes):
        key = str(idx)
        if key not in index_to_quality:
            raise KeyError(f"quality label index {idx} missing in {quality_json_path}")
        quality_labels.append(index_to_quality[key])

    if "N" not in quality_labels:
        raise ValueError("quality vocabulary must contain 'N'.")
    return quality_labels


def extract_label_vocab(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    checkpoint 単体で推論・評価できるように、ラベル語彙を埋め込む。
    現時点では quality 語彙だけを保持する。
    """
    quality_labels = load_quality_labels_from_json(
        cfg["data"]["quality_json_path"],
        int(cfg["model"]["num_quality_classes"]),
    )
    return {"quality": quality_labels}


def load_label_vocab_from_checkpoint(
    checkpoint_or_path: Dict[str, Any] | str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    checkpoint に埋め込まれた label_vocab を読む。
    旧形式へのフォールバックは行わない。
    """
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = torch.load(checkpoint_or_path, map_location=map_location, weights_only=False)
    else:
        checkpoint = checkpoint_or_path

    label_vocab = checkpoint.get("label_vocab")
    if not isinstance(label_vocab, dict):
        raise KeyError("Checkpoint must contain 'label_vocab'.")

    quality_labels = label_vocab.get("quality")
    if not isinstance(quality_labels, list) or not quality_labels:
        raise KeyError("Checkpoint label_vocab must contain non-empty 'quality'.")
    if "N" not in quality_labels:
        raise ValueError("Checkpoint quality vocabulary must contain 'N'.")
    return label_vocab


def build_backbone_from_config(cfg: Dict[str, Any]) -> Backbone:
    model_cfg = cfg["model"]
    classifier_cfg = model_cfg["classifier"]
    backbone_params = model_cfg["backbone"]

    feature_extractor = AudioFeatureExtractor(
        sampling_rate=int(backbone_params["sampling_rate"]),
        n_fft=int(backbone_params["n_fft"]),
        hop_length=int(backbone_params["hop_length"]),
        spec_augment_params=backbone_params["spec_augment_params"],
    )

    return Backbone(
        feature_extractor=feature_extractor,
        hidden_size=int(backbone_params["hidden_size"]),
        base_ch=int(backbone_params.get("base_ch", max(1, int(backbone_params["hidden_size"]) // 4))),
        output_dim=int(classifier_cfg["hidden_size"]),
        num_layers=int(backbone_params.get("num_layers", 1)),
        num_interval_queries=int(backbone_params.get("num_interval_queries", 4)),
        low_bands=int(backbone_params.get("low_bands", 32)),
        dropout=float(backbone_params.get("dropout", 0.0)),
        use_gradient_checkpoint=bool(backbone_params.get("use_gradient_checkpoint", False)),
    )


def build_model_from_config(cfg: Dict[str, Any], use_crf_model: bool = False) -> torch.nn.Module:
    model_cfg = cfg["model"]
    num_bass = int(model_cfg.get("num_bass_classes", 13))
    num_quality = int(model_cfg["num_quality_classes"])
    num_key = int(model_cfg["num_key_classes"])

    base_model = BaseTranscriptionModel(
        backbone=build_backbone_from_config(cfg),
        hidden_size=int(model_cfg["classifier"]["hidden_size"]),
        num_quality_classes=num_quality,
        num_bass_classes=num_bass,
        num_key_classes=num_key,
        dropout_probability=float(model_cfg["classifier"].get("dropout_probability", 0.0)),
        use_layer_norm=bool(model_cfg["classifier"].get("use_layer_norm", True)),
        chord_interval_config=model_cfg["classifier"].get("chord_interval"),
    )

    if not use_crf_model:
        return base_model

    from .crf_model import CRFTranscriptionModel

    return CRFTranscriptionModel(
        base_model=base_model,
        hidden_size=int(model_cfg["classifier"]["hidden_size"]),
        num_quality_classes=num_quality,
        num_bass_classes=num_bass,
        num_key_classes=num_key,
        pitch_chroma_dim=12,
    )


def _select_model_state_dict(
    checkpoint: Dict[str, Any],
    *,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    state_dict = checkpoint.get("ema_state_dict") if prefer_ema else None
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None and prefer_ema:
        state_dict = checkpoint.get("ema_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'ema_state_dict' or 'model_state_dict'.")
    return state_dict


def _filter_state_dict_for_module(
    source_state_dict: Dict[str, Any],
    target_state_dict: Dict[str, Any],
    *,
    prefixes: Optional[Sequence[str]] = None,
    strip_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    for key, value in source_state_dict.items():
        if prefixes is not None and not any(key.startswith(prefix) for prefix in prefixes):
            continue

        target_key = key
        if strip_prefix is not None:
            if not key.startswith(strip_prefix):
                continue
            target_key = key[len(strip_prefix) :]

        if target_key in target_state_dict and target_state_dict[target_key].shape == value.shape:
            filtered[target_key] = value
    return filtered


def load_pretrained_weights(
    module: torch.nn.Module,
    pretrained_model_name_or_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    prefer_ema: bool = True,
    strict: bool = False,
    prefixes: Optional[Sequence[str]] = None,
    strip_prefix: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint_path = resolve_pretrained_checkpoint_path(
        pretrained_model_name_or_path,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    source_state_dict = _select_model_state_dict(checkpoint, prefer_ema=prefer_ema)
    compatible_state_dict = _filter_state_dict_for_module(
        source_state_dict,
        module.state_dict(),
        prefixes=prefixes,
        strip_prefix=strip_prefix,
    )
    if not compatible_state_dict:
        raise ValueError("No compatible tensors were found in the pretrained checkpoint.")

    incompatible = module.load_state_dict(compatible_state_dict, strict=strict)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_tensor_count": len(compatible_state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_pretrained_backbone(
    backbone: Backbone,
    pretrained_model_name_or_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    prefer_ema: bool = True,
    strict: bool = False,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    return load_pretrained_weights(
        backbone,
        pretrained_model_name_or_path,
        map_location=map_location,
        prefer_ema=prefer_ema,
        strict=strict,
        prefixes=("backbone.",),
        strip_prefix="backbone.",
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )


def build_model_from_pretrained(
    pretrained_model_name_or_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    map_location: str | torch.device = "cpu",
    prefer_ema: bool = True,
    strict: bool = False,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> torch.nn.Module:
    checkpoint_path = resolve_pretrained_checkpoint_path(
        pretrained_model_name_or_path,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model_build_config = load_model_build_config_from_checkpoint(checkpoint)
    if model_build_config is None:
        raise KeyError("Checkpoint must contain 'model_build_config'.")

    config = merge_model_build_config({}, model_build_config)
    use_crf_model = str(model_build_config.get("model_kind", "base")).lower() == "crf"
    model = build_model_from_config(config, use_crf_model=use_crf_model).to(torch.device(device))
    load_pretrained_weights(
        model,
        checkpoint_path,
        map_location=map_location,
        prefer_ema=prefer_ema,
        strict=strict,
    )
    model.eval()
    return model


def build_backbone_from_pretrained(
    pretrained_model_name_or_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    map_location: str | torch.device = "cpu",
    prefer_ema: bool = True,
    strict: bool = False,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> Backbone:
    checkpoint_path = resolve_pretrained_checkpoint_path(
        pretrained_model_name_or_path,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model_build_config = load_model_build_config_from_checkpoint(checkpoint)
    if model_build_config is None:
        raise KeyError("Checkpoint must contain 'model_build_config'.")

    config = merge_model_build_config({}, model_build_config)
    backbone = build_backbone_from_config(config).to(torch.device(device))
    load_pretrained_backbone(
        backbone,
        checkpoint_path,
        map_location=map_location,
        prefer_ema=prefer_ema,
        strict=strict,
    )
    backbone.eval()
    return backbone
