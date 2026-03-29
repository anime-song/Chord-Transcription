from __future__ import annotations

import argparse
import copy
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from .data.dataset import ChordDataset, mix_stems
from .data.processing import LabelProcessor
from .chord_transcription.models.cqt import RecursiveCQT
from .chord_transcription.models.factory import load_quality_labels_from_json
from .chord_transcription.models.repeat_ssm import ChordWindowSSM, RepeatPairBuilderCPU


PITCH_CLASS_LABELS_13: List[str] = ["N", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HARMONIC_STEM_NAMES = {"bass", "other", "piano", "guitar"}
PAIR_COLORS: List[str] = [
    "#d7263d",
    "#1b998b",
    "#2e294e",
    "#f46036",
    "#3a86ff",
    "#ff006e",
    "#6a994e",
    "#8338ec",
    "#ffbe0b",
    "#5f0f40",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="repeat_ssm のデバッグ可視化を行うスクリプト")
    parser.add_argument("--config", type=Path, required=True, help="学習設定 YAML")
    parser.add_argument("--split", choices=["train", "valid"], default="valid", help="対象 split")
    parser.add_argument("--index", type=int, default=0, help="対象セグメント index")
    parser.add_argument(
        "--phase",
        choices=["base", "crf"],
        default="base",
        help="segment_seconds の既定値をどの学習 phase から拾うか",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="config の segment_seconds を上書きしたい場合のみ指定",
    )
    parser.add_argument(
        "--feature-type",
        choices=["cqt", "mel", "stft", "chroma", "stem_chroma", "stem_cens"],
        default="cqt",
        help="repeat_ssm に入れるオーディオ特徴",
    )
    parser.add_argument(
        "--label-mode",
        choices=["full_chord", "root_chord"],
        default="full_chord",
        help="GT セグメント化に使うラベル。full_chord は root_chord+bass を結合する",
    )
    parser.add_argument(
        "--window-mode",
        choices=["adaptive_exact", "adaptive_label_rescore", "fixed"],
        default="adaptive_exact",
        help="fixed は固定長窓、adaptive_exact は ChordWindowSSM の adaptive pair、adaptive_label_rescore は GT pair を CPU で作って sparse に audio 再スコアする",
    )
    parser.add_argument("--window-size", type=int, default=4, help="固定長窓の長さ、または adaptive seed の最小長")
    parser.add_argument("--window-hop", type=int, default=1, help="窓のシフト幅（コード単位）")
    parser.add_argument("--min-segment-frames", type=int, default=1, help="有効とみなす最小セグメント長")
    parser.add_argument("--min-window-frames", type=int, default=1, help="有効とみなす最小窓長")
    parser.add_argument(
        "--max-span-ratio",
        type=float,
        default=1.5,
        help="対応する 2 区間の全長フレーム比の上限。大きすぎる長さ差の pair を除外する",
    )
    parser.add_argument(
        "--exclude-neighbor-windows",
        type=int,
        default=None,
        help="対角近傍として除外する window index 距離。省略時は overlap 分を自動除外",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="repeat 候補として残す最小類似度",
    )
    parser.add_argument("--topk", type=int, default=5, help="各 window から残す候補数")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="特徴量の時間・周波数方向の z-score 正規化を無効化する",
    )
    parser.add_argument(
        "--no-log-amplitude",
        action="store_true",
        help="振幅の log1p 圧縮を無効化する",
    )
    parser.add_argument("--cqt-bins-per-octave", type=int, default=36, help="CQT の bins_per_octave")
    parser.add_argument("--cqt-n-bins", type=int, default=36 * 7, help="CQT の総 bin 数")
    parser.add_argument("--n-mels", type=int, default=128, help="Mel 特徴量の周波数 bin 数")
    parser.add_argument("--n-chroma", type=int, default=12, help="Chroma 特徴量の次元数")
    parser.add_argument(
        "--audio-stems",
        choices=["mix", "harmonic"],
        default="mix",
        help="特徴抽出に使うステム群。harmonic は bass/other/piano/guitar のみを mix する",
    )
    parser.add_argument(
        "--label-filter",
        choices=["exact", "none"],
        default="exact",
        help="候補抽出後に GT ラベル列で再フィルタする方法",
    )
    parser.add_argument("--max-pairs", type=int, default=20, help="可視化・TSV に残す最大 pair 数")
    parser.add_argument("--output-dir", type=Path, default=Path("debug/repeat_ssm"), help="出力先ディレクトリ")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="特徴抽出に使う device",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_phase_config(config: Dict[str, Any], phase: str) -> Dict[str, Any]:
    if phase == "base":
        return config["base_model_training"]
    if phase == "crf":
        if "crf_training" not in config:
            raise KeyError("Config does not contain 'crf_training'.")
        return config["crf_training"]
    raise ValueError(f"Unsupported phase: {phase}")


def build_dataset(
    config: Dict[str, Any],
    split: str,
    segment_seconds: float,
) -> tuple[ChordDataset, LabelProcessor]:
    backbone_cfg = config["model"]["backbone"]
    loader_cfg = config["data_loader"]
    data_cfg = config["data"]

    label_processor = LabelProcessor(
        sample_rate=int(loader_cfg["sample_rate"]),
        hop_length=int(backbone_cfg["hop_length"]),
        n_fft=int(backbone_cfg["n_fft"]),
    )

    dataset_config = copy.deepcopy(config)
    dataset_config["data_loader"]["segment_seconds"] = float(segment_seconds)
    dataset_config["data_loader"]["use_beat_annotations"] = False
    jsonl_path = Path(data_cfg["train_jsonl_path"] if split == "train" else data_cfg["valid_jsonl_path"])
    dataset = ChordDataset(
        jsonl_path=jsonl_path,
        label_processor=label_processor,
        config=dataset_config,
        # デバッグ用途では毎回同じセグメントが出てほしいので random_crop は使わない。
        random_crop=False,
    )
    return dataset, label_processor


def crop_length_from_num_samples(num_samples: int, n_fft: int, hop_length: int) -> int:
    return (num_samples - n_fft) // hop_length + 1


def resolve_selected_stems(stem_order: Sequence[str], audio_stems: str) -> List[str]:
    if audio_stems == "mix":
        return list(stem_order)
    if audio_stems == "harmonic":
        selected = [stem_name for stem_name in stem_order if stem_name in HARMONIC_STEM_NAMES]
        if not selected:
            raise ValueError("No harmonic stems found in stem_order.")
        return selected
    raise ValueError(f"Unsupported audio_stems mode: {audio_stems}")


def load_selected_stem_waveform(
    dataset: ChordDataset,
    record: Dict[str, Any],
    segment_start_sec: float,
    segment_seconds: float,
    target_samples: int,
    selected_stems: Sequence[str],
) -> torch.Tensor:
    shift_semitones = dataset._get_default_semitone(record)
    if dataset.audio_backend == "packed":
        stems = dataset._load_packed_stems_segment(
            record=record,
            start_seconds=segment_start_sec,
            segment_seconds=segment_seconds,
            shift_semitones=shift_semitones,
        )
    else:
        stems = dataset._load_wav_stems_segment(
            stems_dir=Path(record["stems_dir"]),
            start_seconds=segment_start_sec,
            segment_seconds=segment_seconds,
            shift_semitones=shift_semitones,
        )

    stem_index_map = {stem_name: idx for idx, stem_name in enumerate(dataset.stem_order)}
    missing = [stem_name for stem_name in selected_stems if stem_name not in stem_index_map]
    if missing:
        raise KeyError(f"Requested stems are not in stem_order: {missing}")

    selected_audio = [stems[stem_index_map[stem_name]] for stem_name in selected_stems]
    mixed_audio = mix_stems(selected_audio)
    mixed_audio = np.asarray(mixed_audio[:, :target_samples], dtype=np.float32)
    return torch.from_numpy(mixed_audio).unsqueeze(0)


def load_selected_stem_tensor(
    dataset: ChordDataset,
    record: Dict[str, Any],
    segment_start_sec: float,
    segment_seconds: float,
    target_samples: int,
    selected_stems: Sequence[str],
) -> torch.Tensor:
    shift_semitones = dataset._get_default_semitone(record)
    if dataset.audio_backend == "packed":
        stems = dataset._load_packed_stems_segment(
            record=record,
            start_seconds=segment_start_sec,
            segment_seconds=segment_seconds,
            shift_semitones=shift_semitones,
        )
    else:
        stems = dataset._load_wav_stems_segment(
            stems_dir=Path(record["stems_dir"]),
            start_seconds=segment_start_sec,
            segment_seconds=segment_seconds,
            shift_semitones=shift_semitones,
        )

    stem_index_map = {stem_name: idx for idx, stem_name in enumerate(dataset.stem_order)}
    missing = [stem_name for stem_name in selected_stems if stem_name not in stem_index_map]
    if missing:
        raise KeyError(f"Requested stems are not in stem_order: {missing}")

    selected_audio = [
        np.asarray(stems[stem_index_map[stem_name]][:, :target_samples], dtype=np.float32)
        for stem_name in selected_stems
    ]
    stacked_audio = np.stack(selected_audio, axis=0)
    return torch.from_numpy(stacked_audio).unsqueeze(0)


def _normalize_spec(spec: torch.Tensor) -> torch.Tensor:
    mean = spec.mean(dim=(2, 3), keepdim=True)
    std = spec.std(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    return (spec - mean) / std


def _compute_cqt_spec(
    flat_waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    cqt_bins_per_octave: int,
    cqt_n_bins: int,
) -> torch.Tensor:
    extractor = RecursiveCQT(
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=cqt_n_bins,
        bins_per_octave=cqt_bins_per_octave,
        filter_scale=0.4375,
    ).to(flat_waveform.device)
    return extractor(flat_waveform, return_complex=False)


def _fold_cqt_to_chroma(cqt_spec: torch.Tensor, bins_per_octave: int, n_chroma: int) -> torch.Tensor:
    if n_chroma != 12:
        raise ValueError("chroma feature currently supports only 12 bins.")
    if bins_per_octave % n_chroma != 0:
        raise ValueError("cqt_bins_per_octave must be a multiple of 12 for chroma.")

    bins_per_pitch_class = bins_per_octave // n_chroma
    usable_bins = (cqt_spec.shape[1] // bins_per_octave) * bins_per_octave
    if usable_bins == 0:
        raise ValueError("Not enough CQT bins to fold into chroma.")

    cqt_spec = cqt_spec[:, :usable_bins]
    num_octaves = usable_bins // bins_per_octave
    # (B, octave, chroma, sub_bins, T) -> 平均して 12 次元 chroma へ落とす。
    cqt_spec = cqt_spec.reshape(cqt_spec.shape[0], num_octaves, n_chroma, bins_per_pitch_class, cqt_spec.shape[-1])
    chroma = cqt_spec.mean(dim=3).mean(dim=1)
    chroma = chroma / chroma.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return chroma


def _compute_cens_from_chroma(chroma: torch.Tensor, smoothing_frames: int = 9) -> torch.Tensor:
    """
    chroma: (B, N, C, T)
    returns: (B, N, C, T)
    """
    chroma = chroma / chroma.sum(dim=2, keepdim=True).clamp_min(1e-8)

    quantized = (
        0.25 * (chroma > 0.4).to(chroma.dtype)
        + 0.25 * (chroma > 0.2).to(chroma.dtype)
        + 0.25 * (chroma > 0.1).to(chroma.dtype)
        + 0.25 * (chroma > 0.05).to(chroma.dtype)
    )

    if smoothing_frames > 1:
        padding = smoothing_frames // 2
        kernel = torch.hann_window(smoothing_frames, device=chroma.device, dtype=chroma.dtype)
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        flat = quantized.reshape(-1, 1, quantized.shape[-1])
        smoothed = torch.nn.functional.conv1d(
            flat,
            kernel.view(1, 1, -1),
            padding=padding,
        )
        quantized = smoothed.reshape_as(quantized)

    quantized = quantized / torch.linalg.norm(quantized, dim=2, keepdim=True).clamp_min(1e-8)
    return quantized


def _zscore_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=(1, 2), keepdim=True)
    std = features.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return (features - mean) / std


def compute_stem_aggregated_audio_features(
    stem_waveforms: torch.Tensor,
    feature_type: str,
    sample_rate: int,
    hop_length: int,
    crop_length: int,
    *,
    cqt_bins_per_octave: int,
    cqt_n_bins: int,
    n_chroma: int,
    log_amplitude: bool,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        stem_waveforms: (B, N, C, S)

    Returns:
        frame_features: (B, T, D)
        feature_map: (T, F)
    """
    if stem_waveforms.dim() != 4:
        raise ValueError("stem_waveforms must have shape (B, N, C, S)")
    if feature_type not in {"stem_chroma", "stem_cens"}:
        raise ValueError(f"Unsupported stem feature_type: {feature_type}")

    batch_size, num_stems, num_channels, _ = stem_waveforms.shape
    flat_waveform = stem_waveforms.float().reshape(batch_size * num_stems * num_channels, -1)
    cqt_spec = _compute_cqt_spec(
        flat_waveform=flat_waveform,
        sample_rate=sample_rate,
        hop_length=hop_length,
        cqt_bins_per_octave=cqt_bins_per_octave,
        cqt_n_bins=cqt_n_bins,
    )
    if log_amplitude:
        cqt_spec = torch.log1p(cqt_spec)

    chroma = _fold_cqt_to_chroma(
        cqt_spec=cqt_spec,
        bins_per_octave=cqt_bins_per_octave,
        n_chroma=n_chroma,
    )
    chroma = chroma.reshape(batch_size, num_stems, num_channels, n_chroma, chroma.shape[-1]).mean(dim=2)
    chroma = chroma[..., :crop_length]

    if feature_type == "stem_cens":
        stem_features = _compute_cens_from_chroma(chroma)
    else:
        stem_features = chroma / chroma.sum(dim=2, keepdim=True).clamp_min(1e-8)

    stem_features = stem_features.permute(0, 1, 3, 2).contiguous()

    stem_energy = stem_waveforms.pow(2).mean(dim=(2, 3)).sqrt()
    active_mask = stem_energy > 1e-4
    if not bool(active_mask.any().item()):
        active_mask = torch.ones_like(active_mask, dtype=torch.bool)
    weights = active_mask.to(stem_features.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    aggregated = (stem_features * weights[:, :, None, None]).sum(dim=1)

    if normalize:
        aggregated = _zscore_feature_tensor(aggregated)

    frame_features = aggregated
    feature_map = aggregated[0]
    return frame_features, feature_map


def compute_audio_features(
    waveform: torch.Tensor,
    feature_type: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    crop_length: int,
    *,
    cqt_bins_per_octave: int,
    cqt_n_bins: int,
    n_mels: int,
    n_chroma: int,
    log_amplitude: bool,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        frame_features: (1, T, D)
        feature_map: (T, F) で可視化用に channel 平均したもの
    """
    if waveform.dim() != 3:
        raise ValueError("waveform must have shape (B, C, S)")

    batch_size, num_channels, _ = waveform.shape
    waveform = waveform.float()
    flat_waveform = waveform.reshape(batch_size * num_channels, -1)

    if feature_type == "cqt":
        spec = _compute_cqt_spec(
            flat_waveform=flat_waveform,
            sample_rate=sample_rate,
            hop_length=hop_length,
            cqt_bins_per_octave=cqt_bins_per_octave,
            cqt_n_bins=cqt_n_bins,
        )
    elif feature_type == "mel":
        extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            power=2.0,
            center=True,
            normalized=False,
        ).to(waveform.device)
        spec = extractor(flat_waveform)
    elif feature_type == "chroma":
        cqt_spec = _compute_cqt_spec(
            flat_waveform=flat_waveform,
            sample_rate=sample_rate,
            hop_length=hop_length,
            cqt_bins_per_octave=cqt_bins_per_octave,
            cqt_n_bins=cqt_n_bins,
        )
        if log_amplitude:
            cqt_spec = torch.log1p(cqt_spec)
        spec = _fold_cqt_to_chroma(
            cqt_spec=cqt_spec,
            bins_per_octave=cqt_bins_per_octave,
            n_chroma=n_chroma,
        )
    else:
        window = torch.hann_window(n_fft, device=waveform.device)
        spec = torch.stft(
            flat_waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()

    spec = spec.reshape(batch_size, num_channels, spec.shape[-2], spec.shape[-1]).permute(0, 1, 3, 2).contiguous()
    spec = spec[:, :, :crop_length]

    if log_amplitude and feature_type != "chroma":
        spec = torch.log1p(spec)
    if normalize:
        spec = _normalize_spec(spec)

    frame_features = spec.reshape(batch_size, spec.shape[2], -1)
    feature_map = spec.mean(dim=1)[0]
    return frame_features, feature_map


def combine_full_chord_labels(
    root_chord_index: torch.Tensor,
    bass_index: torch.Tensor,
    num_bass_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    combined = torch.full_like(root_chord_index, ignore_index)
    valid_mask = (root_chord_index != ignore_index) & (bass_index != ignore_index)
    combined[valid_mask] = root_chord_index[valid_mask] * num_bass_classes + bass_index[valid_mask]
    return combined


def build_root_chord_labels(num_root_classes: int, quality_labels: Sequence[str]) -> List[str]:
    quality_slots = [label for label in quality_labels if label != "N"]
    labels: List[str] = []
    for root_idx in range(1, num_root_classes):
        root_name = PITCH_CLASS_LABELS_13[root_idx]
        for quality in quality_slots:
            labels.append(f"{root_name}{quality}")
    labels.append("N")
    return labels


def split_root_pitch(root_chord_label: str) -> str:
    for pitch in sorted(PITCH_CLASS_LABELS_13[1:], key=len, reverse=True):
        if root_chord_label.startswith(pitch):
            return pitch
    return root_chord_label


def format_chord_label(root_chord_idx: int, bass_idx: int, root_chord_labels: Sequence[str]) -> str:
    if root_chord_idx < 0 or root_chord_idx >= len(root_chord_labels):
        return "?"
    root_label = root_chord_labels[root_chord_idx]
    if root_label == "N":
        return "N"

    bass_label = PITCH_CLASS_LABELS_13[bass_idx] if 0 <= bass_idx < len(PITCH_CLASS_LABELS_13) else "?"
    if bass_label in {"N", split_root_pitch(root_label)}:
        return root_label
    return f"{root_label}/{bass_label}"


def format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes:02d}:{remain:05.2f}"


def make_output_stem(
    split: str,
    index: int,
    feature_type: str,
    label_mode: str,
    window_mode: str,
    audio_stems: str,
) -> str:
    return f"{split}_{index:05d}_{feature_type}_{audio_stems}_{label_mode}_{window_mode}"


def html_escape(text: str) -> str:
    return html.escape(text, quote=True)


def color_for_value(value: float, min_value: float, max_value: float) -> str:
    if max_value - min_value < 1e-12:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    low = (242, 244, 247)
    high = (13, 71, 161)
    rgb = tuple(int(low[idx] + ratio * (high[idx] - low[idx])) for idx in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def save_similarity_svg(
    similarity: torch.Tensor,
    candidate_mask: torch.Tensor,
    output_path: Path,
) -> None:
    num_windows = int(similarity.shape[0])
    if num_windows == 0:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='80'><text x='20' y='45'>No valid windows.</text></svg>",
            encoding="utf-8",
        )
        return

    valid_values = similarity.reshape(-1)
    min_value = float(valid_values.min().item())
    max_value = float(valid_values.max().item())

    cell_size = max(6, min(18, 720 // max(1, num_windows)))
    margin_left = 72
    margin_top = 48
    width = margin_left + cell_size * num_windows + 32
    height = margin_top + cell_size * num_windows + 48
    tick_step = max(1, num_windows // 10)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:monospace;font-size:10px;fill:#222;} .title{font-size:14px;font-weight:bold;}</style>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text class='title' x='{margin_left}' y='24'>Repeat SSM</text>",
    ]

    for row_idx in range(num_windows):
        for col_idx in range(num_windows):
            x = margin_left + col_idx * cell_size
            y = margin_top + row_idx * cell_size
            fill = color_for_value(float(similarity[row_idx, col_idx].item()), min_value, max_value)
            parts.append(
                f"<rect x='{x}' y='{y}' width='{cell_size}' height='{cell_size}' fill='{fill}' stroke='none'/>"
            )
            if bool(candidate_mask[row_idx, col_idx].item()):
                parts.append(
                    f"<rect x='{x + 0.5}' y='{y + 0.5}' width='{cell_size - 1}' height='{cell_size - 1}' "
                    "fill='none' stroke='#d7263d' stroke-width='1.5'/>"
                )

    parts.append(
        f"<rect x='{margin_left}' y='{margin_top}' width='{cell_size * num_windows}' height='{cell_size * num_windows}' "
        "fill='none' stroke='#444' stroke-width='1'/>"
    )

    for tick in range(0, num_windows, tick_step):
        x = margin_left + tick * cell_size + cell_size / 2
        y = margin_top + tick * cell_size + cell_size / 2
        parts.append(f"<text x='{x}' y='{margin_top - 10}' text-anchor='middle'>{tick}</text>")
        parts.append(f"<text x='{margin_left - 10}' y='{y + 3}' text-anchor='end'>{tick}</text>")

    legend_x = margin_left + cell_size * num_windows + 12
    for step in range(100):
        legend_y = margin_top + step * 2
        value = min_value + (max_value - min_value) * (1.0 - step / 99.0)
        parts.append(
            f"<rect x='{legend_x}' y='{legend_y}' width='10' height='2' fill='{color_for_value(value, min_value, max_value)}'/>"
        )
    parts.append(f"<text x='{legend_x + 16}' y='{margin_top + 8}'>{max_value:.3f}</text>")
    parts.append(f"<text x='{legend_x + 16}' y='{margin_top + 198}'>{min_value:.3f}</text>")
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def save_feature_svg(
    feature_map: torch.Tensor,
    output_path: Path,
) -> None:
    num_frames, num_bins = int(feature_map.shape[0]), int(feature_map.shape[1])
    if num_frames == 0 or num_bins == 0:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='80'><text x='20' y='45'>No features.</text></svg>",
            encoding="utf-8",
        )
        return

    min_value = float(feature_map.min().item())
    max_value = float(feature_map.max().item())
    x_scale = max(1, math.ceil(num_frames / 600))
    y_scale = max(1, math.ceil(num_bins / 240))
    width = math.ceil(num_frames / x_scale)
    height = math.ceil(num_bins / y_scale)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width + 80}' height='{height + 60}' viewBox='0 0 {width + 80} {height + 60}'>",
        "<style>text{font-family:monospace;font-size:10px;fill:#222;} .title{font-size:14px;font-weight:bold;}</style>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text class='title' x='60' y='24'>Audio Feature Map</text>",
    ]

    for frame_idx in range(0, num_frames, x_scale):
        for bin_idx in range(0, num_bins, y_scale):
            value = float(feature_map[frame_idx, bin_idx].item())
            fill = color_for_value(value, min_value, max_value)
            x = 60 + frame_idx // x_scale
            y = 40 + (num_bins - 1 - bin_idx) // y_scale
            parts.append(f"<rect x='{x}' y='{y}' width='1' height='1' fill='{fill}' stroke='none'/>")

    parts.append(f"<rect x='60' y='40' width='{width}' height='{height}' fill='none' stroke='#444' stroke-width='1'/>")
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def save_timeline_svg(
    segment_infos: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    total_frames: int,
    seconds_per_frame: float,
    output_path: Path,
) -> None:
    width = 1200
    margin_left = 96
    margin_top = 40
    base_row_height = 28
    pair_row_height = 24
    timeline_width = width - margin_left - 40
    total_seconds = total_frames * seconds_per_frame
    canvas_height = margin_top + base_row_height + max(1, len(pairs)) * pair_row_height + 56

    def frame_to_x(frame_idx: int) -> float:
        if total_frames <= 0:
            return float(margin_left)
        return margin_left + timeline_width * (frame_idx / total_frames)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{canvas_height}' viewBox='0 0 {width} {canvas_height}'>",
        "<style>text{font-family:monospace;font-size:11px;fill:#222;} .title{font-size:14px;font-weight:bold;}</style>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text class='title' x='96' y='22'>Repeat Timeline</text>",
    ]

    parts.append(f"<text x='12' y='{margin_top + 18}'>Segments</text>")
    segment_y = margin_top
    for segment_idx, segment in enumerate(segment_infos):
        x0 = frame_to_x(segment["frame_start"])
        x1 = frame_to_x(segment["frame_end"])
        width_px = max(1.0, x1 - x0)
        fill = "#d9d9d9" if segment_idx % 2 == 0 else "#bdbdbd"
        parts.append(
            f"<rect x='{x0:.2f}' y='{segment_y}' width='{width_px:.2f}' height='18' fill='{fill}' stroke='#666' stroke-width='0.5'/>"
        )
        if width_px >= 36:
            parts.append(
                f"<text x='{(x0 + x1) / 2:.2f}' y='{segment_y + 13}' text-anchor='middle'>{html_escape(segment['label'])}</text>"
            )

    parts.append(
        f"<line x1='{margin_left}' y1='{segment_y + 24}' x2='{margin_left + timeline_width}' y2='{segment_y + 24}' "
        "stroke='#444' stroke-width='1'/>"
    )

    tick_step_sec = max(1.0, total_seconds / 10.0) if total_seconds > 0 else 1.0
    tick_count = int(math.floor(total_seconds / tick_step_sec)) + 1
    for tick_idx in range(tick_count + 1):
        tick_sec = min(total_seconds, tick_idx * tick_step_sec)
        tick_x = frame_to_x(int(round(tick_sec / seconds_per_frame)))
        parts.append(
            f"<line x1='{tick_x:.2f}' y1='{segment_y + 24}' x2='{tick_x:.2f}' y2='{segment_y + 30}' stroke='#444' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{tick_x:.2f}' y='{segment_y + 42}' text-anchor='middle'>{format_seconds(tick_sec)}</text>"
        )

    start_y = segment_y + 56
    if not pairs:
        parts.append(f"<text x='12' y='{start_y + 12}'>No candidate pairs.</text>")
    for pair_idx, pair in enumerate(pairs):
        row_y = start_y + pair_idx * pair_row_height
        color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
        parts.append(
            f"<text x='12' y='{row_y + 12}'>#{pair_idx + 1} {pair['similarity']:.3f} len={pair['length_segments']}</text>"
        )
        for span_key in ("first", "second"):
            span = pair[span_key]
            x0 = frame_to_x(span["frame_start"])
            x1 = frame_to_x(span["frame_end"])
            parts.append(
                f"<rect x='{x0:.2f}' y='{row_y}' width='{max(1.0, x1 - x0):.2f}' height='14' "
                f"fill='{color}' fill-opacity='0.72' stroke='{color}' stroke-width='1'/>"
            )
        parts.append(
            f"<text x='{margin_left + timeline_width + 8}' y='{row_y + 12}'>"
            f"{html_escape(pair['first']['label_seq'])} &lt;-&gt; {html_escape(pair['second']['label_seq'])}</text>"
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def save_pairs_tsv(pairs: List[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        "\t".join(
            [
                "rank",
                "similarity",
                "span_ratio",
                "window_i",
                "window_j",
                "length_segments",
                "start_sec_i",
                "end_sec_i",
                "start_sec_j",
                "end_sec_j",
                "labels_i",
                "labels_j",
            ]
        )
    ]
    for rank, pair in enumerate(pairs, start=1):
        lines.append(
            "\t".join(
                [
                    str(rank),
                    f"{pair['similarity']:.6f}",
                    f"{pair.get('span_ratio', 1.0):.6f}",
                    str(pair["window_i"]),
                    str(pair["window_j"]),
                    str(pair["length_segments"]),
                    f"{pair['first']['start_sec']:.6f}",
                    f"{pair['first']['end_sec']:.6f}",
                    f"{pair['second']['start_sec']:.6f}",
                    f"{pair['second']['end_sec']:.6f}",
                    pair["first"]["label_seq"],
                    pair["second"]["label_seq"],
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_segments_tsv(segment_infos: List[Dict[str, Any]], output_path: Path) -> None:
    lines = ["segment_idx\tframe_start\tframe_end\tstart_sec\tend_sec\tlabel"]
    for segment in segment_infos:
        lines.append(
            "\t".join(
                [
                    str(segment["segment_idx"]),
                    str(segment["frame_start"]),
                    str(segment["frame_end"]),
                    f"{segment['start_sec']:.6f}",
                    f"{segment['end_sec']:.6f}",
                    segment["label"],
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_windows_tsv(windows: List[Dict[str, Any]], output_path: Path) -> None:
    lines = ["window_idx\tframe_start\tframe_end\tstart_sec\tend_sec\tsegment_start\tsegment_end\tlabels"]
    for window in windows:
        lines.append(
            "\t".join(
                [
                    str(window["window_idx"]),
                    str(window["frame_start"]),
                    str(window["frame_end"]),
                    f"{window['start_sec']:.6f}",
                    f"{window['end_sec']:.6f}",
                    str(window["segment_start"]),
                    str(window["segment_end"]),
                    window["label_seq"],
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report_html(
    summary: Dict[str, Any],
    pairs: List[Dict[str, Any]],
    output_path: Path,
    similarity_svg_name: str,
    timeline_svg_name: str,
    feature_svg_name: str,
) -> None:
    rows = []
    for rank, pair in enumerate(pairs, start=1):
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{pair['similarity']:.3f}</td>"
            f"<td>{pair.get('span_ratio', 1.0):.2f}</td>"
            f"<td>{pair['length_segments']}</td>"
            f"<td>{html_escape(format_seconds(pair['first']['start_sec']))} - {html_escape(format_seconds(pair['first']['end_sec']))}</td>"
            f"<td>{html_escape(format_seconds(pair['second']['start_sec']))} - {html_escape(format_seconds(pair['second']['end_sec']))}</td>"
            f"<td>{html_escape(pair['first']['label_seq'])}</td>"
            f"<td>{html_escape(pair['second']['label_seq'])}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Repeat SSM Debug</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; color: #222; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    section {{ margin-top: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
    code, pre {{ background: #f6f8fa; padding: 2px 4px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Repeat SSM Debug</h1>
  <pre>{html_escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>

  <section>
    <h2>Audio Feature</h2>
    <img src="{html_escape(feature_svg_name)}" alt="audio feature map" />
  </section>

  <section>
    <h2>Similarity</h2>
    <img src="{html_escape(similarity_svg_name)}" alt="repeat ssm" />
  </section>

  <section>
    <h2>Timeline</h2>
    <img src="{html_escape(timeline_svg_name)}" alt="repeat timeline" />
  </section>

  <section>
    <h2>Top Pairs</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>sim</th>
          <th>ratio</th>
          <th>len</th>
          <th>span A</th>
          <th>span B</th>
          <th>labels A</th>
          <th>labels B</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows) if rows else '<tr><td colspan="8">No candidate pairs.</td></tr>'}
      </tbody>
    </table>
  </section>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def build_segment_infos(
    repeat_output: Any,
    root_chord_index: torch.Tensor,
    bass_index: torch.Tensor,
    root_chord_labels: Sequence[str],
    seconds_per_frame: float,
    segment_offset_sec: float,
) -> List[Dict[str, Any]]:
    segment_infos: List[Dict[str, Any]] = []
    num_segments = int(repeat_output.segment_counts[0].item())
    for segment_idx in range(num_segments):
        frame_start = int(repeat_output.segment_starts[0, segment_idx].item())
        frame_end = int(repeat_output.segment_ends[0, segment_idx].item())
        label = format_chord_label(
            root_chord_idx=int(root_chord_index[frame_start].item()),
            bass_idx=int(bass_index[frame_start].item()),
            root_chord_labels=root_chord_labels,
        )
        segment_infos.append(
            {
                "segment_idx": segment_idx,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "start_sec": segment_offset_sec + frame_start * seconds_per_frame,
                "end_sec": segment_offset_sec + frame_end * seconds_per_frame,
                "label": label,
            }
        )
    return segment_infos


def build_window_infos(
    repeat_output: Any,
    segment_infos: List[Dict[str, Any]],
    seconds_per_frame: float,
    segment_offset_sec: float,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    num_windows = int(repeat_output.window_counts[0].item())
    for window_idx in range(num_windows):
        segment_start = int(repeat_output.window_segment_starts[0, window_idx].item())
        segment_end = int(repeat_output.window_segment_ends[0, window_idx].item())
        frame_start = int(repeat_output.window_frame_starts[0, window_idx].item())
        frame_end = int(repeat_output.window_frame_ends[0, window_idx].item())
        label_seq = " | ".join(segment["label"] for segment in segment_infos[segment_start:segment_end])
        windows.append(
            {
                "window_idx": window_idx,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "start_sec": segment_offset_sec + frame_start * seconds_per_frame,
                "end_sec": segment_offset_sec + frame_end * seconds_per_frame,
                "label_seq": label_seq,
            }
        )
    return windows


def build_sequence_info(
    segment_infos: List[Dict[str, Any]],
    segment_start: int,
    segment_end: int,
) -> Dict[str, Any]:
    segments = segment_infos[segment_start:segment_end]
    if not segments:
        raise ValueError("segment range must not be empty")
    return {
        "segment_start": segment_start,
        "segment_end": segment_end,
        "frame_start": segments[0]["frame_start"],
        "frame_end": segments[-1]["frame_end"],
        "start_sec": segments[0]["start_sec"],
        "end_sec": segments[-1]["end_sec"],
        "label_seq": " | ".join(segment["label"] for segment in segments),
        "length_segments": segment_end - segment_start,
    }


def compute_span_ratio_frames(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    length_first = max(1, int(first["frame_end"]) - int(first["frame_start"]))
    length_second = max(1, int(second["frame_end"]) - int(second["frame_start"]))
    shorter = min(length_first, length_second)
    longer = max(length_first, length_second)
    return float(longer) / float(shorter)


def filter_pairs_by_span_ratio(
    pairs: List[Dict[str, Any]],
    max_span_ratio: float,
) -> List[Dict[str, Any]]:
    if max_span_ratio < 1.0:
        raise ValueError("max_span_ratio must be >= 1.0")

    filtered_pairs: List[Dict[str, Any]] = []
    for pair in pairs:
        span_ratio = compute_span_ratio_frames(pair["first"], pair["second"])
        pair["span_ratio"] = span_ratio
        if span_ratio <= max_span_ratio:
            filtered_pairs.append(pair)
    return filtered_pairs


def build_adaptive_pairs(
    repeat_output: Any,
    segment_infos: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    ChordWindowSSM が返した adaptive repeat を、そのまま可視化用 dict に変換する。

    `window_i/window_j` には fixed window index ではなく、repeat 区間の segment start を入れる。
    """
    pairs: List[Dict[str, Any]] = []
    num_pairs = int(repeat_output.repeat_pair_counts[0].item())
    if num_pairs == 0:
        return pairs

    pair_segment_starts = repeat_output.repeat_pairs_segment_starts[0, :num_pairs]
    pair_segment_ends = repeat_output.repeat_pairs_segment_ends[0, :num_pairs]
    pair_lengths = repeat_output.repeat_pairs_lengths[0, :num_pairs]
    pair_similarity = repeat_output.repeat_pairs_similarity[0, :num_pairs]
    pair_span_ratio = repeat_output.repeat_pairs_span_ratio[0, :num_pairs]

    for pair_idx in range(num_pairs):
        segment_start_i = int(pair_segment_starts[pair_idx, 0].item())
        segment_start_j = int(pair_segment_starts[pair_idx, 1].item())
        segment_end_i = int(pair_segment_ends[pair_idx, 0].item())
        segment_end_j = int(pair_segment_ends[pair_idx, 1].item())
        length_segments = int(pair_lengths[pair_idx].item())
        pairs.append(
            {
                "window_i": segment_start_i,
                "window_j": segment_start_j,
                "similarity": float(pair_similarity[pair_idx].item()),
                "span_ratio": float(pair_span_ratio[pair_idx].item()),
                "first": build_sequence_info(segment_infos, segment_start_i, segment_end_i),
                "second": build_sequence_info(segment_infos, segment_start_j, segment_end_j),
                "length_segments": length_segments,
            }
        )

    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return pairs


def _compute_sparse_pair_similarity(
    segment_features: torch.Tensor,
    segment_start_i: int,
    segment_start_j: int,
    length_segments: int,
) -> float:
    if length_segments <= 0:
        return 0.0
    sequence_i = segment_features[segment_start_i : segment_start_i + length_segments].reshape(1, -1)
    sequence_j = segment_features[segment_start_j : segment_start_j + length_segments].reshape(1, -1)
    similarity = F.cosine_similarity(sequence_i, sequence_j, dim=-1, eps=1e-8)
    return float(similarity.squeeze(0).item())


def build_adaptive_label_rescored_pairs(
    pair_output: Any,
    segment_infos: List[Dict[str, Any]],
    segment_features: torch.Tensor,
) -> List[Dict[str, Any]]:
    """
    GT ラベルだけで repeat pair を作り、その sparse pair に対してだけ audio 類似度を計算する。

    学習に入れたい `label-only pair generation + sparse audio re-score` を可視化側で再現する。
    """
    pairs: List[Dict[str, Any]] = []
    num_pairs = int(pair_output.repeat_pair_counts[0].item())
    if num_pairs == 0:
        return pairs

    pair_segment_starts = pair_output.repeat_pairs_segment_starts[0, :num_pairs]
    pair_segment_ends = pair_output.repeat_pairs_segment_ends[0, :num_pairs]
    pair_lengths = pair_output.repeat_pairs_lengths[0, :num_pairs]
    pair_span_ratio = pair_output.repeat_pairs_span_ratio[0, :num_pairs]

    for pair_idx in range(num_pairs):
        segment_start_i = int(pair_segment_starts[pair_idx, 0].item())
        segment_start_j = int(pair_segment_starts[pair_idx, 1].item())
        segment_end_i = int(pair_segment_ends[pair_idx, 0].item())
        segment_end_j = int(pair_segment_ends[pair_idx, 1].item())
        length_segments = int(pair_lengths[pair_idx].item())
        similarity = _compute_sparse_pair_similarity(
            segment_features=segment_features,
            segment_start_i=segment_start_i,
            segment_start_j=segment_start_j,
            length_segments=length_segments,
        )
        pairs.append(
            {
                "window_i": segment_start_i,
                "window_j": segment_start_j,
                "similarity": similarity,
                "span_ratio": float(pair_span_ratio[pair_idx].item()),
                "first": build_sequence_info(segment_infos, segment_start_i, segment_end_i),
                "second": build_sequence_info(segment_infos, segment_start_j, segment_end_j),
                "length_segments": length_segments,
            }
        )

    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return pairs


def build_fixed_window_pairs(
    repeat_output: Any,
    windows: List[Dict[str, Any]],
    label_filter: str,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    num_windows = int(repeat_output.window_counts[0].item())
    similarity = repeat_output.similarity[0, :num_windows, :num_windows]
    candidate_mask = repeat_output.candidate_mask[0, :num_windows, :num_windows]
    window_labels = repeat_output.window_labels[0, :num_windows]

    indices = torch.nonzero(torch.triu(candidate_mask, diagonal=1), as_tuple=False)
    for pair_indices in indices.tolist():
        window_i, window_j = int(pair_indices[0]), int(pair_indices[1])
        if label_filter == "exact" and not torch.equal(window_labels[window_i], window_labels[window_j]):
            continue

        pairs.append(
            {
                "window_i": window_i,
                "window_j": window_j,
                "similarity": float(similarity[window_i, window_j].item()),
                "first": windows[window_i],
                "second": windows[window_j],
                "length_segments": windows[window_i]["segment_end"] - windows[window_i]["segment_start"],
            }
        )

    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return pairs


def build_candidate_pairs(
    repeat_output: Any,
    windows: List[Dict[str, Any]],
    segment_infos: List[Dict[str, Any]],
    max_pairs: int,
    label_filter: str,
    window_mode: str,
    max_span_ratio: float,
    pair_output: Any | None = None,
) -> List[Dict[str, Any]]:
    if window_mode == "adaptive_exact":
        pairs = build_adaptive_pairs(
            repeat_output=repeat_output,
            segment_infos=segment_infos,
        )
    elif window_mode == "adaptive_label_rescore":
        if pair_output is None:
            raise ValueError("pair_output is required for adaptive_label_rescore")
        segment_count = int(repeat_output.segment_counts[0].item())
        if int(pair_output.segment_counts[0].item()) != segment_count:
            raise ValueError("pair_output and repeat_output must share the same segment count")
        pairs = build_adaptive_label_rescored_pairs(
            pair_output=pair_output,
            segment_infos=segment_infos,
            segment_features=repeat_output.segment_features[0, :segment_count],
        )
    else:
        pairs = build_fixed_window_pairs(
            repeat_output=repeat_output,
            windows=windows,
            label_filter=label_filter,
        )
        pairs = filter_pairs_by_span_ratio(pairs, max_span_ratio=max_span_ratio)
    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return pairs[:max_pairs]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    phase_config = resolve_phase_config(config, args.phase)
    segment_seconds = float(args.segment_seconds or phase_config["segment_seconds"])

    dataset, label_processor = build_dataset(config, args.split, segment_seconds)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index out of range: {args.index} (dataset length={len(dataset)})")

    device = torch.device(args.device)
    sample = dataset[args.index]
    record, segment_start_sec = dataset._get_segment_info(args.index)

    target_samples = int(sample["target_samples"].item())
    selected_stems = resolve_selected_stems(dataset.stem_order, args.audio_stems)

    sample_rate = int(config["data_loader"]["sample_rate"])
    hop_length = int(config["model"]["backbone"]["hop_length"])
    n_fft = int(config["model"]["backbone"]["n_fft"])
    crop_length = crop_length_from_num_samples(target_samples, n_fft=n_fft, hop_length=hop_length)
    seconds_per_frame = label_processor.hop_sec

    if args.feature_type in {"stem_chroma", "stem_cens"}:
        stem_waveforms = load_selected_stem_tensor(
            dataset=dataset,
            record=record,
            segment_start_sec=segment_start_sec,
            segment_seconds=segment_seconds,
            target_samples=target_samples,
            selected_stems=selected_stems,
        ).to(device)
        frame_features, feature_map = compute_stem_aggregated_audio_features(
            stem_waveforms=stem_waveforms,
            feature_type=args.feature_type,
            sample_rate=sample_rate,
            hop_length=hop_length,
            crop_length=crop_length,
            cqt_bins_per_octave=args.cqt_bins_per_octave,
            cqt_n_bins=args.cqt_n_bins,
            n_chroma=args.n_chroma,
            log_amplitude=not args.no_log_amplitude,
            normalize=not args.no_normalize,
        )
    else:
        waveform = load_selected_stem_waveform(
            dataset=dataset,
            record=record,
            segment_start_sec=segment_start_sec,
            segment_seconds=segment_seconds,
            target_samples=target_samples,
            selected_stems=selected_stems,
        ).to(device)
        frame_features, feature_map = compute_audio_features(
            waveform=waveform,
            feature_type=args.feature_type,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            crop_length=crop_length,
            cqt_bins_per_octave=args.cqt_bins_per_octave,
            cqt_n_bins=args.cqt_n_bins,
            n_mels=args.n_mels,
            n_chroma=args.n_chroma,
            log_amplitude=not args.no_log_amplitude,
            normalize=not args.no_normalize,
        )

    root_chord_index = sample["root_chord_index"][:crop_length].to(torch.long)
    bass_index = sample["bass_index"][:crop_length].to(torch.long)
    ignore_index = label_processor.ignore_index

    if args.label_mode == "full_chord":
        chord_labels = combine_full_chord_labels(
            root_chord_index=root_chord_index,
            bass_index=bass_index,
            num_bass_classes=int(config["model"]["num_bass_classes"]),
            ignore_index=ignore_index,
        )
    else:
        chord_labels = root_chord_index.clone()

    repeat_ssm = ChordWindowSSM(
        window_size=args.window_size,
        window_hop=args.window_hop,
        similarity="cosine",
        min_segment_frames=args.min_segment_frames,
        min_window_frames=args.min_window_frames,
        max_span_ratio=args.max_span_ratio,
        exclude_neighbor_windows=args.exclude_neighbor_windows,
        similarity_threshold=args.similarity_threshold,
        topk=args.topk,
        ignore_index=ignore_index,
    ).to(device)
    repeat_ssm.eval()
    pair_output = None

    with torch.inference_mode():
        repeat_output = repeat_ssm(
            frame_features=frame_features,
            chord_labels=chord_labels.unsqueeze(0).to(device),
        )
        if args.window_mode == "adaptive_label_rescore":
            pair_builder = RepeatPairBuilderCPU(
                window_size=args.window_size,
                max_span_ratio=args.max_span_ratio,
                ignore_index=ignore_index,
            )
            pair_output = pair_builder(chord_labels.unsqueeze(0))

    quality_labels = load_quality_labels_from_json(
        config["data"]["quality_json_path"],
        int(config["model"]["num_quality_classes"]),
    )
    root_chord_labels = build_root_chord_labels(
        num_root_classes=int(config["model"]["num_root_classes"]),
        quality_labels=quality_labels,
    )

    segment_infos = build_segment_infos(
        repeat_output=repeat_output,
        root_chord_index=root_chord_index,
        bass_index=bass_index,
        root_chord_labels=root_chord_labels,
        seconds_per_frame=seconds_per_frame,
        segment_offset_sec=segment_start_sec,
    )
    windows = build_window_infos(
        repeat_output=repeat_output,
        segment_infos=segment_infos,
        seconds_per_frame=seconds_per_frame,
        segment_offset_sec=segment_start_sec,
    )
    pairs = build_candidate_pairs(
        repeat_output=repeat_output,
        windows=windows,
        segment_infos=segment_infos,
        max_pairs=args.max_pairs,
        label_filter=args.label_filter,
        window_mode=args.window_mode,
        max_span_ratio=args.max_span_ratio,
        pair_output=pair_output,
    )

    output_stem = make_output_stem(
        args.split,
        args.index,
        args.feature_type,
        args.label_mode,
        args.window_mode,
        args.audio_stems,
    )
    output_dir = args.output_dir / output_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    similarity_svg_path = output_dir / "similarity.svg"
    timeline_svg_path = output_dir / "timeline.svg"
    feature_svg_path = output_dir / "audio_feature.svg"
    pairs_tsv_path = output_dir / "pairs.tsv"
    segments_tsv_path = output_dir / "segments.tsv"
    windows_tsv_path = output_dir / "windows.tsv"
    summary_json_path = output_dir / "summary.json"
    report_html_path = output_dir / "report.html"

    num_windows = int(repeat_output.window_counts[0].item())
    similarity = repeat_output.similarity[0, :num_windows, :num_windows].detach().cpu()
    candidate_mask = repeat_output.candidate_mask[0, :num_windows, :num_windows].detach().cpu()
    feature_map = feature_map[:crop_length].detach().cpu()

    save_similarity_svg(similarity=similarity, candidate_mask=candidate_mask, output_path=similarity_svg_path)
    save_timeline_svg(
        segment_infos=segment_infos,
        pairs=pairs,
        total_frames=crop_length,
        seconds_per_frame=seconds_per_frame,
        output_path=timeline_svg_path,
    )
    save_feature_svg(feature_map=feature_map, output_path=feature_svg_path)
    save_pairs_tsv(pairs=pairs, output_path=pairs_tsv_path)
    save_segments_tsv(segment_infos=segment_infos, output_path=segments_tsv_path)
    save_windows_tsv(windows=windows, output_path=windows_tsv_path)

    summary = {
        "config": str(args.config.resolve()),
        "split": args.split,
        "index": args.index,
        "phase": args.phase,
        "segment_seconds": segment_seconds,
        "feature_type": args.feature_type,
        "audio_stems": args.audio_stems,
        "selected_stems": list(selected_stems),
        "label_mode": args.label_mode,
        "window_mode": args.window_mode,
        "window_size": args.window_size,
        "window_hop": args.window_hop,
        "topk": args.topk,
        "label_filter": args.label_filter,
        "max_span_ratio": args.max_span_ratio,
        "similarity_threshold": args.similarity_threshold,
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "crop_length_frames": crop_length,
        "seconds_per_frame": seconds_per_frame,
        "segment_start_sec": segment_start_sec,
        "record": {
            "stems_dir": record.get("stems_dir"),
            "chord_label_path": record.get("chord_label_path"),
            "key_label_path": record.get("key_label_path"),
        },
        "num_segments": len(segment_infos),
        "num_windows": len(windows),
        "num_pairs": len(pairs),
        "pair_generation": "cpu_label_only" if args.window_mode == "adaptive_label_rescore" else "repeat_ssm",
        "pair_scoring": "sparse_audio_sequence_cosine" if args.window_mode == "adaptive_label_rescore" else "repeat_ssm",
        "similarity_visualization": "fixed_window_debug_ssm",
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_report_html(
        summary=summary,
        pairs=pairs,
        output_path=report_html_path,
        similarity_svg_name=similarity_svg_path.name,
        timeline_svg_name=timeline_svg_path.name,
        feature_svg_name=feature_svg_path.name,
    )

    print(f"[OK] report: {report_html_path}")
    print(f"[OK] similarity: {similarity_svg_path}")
    print(f"[OK] timeline: {timeline_svg_path}")
    print(f"[OK] feature: {feature_svg_path}")
    print(f"[OK] pairs: {pairs_tsv_path}")


if __name__ == "__main__":
    main()
