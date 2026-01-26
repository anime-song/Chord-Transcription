#!/usr/bin/env python3
"""
Evaluate a trained transcription model on the full-length validation songs.

Outputs
-------
- JSON report (overall + per-song accuracies only).
- Optional PNG for the quality confusion matrix (raw counts + row-normalised).
- Per-song chord label TSVs (`start_time\tend_time\tchord`).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import re

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from dlchordx import Tone  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from src.data.dataset import read_chords_jsonl, read_tsv  # noqa: E402
from src.data.processing import ChordEvent, EventSpan  # noqa: E402
from src.utils import build_label_processor, build_model_from_config, load_config  # noqa: E402

PITCH_CLASS_LABELS_13: List[str] = [
    "N",
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

TASK_SPECS: Dict[str, Dict[str, str]] = {
    "root": {"head": "initial_root_logits", "target": "root_index"},
    "bass": {"head": "initial_bass_logits", "target": "bass_index"},
    "root_chord": {"head": "initial_root_chord_logits", "target": "root_chord_index"},
    "quality": {"head": "initial_quality_logits", "target": "quality_index"},
    "key": {"head": "initial_key_logits", "target": "key_index"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-length validation evaluator.")
    parser.add_argument("--config", required=True, help="Training config YAML (same as used for training).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file (EMA weights recommended).")
    parser.add_argument(
        "--output-json", default="results/validation_full.json", help="Path to save the evaluation JSON."
    )
    parser.add_argument(
        "--quality-confusion-path",
        default=None,
        help="Path to save the quality confusion matrix PNG (omit to skip plotting).",
    )
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda or cpu (defaults to auto).")
    parser.add_argument(
        "--validation-jsonl",
        default=None,
        help="Override validation JSONL path (defaults to config.data.valid_jsonl_path).",
    )
    parser.add_argument(
        "--chord-output-dir",
        default="results/predicted_chords",
        help="Directory to store per-song chord label TSV files.",
    )
    return parser.parse_args()


def load_full_stems(
    stems_dir: Path,
    sample_rate: int,
    stem_order: Iterable[str],
) -> np.ndarray:
    """Load every stem for the song, resample to `sample_rate`, and stack into (channels, samples)."""
    waves: List[np.ndarray] = []
    max_len = 0

    for stem in stem_order:
        base_name = stems_dir.name
        stem_path = stems_dir / f"{base_name}_{stem}.wav"
        if not stem_path.exists():
            raise FileNotFoundError(f"Stem file not found: {stem_path}")
        audio, sr = sf.read(stem_path, dtype="float32", always_2d=True)
        audio = audio.T  # (channels, samples)

        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)  # enforce stereo per stem

        waves.append(audio)
        max_len = max(max_len, audio.shape[1])

    padded: List[np.ndarray] = []
    for w in waves:
        if w.shape[1] < max_len:
            pad = max_len - w.shape[1]
            w = np.pad(w, ((0, 0), (0, pad)))
        else:
            w = w[:, :max_len]
        padded.append(w)

    return np.concatenate(padded, axis=0)


def tone_to_index(note: str) -> int:
    if note == "N":
        return 0
    try:
        return int(Tone(note).get_interval()) + 1
    except Exception:
        return 0


def chord_events_to_index_spans(
    events: List[ChordEvent],
    quality_to_index: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    root_spans = [{"start_time": e.start_time, "end_time": e.end_time, "idx": tone_to_index(e.root)} for e in events]
    bass_spans = [{"start_time": e.start_time, "end_time": e.end_time, "idx": tone_to_index(e.bass)} for e in events]
    quality_spans = [
        {"start_time": e.start_time, "end_time": e.end_time, "idx": quality_to_index.get(e.quality, 0)} for e in events
    ]
    return root_spans, bass_spans, quality_spans


def events_to_key_spans(events: List[EventSpan], key_map: Dict[str, int]) -> List[Dict[str, Any]]:
    return [{"start_time": e.start_time, "end_time": e.end_time, "idx": key_map.get(e.label, 0)} for e in events]


def build_quality_maps(quality_json: Path, num_quality_classes: int) -> Tuple[Dict[str, int], List[str], int]:
    with quality_json.open("r", encoding="utf-8") as f:
        index_to_quality = json.load(f)

    quality_labels: List[str] = []
    for idx in range(num_quality_classes):
        key = str(idx)
        if key not in index_to_quality:
            raise KeyError(f"quality label index {idx} missing in {quality_json}")
        quality_labels.append(index_to_quality[key])

    quality_to_index = {label: idx for idx, label in enumerate(quality_labels)}
    try:
        non_chord_index = next(idx for idx, label in enumerate(quality_labels) if label == "N")
    except StopIteration as exc:
        raise ValueError("quality labels must contain 'N'.") from exc
    return quality_to_index, quality_labels, non_chord_index


def build_root_chord_labels(num_root_classes: int, quality_labels: List[str]) -> List[str]:
    quality_slots = [label for label in quality_labels if label != "N"]
    labels: List[str] = []
    for root_idx in range(1, num_root_classes):
        root_name = PITCH_CLASS_LABELS_13[root_idx]
        for quality in quality_slots:
            labels.append(f"{root_name}{quality}")
    labels.append("N")
    expected = (num_root_classes - 1) * len(quality_slots) + 1
    if len(labels) != expected:
        raise ValueError(
            f"root_chord label count mismatch (expected {expected}, got {len(labels)}). "
            "Check model.num_root_classes / num_quality_classes."
        )
    return labels


def combine_root_quality(
    root_frames: np.ndarray,
    quality_frames: np.ndarray,
    quality_idx_lookup: np.ndarray,
    quality_non_chord_idx: int,
    num_quality_without_nc: int,
    num_root_without_n: int,
    ignore_index: int,
) -> np.ndarray:
    combined = np.full(root_frames.shape, ignore_index, dtype=np.int64)
    valid_mask = (root_frames != ignore_index) & (quality_frames != ignore_index)
    if not np.any(valid_mask):
        return combined

    root_valid = root_frames[valid_mask]
    quality_valid = quality_frames[valid_mask]
    slot_values = quality_idx_lookup[quality_valid]
    no_chord = (root_valid == 0) | (quality_valid == quality_non_chord_idx) | (slot_values == -1)

    combined_values = np.full(root_valid.shape, num_quality_without_nc * num_root_without_n, dtype=np.int64)
    valid_combo = ~no_chord
    if np.any(valid_combo):
        root_offsets = np.clip(root_valid[valid_combo] - 1, 0, None)
        combined_values[valid_combo] = root_offsets * num_quality_without_nc + slot_values[valid_combo]

    combined[valid_mask] = combined_values
    return combined


def plot_confusion_matrices(counts: np.ndarray, normalized: np.ndarray, labels: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(14.0, len(labels) * 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.6), sharey=True)

    im_counts = axes[0].imshow(counts, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Counts")
    fig.colorbar(im_counts, ax=axes[0], fraction=0.046, pad=0.04)

    im_norm = axes[1].imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[1].set_title("Row-normalised")
    fig.colorbar(im_norm, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6, ha="center")
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def indices_to_labels(indices: Iterable[int], vocab: List[str], ignore_index: int) -> List[str]:
    labels: List[str] = []
    for idx in indices:
        value = int(idx)
        if value == ignore_index:
            labels.append("IGNORE")
        elif 0 <= value < len(vocab):
            labels.append(vocab[value])
        else:
            labels.append("UNK")
    return labels


def compute_accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return correct / total


def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", name)


def _normalize_label(label: str) -> str:
    normalized = (label or "").strip()
    if normalized in {"", "IGNORE", "UNK", "N/A"}:
        return "N"
    return normalized


def _extract_root(label: str) -> str:
    if len(label) >= 2 and label[1] in {"#", "b"}:
        return label[:2]
    return label[:1]


def build_chord_events(
    frame_times: np.ndarray,
    hop_seconds: float,
    root_chord_labels: List[str],
    bass_labels: List[str],
) -> List[Dict[str, Any]]:
    num_frames = len(frame_times)
    if num_frames == 0:
        return []

    frame_duration = hop_seconds if num_frames == 1 else float(frame_times[1] - frame_times[0])

    def format_chord(root_label: str, bass_label: str) -> str:
        root_norm = _normalize_label(root_label)
        bass_norm = _normalize_label(bass_label)
        if root_norm == "N":
            return "N"
        if bass_norm == "N":
            return root_norm
        if _extract_root(root_norm) == bass_norm:
            return root_norm
        return f"{root_norm}/{bass_norm}"

    current_label = format_chord(root_chord_labels[0], bass_labels[0])
    segment_start = float(frame_times[0])
    events: List[Dict[str, Any]] = []

    for i in range(1, num_frames):
        next_label = format_chord(root_chord_labels[i], bass_labels[i])
        if next_label != current_label:
            events.append(
                {
                    "start_time": segment_start,
                    "end_time": float(frame_times[i]),
                    "chord": current_label,
                }
            )
            segment_start = float(frame_times[i])
            current_label = next_label

    end_time = float(frame_times[-1] + frame_duration)
    events.append({"start_time": segment_start, "end_time": end_time, "chord": current_label})
    return events


def write_chord_events(events: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(f"{event['start_time']:.6f}\t{event['end_time']:.6f}\t{event['chord']}\n")


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    label_processor = build_label_processor(config)
    ignore_index = label_processor.ignore_index
    hop_seconds = label_processor.hop_sec

    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'ema_state_dict' or 'model_state_dict'.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading: {unexpected}")
    model.eval()

    data_cfg = config["data"]
    loader_cfg = config["data_loader"]
    sample_rate = int(loader_cfg["sample_rate"])
    stem_order = loader_cfg["stem_order"]

    valid_jsonl = Path(args.validation_jsonl or data_cfg["valid_jsonl_path"])
    with valid_jsonl.open("r", encoding="utf-8") as f:
        validation_records = [json.loads(line) for line in f if line.strip()]

    num_quality_classes = int(config["model"]["num_quality_classes"])
    num_root_classes = int(config["model"]["num_root_classes"])

    quality_to_index, quality_labels, quality_non_chord_idx = build_quality_maps(
        Path(data_cfg["quality_json_path"]),
        num_quality_classes,
    )
    root_chord_labels = build_root_chord_labels(num_root_classes, quality_labels)

    quality_idx_lookup = np.full((max(quality_to_index.values()) + 1,), -1, dtype=np.int64)
    slot = 0
    for quality_label, idx in sorted(quality_to_index.items(), key=lambda item: item[1]):
        if idx == quality_non_chord_idx:
            continue
        if idx >= quality_idx_lookup.shape[0]:
            quality_idx_lookup.resize(idx + 1, refcheck=False)
            quality_idx_lookup[idx] = -1
        quality_idx_lookup[idx] = slot
        slot += 1

    num_quality_without_nc = num_quality_classes - 1
    num_root_without_n = num_root_classes - 1

    key_to_index = {
        "N": 0,
        "C": 1,
        "Db": 2,
        "D": 3,
        "Eb": 4,
        "E": 5,
        "F": 6,
        "Gb": 7,
        "G": 8,
        "Ab": 9,
        "A": 10,
        "Bb": 11,
        "B": 12,
        "Cm": 4,
        "C#m": 5,
        "Dm": 6,
        "Ebm": 7,
        "Em": 8,
        "Fm": 9,
        "F#m": 10,
        "Gm": 11,
        "G#m": 12,
        "Am": 1,
        "Bbm": 2,
        "Bm": 3,
    }

    chord_output_dir = Path(args.chord_output_dir)
    chord_output_dir.mkdir(parents=True, exist_ok=True)

    per_song_results: List[Dict[str, Any]] = []
    overall_counts = {task: {"correct": 0, "total": 0} for task in TASK_SPECS}
    quality_conf_counts = np.zeros((num_quality_classes, num_quality_classes), dtype=np.int64)

    iterator = validation_records
    if tqdm:
        iterator = tqdm(validation_records, desc="Evaluating songs", unit="song")

    with torch.inference_mode():
        for record in iterator:
            basename = record.get("basename") or Path(record["stems_dir"]).name
            stems_dir = Path(record["stems_dir"])

            waveform = load_full_stems(stems_dir, sample_rate, stem_order)
            num_samples = waveform.shape[-1]
            if num_samples < label_processor.n_fft:
                raise ValueError(f"Audio too short for STFT (basename={basename}).")

            chord_events = read_chords_jsonl(Path(record["chord_label_path"]))
            key_events = read_tsv(Path(record["key_label_path"]))

            num_frames = label_processor.get_num_frames(num_samples)

            root_spans, bass_spans, quality_spans = chord_events_to_index_spans(chord_events, quality_to_index)
            key_spans = events_to_key_spans(key_events, key_to_index)

            root_frames = label_processor.spans_to_frames(root_spans, num_frames, seg_start_sec=0.0)
            bass_frames = label_processor.spans_to_frames(bass_spans, num_frames, seg_start_sec=0.0)
            quality_frames = label_processor.spans_to_frames(quality_spans, num_frames, seg_start_sec=0.0)
            key_frames = label_processor.spans_to_frames(key_spans, num_frames, seg_start_sec=0.0)

            root_chord_frames = combine_root_quality(
                root_frames,
                quality_frames,
                quality_idx_lookup,
                quality_non_chord_idx,
                num_quality_without_nc,
                num_root_without_n,
                ignore_index,
            )

            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device=device, dtype=torch.float32)
            outputs = model(waveform_tensor)

            task_predictions: Dict[str, np.ndarray] = {}
            task_targets: Dict[str, np.ndarray] = {
                "root": root_frames,
                "bass": bass_frames,
                "quality": quality_frames,
                "key": key_frames,
                "root_chord": root_chord_frames,
            }

            song_counts = defaultdict(lambda: {"correct": 0, "total": 0})

            for task, spec in TASK_SPECS.items():
                logits = outputs[spec["head"]].squeeze(0).detach().cpu()
                preds = logits.argmax(dim=-1).numpy().astype(np.int64)
                task_predictions[task] = preds

                targets = task_targets[task]
                mask = targets != ignore_index
                total = int(mask.sum())
                correct = int((preds[mask] == targets[mask]).sum())
                song_counts[task]["correct"] += correct
                song_counts[task]["total"] += total
                overall_counts[task]["correct"] += correct
                overall_counts[task]["total"] += total

            mask_quality = quality_frames != ignore_index
            if np.any(mask_quality):
                true_indices = quality_frames[mask_quality]
                pred_indices = task_predictions["quality"][mask_quality]
                np.add.at(quality_conf_counts, (true_indices, pred_indices), 1)

            frame_times = np.arange(num_frames, dtype=np.float64) * hop_seconds
            metrics = {
                task: compute_accuracy(song_counts[task]["correct"], song_counts[task]["total"]) for task in TASK_SPECS
            }

            root_chord_label_seq = indices_to_labels(task_predictions["root_chord"], root_chord_labels, ignore_index)
            bass_label_seq = indices_to_labels(task_predictions["bass"], PITCH_CLASS_LABELS_13, ignore_index)
            chord_events = build_chord_events(frame_times, hop_seconds, root_chord_label_seq, bass_label_seq)

            chord_filename = sanitize_filename(basename) + ".chords.txt"
            chord_path = chord_output_dir / chord_filename
            write_chord_events(chord_events, chord_path)

            per_song_results.append(
                {
                    "basename": basename,
                    "frame_count": int(num_frames),
                    "metrics": metrics,
                    "chord_path": str(chord_path.resolve()),
                }
            )

    overall_metrics = {
        task: compute_accuracy(overall_counts[task]["correct"], overall_counts[task]["total"]) for task in TASK_SPECS
    }
    overall_counts_out = {
        task: {"correct": int(values["correct"]), "total": int(values["total"])}
        for task, values in overall_counts.items()
    }

    quality_conf_norm = quality_conf_counts.astype(np.float64)
    row_sums = quality_conf_norm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        quality_conf_norm = np.divide(
            quality_conf_norm,
            row_sums,
            out=np.zeros_like(quality_conf_norm),
            where=row_sums != 0,
        )

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "validation_jsonl": str(valid_jsonl.resolve()),
        "frame_hop_seconds": hop_seconds,
        "overall": {"accuracy": overall_metrics, "counts": overall_counts_out},
        "per_track": per_song_results,
        "quality_confusion_matrix": {
            "labels": quality_labels,
            "counts": quality_conf_counts.tolist(),
            "normalized": quality_conf_norm.tolist(),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation report to {output_path}")

    if args.quality_confusion_path:
        plot_confusion_matrices(
            quality_conf_counts, quality_conf_norm, quality_labels, Path(args.quality_confusion_path)
        )
        print(f"[INFO] Saved confusion matrix plot to {args.quality_confusion_path}")

    print("[INFO] Overall accuracies:")
    for task, value in overall_metrics.items():
        display = "N/A" if value is None else f"{value:.4%}"
        print(f"  {task:>11}: {display}")


if __name__ == "__main__":
    main()
