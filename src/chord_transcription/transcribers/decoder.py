from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..hub import resolve_pretrained_checkpoint_path
from ..models.factory import load_label_vocab_from_checkpoint, load_model_build_config_from_checkpoint
from .types import (
    DecodedFramePrediction,
    TranscriptionEvents,
    TranscriptionMetadata,
    TranscriptionPrediction,
)


PITCH_CLASS_LABELS_13: List[str] = ["N", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


class PredictionDecoder:
    """
    predictor の index 出力を、文字列ラベルやイベント列へ変換する。

    推論本体は predictor に任せ、ここでは「人間が読む形式に整える」ことだけを担当する。
    """

    def __init__(self, metadata: TranscriptionMetadata) -> None:
        self.metadata = metadata
        self.root_chord_labels = self._build_root_chord_labels()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "PredictionDecoder":
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_build_config = load_model_build_config_from_checkpoint(checkpoint)
        if model_build_config is None:
            raise KeyError("Checkpoint must contain 'model_build_config'.")
        label_vocab = load_label_vocab_from_checkpoint(checkpoint)

        # predictor を通さず decoder だけ作る場合でも、
        # 時間軸と語彙を復元できるだけの情報は checkpoint から取れるようにする。
        data_loader_cfg = model_build_config["data_loader"]
        if fps := data_loader_cfg.get("frames_per_second") or data_loader_cfg.get("fps"):
            seconds_per_frame = 1.0 / float(fps)
        else:
            seconds_per_frame = float(data_loader_cfg["hop_length"]) / float(data_loader_cfg["sample_rate"])

        metadata = TranscriptionMetadata(
            checkpoint_path=checkpoint_path,
            model_kind=str(model_build_config.get("model_kind", "base")).lower(),
            sample_rate=int(data_loader_cfg["sample_rate"]),
            seconds_per_frame=seconds_per_frame,
            num_root_classes=int(model_build_config["model"]["num_root_classes"]),
            num_quality_classes=int(model_build_config["model"]["num_quality_classes"]),
            quality_labels=tuple(label_vocab["quality"]),
        )
        return cls(metadata)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        local_files_only: bool = False,
        token: Optional[str] = None,
    ) -> "PredictionDecoder":
        checkpoint_path = resolve_pretrained_checkpoint_path(
            pretrained_model_name_or_path,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        return cls.from_checkpoint(checkpoint_path)

    @classmethod
    def from_metadata(cls, metadata: TranscriptionMetadata) -> "PredictionDecoder":
        return cls(metadata)

    def decode_frames(self, prediction: TranscriptionPrediction) -> DecodedFramePrediction:
        # 1. index -> 文字列ラベルへ変換する。
        root_chord = None
        if prediction.root_chord_index is not None:
            root_chord = []
            for index in prediction.root_chord_index.tolist():
                if 0 <= index < len(self.root_chord_labels):
                    root_chord.append(self.root_chord_labels[index])
                else:
                    root_chord.append("N/A")

        bass = None
        if prediction.bass_index is not None:
            bass = []
            for index in prediction.bass_index.tolist():
                bass.append(PITCH_CLASS_LABELS_13[int(index)] if 0 <= int(index) < 13 else "N/A")

        key = None
        if prediction.key_index is not None:
            key = []
            for index in prediction.key_index.tolist():
                key.append(PITCH_CLASS_LABELS_13[int(index)] if 0 <= int(index) < 13 else "N/A")

        # 2. root_chord と bass から、最終的な chord 表記を作る。
        chord = None
        if root_chord is not None:
            chord = []
            if bass is None:
                chord = list(root_chord)
            else:
                for root_label, bass_label in zip(root_chord, bass):
                    if root_label in {"N", "N/A"} or root_label is None:
                        chord.append("N")
                        continue
                    if bass_label in {None, "", "N"}:
                        chord.append(root_label)
                        continue

                    if len(root_label) >= 2 and root_label[1] in {"#", "b"}:
                        root_name = root_label[:2]
                    else:
                        root_name = root_label[:1]

                    if root_name == bass_label:
                        chord.append(root_label)
                    else:
                        chord.append(f"{root_label}/{bass_label}")

        beat_prob = self._sigmoid(prediction.beat_scores)
        downbeat_prob = self._sigmoid(prediction.downbeat_scores)

        return DecodedFramePrediction(
            time_sec=prediction.time_sec,
            root_chord=root_chord,
            bass=bass,
            key=key,
            chord=chord,
            beat_prob=beat_prob,
            downbeat_prob=downbeat_prob,
        )

    def to_events(
        self,
        frames: DecodedFramePrediction,
        *,
        min_duration_chord: float = 0.1,
        min_duration_key: float = 0.5,
        romanize: bool = True,
    ) -> TranscriptionEvents:
        # フレーム列からイベント列を作る処理はここにまとめる。
        def build_events(labels: List[str] | None, label_key: str) -> List[Dict[str, Any]]:
            if labels is None or len(labels) == 0 or len(frames.time_sec) == 0:
                return []

            limit = min(len(labels), len(frames.time_sec))
            frame_duration = float(frames.time_sec[1] - frames.time_sec[0]) if limit > 1 else 0.1

            events: List[Dict[str, Any]] = []
            last_label = labels[0]
            segment_start_time = float(frames.time_sec[0])
            for index in range(1, limit):
                current_label = labels[index]
                if current_label != last_label:
                    events.append(
                        {
                            "start_time": segment_start_time,
                            "end_time": float(frames.time_sec[index]),
                            label_key: last_label,
                        }
                    )
                    segment_start_time = float(frames.time_sec[index])
                    last_label = current_label

            events.append(
                {
                    "start_time": segment_start_time,
                    "end_time": float(frames.time_sec[limit - 1] + frame_duration),
                    label_key: last_label,
                }
            )
            return events

        def filter_short_events(
            events: List[Dict[str, Any]],
            min_duration: float,
            label_key: str,
        ) -> List[Dict[str, Any]]:
            if not events:
                return []

            filtered: List[Dict[str, Any]] = []
            for event in events:
                duration = float(event["end_time"] - event["start_time"])
                if duration < min_duration and filtered:
                    filtered[-1]["end_time"] = event["end_time"]
                    if len(filtered) >= 2 and filtered[-2][label_key] == filtered[-1][label_key]:
                        filtered[-2]["end_time"] = filtered[-1]["end_time"]
                        filtered.pop()
                else:
                    if filtered and filtered[-1][label_key] == event[label_key]:
                        filtered[-1]["end_time"] = event["end_time"]
                    else:
                        filtered.append(dict(event))
            return filtered

        # 1. フレーム列から chord / key のイベント列を作る。
        chord_events = build_events(frames.chord, "chord")
        key_events = build_events(frames.key, "key")
        beat_events = self._build_beat_events(frames.time_sec, frames.beat_prob, frames.downbeat_prob)

        # 2. 極端に短いイベントを前後へ吸収して見やすくする。
        if min_duration_chord > 0:
            chord_events = filter_short_events(chord_events, min_duration_chord, "chord")
        if min_duration_key > 0:
            key_events = filter_short_events(key_events, min_duration_key, "key")

        # 3. chord-romanizer が入っていれば、綴りを少し自然に寄せる。
        if romanize and chord_events and key_events:
            self._apply_chord_romanizer(chord_events, key_events)

        return TranscriptionEvents(chord_events=chord_events, key_events=key_events, beat_events=beat_events)

    @staticmethod
    def _sigmoid(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-array))

    @staticmethod
    def _pick_peak_indices(
        scores: np.ndarray | None,
        *,
        threshold: float,
        suppress_radius_frames: int = 0,
    ) -> np.ndarray:
        if scores is None:
            return np.zeros((0,), dtype=np.int64)

        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return np.zeros((0,), dtype=np.int64)

        if values.size == 1:
            if float(values[0]) >= threshold:
                return np.asarray([0], dtype=np.int64)
            return np.zeros((0,), dtype=np.int64)

        prev_values = np.concatenate(([-np.inf], values[:-1]))
        next_values = np.concatenate((values[1:], [-np.inf]))
        candidate_mask = (values >= threshold) & (values >= prev_values) & (values >= next_values)
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return np.zeros((0,), dtype=np.int64)

        selected: List[int] = []
        blocked = np.zeros((values.shape[0],), dtype=bool)
        ranked_indices = candidate_indices[np.argsort(values[candidate_indices])[::-1]]
        for candidate_index in ranked_indices.tolist():
            left = max(0, int(candidate_index) - suppress_radius_frames)
            right = min(values.shape[0], int(candidate_index) + suppress_radius_frames + 1)
            if blocked[left:right].any():
                continue
            selected.append(int(candidate_index))
            blocked[left:right] = True

        selected.sort()
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _estimate_beats_per_bar(beat_indices: np.ndarray, downbeat_indices: np.ndarray) -> int:
        beats = np.asarray(beat_indices, dtype=np.int64).reshape(-1)
        downbeats = np.asarray(downbeat_indices, dtype=np.int64).reshape(-1)

        if downbeats.size >= 2 and beats.size > 0:
            counts: List[int] = []
            for start, end in zip(downbeats[:-1].tolist(), downbeats[1:].tolist()):
                count = int(np.sum((beats >= start) & (beats < end)))
                if count > 0:
                    counts.append(count)
            if counts:
                estimated = int(round(float(np.median(np.asarray(counts, dtype=np.float32)))))
                if 1 <= estimated <= 12:
                    return estimated

        if downbeats.size >= 2 and beats.size >= 2:
            beat_gaps = np.diff(beats)
            downbeat_gaps = np.diff(downbeats)
            beat_gaps = beat_gaps[beat_gaps > 0]
            downbeat_gaps = downbeat_gaps[downbeat_gaps > 0]
            if beat_gaps.size > 0 and downbeat_gaps.size > 0:
                estimated = int(round(float(np.median(downbeat_gaps) / np.median(beat_gaps))))
                if 1 <= estimated <= 12:
                    return estimated

        return 4

    def _build_beat_events(
        self,
        time_sec: np.ndarray,
        beat_prob: np.ndarray | None,
        downbeat_prob: np.ndarray | None,
    ) -> List[Dict[str, Any]]:
        if len(time_sec) == 0:
            return []

        beat_indices = self._pick_peak_indices(beat_prob, threshold=0.5)
        downbeat_indices = self._pick_peak_indices(downbeat_prob, threshold=0.5)
        downbeat_set = {int(index) for index in downbeat_indices.tolist()}

        merged_indices: List[int] = list(downbeat_set)
        for beat_index in beat_indices.tolist():
            if any(abs(int(beat_index) - downbeat_index) <= 1 for downbeat_index in downbeat_set):
                continue
            merged_indices.append(int(beat_index))

        if not merged_indices:
            return []

        merged_indices = sorted(set(merged_indices))
        beats_per_bar = self._estimate_beats_per_bar(np.asarray(merged_indices), downbeat_indices)
        downbeat_positions = [pos for pos, frame_index in enumerate(merged_indices) if frame_index in downbeat_set]
        first_downbeat_position = downbeat_positions[0] if downbeat_positions else None
        downbeat_position_set = set(downbeat_positions)

        beat_numbers: List[int] = []
        last_downbeat_position = None
        for position in range(len(merged_indices)):
            if position in downbeat_position_set:
                last_downbeat_position = position

            if last_downbeat_position is not None:
                beat_numbers.append(((position - last_downbeat_position) % beats_per_bar) + 1)
            elif first_downbeat_position is not None:
                beat_numbers.append(((position - first_downbeat_position) % beats_per_bar) + 1)
            else:
                beat_numbers.append((position % beats_per_bar) + 1)

        return [
            {
                "time_sec": float(time_sec[frame_index]),
                "beat": int(beat_number),
            }
            for frame_index, beat_number in zip(merged_indices, beat_numbers)
        ]

    def _build_root_chord_labels(self) -> List[str]:
        quality_list = list(self.metadata.quality_labels[: self.metadata.num_quality_classes])
        try:
            non_chord_idx = next(i for i, label in enumerate(quality_list) if label == "N")
        except StopIteration as exc:
            raise ValueError("quality語彙に 'N' が存在しません") from exc

        quality_slots = [label for i, label in enumerate(quality_list) if i != non_chord_idx]
        root_chord_labels: List[str] = []
        for root_idx in range(1, self.metadata.num_root_classes):
            root_name = PITCH_CLASS_LABELS_13[root_idx]
            for quality in quality_slots:
                root_chord_labels.append(f"{root_name}{quality}")
        root_chord_labels.append("N")
        return root_chord_labels

    @staticmethod
    def _apply_chord_romanizer(
        chord_events: List[Dict[str, Any]],
        key_events: List[Dict[str, Any]],
    ) -> None:
        try:
            from chord_romanizer import Romanizer, ChordParser
        except ImportError:
            return

        try:
            romanizer = Romanizer(simplify_accidentals=True)
            progression_sequence = []
            parsed_chord_map = {}

            for index, chord_event in enumerate(chord_events):
                best_key = "C"
                max_overlap = -1.0
                for key_event in key_events:
                    overlap_start = max(chord_event["start_time"], key_event["start_time"])
                    overlap_end = min(chord_event["end_time"], key_event["end_time"])
                    overlap = max(0.0, overlap_end - overlap_start)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_key = key_event["key"]

                if best_key == "N":
                    best_key = "C"

                parsed = ChordParser.parse(chord_event["chord"])
                if parsed is None:
                    parsed = ChordParser.parse("N")
                if parsed is None:
                    continue

                parsed_chord_map[id(parsed)] = index
                progression_sequence.append((parsed, best_key))

            annotated = romanizer.annotate_progression(progression_sequence)
            for result in annotated:
                if result and result.chord:
                    event_index = parsed_chord_map.get(id(result.chord))
                    if event_index is not None and result.symbol_fixed:
                        chord_events[event_index]["chord"] = result.symbol_fixed
        except Exception:
            return
