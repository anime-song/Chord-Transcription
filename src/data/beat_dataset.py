from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BeatSongData:
    """1曲ぶんの beat/downbeat 情報を保持する。"""

    song_id: str
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    metadata: Dict[str, Any]


class BeatAnnotationLoader:
    """
    precompute_beats.py が保存した beat/downbeat を読むための軽量 loader。

    ChordDataset 本体に beat 読み込みの責務を持たせると見通しが悪くなるので、
    ここでは以下だけを担当する。

    1. 曲IDから beat ファイルを解決する
    2. 1曲ぶんの秒単位イベントをキャッシュ付きで読む
    3. 任意のセグメントに切り出し、必要なら現在の hop_length で frame 化する
    """

    def __init__(
        self,
        beat_root_dir: Path,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        use_cache: bool = True,
        max_cache_files: int = 128,
    ) -> None:
        self.beat_root_dir = beat_root_dir.expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.use_cache = bool(use_cache)
        self.max_cache_files = max(1, int(max_cache_files))
        self._song_cache: OrderedDict[str, BeatSongData] = OrderedDict()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_song_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._song_cache = OrderedDict()

    def close(self) -> None:
        self._song_cache.clear()

    def has_record(self, record_or_song: Any) -> bool:
        _, npz_path, metadata_path = self._resolve_paths(record_or_song)
        return npz_path.exists() and metadata_path.exists()

    def load_song(self, record_or_song: Any) -> BeatSongData:
        song_id, npz_path, metadata_path = self._resolve_paths(record_or_song)
        cache_key = str(npz_path)

        if self.use_cache:
            cached = self._song_cache.pop(cache_key, None)
            if cached is not None:
                self._song_cache[cache_key] = cached
                return cached

        if not npz_path.exists():
            raise FileNotFoundError(f"Beat annotation not found: {npz_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Beat annotation metadata not found: {metadata_path}")

        # 保存済みの frame cache は参照せず、秒単位イベントを正本として読む。
        with np.load(npz_path) as data:
            song_data = BeatSongData(
                song_id=song_id,
                beat_times=np.asarray(data["beat_times"], dtype=np.float32),
                downbeat_times=np.asarray(data["downbeat_times"], dtype=np.float32),
                metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
            )

        if self.use_cache:
            self._song_cache[cache_key] = song_data
            while len(self._song_cache) > self.max_cache_files:
                self._song_cache.popitem(last=False)

        return song_data

    def load_segment(
        self,
        record_or_song: Any,
        segment_start_sec: float,
        segment_duration_sec: float,
        *,
        scale_factor: float = 1.0,
        target_num_frames: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        1セグメントぶんの beat/downbeat を返す。

        Args:
            record_or_song:
                record dict か song_id 文字列。record の場合は stems_dir.name を優先して曲IDに使う。
            segment_start_sec:
                元の曲上でのセグメント開始位置。
            segment_duration_sec:
                元の曲上で読む長さ。time-stretch 前の長さを入れる。
            scale_factor:
                セグメント内で時間をどれだけ伸縮して frame へ落とすか。
                time-stretch 後に元の segment 長へ戻す場合は 1 / stretch_rate を渡す。
            target_num_frames:
                出力フレーム数を明示したい場合に指定する。
                None の場合は segment_duration_sec * scale_factor から計算する。
        """
        song = self.load_song(record_or_song)

        segment_start_sec = float(segment_start_sec)
        segment_duration_sec = float(segment_duration_sec)
        scale_factor = float(scale_factor)
        segment_end_sec = segment_start_sec + segment_duration_sec

        beat_times = self._slice_event_times(song.beat_times, segment_start_sec, segment_end_sec, scale_factor)
        downbeat_times = self._slice_event_times(song.downbeat_times, segment_start_sec, segment_end_sec, scale_factor)

        if target_num_frames is None:
            target_duration_sec = max(segment_duration_sec * scale_factor, 0.0)
            target_num_samples = int(round(target_duration_sec * self.sample_rate))
            target_num_frames = self._num_frames_from_samples(target_num_samples)

        beat_frame_indices, beat_mask = self._times_to_frame_targets(beat_times, target_num_frames)
        downbeat_frame_indices, downbeat_mask = self._times_to_frame_targets(downbeat_times, target_num_frames)

        return {
            "beat_times": beat_times,
            "downbeat_times": downbeat_times,
            "beat_frame_indices": beat_frame_indices,
            "downbeat_frame_indices": downbeat_frame_indices,
            "beat_mask": beat_mask,
            "downbeat_mask": downbeat_mask,
        }

    def _resolve_paths(self, record_or_song: Any) -> tuple[str, Path, Path]:
        # ChordDataset から呼ぶときは record dict をそのまま渡せるようにしておく。
        if isinstance(record_or_song, dict):
            beat_label_path = record_or_song.get("beat_label_path")
            if beat_label_path:
                npz_path = Path(beat_label_path).expanduser().resolve()
                song_id = npz_path.stem.removesuffix("_beat_this")
                return song_id, npz_path, npz_path.with_suffix(".json")

            stems_dir = record_or_song.get("stems_dir")
            if stems_dir:
                song_id = Path(stems_dir).expanduser().resolve().name
            else:
                basename = record_or_song.get("basename")
                if not basename:
                    raise KeyError("record must contain either stems_dir, basename, or beat_label_path")
                song_id = str(basename)
        elif isinstance(record_or_song, Path):
            song_id = record_or_song.expanduser().resolve().name
        else:
            song_id = str(record_or_song)

        song_dir = self.beat_root_dir / song_id
        base_name = f"{song_id}_beat_this"
        return song_id, song_dir / f"{base_name}.npz", song_dir / f"{base_name}.json"

    def _slice_event_times(
        self,
        absolute_times: np.ndarray,
        segment_start_sec: float,
        segment_end_sec: float,
        scale_factor: float,
    ) -> np.ndarray:
        if absolute_times.size == 0:
            return np.zeros((0,), dtype=np.float32)

        # 区間内イベントだけを取り出し、セグメント先頭基準の相対秒へ変換する。
        valid = (absolute_times >= segment_start_sec) & (absolute_times <= segment_end_sec + 1e-7)
        if not np.any(valid):
            return np.zeros((0,), dtype=np.float32)

        relative_times = (absolute_times[valid] - segment_start_sec) * scale_factor
        return np.asarray(relative_times, dtype=np.float32)

    def _times_to_frame_targets(self, event_times: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
        if num_frames <= 0 or event_times.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((num_frames,), dtype=np.uint8)

        frame_indices = np.rint(event_times * self.sample_rate / float(self.hop_length)).astype(np.int64)
        frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < num_frames)]
        if frame_indices.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((num_frames,), dtype=np.uint8)

        frame_indices = np.unique(frame_indices)
        mask = np.zeros((num_frames,), dtype=np.uint8)
        mask[frame_indices] = 1
        return frame_indices, mask

    def _num_frames_from_samples(self, num_samples: int) -> int:
        if num_samples < self.n_fft:
            return 0
        return (num_samples - self.n_fft) // self.hop_length + 1
