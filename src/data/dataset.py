import json
import random
import bisect
import glob
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional, Set, Sequence, Callable

import librosa
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset
from dlchordx import Tone

from .audio_augmentation import StereoWaveformAugmentation
from .beat_dataset import BeatAnnotationLoader
from .processing import LabelProcessor, ChordEvent, EventSpan
from .label_shifter import LabelShifter

STEM_ORDER_DEFAULT: Tuple[str, ...] = ("vocals", "drums", "bass", "other", "piano", "guitar")
REQUIRED_CHORD_STEMS: Tuple[str, ...] = ("other", "piano", "guitar")
REQUIRED_STEM_MIN_RMS: float = 1e-4
PACKED_PITCH_SUFFIX_PATTERN = re.compile(r"_pitch_(-?\d+)st$")
TARGET_AUDIO_CHANNELS: int = 2


@dataclass(frozen=True)
class PackedAudioEntry:
    array_path: Path
    metadata_path: Path
    sample_rate: int
    channels_per_stem: int
    num_channels: int
    num_frames: int
    storage_dtype: str


def sample_stem_dropout(
    stem_order: Sequence[str],
    dropout_cfg: Dict[str, Any],
    required_indices: Sequence[int],
) -> Set[str]:
    stems_to_mute: Set[str] = set()
    if not stem_order or not dropout_cfg:
        return stems_to_mute

    prob = float(dropout_cfg.get("p", 0.0))
    if prob <= 0.0 or random.random() >= prob:
        return stems_to_mute

    total_indices = list(range(len(stem_order)))
    min_stems = int(dropout_cfg.get("min_stems", 1))
    min_stems = max(1, min(min_stems, len(total_indices)))
    num_to_keep = random.randint(min_stems, len(total_indices))

    keep_indices = set(random.sample(total_indices, num_to_keep))
    if required_indices and not keep_indices.intersection(required_indices):
        required_choice = random.choice(required_indices)
        keep_indices.add(required_choice)
        if len(keep_indices) > num_to_keep:
            removable = [idx for idx in keep_indices if idx != required_choice]
            if removable:
                keep_indices.remove(random.choice(removable))

    stems_to_mute = {stem_order[idx] for idx in total_indices if idx not in keep_indices}
    return stems_to_mute


def stem_has_energy(audio: np.ndarray, min_rms: float) -> bool:
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return rms >= min_rms


def apply_stem_mutes(
    stems: List[np.ndarray],
    stem_order: Sequence[str],
    stems_to_mute: Set[str],
) -> List[np.ndarray]:
    muted_stems: List[np.ndarray] = []
    mute_set = set(stems_to_mute)
    for stem_name, stem_audio in zip(stem_order, stems):
        if stem_name in mute_set:
            muted_stems.append(np.zeros_like(stem_audio, dtype=np.float32))
        else:
            muted_stems.append(np.array(stem_audio, copy=True))
    return muted_stems


def mix_stems(stems: Sequence[np.ndarray]) -> np.ndarray:
    if not stems:
        raise ValueError("stems must not be empty")
    return np.sum(np.stack(stems, axis=0), axis=0, dtype=np.float32)


def ensure_required_chord_stem(
    stems: List[np.ndarray],
    stem_order: Sequence[str],
    required_indices: Sequence[int],
    stems_to_mute: Set[str],
    reload_stems: Callable[[Set[str]], List[np.ndarray]],
) -> Tuple[List[np.ndarray], Set[str]]:
    def has_active_required(audio_list: List[np.ndarray], mute_set: Set[str]) -> bool:
        for idx in required_indices:
            stem_name = stem_order[idx]
            if stem_name in mute_set:
                continue
            if stem_has_energy(audio_list[idx], REQUIRED_STEM_MIN_RMS):
                return True
        return False

    if not required_indices:
        return stems, stems_to_mute

    mute_set = set(stems_to_mute)
    if has_active_required(stems, mute_set):
        return stems, mute_set

    for idx in required_indices:
        stem_name = stem_order[idx]
        if stem_name not in mute_set:
            continue
        mute_set.remove(stem_name)
        stems = reload_stems(mute_set)
        if has_active_required(stems, mute_set):
            return stems, mute_set

    return stems, mute_set


def _discover_packed_variants(
    packed_song_dir: Path,
    song_id: str,
    stem_names: Sequence[str],
) -> Dict[int, PackedAudioEntry]:
    variants: Dict[int, PackedAudioEntry] = {}
    if not packed_song_dir.exists():
        return variants

    prefix = f"{song_id}_stems"
    for metadata_path in packed_song_dir.glob(f"{glob.escape(song_id)}_stems_pitch_*.json"):
        metadata_stem = metadata_path.stem
        if not metadata_stem.startswith(prefix):
            continue

        suffix = metadata_stem[len(prefix) :]
        match = PACKED_PITCH_SUFFIX_PATTERN.fullmatch(suffix)
        if match is None:
            continue

        array_path = metadata_path.with_suffix(".npy")
        if not array_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("stem_names", [])) != tuple(stem_names):
            continue

        semitone = int(match.group(1))
        variants[semitone] = PackedAudioEntry(
            array_path=array_path,
            metadata_path=metadata_path,
            sample_rate=int(metadata["sample_rate"]),
            channels_per_stem=int(metadata["channels_per_stem"]),
            num_channels=int(metadata["num_channels"]),
            num_frames=int(metadata["num_frames"]),
            storage_dtype=str(metadata.get("storage_dtype", "float32")),
        )

    return variants


def read_chords_jsonl(file_path: Path) -> List[ChordEvent]:
    """JSONL形式のコードファイルを読み込み、ChordEventのリストを返します。"""
    events: List[ChordEvent] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            events.append(
                ChordEvent(
                    start_time=float(obj["start_time"]),
                    end_time=float(obj["end_time"]),
                    root=str(obj["root"]),
                    bass=str(obj["bass"]),
                    quality=str(obj["quality"]).replace(" ", ""),
                )
            )
    return events


def read_tsv(path: Path) -> List[EventSpan]:
    """TSV形式のキー/テンポファイルを読み込み、EventSpanのリストを返します。"""
    events: List[EventSpan] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            if ":" in line:
                parts = line.strip().split(":", maxsplit=2)
            else:
                parts = line.strip().split("\t", maxsplit=2)
            if len(parts) < 3:
                continue
            events.append(EventSpan(start_time=float(parts[0]), end_time=float(parts[1]), label=parts[2].strip()))
    return events


class ChordDataset(Dataset):
    """
    オーディオセグメントと、それに対応するフレーム化されたラベルを提供するDataset。
    """

    def __init__(
        self,
        jsonl_path: Path,
        label_processor: LabelProcessor,
        config: Dict[str, Any],
        random_crop: bool = True,
    ):
        super().__init__()
        # __init__ の途中で例外が起きても __del__ から安全に close できるよう、
        # 後段で使う属性は先に最低限だけ作っておく。
        self._audio_file_cache: OrderedDict[str, sf.SoundFile] = OrderedDict()
        self._packed_array_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._packed_variant_cache: Dict[str, Dict[int, PackedAudioEntry]] = {}
        self.packed_audio_dir: Optional[Path] = None
        self.beat_loader: Optional[BeatAnnotationLoader] = None
        self.waveform_augmentation: Optional[StereoWaveformAugmentation] = None
        self.waveform_aug_backend = "cpu"

        self.records = self._load_jsonl(jsonl_path)
        if len(self.records) == 0:
            raise RuntimeError(f"No records in {jsonl_path}")

        self.label_processor = label_processor

        # 設定ファイルから必要なパラメータを取得
        self.data_cfg = config["data_loader"]
        self.sample_rate = self.data_cfg["sample_rate"]
        self.segment_seconds = self.data_cfg["segment_seconds"]
        self.segment_samples = int(self.segment_seconds * self.sample_rate)
        self.stem_order = self.data_cfg["stem_order"]
        self.random_crop = random_crop
        self.stem_channels = TARGET_AUDIO_CHANNELS
        self.audio_backend = str(self.data_cfg.get("audio_backend", "wav")).lower()
        if self.audio_backend not in {"wav", "packed"}:
            raise ValueError("data_loader.audio_backend must be either 'wav' or 'packed'")
        self.use_file_handle_cache = bool(self.data_cfg.get("use_file_handle_cache", True))
        self.max_open_files = max(1, int(self.data_cfg.get("max_open_files", 64)))
        if self.audio_backend == "packed":
            self.packed_audio_dir = self._resolve_packed_audio_dir()
            if not self.packed_audio_dir.exists():
                raise FileNotFoundError(f"Packed audio directory not found: {self.packed_audio_dir}")

        # コードチェンジ中心サンプリング用キャッシュ（random_crop時のみ使用）
        self.chord_change_cache: Dict[str, List[float]] = {}

        # ステム拡張の設定
        self.stem_aug_cfg = self.data_cfg.get("stem_augmentation", {"enabled": False})
        self.stem_aug_enabled = self.stem_aug_cfg.get("enabled", False) and self.random_crop
        if self.stem_aug_enabled:
            self.dropout_cfg = self.stem_aug_cfg.get("dropout", {})
            self.mixing_cfg = self.stem_aug_cfg.get("mixing", {})
            print(
                f"Stem augmentation enabled: Dropout(p={self.dropout_cfg.get('p', 0)}), Mixing(p={self.mixing_cfg.get('p', 0)})"
            )

        # stem を mix する前の 2ch waveform に対する拡張。
        # mute 済み stem はここでは触らず、音のある stem にだけ適用する。
        waveform_aug_cfg = self.data_cfg.get("waveform_augmentation", {"enabled": False})
        self.waveform_aug_backend = str(waveform_aug_cfg.get("backend", "cpu")).lower()
        if self.waveform_aug_backend not in {"cpu", "cuda"}:
            raise ValueError("data_loader.waveform_augmentation.backend must be either 'cpu' or 'cuda'")
        if bool(waveform_aug_cfg.get("enabled", False)) and self.random_crop:
            if self.waveform_aug_backend == "cpu":
                self.waveform_augmentation = StereoWaveformAugmentation(
                    sample_rate=self.sample_rate,
                    config=waveform_aug_cfg,
                )
                print(f"{self.waveform_augmentation.summary()} [cpu stem]")
            else:
                print("Waveform augmentation enabled: deferred to trainer on CUDA after batch transfer.")

        # ピッチ拡張の設定
        self.pitch_aug_cfg = self.data_cfg.get("pitch_augmentation", {"enabled": False})
        self.pitch_aug_enabled = self.pitch_aug_cfg.get("enabled", False) and self.random_crop
        if self.pitch_aug_enabled:
            self.pitch_semitone_range = self.pitch_aug_cfg["semitone_range"]
            self.pitch_aug_p = self.pitch_aug_cfg.get("p", 0.5)
            print(f"Pitch augmentation enabled: p={self.pitch_aug_p}, semitone_range={self.pitch_semitone_range}")

        # タイムストレッチ拡張の設定
        self.time_stretch_cfg = self.data_cfg.get("time_stretch", {"enabled": False})
        self.time_stretch_enabled = self.time_stretch_cfg.get("enabled", False) and self.random_crop
        if self.time_stretch_enabled:
            self.ts_min_rate = float(self.time_stretch_cfg.get("min_rate", 0.75))
            self.ts_max_rate = float(self.time_stretch_cfg.get("max_rate", 1.25))
            self.ts_p = float(self.time_stretch_cfg.get("p", 0.5))
            print(f"Time stretch augmentation enabled: p={self.ts_p}, rate=[{self.ts_min_rate}, {self.ts_max_rate}]")

        # beat / downbeat 教師の設定
        self.use_beat_annotations = bool(self.data_cfg.get("use_beat_annotations", False))
        if self.use_beat_annotations:
            beat_root_dir = Path(self.data_cfg.get("beat_root_dir", "dataset/beats"))
            if not beat_root_dir.exists():
                raise FileNotFoundError(f"Beat annotation directory not found: {beat_root_dir}")
            self.beat_loader = BeatAnnotationLoader(
                beat_root_dir=beat_root_dir,
                sample_rate=self.sample_rate,
                hop_length=int(self.data_cfg.get("hop_length", config["model"]["backbone"]["hop_length"])),
                n_fft=int(config["model"]["backbone"]["n_fft"]),
                use_cache=bool(self.data_cfg.get("use_beat_cache", True)),
                max_cache_files=int(self.data_cfg.get("max_beat_cache_files", 128)),
            )
            print(f"Beat annotations enabled: root_dir={beat_root_dir}")

        self.label_shifter = LabelShifter()

        # ドラム以外のステムのインデックスを特定
        self.nondrum_indices = [i for i, stem in enumerate(self.stem_order) if stem != "drums"]
        self.required_stem_indices = [i for i, stem in enumerate(self.stem_order) if stem in REQUIRED_CHORD_STEMS]

        self.quality_to_index = self._load_json_map(Path(config["data"]["quality_json_path"]))

        # root-chord作成の準備
        self.quality_non_chord_index = self.quality_to_index.get("N")
        if self.quality_non_chord_index is None:
            raise ValueError("quality vocabulary must contain 'N' for non-chord handling")

        quality_lookup = torch.full((max(self.quality_to_index.values()) + 1,), -1, dtype=torch.long)
        slot = 0
        for _, idx in sorted(self.quality_to_index.items(), key=lambda item: item[1]):
            if idx == self.quality_non_chord_index:
                continue
            quality_lookup[idx] = slot
            slot += 1
        self.quality_idx_lookup = quality_lookup
        self.num_quality_without_non_chord = config["model"]["num_quality_classes"] - 1
        self.num_root_without_non_root = config["model"]["num_root_classes"] - 1
        self.root_chord_n_index = self.num_quality_without_non_chord * self.num_root_without_non_root
        self.num_root_chord_classes = self.root_chord_n_index + 1

        self.key_to_index = {
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

        # データセットの全長を計算するためのセグメントマッピング
        self._create_segment_map()

    def __len__(self) -> int:
        return self.cumulative_segments[-1]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_audio_file_cache"] = OrderedDict()
        state["_packed_array_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._audio_file_cache = OrderedDict()
        self._packed_array_cache = OrderedDict()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        audio_file_cache = getattr(self, "_audio_file_cache", None)
        while audio_file_cache:
            _, audio_file = self._audio_file_cache.popitem(last=False)
            audio_file.close()
        packed_array_cache = getattr(self, "_packed_array_cache", None)
        while packed_array_cache:
            _, packed_array = self._packed_array_cache.popitem(last=False)
            mmap_handle = getattr(packed_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        beat_loader = getattr(self, "beat_loader", None)
        if beat_loader is not None:
            beat_loader.close()

    def _resolve_packed_audio_dir(self) -> Path:
        configured_dir = self.data_cfg.get("packed_audio_dir")
        if configured_dir:
            return Path(configured_dir).expanduser().resolve()

        sample_stems_dir = Path(self.records[0]["stems_dir"]).expanduser().resolve()
        return sample_stems_dir.parent.parent / "songs_packed"

    def _record_cache_key(self, record: Dict[str, Any]) -> str:
        return str(Path(record["stems_dir"]).expanduser().resolve())

    def _get_record_packed_variants(self, record: Dict[str, Any]) -> Dict[int, PackedAudioEntry]:
        if self.audio_backend != "packed":
            return {}

        record_key = self._record_cache_key(record)
        cached = self._packed_variant_cache.get(record_key)
        if cached is not None:
            return cached

        stems_dir = Path(record["stems_dir"]).expanduser().resolve()
        song_id = stems_dir.name
        assert self.packed_audio_dir is not None
        variants = _discover_packed_variants(
            packed_song_dir=self.packed_audio_dir / song_id,
            song_id=song_id,
            stem_names=self.stem_order,
        )
        self._packed_variant_cache[record_key] = variants
        return variants

    def _get_default_semitone(self, record: Dict[str, Any]) -> int:
        if self.audio_backend != "packed":
            return 0

        variants = self._get_record_packed_variants(record)
        if not variants:
            raise FileNotFoundError(f"No packed audio variants found for {record['stems_dir']}")
        return 0 if 0 in variants else sorted(variants)[0]

    def _normalize_stem_channels(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise ValueError(f"Expected 2D audio array [channels, samples], got shape {audio.shape}")

        if audio.shape[0] == self.stem_channels:
            return audio
        if audio.shape[0] == 1 and self.stem_channels == 2:
            return np.repeat(audio, 2, axis=0)
        if audio.shape[0] > self.stem_channels:
            return audio[: self.stem_channels]
        raise ValueError(f"Expected up to {self.stem_channels} channels per stem, got {audio.shape[0]}")

    def _get_cached_audio_file(self, wav_path: Path) -> sf.SoundFile:
        cache_key = str(wav_path)
        audio_file = self._audio_file_cache.pop(cache_key, None)
        if audio_file is None:
            audio_file = sf.SoundFile(cache_key, mode="r")
        self._audio_file_cache[cache_key] = audio_file

        while len(self._audio_file_cache) > self.max_open_files:
            _, oldest_audio_file = self._audio_file_cache.popitem(last=False)
            oldest_audio_file.close()
        return audio_file

    def _get_cached_packed_array(self, array_path: Path) -> np.ndarray:
        cache_key = str(array_path)
        packed_array = self._packed_array_cache.pop(cache_key, None)
        if packed_array is None:
            packed_array = np.load(array_path, mmap_mode="r")
        self._packed_array_cache[cache_key] = packed_array

        while len(self._packed_array_cache) > self.max_open_files:
            _, oldest_array = self._packed_array_cache.popitem(last=False)
            mmap_handle = getattr(oldest_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        return packed_array

    def _load_wav_stems_segment(
        self,
        stems_dir: Path,
        start_seconds: float,
        segment_seconds: float,
        stems_to_mute: Optional[Set[str]] = None,
        shift_semitones: int = 0,
    ) -> List[np.ndarray]:
        mute_set = set(stems_to_mute or set())
        num_samples = int(segment_seconds * self.sample_rate)
        waves: List[np.ndarray] = []

        for stem_name in self.stem_order:
            if stem_name in mute_set:
                waves.append(np.zeros((self.stem_channels, num_samples), dtype=np.float32))
                continue

            stem_filename_base = f"{stems_dir.name}_{stem_name}"
            if shift_semitones != 0:
                stem_filename = f"{stem_filename_base}_pitch_{shift_semitones}st.wav"
            else:
                stem_filename = f"{stem_filename_base}.wav"

            stem_path = stems_dir / stem_filename
            if not stem_path.exists():
                raise FileNotFoundError(f"Stem file not found: {stem_path}")

            if self.use_file_handle_cache:
                audio_file = self._get_cached_audio_file(stem_path)
                source_sample_rate = int(audio_file.samplerate)
                start_frame = int(start_seconds * source_sample_rate)
                num_frames = int(segment_seconds * source_sample_rate)
                audio_file.seek(start_frame)
                audio = audio_file.read(num_frames, dtype="float32", always_2d=True).T
            else:
                with sf.SoundFile(str(stem_path), mode="r") as audio_file:
                    source_sample_rate = int(audio_file.samplerate)
                    start_frame = int(start_seconds * source_sample_rate)
                    num_frames = int(segment_seconds * source_sample_rate)
                    audio_file.seek(start_frame)
                    audio = audio_file.read(num_frames, dtype="float32", always_2d=True).T

            audio = self._normalize_stem_channels(audio)
            if source_sample_rate != self.sample_rate:
                audio = librosa.resample(y=audio, orig_sr=source_sample_rate, target_sr=self.sample_rate)
            waves.append(np.asarray(audio, dtype=np.float32))

        max_len = int(segment_seconds * self.sample_rate)
        padded_waves: List[np.ndarray] = []
        for wave in waves:
            pad_len = max_len - wave.shape[1]
            if pad_len > 0:
                padded_waves.append(np.pad(wave, ((0, 0), (0, pad_len))))
            else:
                padded_waves.append(wave[:, :max_len])
        return padded_waves

    def _load_packed_stems_segment(
        self,
        record: Dict[str, Any],
        start_seconds: float,
        segment_seconds: float,
        shift_semitones: int = 0,
    ) -> List[np.ndarray]:
        variants = self._get_record_packed_variants(record)
        packed_entry = variants.get(shift_semitones)
        if packed_entry is None:
            raise FileNotFoundError(
                f"Packed audio variant not found for {record['stems_dir']} (semitone={shift_semitones})"
            )

        source_sample_rate = packed_entry.sample_rate
        frame_offset = int(start_seconds * source_sample_rate)
        num_frames = int(segment_seconds * source_sample_rate)
        frame_end = min(frame_offset + num_frames, packed_entry.num_frames)

        if self.use_file_handle_cache:
            packed_array = self._get_cached_packed_array(packed_entry.array_path)
            crop = (
                np.zeros((packed_entry.num_channels, 0), dtype=np.float32)
                if frame_offset >= packed_entry.num_frames
                else np.array(packed_array[:, frame_offset:frame_end], copy=True)
            )
        else:
            packed_array = np.load(packed_entry.array_path, mmap_mode="r")
            crop = (
                np.zeros((packed_entry.num_channels, 0), dtype=np.float32)
                if frame_offset >= packed_entry.num_frames
                else np.array(packed_array[:, frame_offset:frame_end], copy=True)
            )
            mmap_handle = getattr(packed_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

        audio = np.asarray(crop, dtype=np.float32)
        if source_sample_rate != self.sample_rate:
            audio = librosa.resample(y=audio, orig_sr=source_sample_rate, target_sr=self.sample_rate)

        expected_channels = len(self.stem_order) * packed_entry.channels_per_stem
        if audio.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} packed channels in {packed_entry.array_path}, found {audio.shape[0]}"
            )

        max_len = int(segment_seconds * self.sample_rate)
        padded_waves: List[np.ndarray] = []
        for stem_index in range(len(self.stem_order)):
            channel_start = stem_index * packed_entry.channels_per_stem
            channel_end = channel_start + packed_entry.channels_per_stem
            stem_audio = self._normalize_stem_channels(audio[channel_start:channel_end])
            pad_len = max_len - stem_audio.shape[1]
            if pad_len > 0:
                padded_waves.append(np.pad(stem_audio, ((0, 0), (0, pad_len))))
            else:
                padded_waves.append(stem_audio[:, :max_len])
        return padded_waves

    def __getitem__(self, index: int) -> Dict:
        # グローバルなindexから、どのファイルのどのセグメントかを特定
        record, segment_start_sec = self._get_segment_info(index)
        stems_dir = Path(record["stems_dir"])

        # ピッチ拡張
        shift_semitones = self._get_default_semitone(record)
        if self.pitch_aug_enabled and random.random() < self.pitch_aug_p:
            if self.audio_backend == "packed":
                variants = self._get_record_packed_variants(record)
                candidate_shifts = [
                    semitone
                    for semitone in variants
                    if self.pitch_semitone_range[0] <= semitone <= self.pitch_semitone_range[1]
                ]
                if candidate_shifts:
                    shift_semitones = random.choice(candidate_shifts)
            else:
                shift_semitones = random.randint(self.pitch_semitone_range[0], self.pitch_semitone_range[1])

        # タイムストレッチ拡張
        stretch_rate = 1.0
        if self.time_stretch_enabled and random.random() < self.ts_p:
            stretch_rate = random.uniform(self.ts_min_rate, self.ts_max_rate)

        # モデル側でタイムストレッチ後に元の segment_seconds になるように、
        # 読み込むオーディオの長さは rate 倍とする。
        # 例：rate=1.25 (速くなる) の場合、1.25倍の長さを読み込みテンソルとして渡す。
        read_segment_seconds = self.segment_seconds * stretch_rate

        # ステム・ドロップアウト
        stems_to_mute: Set[str] = set()
        if self.stem_aug_enabled:
            stems_to_mute = sample_stem_dropout(self.stem_order, self.dropout_cfg, self.required_stem_indices)

        # オーディオの読み込み
        if self.audio_backend == "packed":
            full_stems = self._load_packed_stems_segment(
                record=record,
                start_seconds=segment_start_sec,
                segment_seconds=read_segment_seconds,
                shift_semitones=shift_semitones,
            )

            def reload_stems(mute_set: Set[str]) -> List[np.ndarray]:
                return apply_stem_mutes(full_stems, self.stem_order, mute_set)

        else:

            def reload_stems(mute_set: Set[str]) -> List[np.ndarray]:
                return self._load_wav_stems_segment(
                    stems_dir=stems_dir,
                    start_seconds=segment_start_sec,
                    segment_seconds=read_segment_seconds,
                    stems_to_mute=mute_set,
                    shift_semitones=shift_semitones,
                )

        stems = reload_stems(stems_to_mute)

        if self.stem_aug_enabled:
            stems, stems_to_mute = ensure_required_chord_stem(
                stems=stems,
                stem_order=self.stem_order,
                required_indices=self.required_stem_indices,
                stems_to_mute=stems_to_mute,
                reload_stems=reload_stems,
            )

        # ステム拡張
        if self.stem_aug_enabled:
            # ランダム・ゲイン
            if random.random() < self.mixing_cfg.get("p", 0.0):
                min_db = self.mixing_cfg.get("min_gain_db", -6.0)
                max_db = self.mixing_cfg.get("max_gain_db", 3.0)

                for i in self.nondrum_indices:
                    gain_db = random.uniform(min_db, max_db)
                    gain_linear = 10.0 ** (gain_db / 20.0)
                    stems[i] *= gain_linear

        # mix 前に stem ごとへ波形拡張を掛ける。
        # mute stem に noise や saturation が乗ると dropout の意味が変わるため、
        # 十分なエネルギーがある stem だけを対象にする。
        if self.waveform_augmentation is not None:
            stems = [
                self.waveform_augmentation(stem) if stem_has_energy(stem, 1e-8) else stem
                for stem in stems
            ]

        # stem ごとの拡張が終わったら、モデル入力用の stereo mix に落とす。
        audio = mix_stems(stems)

        # DataLoaderでのバッチ化（次元合わせ）のために、最大長でパディングを行う。
        # 実際の有効なサンプル数は batch["target_samples"] としてモデルに渡す。
        actual_samples = audio.shape[1]
        max_possible_rate = self.ts_max_rate if self.time_stretch_enabled else 1.0
        max_len_samples = int(math.ceil(self.segment_samples * max_possible_rate))

        if actual_samples < max_len_samples:
            pad_len = max_len_samples - actual_samples
            audio = np.pad(audio, ((0, 0), (0, pad_len)))
        elif actual_samples > max_len_samples:
            audio = audio[:, :max_len_samples]
            actual_samples = max_len_samples

        # ラベル読み込み
        chord_events = read_chords_jsonl(Path(record["chord_label_path"]))
        key_events = read_tsv(Path(record["key_label_path"]))

        # オーディオのサンプル長から、ズレのない正確なフレーム数を計算
        num_frames = self.label_processor.get_num_frames(self.segment_samples)

        # ラベルイベントを、Processorが扱える「インデックス付きスパン」に変換
        # タイムストレッチによるラベルの伸縮 (rateが変化した分だけ長さをスケーリングする)
        # ※ モデルに入力する段階では長さが segment_seconds に戻るため、
        # ここでは stretch_rate に反比例させて時間を縮める (1.0 / stretch_rate 倍) ことで、
        # モデルの出力フレームにぴったり合うようにする。
        label_scale_factor = 1.0 / stretch_rate

        root_spans, bass_spans, quality_spans = self._index_chord_events(chord_events)
        key_spans = self._index_key_events(key_events)

        # Processorを使って、各ターゲットのフレーム系列を生成
        # イベントの時間を scale_factor でスケールするように変更
        root_frames = self.label_processor.spans_to_frames(
            root_spans, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )
        bass_frames = self.label_processor.spans_to_frames(
            bass_spans, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )
        quality_frames = self.label_processor.spans_to_frames(
            quality_spans, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )
        key_frames = self.label_processor.spans_to_frames(
            key_spans, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )

        chord25_frames = self.label_processor.chords_to_25d_frames(
            chord_events, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )
        boundary_frames = self.label_processor.chords_to_boundary_frames(
            chord_events, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )
        key_boundary_frames = self.label_processor.sections_to_boundary_frames(
            key_spans, num_frames, segment_start_sec, scale_factor=label_scale_factor
        )

        batch = {
            "audio": torch.from_numpy(audio),
            "root_index": torch.from_numpy(root_frames).to(torch.long),
            "bass_index": torch.from_numpy(bass_frames).to(torch.long),
            "quality_index": torch.from_numpy(quality_frames).to(torch.long),
            "key_index": torch.from_numpy(key_frames).to(torch.long),
            "chord25": torch.from_numpy(chord25_frames),
            "boundary": torch.from_numpy(boundary_frames),
            "key_boundary": torch.from_numpy(key_boundary_frames),
            "time_stretch_rate": torch.tensor(stretch_rate, dtype=torch.float32),
            "target_samples": torch.tensor(actual_samples, dtype=torch.long),
        }

        # beat/downbeat は秒単位の事前計算結果から、現在の frame grid に再量子化して使う。
        if self.beat_loader is not None:
            beat_targets = self.beat_loader.load_segment(
                record,
                segment_start_sec=segment_start_sec,
                segment_duration_sec=read_segment_seconds,
                scale_factor=label_scale_factor,
                target_num_frames=num_frames,
            )
            batch["beat"] = torch.from_numpy(beat_targets["beat_mask"].astype(np.float32, copy=False)[:, None])
            batch["downbeat"] = torch.from_numpy(beat_targets["downbeat_mask"].astype(np.float32, copy=False)[:, None])

        # ラベルのピッチシフト
        batch = self.label_shifter(batch, shift_pitch=shift_semitones)
        batch["root_chord_index"] = self._combine_root_quality(
            batch["root_index"],
            batch["quality_index"],
        )
        return batch

    def _combine_root_quality(self, root_tensor: torch.Tensor, quality_tensor: torch.Tensor) -> torch.Tensor:
        """rootとqualityをくっつけてroot-chordのラベルを作成します"""

        combined = torch.full(
            root_tensor.shape,
            self.label_processor.ignore_index,
            dtype=torch.long,
            device=root_tensor.device,
        )

        # ２つのラベルのマスクを合成
        valid_mask = (root_tensor != self.label_processor.ignore_index) & (
            quality_tensor != self.label_processor.ignore_index
        )
        if not torch.any(valid_mask):
            return combined

        # 有効な部分のみ抜き出し
        root_valid = root_tensor[valid_mask]
        quality_valid = quality_tensor[valid_mask]

        # qualityをlookupでidxに変換
        slot_values = self.quality_idx_lookup[quality_valid].to(root_tensor.device)
        # Non Chordを検出 (root: N or quality: N or quality: -1 (N))
        no_chord_mask = (root_valid == 0) | (quality_valid == self.quality_non_chord_index) | (slot_values == -1)

        # ルートとクオリティの合成
        combined_values = torch.full(
            root_valid.shape,
            self.root_chord_n_index,
            dtype=torch.long,
            device=root_tensor.device,
        )
        valid_combination_mask = ~no_chord_mask
        if torch.any(valid_combination_mask):
            root_offsets = torch.clamp(root_valid[valid_combination_mask] - 1, min=0)
            combined_values[valid_combination_mask] = (
                root_offsets * self.num_quality_without_non_chord + slot_values[valid_combination_mask]
            )

        combined[valid_mask] = combined_values
        return combined

    def _index_chord_events(self, events: List[ChordEvent]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """ChordEventを、root/bass/qualityそれぞれのインデックス付きスパンに変換します。"""

        # dlchordx.Toneを使ったインデックス変換ロジックをここに集約
        def tone_to_idx(note: str) -> int:
            if note == "N":
                return 0
            try:
                return int(Tone(note).get_interval()) + 1
            except Exception:
                return 0

        root_spans = [{"start_time": e.start_time, "end_time": e.end_time, "idx": tone_to_idx(e.root)} for e in events]
        bass_spans = [{"start_time": e.start_time, "end_time": e.end_time, "idx": tone_to_idx(e.bass)} for e in events]
        quality_spans = [
            {"start_time": e.start_time, "end_time": e.end_time, "idx": self.quality_to_index.get(e.quality, 0)}
            for e in events
        ]
        return root_spans, bass_spans, quality_spans

    def _index_key_events(self, events: List[EventSpan]) -> List[Dict]:
        """EventSpanを、キーのインデックス付きスパンに変換します。"""
        return [
            {"start_time": e.start_time, "end_time": e.end_time, "idx": self.key_to_index.get(e.label, 0)}
            for e in events
        ]

    def _create_segment_map(self):
        """データセットの全長を計算するために、各ファイルから何セグメント取れるかを事前に計算します。"""
        self.segment_map = []
        self.cumulative_segments = [0]
        for rec_idx, record in enumerate(self.records):
            try:
                if self.audio_backend == "packed":
                    variants = self._get_record_packed_variants(record)
                    if not variants:
                        raise FileNotFoundError(f"No packed variants found for {record['stems_dir']}")
                    reference_variant = variants[0] if 0 in variants else variants[sorted(variants)[0]]
                    total_frames = min(variant.num_frames for variant in variants.values())
                    total_sec = total_frames / float(reference_variant.sample_rate)
                else:
                    stems_dir = Path(record["stems_dir"]).expanduser()
                    info = sf.info(str(stems_dir / f"{stems_dir.name}_{self.stem_order[0]}.wav"))
                    total_sec = info.frames / info.samplerate

                # random_crop=Trueの場合は、曲長がセグメント長より長ければ1セグメントとする
                if self.random_crop:
                    num_segments = 1 if total_sec >= self.segment_seconds else 0
                else:
                    num_segments = int(total_sec // self.segment_seconds)

                if num_segments > 0:
                    self.segment_map.append((rec_idx, total_sec))
                    self.cumulative_segments.append(self.cumulative_segments[-1] + num_segments)
            except Exception as e:
                print(f"Warning: Could not read info for {record['stems_dir']}. Skipping. Error: {e}")

    def _get_segment_info(self, index: int) -> Tuple[Dict, float]:
        """グローバルindexから、対象レコードとセグメント開始時間を返します。"""
        record_group_idx = bisect.bisect_right(self.cumulative_segments, index) - 1
        record_idx, total_sec = self.segment_map[record_group_idx]
        record = self.records[record_idx]

        if self.random_crop:
            change_times = self._get_chord_change_times(record)
            segment_start_sec = self._sample_chord_change_start(total_sec, change_times)
        else:
            start_index_of_group = self.cumulative_segments[record_group_idx]
            local_segment_idx = index - start_index_of_group
            segment_start_sec = float(local_segment_idx * self.segment_seconds)

        return record, segment_start_sec

    def _sample_chord_change_start(self, total_sec: float, change_times: List[float]) -> float:
        """コードチェンジ時刻をセグメント開始位置として返す。"""
        max_start = max(float(total_sec - self.segment_seconds), 0.0)
        if max_start <= 0.0:
            return 0.0
        if not change_times:
            # コードチェンジがない曲はフォールバックで一様サンプリング
            return float(np.random.uniform(0.0, max_start))

        start = float(random.choice(change_times))
        return float(np.clip(start, 0.0, max_start))

    def _get_chord_change_times(self, record: Dict[str, Any]) -> List[float]:
        """コードラベルからコードが変化する時刻のリストを取得する（キャッシュ付き）。"""
        chord_label_path = str(record.get("chord_label_path", ""))
        if not chord_label_path:
            return []

        cached = self.chord_change_cache.get(chord_label_path)
        if cached is not None:
            return cached

        try:
            events = read_chords_jsonl(Path(chord_label_path))
        except Exception as e:
            print(f"Warning: Failed to read chord label file '{chord_label_path}': {e}")
            self.chord_change_cache[chord_label_path] = []
            return []

        if len(events) < 2:
            self.chord_change_cache[chord_label_path] = []
            return []

        events = sorted(events, key=lambda e: e.start_time)
        change_times: List[float] = []
        prev = events[0]
        for event in events[1:]:
            # root bass または quality が変わった時点をコードチェンジとみなす
            if event.root != prev.root or event.bass != prev.bass or event.quality != prev.quality:
                change_times.append(float(event.start_time))
            prev = event

        self.chord_change_cache[chord_label_path] = change_times
        return change_times

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _load_json_map(path: Path) -> Dict[str, int]:
        with path.open("r", encoding="utf-8") as f:
            # quality.jsonは {"0": "maj", ...} という形式なので、キーと値を反転させる
            str_to_int = {v: int(k) for k, v in json.load(f).items()}
        return str_to_int
