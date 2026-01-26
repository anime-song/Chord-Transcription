import json
import random
import bisect
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional, Set, Sequence

import librosa
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset
from dlchordx import Tone

from .processing import LabelProcessor, ChordEvent, EventSpan
from .label_shifter import LabelShifter

STEM_ORDER_DEFAULT: Tuple[str, ...] = ("vocals", "drums", "bass", "other", "piano", "guitar")
REQUIRED_CHORD_STEMS: Tuple[str, ...] = ("other", "piano", "guitar")
REQUIRED_STEM_MIN_RMS: float = 1e-4


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


def ensure_required_chord_stem(
    stems: List[np.ndarray],
    stem_order: Sequence[str],
    required_indices: Sequence[int],
    stems_to_mute: Set[str],
    stems_dir: Path,
    target_sample_rate: int,
    start_seconds: float,
    segment_seconds: float,
    shift_semitones: int,
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
        stems = load_audio_stems_segment(
            stems_dir=stems_dir,
            target_sample_rate=target_sample_rate,
            start_seconds=start_seconds,
            segment_seconds=segment_seconds,
            stem_order=stem_order,
            stems_to_mute=mute_set,
            shift_semitones=shift_semitones,
        )
        if has_active_required(stems, mute_set):
            return stems, mute_set

    return stems, mute_set


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


def load_audio_stems_segment(
    stems_dir: Path,
    target_sample_rate: int,
    start_seconds: float,
    segment_seconds: float,
    stem_order: Iterable[str],
    stems_to_mute: Optional[Set[str]] = None,
    shift_semitones: int = 0,
) -> np.ndarray:
    """
    各ステムを個別のNumpy配列としてリストで読み込みます。
    """
    waves: List[np.ndarray] = []
    for stem in stem_order:
        # このステムをミュートすべきかチェック
        if stem in stems_to_mute:
            # ファイルを読まずに無音の配列を生成
            num_samples = int(segment_seconds * target_sample_rate)
            silent_audio = np.zeros((2, num_samples), dtype="float32")
            waves.append(silent_audio)
            continue

        stem_filename_base = f"{stems_dir.name}_{stem}"
        if shift_semitones != 0:
            pitch_suffix = f"_pitch_{shift_semitones}st"
            stem_filename = f"{stem_filename_base}{pitch_suffix}.wav"
        else:
            stem_filename = f"{stem_filename_base}.wav"

        stem_path = stems_dir / stem_filename
        if not stem_path.exists():
            raise FileNotFoundError(f"Stem file not found: {stem_path}")

        # soundfileで必要な区間だけを効率的に読み込み
        info = sf.info(str(stem_path))
        start_frame = int(start_seconds * info.samplerate)
        num_frames = int(segment_seconds * info.samplerate)
        audio, sr = sf.read(stem_path, start=start_frame, frames=num_frames, dtype="float32", always_2d=True)
        audio = audio.T  # (channels, samples)

        # 必要であればリサンプリング
        if sr != target_sample_rate:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sample_rate)

        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)  # モノラルならステレオに変換

        waves.append(audio)

    # 長さを揃える
    max_len = int(segment_seconds * target_sample_rate)
    padded_waves = []
    for w in waves:
        pad_len = max_len - w.shape[1]
        if pad_len > 0:
            padded_w = np.pad(w, ((0, 0), (0, pad_len)))
            padded_waves.append(padded_w)
        else:
            padded_waves.append(w[:, :max_len])

    return padded_waves


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
        require_structure_label: bool = False,
    ):
        super().__init__()
        self.records = self._load_jsonl(jsonl_path)
        if require_structure_label:
            before_count = len(self.records)
            self.records = [r for r in self.records if "section_label_path" in r]
            print(f"Filtered records by structure label: {before_count} -> {len(self.records)}")

        if len(self.records) == 0:
            raise RuntimeError(f"No records in {jsonl_path} (after filter: {require_structure_label})")

        self.label_processor = label_processor

        # 設定ファイルから必要なパラメータを取得
        self.data_cfg = config["data_loader"]
        self.sample_rate = self.data_cfg["sample_rate"]
        self.segment_seconds = self.data_cfg["segment_seconds"]
        self.segment_samples = int(self.segment_seconds * self.sample_rate)
        self.stem_order = self.data_cfg["stem_order"]
        self.random_crop = random_crop

        # ステム拡張の設定
        self.stem_aug_cfg = self.data_cfg.get("stem_augmentation", {"enabled": False})
        self.stem_aug_enabled = self.stem_aug_cfg.get("enabled", False) and self.random_crop
        if self.stem_aug_enabled:
            self.dropout_cfg = self.stem_aug_cfg.get("dropout", {})
            self.mixing_cfg = self.stem_aug_cfg.get("mixing", {})
            print(
                f"Stem augmentation enabled: Dropout(p={self.dropout_cfg.get('p', 0)}), Mixing(p={self.mixing_cfg.get('p', 0)})"
            )

        # ピッチ拡張の設定
        self.pitch_aug_cfg = self.data_cfg.get("pitch_augmentation", {"enabled": False})
        self.pitch_aug_enabled = self.pitch_aug_cfg.get("enabled", False) and self.random_crop
        if self.pitch_aug_enabled:
            self.pitch_semitone_range = self.pitch_aug_cfg["semitone_range"]
            self.pitch_aug_p = self.pitch_aug_cfg.get("p", 0.5)
            print(f"Pitch augmentation enabled: p={self.pitch_aug_p}, semitone_range={self.pitch_semitone_range}")
        self.label_shifter = LabelShifter()

        # ドラム以外のステムのインデックスを特定
        self.nondrum_indices = [i for i, stem in enumerate(self.stem_order) if stem != "drums"]
        self.required_stem_indices = [i for i, stem in enumerate(self.stem_order) if stem in REQUIRED_CHORD_STEMS]

        self.quality_to_index = self._load_json_map(Path(config["data"]["quality_json_path"]))

        # 音楽構造のラベル定義を読み込む (あれば)
        self.structure_to_index = {}
        if "structure_json_path" in config["data"]:
            self.structure_to_index = self._load_json_map(Path(config["data"]["structure_json_path"]))
            # 'N' (None) がなければ 0 に割り当てる等の処理が必要だが、
            # music_structures.json に 'N' が含まれていることを前提とする
            print(f"Loaded structure labels: {len(self.structure_to_index)} classes")
        else:
            # フォールバック: data/music_structures.json があれば読む
            default_struct_json = Path("data/music_structures.json")
            if default_struct_json.exists():
                self.structure_to_index = self._load_json_map(default_struct_json)
                print(f"Loaded structure labels from default path: {len(self.structure_to_index)} classes")

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

    def __getitem__(self, index: int) -> Dict:
        # グローバルなindexから、どのファイルのどのセグメントかを特定
        record, segment_start_sec = self._get_segment_info(index)
        stems_dir = Path(record["stems_dir"])

        # ピッチ拡張
        shift_semitones = 0
        if self.pitch_aug_enabled and random.random() < self.pitch_aug_p:
            shift_semitones = random.randint(self.pitch_semitone_range[0], self.pitch_semitone_range[1])

        # ステム・ドロップアウト
        stems_to_mute: Set[str] = set()
        if self.stem_aug_enabled:
            stems_to_mute = sample_stem_dropout(self.stem_order, self.dropout_cfg, self.required_stem_indices)

        # オーディオの読み込み
        stems = load_audio_stems_segment(
            stems_dir=stems_dir,
            target_sample_rate=self.sample_rate,
            start_seconds=segment_start_sec,
            segment_seconds=self.segment_seconds,
            stem_order=self.stem_order,
            stems_to_mute=stems_to_mute,
            shift_semitones=shift_semitones,
        )

        if self.stem_aug_enabled:
            stems, stems_to_mute = ensure_required_chord_stem(
                stems=stems,
                stem_order=self.stem_order,
                required_indices=self.required_stem_indices,
                stems_to_mute=stems_to_mute,
                stems_dir=stems_dir,
                target_sample_rate=self.sample_rate,
                start_seconds=segment_start_sec,
                segment_seconds=self.segment_seconds,
                shift_semitones=shift_semitones,
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

        # 全ステムを結合
        audio = np.concatenate(stems, axis=0)

        # ラベル読み込み
        chord_events = read_chords_jsonl(Path(record["chord_label_path"]))
        key_events = read_tsv(Path(record["key_label_path"]))
        tempo_events = read_tsv(Path(record["tempo_label_path"]))

        # 音楽構造ラベルの読み込み (任意)
        structure_events = []
        if "section_label_path" in record and self.structure_to_index:
            try:
                structure_events = read_tsv(Path(record["section_label_path"]))
            except Exception as e:
                print(f"Warning: Failed to read section label for {record['basename']}: {e}")

        # オーディオのサンプル長から、ズレのない正確なフレーム数を計算
        num_frames = self.label_processor.get_num_frames(self.segment_samples)

        # ラベルイベントを、Processorが扱える「インデックス付きスパン」に変換
        root_spans, bass_spans, quality_spans = self._index_chord_events(chord_events)
        key_spans = self._index_key_events(key_events)
        tempo_spans = self._value_tempo_events(tempo_events)
        structure_spans = self._index_structure_events(structure_events)

        # Processorを使って、各ターゲットのフレーム系列を生成
        root_frames = self.label_processor.spans_to_frames(root_spans, num_frames, segment_start_sec)
        bass_frames = self.label_processor.spans_to_frames(bass_spans, num_frames, segment_start_sec)
        quality_frames = self.label_processor.spans_to_frames(quality_spans, num_frames, segment_start_sec)
        key_frames = self.label_processor.spans_to_frames(key_spans, num_frames, segment_start_sec)
        tempo_frames = self.label_processor.spans_to_float_frames(tempo_spans, num_frames, segment_start_sec)

        chord25_frames = self.label_processor.chords_to_25d_frames(chord_events, num_frames, segment_start_sec)
        boundary_frames = self.label_processor.chords_to_boundary_frames(chord_events, num_frames, segment_start_sec)

        batch = {
            "audio": torch.from_numpy(audio),
            "root_index": torch.from_numpy(root_frames).to(torch.long),
            "bass_index": torch.from_numpy(bass_frames).to(torch.long),
            "quality_index": torch.from_numpy(quality_frames).to(torch.long),
            "key_index": torch.from_numpy(key_frames).to(torch.long),
            "tempo_value": torch.from_numpy(tempo_frames),
            "chord25": torch.from_numpy(chord25_frames),
            "boundary": torch.from_numpy(boundary_frames),
        }

        # 音楽構造のフレーム生成 (Function & Boundary)
        if self.structure_to_index:
            structure_function_frames = self.label_processor.spans_to_frames(
                structure_spans, num_frames, segment_start_sec
            )
            structure_boundary_frames = self.label_processor.sections_to_boundary_frames(
                structure_spans, num_frames, segment_start_sec
            )
            batch["structure_function_index"] = torch.from_numpy(structure_function_frames).to(torch.long)
            batch["structure_boundary"] = torch.from_numpy(structure_boundary_frames)

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

    def _value_tempo_events(self, events: List[EventSpan]) -> List[Dict]:
        """EventSpanを、テンポのfloat値付きスパンに変換します。"""
        return [{"start_time": e.start_time, "end_time": e.end_time, "value": float(e.label)} for e in events]

    def _index_structure_events(self, events: List[EventSpan]) -> List[Dict]:
        """EventSpanを、構造ラベルのインデックス付きスパンに変換します。"""
        # structure_to_indexが空なら空リストを返す
        if not self.structure_to_index:
            return []

        return [
            {
                "start_time": e.start_time,
                "end_time": e.end_time,
                "idx": self.structure_to_index.get(e.label, self.label_processor.ignore_index),
            }
            for e in events
        ]

    def _create_segment_map(self):
        """データセットの全長を計算するために、各ファイルから何セグメント取れるかを事前に計算します。"""
        self.segment_map = []
        self.cumulative_segments = [0]
        for rec_idx, record in enumerate(self.records):
            try:
                stems_dir = Path(record["stems_dir"])
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
            segment_start_sec = np.random.randint(0, max(total_sec - self.segment_seconds, 1))
        else:
            start_index_of_group = self.cumulative_segments[record_group_idx]
            local_segment_idx = index - start_index_of_group
            segment_start_sec = float(local_segment_idx * self.segment_seconds)

        return record, segment_start_sec

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


class UnlabeledChordDataset(Dataset):
    """ラベルなしデータセット"""

    def __init__(
        self,
        root_dir: Path,
        config: Dict[str, Any],
        random_crop: bool = True,
    ) -> None:
        super().__init__()

        self.data_cfg = config["data_loader"]
        self.sample_rate = self.data_cfg["sample_rate"]
        self.segment_seconds = self.data_cfg["segment_seconds"]
        self.stem_order = tuple(self.data_cfg["stem_order"])
        self.random_crop = random_crop

        self.stem_aug_cfg = self.data_cfg.get("stem_augmentation", {"enabled": False})
        self.stem_aug_enabled = self.stem_aug_cfg.get("enabled", False) and self.random_crop
        if self.stem_aug_enabled:
            self.dropout_cfg = self.stem_aug_cfg.get("dropout", {})
            self.mixing_cfg = self.stem_aug_cfg.get("mixing", {})
        else:
            self.dropout_cfg = {}
            self.mixing_cfg = {}

        self.nondrum_indices = [i for i, stem in enumerate(self.stem_order) if stem != "drums"]
        self.required_stem_indices = [i for i, stem in enumerate(self.stem_order) if stem in REQUIRED_CHORD_STEMS]

        self.records = self._discover_records(root_dir)
        if len(self.records) == 0:
            raise RuntimeError(f"No valid stem folders found under {root_dir}")

        self._create_segment_map()

    def __len__(self) -> int:
        return self.cumulative_segments[-1]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record, segment_start_sec = self._get_segment_info(index)
        stems_dir = Path(record["stems_dir"])

        audio = self._load_augmented_audio(stems_dir, segment_start_sec)

        batch = {"audio": torch.from_numpy(audio)}

        return batch

    def _load_augmented_audio(self, stems_dir: Path, segment_start_sec: float) -> np.ndarray:
        stems_to_mute: Set[str] = set()
        if self.stem_aug_enabled:
            stems_to_mute = sample_stem_dropout(self.stem_order, self.dropout_cfg, self.required_stem_indices)

        stems = load_audio_stems_segment(
            stems_dir=stems_dir,
            target_sample_rate=self.sample_rate,
            start_seconds=segment_start_sec,
            segment_seconds=self.segment_seconds,
            stem_order=self.stem_order,
            stems_to_mute=stems_to_mute,
            shift_semitones=0,
        )

        if self.stem_aug_enabled:
            stems, stems_to_mute = ensure_required_chord_stem(
                stems=stems,
                stem_order=self.stem_order,
                required_indices=self.required_stem_indices,
                stems_to_mute=stems_to_mute,
                stems_dir=stems_dir,
                target_sample_rate=self.sample_rate,
                start_seconds=segment_start_sec,
                segment_seconds=self.segment_seconds,
                shift_semitones=0,
            )

        self._apply_random_gain(stems)

        audio = np.concatenate(stems, axis=0)
        return audio

    def _apply_random_gain(self, stems: List[np.ndarray]) -> None:
        if not self.stem_aug_enabled or not self.nondrum_indices:
            return

        mixing_cfg = self.mixing_cfg
        prob = float(mixing_cfg.get("p", 0.0))
        if prob <= 0.0 or random.random() >= prob:
            return

        min_db = float(mixing_cfg.get("min_gain_db", -6.0))
        max_db = float(mixing_cfg.get("max_gain_db", 3.0))

        for idx in self.nondrum_indices:
            gain_db = random.uniform(min_db, max_db)
            gain_linear = 10.0 ** (gain_db / 20.0)
            stems[idx] *= gain_linear

    def _create_segment_map(self) -> None:
        self.segment_map: List[Tuple[int, float]] = []
        self.cumulative_segments: List[int] = [0]

        for rec_idx, record in enumerate(self.records):
            try:
                stems_dir = Path(record["stems_dir"])
                info = sf.info(str(stems_dir / f"{stems_dir.name}_{self.stem_order[0]}.wav"))
                total_sec = info.frames / info.samplerate

                if self.random_crop:
                    num_segments = 1 if total_sec >= self.segment_seconds else 0
                else:
                    num_segments = int(total_sec // self.segment_seconds)

                if num_segments > 0:
                    self.segment_map.append((rec_idx, total_sec))
                    self.cumulative_segments.append(self.cumulative_segments[-1] + num_segments)
            except Exception as e:  # pragma: no cover - データ欠損時のフォールバック
                print(f"Warning: Could not read info for {record['stems_dir']}. Skipping. Error: {e}")

        if len(self.cumulative_segments) == 1:
            raise RuntimeError("All unlabeled records were skipped due to missing audio stems")

    def _get_segment_info(self, index: int) -> Tuple[Dict[str, Any], float]:
        record_group_idx = bisect.bisect_right(self.cumulative_segments, index) - 1
        record_idx, total_sec = self.segment_map[record_group_idx]
        record = self.records[record_idx]

        if self.random_crop:
            segment_start_sec = np.random.randint(0, max(total_sec - self.segment_seconds, 1))
        else:
            start_index_of_group = self.cumulative_segments[record_group_idx]
            local_segment_idx = index - start_index_of_group
            segment_start_sec = float(local_segment_idx * self.segment_seconds)

        return record, segment_start_sec

    def _discover_records(self, root_dir: Path) -> List[Dict[str, Any]]:
        if not root_dir.exists():
            raise FileNotFoundError(f"Unlabeled root directory not found: {root_dir}")

        records: List[Dict[str, Any]] = []

        for subdir in sorted(root_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            if self._has_all_stems(subdir, self.stem_order):
                records.append({"stems_dir": str(subdir)})

        return records

    @staticmethod
    def _has_all_stems(stems_dir: Path, stem_order: Iterable[str]) -> bool:
        name = stems_dir.name
        for stem in stem_order:
            stem_file = stems_dir / f"{name}_{stem}.wav"
            if not stem_file.exists():
                return False
        return True
