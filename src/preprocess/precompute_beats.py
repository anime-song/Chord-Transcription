from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_STEM_NAMES = ("vocals", "drums", "bass", "other", "piano", "guitar")


@dataclass(frozen=True)
class SongTask:
    song_id: str
    stems_dir: Optional[Path]
    packed_song_dir: Optional[Path]


@dataclass(frozen=True)
class AudioSource:
    mix: np.ndarray
    sample_rate: int
    num_samples: int
    source_kind: str
    metadata: dict


def normalize_song_filter_key(song_id: str) -> str:
    """
    CLI からの --song-id 指定では、末尾空白の有無で取りこぼしやすい。
    内部の実ファイル名は保持しつつ、フィルタ比較だけは前後空白を無視する。
    """
    return str(song_id).strip().casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute beat_this beat/downbeat predictions from separated stems or packed stems and save "
            "both event times and frame-aligned masks for the chord transcription model."
        )
    )
    parser.add_argument("--pairs-jsonl", type=Path, default=None)
    parser.add_argument("--songs-separated-dir", type=Path, default=Path("./dataset/songs_separated"))
    parser.add_argument("--songs-packed-dir", type=Path, default=Path("./dataset/songs_packed"))
    parser.add_argument("--output-dir", type=Path, default=Path("./dataset/beats"))
    parser.add_argument("--audio-backend", choices=("auto", "wav", "packed"), default="auto")
    parser.add_argument("--song-id", dest="song_ids", action="append", default=None)
    parser.add_argument("--song-limit", type=int, default=None)
    parser.add_argument("--stem-names", nargs="+", default=list(DEFAULT_STEM_NAMES))
    parser.add_argument("--checkpoint", default="final0", help="beat_this checkpoint name or local checkpoint path")
    parser.add_argument("--device", default=None, help="Torch device. Default: cuda if available, else cpu.")
    parser.add_argument("--dbn", action="store_true", help="Use beat_this DBN post-processing.")
    parser.add_argument("--float16", action="store_true", help="Use torch autocast float16 inside beat_this.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate used by the chord model.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length used by the chord model.")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size used by the chord model.")
    parser.add_argument("--chunk-frames", type=int, default=1_048_576, help="Chunk size for reading and mixing audio.")
    parser.add_argument("--force", action="store_true", help="Recompute even when a matching output already exists.")
    return parser.parse_args()


def iter_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_song_tasks(args: argparse.Namespace) -> list[SongTask]:
    separated_dir = args.songs_separated_dir.expanduser().resolve()
    packed_dir = args.songs_packed_dir.expanduser().resolve()
    task_map: dict[str, SongTask] = {}

    # pairs.jsonl がある場合はそれを正として曲一覧を作る。
    if args.pairs_jsonl is not None:
        pairs_jsonl = args.pairs_jsonl.expanduser().resolve()
        if not pairs_jsonl.exists():
            raise FileNotFoundError(f"pairs_jsonl not found: {pairs_jsonl}")

        for record in load_jsonl(pairs_jsonl):
            stems_dir_value = record.get("stems_dir")
            if stems_dir_value is None:
                continue

            stems_dir = Path(stems_dir_value).expanduser().resolve()
            song_id = stems_dir.name
            packed_song_dir = packed_dir / song_id
            task_map[song_id.casefold()] = SongTask(
                song_id=song_id,
                stems_dir=stems_dir if stems_dir.exists() else None,
                packed_song_dir=packed_song_dir if packed_song_dir.exists() else None,
            )
    else:
        # pairs.jsonl が無い場合は songs_separated / songs_packed を走査して候補を集める。
        if separated_dir.exists():
            for song_dir in sorted(separated_dir.iterdir()):
                if not song_dir.is_dir():
                    continue
                key = song_dir.name.casefold()
                existing = task_map.get(key)
                task_map[key] = SongTask(
                    song_id=song_dir.name,
                    stems_dir=song_dir.resolve(),
                    packed_song_dir=existing.packed_song_dir if existing is not None else None,
                )

        if packed_dir.exists():
            for song_dir in sorted(packed_dir.iterdir()):
                if not song_dir.is_dir():
                    continue
                key = song_dir.name.casefold()
                existing = task_map.get(key)
                task_map[key] = SongTask(
                    song_id=song_dir.name,
                    stems_dir=existing.stems_dir if existing is not None else None,
                    packed_song_dir=song_dir.resolve(),
                )

    tasks = sorted(task_map.values(), key=lambda task: task.song_id.casefold())

    if args.song_ids:
        requested = {normalize_song_filter_key(song_id) for song_id in args.song_ids}
        tasks = [task for task in tasks if normalize_song_filter_key(task.song_id) in requested]
    if args.song_limit is not None:
        tasks = tasks[: args.song_limit]
    if not tasks:
        raise ValueError("No songs matched the requested filters.")
    return tasks


def load_mono_mix_from_wav_stems(stems_dir: Path, stem_names: Sequence[str], chunk_frames: int) -> AudioSource:
    stem_paths = [stems_dir / f"{stems_dir.name}_{stem_name}.wav" for stem_name in stem_names]
    missing = [str(path) for path in stem_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing stem WAV files for {stems_dir}: {missing[:3]}")

    infos = [sf.info(str(path)) for path in stem_paths]
    sample_rate = int(infos[0].samplerate)
    if any(int(info.samplerate) != sample_rate for info in infos):
        raise ValueError(f"Mismatched sample rates in {stems_dir}")

    # 各 stem をモノラル化して足し合わせる。
    num_samples = max(int(info.frames) for info in infos)
    mix = np.zeros(num_samples, dtype=np.float32)

    for stem_path in stem_paths:
        with sf.SoundFile(str(stem_path), mode="r") as audio_file:
            write_offset = 0
            while True:
                block = audio_file.read(chunk_frames, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                mono = block.mean(axis=1, dtype=np.float32)
                block_len = mono.shape[0]
                mix[write_offset : write_offset + block_len] += mono
                write_offset += block_len

    # 和音後にピークが 1 を超えることがあるので、必要なときだけ正規化する。
    peak = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
    if peak > 1.0:
        mix = mix / peak

    return AudioSource(
        mix=mix.astype(np.float32, copy=False),
        sample_rate=sample_rate,
        num_samples=num_samples,
        source_kind="wav",
        metadata={"stems_dir": str(stems_dir)},
    )


def load_mono_mix_from_packed(packed_song_dir: Path, stem_names: Sequence[str], chunk_frames: int) -> AudioSource:
    # packed は複数 semitone 版を持ちうるので、まず 0st を優先して選ぶ。
    selected_metadata_path: Optional[Path] = None
    selected_metadata: Optional[dict] = None
    fallback_metadata_path: Optional[Path] = None
    fallback_metadata: Optional[dict] = None

    for metadata_path in sorted(packed_song_dir.glob("*.json")):
        array_path = metadata_path.with_suffix(".npy")
        if not array_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        available_stems = tuple(str(stem_name) for stem_name in metadata.get("stem_names", []))
        if any(stem_name not in available_stems for stem_name in stem_names):
            continue

        semitone = int(metadata.get("semitone", 0))
        if semitone == 0:
            selected_metadata_path = metadata_path
            selected_metadata = metadata
            break
        if fallback_metadata_path is None:
            fallback_metadata_path = metadata_path
            fallback_metadata = metadata

    if selected_metadata_path is None:
        selected_metadata_path = fallback_metadata_path
        selected_metadata = fallback_metadata
    if selected_metadata_path is None or selected_metadata is None:
        raise FileNotFoundError(f"No packed variants were found for {packed_song_dir}")

    array_path = selected_metadata_path.with_suffix(".npy")
    packed = np.load(array_path, mmap_mode="r")
    sample_rate = int(selected_metadata["sample_rate"])
    channels_per_stem = int(selected_metadata["channels_per_stem"])
    num_samples = int(selected_metadata["num_frames"])
    available_stems = tuple(str(stem_name) for stem_name in selected_metadata["stem_names"])

    stem_index_map = {stem_name: idx for idx, stem_name in enumerate(available_stems)}
    mix = np.zeros(num_samples, dtype=np.float32)

    # packed は [stem*channel, time] なので、必要な stem のチャネル範囲を順に読む。
    for stem_name in stem_names:
        stem_index = stem_index_map[stem_name]
        channel_start = stem_index * channels_per_stem
        channel_end = channel_start + channels_per_stem

        for start in range(0, num_samples, chunk_frames):
            end = min(start + chunk_frames, num_samples)
            chunk = np.asarray(packed[channel_start:channel_end, start:end], dtype=np.float32)
            mix[start:end] += chunk.mean(axis=0, dtype=np.float32)

    peak = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
    if peak > 1.0:
        mix = mix / peak

    return AudioSource(
        mix=mix.astype(np.float32, copy=False),
        sample_rate=sample_rate,
        num_samples=num_samples,
        source_kind="packed",
        metadata={
            "packed_array_path": str(array_path),
            "packed_metadata_path": str(selected_metadata_path),
            "source_semitone": int(selected_metadata.get("semitone", 0)),
        },
    )


def resolve_audio_source(
    task: SongTask,
    stem_names: Sequence[str],
    audio_backend: str,
    chunk_frames: int,
) -> AudioSource:
    wav_error: Optional[Exception] = None
    packed_error: Optional[Exception] = None

    # auto のときは wav を先に試し、無ければ packed にフォールバックする。
    if audio_backend in {"auto", "wav"} and task.stems_dir is not None:
        try:
            return load_mono_mix_from_wav_stems(task.stems_dir, stem_names, chunk_frames)
        except Exception as exc:
            wav_error = exc
            if audio_backend == "wav":
                raise

    if audio_backend in {"auto", "packed"} and task.packed_song_dir is not None:
        try:
            return load_mono_mix_from_packed(task.packed_song_dir, stem_names, chunk_frames)
        except Exception as exc:
            packed_error = exc
            if audio_backend == "packed":
                raise

    raise RuntimeError(
        f"Could not resolve audio source for {task.song_id}. " f"wav_error={wav_error!r}, packed_error={packed_error!r}"
    )


def normalize_beat_output(first, second) -> tuple[np.ndarray, np.ndarray]:
    # beat_this の返り順の差異に備え、本数が少ない側を downbeat とみなす。
    arr1 = np.sort(np.asarray(first, dtype=np.float32).reshape(-1))
    arr2 = np.sort(np.asarray(second, dtype=np.float32).reshape(-1))

    if arr1.size <= arr2.size:
        downbeats, beats = arr1, arr2
    else:
        beats, downbeats = arr1, arr2
    return beats, downbeats


def event_times_to_frame_targets(
    event_times: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_frames <= 0 or event_times.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((num_frames,), dtype=np.uint8)

    # 秒単位イベントを chord model のフレーム格子へ量子化する。
    frame_indices = np.rint(event_times * sample_rate / float(hop_length)).astype(np.int64)
    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < num_frames)]
    if frame_indices.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((num_frames,), dtype=np.uint8)

    frame_indices = np.unique(frame_indices)
    mask = np.zeros((num_frames,), dtype=np.uint8)
    mask[frame_indices] = 1
    return frame_indices, mask


def should_skip_existing(
    npz_path: Path,
    metadata_path: Path,
    *,
    song_id: str,
    args: argparse.Namespace,
    selected_stems: Sequence[str],
) -> bool:
    if args.force or not npz_path.exists() or not metadata_path.exists():
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("song_id") == song_id
        and tuple(metadata.get("selected_stems", [])) == tuple(selected_stems)
        and int(metadata.get("target_sample_rate", -1)) == int(args.sample_rate)
        and int(metadata.get("hop_length", -1)) == int(args.hop_length)
        and int(metadata.get("n_fft", -1)) == int(args.n_fft)
        and str(metadata.get("checkpoint")) == str(args.checkpoint)
        and bool(metadata.get("dbn")) == bool(args.dbn)
        and bool(metadata.get("float16")) == bool(args.float16)
    )


def build_tracker(checkpoint: str, device: str, float16: bool, dbn: bool):
    try:
        from beat_this.inference import Audio2Beats
    except ImportError as exc:
        raise ImportError(
            "beat_this is required for precompute_beats.py. Install it first, for example: "
            "pip install https://github.com/CPJKU/beat_this/archive/main.zip"
        ) from exc

    return Audio2Beats(checkpoint_path=checkpoint, device=device, float16=float16, dbn=dbn)


def process_song(task: SongTask, tracker, args: argparse.Namespace, stem_names: Sequence[str]) -> str:
    song_output_dir = args.output_dir / task.song_id
    base_name = f"{task.song_id}_beat_this"
    npz_path = song_output_dir / f"{base_name}.npz"
    metadata_path = song_output_dir / f"{base_name}.json"

    if should_skip_existing(
        npz_path,
        metadata_path,
        song_id=task.song_id,
        args=args,
        selected_stems=stem_names,
    ):
        return "skipped"

    # 1. 学習に使う stem 群を 1 本のモノラル波形へまとめる。
    source = resolve_audio_source(
        task=task,
        stem_names=stem_names,
        audio_backend=args.audio_backend,
        chunk_frames=args.chunk_frames,
    )

    # 2. beat_this で beat / downbeat の秒位置を推定する。
    first, second = tracker(source.mix, source.sample_rate)
    beat_times, downbeat_times = normalize_beat_output(first, second)

    # 3. 音声長をはみ出したイベントを落とし、必要なら後段用にフレーム化する。
    duration_sec = source.num_samples / float(source.sample_rate)
    beat_times = beat_times[(beat_times >= 0.0) & (beat_times <= duration_sec + 1e-4)]
    downbeat_times = downbeat_times[(downbeat_times >= 0.0) & (downbeat_times <= duration_sec + 1e-4)]

    if source.num_samples < args.n_fft:
        num_label_frames = 0
    else:
        num_label_frames = (source.num_samples - args.n_fft) // args.hop_length + 1

    beat_frame_indices, beat_mask = event_times_to_frame_targets(
        beat_times,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        num_frames=num_label_frames,
    )
    downbeat_frame_indices, downbeat_mask = event_times_to_frame_targets(
        downbeat_times,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        num_frames=num_label_frames,
    )

    # 4. 秒単位イベントを正本として保存し、同時に現在設定のフレーム cache も残す。
    song_output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        beat_times=beat_times.astype(np.float32, copy=False),
        downbeat_times=downbeat_times.astype(np.float32, copy=False),
        beat_frame_indices=beat_frame_indices,
        downbeat_frame_indices=downbeat_frame_indices,
        beat_mask=beat_mask,
        downbeat_mask=downbeat_mask,
    )

    metadata = {
        "song_id": task.song_id,
        "source_kind": source.source_kind,
        "selected_stems": list(stem_names),
        "checkpoint": str(args.checkpoint),
        "dbn": bool(args.dbn),
        "float16": bool(args.float16),
        "target_sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "n_fft": int(args.n_fft),
        "source_sample_rate": int(source.sample_rate),
        "num_samples": int(source.num_samples),
        "duration_sec": float(duration_sec),
        "num_label_frames": int(num_label_frames),
        "num_beats": int(beat_times.size),
        "num_downbeats": int(downbeat_times.size),
        **source.metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return "written"


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.float16 and str(args.device).startswith("cpu"):
        print("Warning: --float16 on CPU is disabled.")
        args.float16 = False

    stem_names = tuple(str(stem_name) for stem_name in args.stem_names)
    tasks = collect_song_tasks(args)
    tracker = build_tracker(
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        float16=bool(args.float16),
        dbn=bool(args.dbn),
    )

    print(
        f"songs={len(tasks)}, backend={args.audio_backend}, checkpoint={args.checkpoint}, "
        f"device={args.device}, output_dir={args.output_dir}"
    )

    written_count = 0
    skipped_count = 0
    failed_count = 0

    for task in iter_progress(tasks, desc="Precomputing beats", unit="song"):
        try:
            status = process_song(task, tracker, args, stem_names)
            if status == "written":
                written_count += 1
            else:
                skipped_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"Failed: {task.song_id}: {exc}")

    print(f"Written: {written_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
