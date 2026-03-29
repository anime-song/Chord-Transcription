from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import soundfile as sf

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_STEM_NAMES = ("vocals", "drums", "bass", "other", "piano", "guitar")
PITCH_SUFFIX_PATTERN = re.compile(r"_pitch_(-?\d+)st$")


@dataclass(frozen=True)
class PackTask:
    song_id: str
    semitone: int
    stem_names: tuple[str, ...]
    stem_paths: tuple[str, ...]
    output_array_path: str
    output_metadata_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack separated stem WAV files into per-song npy/json files for faster training reads."
    )
    parser.add_argument("--songs-separated-dir", type=Path, default=Path("./dataset/songs_separated"))
    parser.add_argument("--output-dir", type=Path, default=Path("./dataset/songs_packed"))
    parser.add_argument("--song-id", dest="song_ids", action="append", default=None)
    parser.add_argument("--song-limit", type=int, default=None)
    parser.add_argument("--stem-names", nargs="+", default=list(DEFAULT_STEM_NAMES))
    parser.add_argument("--allowed-pitch-shifts", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-original", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--chunk-frames", type=int, default=1048576)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--delete-packed-stems",
        "--delete-source-stems",
        dest="delete_source_stems",
        action="store_true",
        help="Delete source stem WAV files after packed output is created or a valid packed file is confirmed.",
    )
    return parser.parse_args()


def iter_song_dirs(songs_separated_dir: Path) -> Iterable[Path]:
    for song_dir in sorted(songs_separated_dir.iterdir()):
        if song_dir.is_dir():
            yield song_dir


def discover_stem_variants(song_dir: Path, song_id: str, stem_names: Sequence[str]) -> Dict[int, Dict[str, Path]]:
    variants: Dict[int, Dict[str, Path]] = {}
    for stem_name in stem_names:
        prefix = f"{song_id}_{stem_name}"
        for wav_path in song_dir.glob(f"{glob.escape(prefix)}*.wav"):
            stem_id = wav_path.stem
            if not stem_id.startswith(prefix):
                continue

            suffix = stem_id[len(prefix) :]
            if suffix == "":
                semitone = 0
            else:
                match = PITCH_SUFFIX_PATTERN.fullmatch(suffix)
                if match is None:
                    continue
                semitone = int(match.group(1))

            variants.setdefault(semitone, {})[stem_name] = wav_path

    complete_variants: Dict[int, Dict[str, Path]] = {}
    for semitone, stem_map in variants.items():
        if all(stem_name in stem_map for stem_name in stem_names):
            complete_variants[semitone] = {stem_name: stem_map[stem_name] for stem_name in stem_names}
    return complete_variants


def select_song_dirs(song_dirs: Iterable[Path], args: argparse.Namespace) -> list[Path]:
    selected = list(song_dirs)
    if args.song_ids:
        requested = set(args.song_ids)
        selected = [song_dir for song_dir in selected if song_dir.name in requested]
    if args.song_limit is not None:
        selected = selected[: args.song_limit]
    return selected


def select_semitones(variants: Dict[int, Dict[str, Path]], args: argparse.Namespace) -> list[int]:
    semitones = sorted(variants)
    if args.exclude_original:
        semitones = [semitone for semitone in semitones if semitone != 0]
    if args.allowed_pitch_shifts is not None:
        allowed = {int(semitone) for semitone in args.allowed_pitch_shifts}
        semitones = [semitone for semitone in semitones if semitone in allowed]
    return semitones


def output_paths(output_dir: Path, song_id: str, semitone: int) -> tuple[Path, Path]:
    file_stem = f"{song_id}_stems_pitch_{semitone}st"
    song_output_dir = output_dir / song_id
    return song_output_dir / f"{file_stem}.npy", song_output_dir / f"{file_stem}.json"


def build_tasks(
    song_dirs: Iterable[Path], stem_names: tuple[str, ...], output_dir: Path, args: argparse.Namespace
) -> list[PackTask]:
    tasks: list[PackTask] = []
    for song_dir in song_dirs:
        song_id = song_dir.name
        variants = discover_stem_variants(song_dir, song_id, stem_names)
        for semitone in select_semitones(variants, args):
            array_path, metadata_path = output_paths(output_dir, song_id, semitone)
            tasks.append(
                PackTask(
                    song_id=song_id,
                    semitone=semitone,
                    stem_names=stem_names,
                    stem_paths=tuple(str(variants[semitone][stem_name]) for stem_name in stem_names),
                    output_array_path=str(array_path),
                    output_metadata_path=str(metadata_path),
                )
            )
    return tasks


def load_existing_metadata(metadata_path: Path) -> Optional[dict]:
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def should_skip_existing(task: PackTask, storage_dtype: np.dtype, force: bool) -> bool:
    array_path = Path(task.output_array_path)
    metadata_path = Path(task.output_metadata_path)
    if force or not array_path.exists() or not metadata_path.exists():
        return False

    metadata = load_existing_metadata(metadata_path)
    if metadata is None:
        return False

    return (
        metadata.get("song_id") == task.song_id
        and int(metadata.get("semitone", 9999)) == task.semitone
        and tuple(metadata.get("stem_names", [])) == task.stem_names
        and metadata.get("storage_dtype") == storage_dtype.name
    )


def delete_task_source_stems(task: PackTask) -> int:
    deleted_count = 0
    for stem_path_str in task.stem_paths:
        stem_path = Path(stem_path_str)
        try:
            stem_path.unlink()
            deleted_count += 1
        except FileNotFoundError:
            continue
    return deleted_count


def pack_song_variant(
    task: PackTask,
    storage_dtype: np.dtype,
    chunk_frames: int,
    force: bool,
    delete_source_stems: bool,
) -> tuple[str, Path, int]:
    array_path = Path(task.output_array_path)
    metadata_path = Path(task.output_metadata_path)

    if should_skip_existing(task, storage_dtype=storage_dtype, force=force):
        deleted_count = delete_task_source_stems(task) if delete_source_stems else 0
        return "skipped", array_path, deleted_count

    array_path.parent.mkdir(parents=True, exist_ok=True)

    stem_paths = [Path(stem_path) for stem_path in task.stem_paths]
    stem_infos = [sf.info(str(stem_path)) for stem_path in stem_paths]
    sample_rate = int(stem_infos[0].samplerate)
    channels_per_stem = int(stem_infos[0].channels)
    num_frames = min(int(stem_info.frames) for stem_info in stem_infos)
    num_channels = len(task.stem_names) * channels_per_stem

    if any(int(stem_info.samplerate) != sample_rate for stem_info in stem_infos):
        raise ValueError(f"Mismatched sample rates in {task.song_id}")
    if any(int(stem_info.channels) != channels_per_stem for stem_info in stem_infos):
        raise ValueError(f"Mismatched channel counts in {task.song_id}")

    packed = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(num_channels, num_frames),
    )

    for stem_index, stem_path in enumerate(stem_paths):
        channel_start = stem_index * channels_per_stem
        written_frames = 0

        with sf.SoundFile(str(stem_path), mode="r") as audio_file:
            while written_frames < num_frames:
                frames_to_read = min(chunk_frames, num_frames - written_frames)
                block = audio_file.read(frames_to_read, dtype="float32", always_2d=True)
                if block.size == 0:
                    break

                block = block.T
                if block.shape[0] == 1 and channels_per_stem == 2:
                    block = np.repeat(block, 2, axis=0)
                elif block.shape[0] != channels_per_stem:
                    raise ValueError(f"Expected {channels_per_stem} channels in {stem_path}, found {block.shape[0]}")

                if block.dtype != storage_dtype:
                    block = block.astype(storage_dtype, copy=False)

                block_frames = block.shape[1]
                packed[
                    channel_start : channel_start + channels_per_stem,
                    written_frames : written_frames + block_frames,
                ] = block
                written_frames += block_frames

        if written_frames < num_frames:
            packed[
                channel_start : channel_start + channels_per_stem,
                written_frames:num_frames,
            ] = 0

    del packed

    metadata = {
        "song_id": task.song_id,
        "semitone": task.semitone,
        "sample_rate": sample_rate,
        "channels_per_stem": channels_per_stem,
        "num_channels": num_channels,
        "num_frames": num_frames,
        "storage_dtype": storage_dtype.name,
        "stem_names": list(task.stem_names),
        "source_kind": "songs_separated",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    deleted_count = delete_task_source_stems(task) if delete_source_stems else 0
    return "packed", array_path, deleted_count


def execute_pack_task(
    task: PackTask,
    storage_dtype_name: str,
    chunk_frames: int,
    force: bool,
    delete_source_stems: bool,
) -> tuple[str, str, int]:
    status, array_path, deleted_count = pack_song_variant(
        task=task,
        storage_dtype=np.dtype(storage_dtype_name),
        chunk_frames=chunk_frames,
        force=force,
        delete_source_stems=delete_source_stems,
    )
    return status, str(array_path), deleted_count


def summarize_directory_size(paths: Iterable[Path]) -> float:
    total_bytes = 0
    for path in paths:
        if path.exists():
            total_bytes += path.stat().st_size
    return total_bytes / (1024.0 * 1024.0)


def iter_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def main() -> None:
    args = parse_args()
    songs_separated_dir = args.songs_separated_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    storage_dtype = np.dtype(args.dtype)
    stem_names = tuple(str(stem_name) for stem_name in args.stem_names)
    num_jobs = max(1, int(args.jobs))

    if not songs_separated_dir.exists():
        raise FileNotFoundError(f"songs_separated_dir not found: {songs_separated_dir}")

    song_dirs = select_song_dirs(iter_song_dirs(songs_separated_dir), args)
    if not song_dirs:
        raise ValueError("No song directories matched the requested filters")

    tasks = build_tasks(song_dirs, stem_names=stem_names, output_dir=output_dir, args=args)
    if not tasks:
        raise ValueError("No song/semitone combinations matched the requested filters")

    print(
        f"songs={len(song_dirs)}, variants={len(tasks)}, dtype={storage_dtype.name}, "
        f"jobs={num_jobs}, chunk_frames={args.chunk_frames}, output_dir={output_dir}"
    )

    packed_paths: list[Path] = []
    packed_count = 0
    skipped_count = 0
    deleted_stem_count = 0

    if num_jobs == 1:
        for task in iter_progress(tasks, desc="Packing stems", unit="variant"):
            status, array_path, deleted_count = pack_song_variant(
                task=task,
                storage_dtype=storage_dtype,
                chunk_frames=args.chunk_frames,
                force=args.force,
                delete_source_stems=args.delete_source_stems,
            )
            packed_paths.append(array_path)
            deleted_stem_count += deleted_count
            if status == "packed":
                packed_count += 1
            else:
                skipped_count += 1
    else:
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            futures = {
                executor.submit(
                    execute_pack_task,
                    task,
                    storage_dtype.name,
                    args.chunk_frames,
                    args.force,
                    args.delete_source_stems,
                ): task
                for task in tasks
            }
            for future in iter_progress(
                as_completed(futures), total=len(futures), desc="Packing stems", unit="variant"
            ):
                status, array_path_str, deleted_count = future.result()
                packed_paths.append(Path(array_path_str))
                deleted_stem_count += deleted_count
                if status == "packed":
                    packed_count += 1
                else:
                    skipped_count += 1

    packed_size_mib = summarize_directory_size(packed_paths)
    print(f"Packed variants: {packed_count}")
    print(f"Skipped variants: {skipped_count}")
    print(f"Deleted source stems: {deleted_stem_count}")
    print(f"Packed size: {packed_size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
