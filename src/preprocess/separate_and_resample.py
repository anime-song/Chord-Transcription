from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import inspect
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Sequence

import torch
import torchaudio
import torchaudio.functional as AF
from stem_splitter.inference import (
    SeparationConfig,
    _separate_one_file,
    load_mss_model,
    resolve_device,
)


def resample_in_place(file_path: Path, target_sample_rate: int) -> Path:
    """
    単一ファイルを読み込み、target_sample_rate へ変換して同じパスへ上書き保存します。
    """
    waveform, source_sample_rate = torchaudio.load(str(file_path))  # (channels, samples)
    if source_sample_rate != target_sample_rate:
        waveform = AF.resample(waveform, orig_freq=source_sample_rate, new_freq=target_sample_rate)
    torchaudio.save(str(file_path), waveform, sample_rate=target_sample_rate)
    return file_path


def has_valid_packed_output(packed_root: Path, song_id: str, stem_names: Sequence[str]) -> bool:
    """packed 側にこの曲の有効な成果物が 1 つでもあれば True を返します。"""
    packed_song_dir = packed_root / song_id
    if not packed_song_dir.exists():
        return False

    for metadata_path in packed_song_dir.glob(f"{glob.escape(song_id)}_stems_pitch_*.json"):
        array_path = metadata_path.with_suffix(".npy")
        if not array_path.exists():
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if metadata.get("song_id") != song_id:
            continue
        # packed 側と separator 側で stem 順が違っていても、同じ stem 集合ならスキップ対象にします。
        metadata_stem_names = tuple(str(name) for name in metadata.get("stem_names", []))
        if len(metadata_stem_names) != len(stem_names):
            continue
        if set(metadata_stem_names) != set(stem_names):
            continue
        return True

    return False


def build_separation_config(batch_size: int) -> SeparationConfig:
    """stem_splitter 側のバージョン差を吸収して SeparationConfig を作ります。"""
    config_kwargs = {"skip_existing": True}
    if "batch_size" in inspect.signature(SeparationConfig).parameters:
        config_kwargs["batch_size"] = batch_size
    return SeparationConfig(**config_kwargs)


def main() -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="指定したオーディオファイルをステム分離し、分離されたすべてのファイルを指定したサンプルレートへ上書きリサンプリングします。"
    )
    parser.add_argument(
        "--input", type=Path, default="./dataset/songs", help="入力の [.wav, .mp3] または [.wav, .mp3] を含むフォルダ"
    )
    parser.add_argument("--out-dir", type=Path, default="./dataset/songs_separated", help="ステム出力先フォルダ")
    parser.add_argument("--resample-rate", type=int, default=22050, help="上書きリサンプリング先のサンプルレート（Hz）")
    parser.add_argument("--workers", type=int, default=4, help="並列ワーカー数")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument(
        "--skip-if-packed",
        action="store_true",
        help="対応する packed 音源が既に存在する曲は、分離とリサンプリングを行わずにスキップします。",
    )
    parser.add_argument(
        "--packed-dir",
        type=Path,
        default=Path("./dataset/songs_packed"),
        help="--skip-if-packed 時に参照する packed 音源ディレクトリ",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_root: Path = args.out_dir
    output_root.mkdir(parents=True, exist_ok=True)
    packed_root = args.packed_dir.expanduser().resolve()

    if input_path.is_file():
        target_files = [input_path]
    elif input_path.is_dir():
        target_files = sorted(
            [
                p
                for p in input_path.rglob("*")
                if p.is_file() and (p.suffix.lower() == ".wav" or p.suffix.lower() == ".mp3")
            ]
        )
    else:
        raise FileNotFoundError(f"入力パスが存在しません: {input_path}")

    if not target_files:
        print("分離対象ファイルが見つかりませんでした。処理を終了します。")
        return

    separate_config = build_separation_config(batch_size=args.batch_size)

    # packed 済みの曲は、分離済み stem が消えていてもここで先に除外します。
    if args.skip_if_packed:
        if not packed_root.exists():
            raise FileNotFoundError(f"packed_dir が存在しません: {packed_root}")

        filtered_target_files: List[Path] = []
        skipped_packed = 0
        for audio_path in target_files:
            song_id = audio_path.stem
            if has_valid_packed_output(packed_root, song_id=song_id, stem_names=separate_config.stem_names):
                skipped_packed += 1
                continue
            filtered_target_files.append(audio_path)

        print(f"Packed 済みとしてスキップ: {skipped_packed} 曲")
        target_files = filtered_target_files

    if not target_files:
        print("すべての曲が packed 済みのため、分離をスキップしました。")
        return

    device = resolve_device(separate_config.device_preference)
    dtype = torch.float16 if (separate_config.use_half_precision and device.type == "cuda") else torch.float32
    model = load_mss_model(separate_config, device=device)

    total = len(target_files)
    separated = 0
    separated_paths: List[Path] = []

    for idx, wav_path in enumerate(target_files, start=1):
        try:
            result = _separate_one_file(
                wav_path,
                output_root,
                separate_config,
                model,
                device,
                dtype,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to separate {wav_path}: {exc}")
            continue

        if result:
            separated += 1
            separated_paths.extend(result.values())
        print(f"Separated: {idx}/{total} (new outputs: {separated})")

    if not separated_paths:
        print("新規のステム出力が見つからなかったため、リサンプリングをスキップしました。")
        return

    print(f"リサンプリング対象: {len(separated_paths)} ファイル")
    worker_count = min(args.workers, len(separated_paths)) or 1
    resampled = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(resample_in_place, path, args.resample_rate) for path in separated_paths]
        for future in as_completed(futures):
            future.result()
            resampled += 1
            if resampled % 10 == 0 or resampled == len(separated_paths):
                print(f"Resampled: {resampled}/{len(separated_paths)}")

    print("分離とリサンプリングが完了しました。")


if __name__ == "__main__":
    main()
