from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import List

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
    args = parser.parse_args()

    input_path: Path = args.input
    output_root: Path = args.out_dir
    output_root.mkdir(parents=True, exist_ok=True)

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

    separate_config = SeparationConfig(skip_existing=True)
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
