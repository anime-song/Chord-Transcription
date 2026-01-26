from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import torchaudio
import torchaudio.functional as AF


def _iter_audio_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    if not path.exists():
        raise FileNotFoundError(f"入力パスが存在しません: {path}")

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".wav", ".mp3"}:
            yield candidate


def resample_in_place(file_path: Path, target_sample_rate: int) -> bool:
    metadata = torchaudio.info(str(file_path))
    source_sample_rate = metadata.sample_rate
    if source_sample_rate == target_sample_rate:
        return False

    waveform, _ = torchaudio.load(str(file_path))
    waveform = AF.resample(waveform, orig_freq=source_sample_rate, new_freq=target_sample_rate)
    torchaudio.save(str(file_path), waveform, sample_rate=target_sample_rate)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定したオーディオファイル（単体またはフォルダ配下）を上書きリサンプリングします。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./dataset/songs_separated"),
        help="リサンプリング対象のファイル、またはフォルダ",
    )
    parser.add_argument("--resample-rate", type=int, default=22050, help="上書きリサンプリング先のサンプルレート（Hz）")
    parser.add_argument("--workers", type=int, default=4, help="並列ワーカー数")
    args = parser.parse_args()

    try:
        targets: List[Path] = list(_iter_audio_files(args.input))
    except FileNotFoundError as exc:
        print(exc)
        return

    if not targets:
        print("リサンプリング対象ファイルが見つかりませんでした。")
        return

    worker_count = min(args.workers, len(targets)) or 1
    processed = 0
    resampled = 0
    skipped = 0
    print(f"リサンプリング対象: {len(targets)} ファイル (sample_rate -> {args.resample_rate})")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(resample_in_place, path, args.resample_rate): path for path in targets
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            updated: bool | None = None
            try:
                updated = future.result()
            except Exception as exc:
                print(f"[ERROR] Failed to resample {path}: {exc}")
            else:
                if updated:
                    resampled += 1
                else:
                    skipped += 1
            finally:
                processed += 1
                if processed % 10 == 0 or processed == len(targets):
                    print(f"Progress: {processed}/{len(targets)}")

    print(f"リサンプリングが完了しました。 resampled={resampled}, skipped={skipped}")


if __name__ == "__main__":
    main()
