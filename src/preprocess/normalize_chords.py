from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable

from dlchordx import Chord


@dataclass
class LabeledChordSpan:
    start_time: float
    end_time: float
    chord_symbol: str


def read_tsv(tsv_path: Path) -> List[LabeledChordSpan]:
    """
    タブ区切り: start_time\tend_time\tchord を読み込みます。
    空行とコメント行（#, //）は無視します。
    """
    tsv_path = tsv_path.expanduser().resolve()
    spans: List[LabeledChordSpan] = []
    with tsv_path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            if ":" in line:
                parts = line.split(":")
            else:
                parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{tsv_path}:{line_number}: 列数が3ではありません（start\\tend\\tchord）: {raw_line!r}"
                )
            start_text, end_text, chord_text = parts
            start_time = float(start_text)
            end_time = float(end_text)
            if end_time < start_time:
                raise ValueError(f"{tsv_path}:{line_number}: end_time({end_time}) < start_time({start_time})")
            spans.append(LabeledChordSpan(start_time, end_time, chord_text.strip()))
    return spans


def convert_to_jsonl_records(spans: List[LabeledChordSpan]) -> Tuple[List[dict], int]:
    """
    TSV の各行を DLchordX で解析し、JSONL レコードに変換します。
    解析に失敗した行はスキップし、その件数を返します。
    """
    records: List[dict] = []
    skipped = 0
    for span in spans:
        try:
            if span.chord_symbol == "N.C.":
                root_text = "N"
                bass_text = "N"
                quality_text = "N"
            else:
                chord = Chord(span.chord_symbol)

                root_text = chord.root.name
                bass_text = chord.bass.name
                quality_text = chord.quality.name

        except Exception:
            skipped += 1
            continue

        records.append(
            {
                "start_time": span.start_time,
                "end_time": span.end_time,
                "root": root_text,
                "bass": bass_text,
                "quality": quality_text,
            }
        )
    return records, skipped


def write_jsonl(output_path: Path, records: Iterable[dict]) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_input_files(input_dir: Path, patterns: List[str], recursive: bool) -> List[Path]:
    """
    入力フォルダから対象ファイルを列挙します。
    """
    input_dir = input_dir.expanduser().resolve()
    files: List[Path] = []
    walker = input_dir.rglob if recursive else input_dir.glob
    for pattern in patterns:
        files.extend(p for p in walker(pattern) if p.is_file())
    files = sorted(set(files))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="入力フォルダ内の TSV/TXT をまとめて解析し、同名の JSONL を出力フォルダへ保存します。"
    )
    parser.add_argument("--input_dir", default="./dataset/chords", type=Path, help="入力フォルダ（TSV/TXT）")
    parser.add_argument(
        "--output_dir", default="./dataset/chords_normalize", type=Path, help="出力フォルダ（JSONL を生成）"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.tsv", "*.txt"],
        help="入力ファイルのパターン（デフォルト: *.tsv *.txt）",
    )
    parser.add_argument("--recursive", action="store_true", help="サブフォルダも再帰的に処理する")

    args = parser.parse_args()

    input_files = find_input_files(args.input_dir, args.patterns, args.recursive)
    if not input_files:
        print("[INFO] 入力ファイルが見つかりませんでした。")
        return

    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_records = 0
    total_skipped = 0

    for input_path in input_files:
        spans = read_tsv(input_path)
        records, skipped = convert_to_jsonl_records(spans)

        output_path = (args.output_dir / input_path.with_suffix(".jsonl").name).expanduser().resolve()
        write_jsonl(output_path, records)

        print(f"[OK] {input_path} -> {output_path}  (parsed={len(records)}, skipped={skipped})")

        total_files += 1
        total_records += len(records)
        total_skipped += skipped

    print("\n=== Summary ===")
    print(f"Files processed : {total_files}")
    print(f"Records parsed  : {total_records}")
    print(f"Lines skipped   : {total_skipped}")


if __name__ == "__main__":
    main()
