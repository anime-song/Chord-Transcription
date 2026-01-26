from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

STEM_NAMES: Tuple[str, ...] = ("vocals", "drums", "bass", "other", "piano", "guitar")


def index_json_labels(directory: Path) -> Dict[str, Path]:
    """
    指定フォルダ直下の *.jsonl を、ベース名(小文字) -> パス で引ける辞書にします。
    """
    directory = directory.expanduser().resolve()
    index: Dict[str, Path] = {}
    for file_path in directory.glob("*.jsonl"):
        if file_path.is_file():
            index[file_path.stem.lower()] = file_path
    return index


def index_txt_labels(directory: Path) -> Dict[str, Path]:
    """
    指定フォルダ直下の *.txt を、ベース名(小文字) -> パス で引ける辞書にします。
    """
    directory = directory.expanduser().resolve()
    index: Dict[str, Path] = {}
    for file_path in directory.glob("*.txt"):
        if file_path.is_file():
            index[file_path.stem.lower()] = file_path
    return index


def index_subfolders_by_lower_name(root: Path) -> Dict[str, Path]:
    """
    直下のサブフォルダを、フォルダ名(小文字) -> パス で引ける辞書にします。
    """
    root = root.expanduser().resolve()
    folder_index: Dict[str, Path] = {}
    for child in root.iterdir():
        if child.is_dir():
            folder_index[child.name.lower()] = child
    return folder_index


def folder_has_all_exact_stems(stem_folder: Path) -> bool:
    """
    stem_folder に、完全一致の {stem}.wav が 6 個すべて存在するか確認します。
    """
    for stem in STEM_NAMES:
        expected_file = stem_folder / f"{stem_folder.name}_{stem}.wav"
        if not expected_file.is_file():
            return False
    return True


def write_jsonl(output_path: Path, records: List[dict]) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)  # pairs.jsonlが生成されてしまう原因
    with output_path.open("w", encoding="utf-8") as out_file:
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_validation_basenames(file_path: Path) -> Set[str]:
    """
    validation用に固定したいbasename群をファイルから読み込み、全て小文字にして返します。
    *.jsonl の場合は {"basename": "..."} を参照し、それ以外は1行1basenameのテキストとして解釈します。
    """
    file_path = file_path.expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Validation list file not found: {file_path}")

    basenames: Set[str] = set()
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                basename = record.get("basename")
                if not basename:
                    continue
                basenames.add(basename.lower())
    else:
        with file_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                basename = line.strip()
                if basename:
                    basenames.add(basename.lower())
    return basenames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="chordsの *.jsonl と keys/tempos の *.txt と songs_separated/<basename>/ を突き合わせ、stems_dir を JSONL 出力（ステムは完全一致必須）"
    )
    parser.add_argument("--chords_dir", default="./dataset/chords_normalize", type=Path)
    parser.add_argument("--keys_dir", default="./dataset/keys", type=Path)
    parser.add_argument("--tempos_dir", default="./dataset/tempos", type=Path)
    parser.add_argument("--sections_dir", default="./dataset/sections", type=Path)
    parser.add_argument("--songs_separated_dir", default="./dataset/songs_separated", type=Path)
    parser.add_argument("--output_jsonl", default="./pairs.jsonl", type=Path)
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="検証データの比率（0~1）")
    parser.add_argument(
        "--validation_list",
        type=Path,
        default=None,
        help="固定したいvalidation用basenameのリスト（*.jsonl もしくは1行1basenameのテキスト）。指定時はvalidation_ratioを無視します。",
    )
    parser.add_argument("--seed", type=int, default=42, help="シャッフル用乱数シード")

    args = parser.parse_args()

    if args.validation_list is None and not (0.0 < args.validation_ratio < 1.0):
        raise ValueError("--validation_ratio は 0 と 1 の間で指定してください。")

    chord_index = index_json_labels(args.chords_dir)
    key_index = index_txt_labels(args.keys_dir)
    tempo_index = index_txt_labels(args.tempos_dir)
    section_index: Dict[str, Path] = {}
    sections_dir = args.sections_dir.expanduser() if args.sections_dir is not None else None
    if sections_dir is not None and sections_dir.exists():
        section_index = index_txt_labels(sections_dir)
    stem_folder_index = index_subfolders_by_lower_name(args.songs_separated_dir)

    # 3つのラベルが揃っているベース名だけを候補にする
    candidate_basenames = sorted(set(chord_index) & set(key_index) & set(tempo_index))

    output_path = args.output_jsonl.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched_count = 0
    matched_records: List[dict] = []
    skipped_no_folder = 0
    skipped_missing_stems = 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for base in candidate_basenames:
            stems_dir = stem_folder_index.get(base)
            if stems_dir is None:
                skipped_no_folder += 1
                continue

            if not folder_has_all_exact_stems(stems_dir):
                skipped_missing_stems += 1
                continue

            record = {
                "basename": base,
                "chord_label_path": str(chord_index[base].resolve()),
                "key_label_path": str(key_index[base].resolve()),
                "tempo_label_path": str(tempo_index[base].resolve()),
                "stems_dir": str(stems_dir.resolve()),
            }
            section_path = section_index.get(base)
            if section_path is not None:
                record["section_label_path"] = str(section_path.resolve())
            matched_records.append(record)
            matched_count += 1

    random.seed(args.seed)
    random.shuffle(matched_records)

    total = len(matched_records)
    fixed_validation_basenames: Set[str] = set()
    if args.validation_list is not None:
        fixed_validation_basenames = load_validation_basenames(args.validation_list)
        missing = sorted(fixed_validation_basenames - {rec["basename"] for rec in matched_records})
        if missing:
            print("警告: validation_listに含まれるが今回マッチしなかったbasename:", ", ".join(missing))

    if fixed_validation_basenames:
        validation_records = [rec for rec in matched_records if rec["basename"] in fixed_validation_basenames]
        train_records = [rec for rec in matched_records if rec["basename"] not in fixed_validation_basenames]
    else:
        validation_count = max(1 if total > 0 else 0, int(total * args.validation_ratio))
        train_count = total - validation_count
        train_records = matched_records[:train_count]
        validation_records = matched_records[train_count:]

    # 出力ファイル名を決定
    base_out = args.output_jsonl.expanduser().resolve()
    train_out = base_out.with_name(f"{base_out.stem}.train.jsonl")
    valid_out = base_out.with_name(f"{base_out.stem}.validation.jsonl")

    write_jsonl(train_out, train_records)
    write_jsonl(valid_out, validation_records)

    print(f"Train  JSONL: {train_out}  ({len(train_records)} records)")
    print(f"Valid  JSONL: {valid_out}  ({len(validation_records)} records)")
    print(f"Total matched (before split): {total}")
    print(f"Skipped (no stem folder): {skipped_no_folder}")
    print(f"Skipped (missing exact stems): {skipped_missing_stems}")


if __name__ == "__main__":
    main()
