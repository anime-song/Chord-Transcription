import json
import os
import sys
import argparse
from tqdm.auto import tqdm


def create_quality_count_file(data_folder: str, quality_definition_file: str, output_file: str):
    """
    指定されたフォルダ内のJSONLファイルをスキャンし、コードクオリティの出現回数を集計して、
    結果をJSONファイルとして保存する関数。

    Args:
        data_folder (str): JSONLファイルが格納されているフォルダへのパス。
        quality_definition_file (str): クオリティの定義が記述されたJSONファイルへのパス。
        output_file (str): 集計結果を保存するJSONファイルへのパス。
    """
    # quality.jsonを読み込み、カウンターを初期化する
    try:
        with open(quality_definition_file, "r", encoding="utf-8") as f:
            # all_qualities は {"0": "maj", "1": "min", ...} のような辞書を想定
            all_qualities_map = json.load(f)

        # クオリティ名（スペース除去後）をキーとして、カウントを0で初期化
        quality_counts = {quality.replace(" ", ""): 0 for quality in all_qualities_map.values()}
        print(f"カウンターを初期化しました。対象クオリティ数: {len(quality_counts)}")

    except FileNotFoundError:
        print(f"エラー: クオリティ定義ファイル '{quality_definition_file}' が見つかりません。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"エラー: '{quality_definition_file}' は有効なJSONファイルではありません。")
        sys.exit(1)

    # 指定されたフォルダ内の.jsonlファイルを検索
    try:
        jsonl_files = [f for f in os.listdir(data_folder) if f.endswith(".jsonl")]
        if not jsonl_files:
            print(f"警告: '{data_folder}' 内に.jsonlファイルが見つかりませんでした。")
            return
        print(f"{len(jsonl_files)}個の.jsonlファイルを検出しました。集計を開始します...")
    except FileNotFoundError:
        print(f"エラー: データフォルダ '{data_folder}' が見つかりません。")
        sys.exit(1)

    # 各ファイルを読み込み、qualityをカウント
    total_lines = 0
    for filename in tqdm(jsonl_files, desc="ファイルを処理中"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    # qualityキーが存在しない場合も考慮
                    quality = data.get("quality")
                    if quality is not None:
                        quality = quality.replace(" ", "")
                        quality_counts[quality] += 1

                except Exception:
                    print(f"{filepath} QUALITY:{quality}が正しく読み込めません")
                    exit()

    print(f"全ファイルの読み込みが完了しました。総行数: {total_lines}")

    # カウント結果をquality.jsonのID順に並べ替えたリストを作成
    # all_qualities_mapのキーは文字列の"0", "1", ... なので、数値に変換してソート
    sorted_ids = sorted(all_qualities_map.keys(), key=int)

    final_counts_list = []
    for quality_id in sorted_ids:
        quality_name = all_qualities_map[quality_id].replace(" ", "")
        count = quality_counts.get(quality_name, 0)
        final_counts_list.append(count)

    # 結果をJSONファイルに保存
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_counts_list, f, indent=4)
        print(f"集計結果を '{output_file}' に保存しました。")
        print(f"   最初の5件: {final_counts_list[:5]}")

    except IOError as e:
        print(f"エラー: ファイル '{output_file}' の書き込みに失敗しました。: {e}")
        sys.exit(1)


def main():
    """
    コマンドライン引数を処理し、メインのカウント関数を実行します。
    """
    parser = argparse.ArgumentParser(
        description="データセット内のコードクオリティの出現回数を集計し、JSONファイルとして保存します。"
    )
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        default="./dataset/chords_normalize",
        help="JSONLファイルが格納されているフォルダへのパス。",
    )
    parser.add_argument(
        "-q",
        "--quality_definition",
        type=str,
        default="./data/quality.json",
        help="クオリティの定義が記述されたJSONファイルへのパス (例: quality.json)。",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data/quality_freq_count.json",
        help="集計結果を保存するJSONファイルへのパス。",
    )
    args = parser.parse_args()

    create_quality_count_file(args.data_folder, args.quality_definition, args.output)


if __name__ == "__main__":
    main()
