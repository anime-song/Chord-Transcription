# Chord-Transcription
[English](README.md) | **日本語**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/Chord-Transcription/blob/main/Chord_Transcription.ipynb)

コード採譜モデルを学習するリポジトリ。

SegmentModelによる文脈解釈で高精度な推論が可能です。

# データセット作成パイプライン

このドキュメントは、モデルの学習に使用するデータセットを準備するための一連の前処理パイプラインを説明します。各ステップを順番に実行してください。

-----

### Step 1. ステム分離とリサンプリング

楽曲の音声ファイルから各楽器のステム（ボーカル、ドラム、ベース、その他）を分離し、指定されたサンプリングレートに変換します。

```bash
uv run python -m src.preprocess.separate_and_resample --input <input_dir> --out-dir <output_dir>
```

  * `--input_dir`: 処理対象の音声ファイルが格納されているディレクトリ。
      * **Default**: `./dataset/songs`
  * `--out-dir`: ステム分離後の音声ファイルの保存先。
      * **Default**: `./dataset/songs_separated`

-----

### Step 2. ピッチシフトによるデータ拡張

ステム分離後の音声に対し、ピッチシフトを適用して学習データを水増しします（データ拡張）。

```bash
uv run python -m src.preprocess.pitch_shift_augment --target_dir <target_dir>
```

  * `--target_dir`: ピッチシフトを適用する音声ファイルが格納されているディレクトリ。
      * **Default**: `./dataset/songs_separated`

-----

### Step 3. コードデータの正規化

元のコード表記（例: `CM7`, `Gm`, etc.）を、モデルが学習しやすい統一された形式に正規化します。

```bash
uv run python -m src.preprocess.normalize_chords --input_dir <input_dir> --output_dir <output_dir>
```

  * `--input_dir`: 正規化前のコードデータが保存されているディレクトリ。
      * **Default**: `./dataset/chords`
  * `--output_dir`: 正規化後のコードデータの保存先。
      * **Default**: `./dataset/chords_normalize`

-----

### Step 4. 学習用ペアの作成

処理済みの音声ファイルと、対応するコード、キー、テンポの各ラベル情報を紐付けたCSVファイル（学習用・検証用ペアリスト）を作成します。

```bash
uv run python -m src.preprocess.make_pairs_csv --chords_dir <chords_dir> --keys_dir <keys_dir> --tempos_dir <tempos_dir> --songs_separated_dir <songs_separated_dir> --validation_ratio <validation_ratio>
```

  * `--chords_dir`: 正規化後のコードが保存されているディレクトリ。
  * `--keys_dir`: キー情報が保存されているディレクトリ。
  * `--tempos_dir`: テンポ情報が保存されているディレクトリ。
  * `--songs_separated_dir`: ステム分離後の音声が保存されているディレクトリ。
  * `--validation_ratio`: 全データのうち、検証用データとして分割する割合。

-----

### Step 5. コードクオリティの出現頻度計算

学習時の損失関数で使用するために、データセット全体における各コードクオリティ（`Major`, `minor`など）の出現頻度を計算します。

```bash
uv run python -m src.preprocess.count_quality_freq --data_folder <data_folder> --quality_definition <quality_definition> --output <output>
```

  * `--data_folder`: 正規化されたコードが保存されているディレクトリ。
      * **Default**: `./dataset/chords_normalize`
  * `--quality_definition`: コードクオリティの定義ファイル。
      * **Default**: `./data/quality.json`
  * `--output`: 計算結果の出力先ファイルパス。
      * **Default**: `./data/quality_freq_count.json`

-----


# 学習
### Step 1. 1段目のモデル学習

```bash
uv run python -m src.train_transcription --config ./configs/train.yaml
```

### Step 2. 2段目のモデル学習

checkpointには1段目のモデルの重みを指定します。

```bash
uv run python -m src.train_segment_transcription --config ./configs/train.yaml --checkpoint <base_transcription.pt> --training_backbone
```

### Step 3. CRFの学習

checkpointには2段目のモデルの重みを指定します。

```bash
uv run python -m src.train_crf --config ./configs/train.yaml --checkpoint <segment_model.pt>
```

# 推論

### 1段目のモデルで推論する場合

```bash
uv run python -m src.inference --config ./configs/train.yaml --checkpoint <base_transcription.pt> --audio <audio_path>
```

### 2段目のモデルで推論する場合

```bash
uv run python -m src.inference --config ./configs/train.yaml --checkpoint <segment_model.pt> --audio <audio_path> --use_segment_model
```

### CRFで推論する場合

```bash
uv run python -m src.inference --config ./configs/train.yaml --checkpoint <segment_model.pt> --crf_checkpoint <crf_model.pt> --audio <audio_path> --use_segment_model
```

# 学習済みモデル

[ここ](https://huggingface.co/anime-song/Chord-Transcription/tree/main)からダウンロードできます。