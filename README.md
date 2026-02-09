# Chord-Transcription

**English** | [日本語](README.ja.md)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/Chord-Transcription/blob/main/Chord_Transcription.ipynb)

A repository for training chord transcription models.
It enables high-precision inference through context interpretation using a **SegmentModel**.

# Dataset Creation Pipeline

This document describes the preprocessing pipeline required to prepare the dataset for model training. Please execute each step in order.

---

### Step 1. Stem Separation and Resampling

Separates audio files into individual instrument stems (vocals, drums, bass, and others) and resamples them to the specified sampling rate.

```bash
uv run python -m src.preprocess.separate_and_resample --input <input_dir> --out-dir <output_dir>

```

* `--input_dir`: Directory containing the source audio files.
* **Default**: `./dataset/songs`


* `--out-dir`: Destination directory for the separated stem files.
* **Default**: `./dataset/songs_separated`



---

### Step 2. Data Augmentation via Pitch Shifting

Applies pitch shifting to the separated stems to increase the volume and variety of the training data (Data Augmentation).

```bash
uv run python -m src.preprocess.pitch_shift_augment --target_dir <target_dir>

```

* `--target_dir`: Directory containing the audio files to be pitch-shifted.
* **Default**: `./dataset/songs_separated`



---

### Step 3. Chord Data Normalization

Normalizes original chord notations (e.g., `CM7`, `Gm`) into a consistent format optimized for model training.

```bash
uv run python -m src.preprocess.normalize_chords --input_dir <input_dir> --output_dir <output_dir>

```

* `--input_dir`: Directory containing the raw chord data.
* **Default**: `./dataset/chords`


* `--output_dir`: Destination directory for the normalized chord data.
* **Default**: `./dataset/chords_normalize`



---

### Step 4. Creating Training Pairs

Generates a CSV file (training/validation pair list) that maps processed audio files to their corresponding chord, key, and tempo labels.

```bash
uv run python -m src.preprocess.make_pairs_csv --chords_dir <chords_dir> --keys_dir <keys_dir> --tempos_dir <tempos_dir> --songs_separated_dir <songs_separated_dir> --validation_ratio <validation_ratio>

```

* `--chords_dir`: Directory containing normalized chords.
* `--keys_dir`: Directory containing key information.
* `--tempos_dir`: Directory containing tempo information.
* `--songs_separated_dir`: Directory containing separated stems.
* `--validation_ratio`: The proportion of the dataset to be used for validation.

---

### Step 5. Calculating Chord Quality Frequency

Calculates the frequency of each chord quality (e.g., `Major`, `minor`) across the dataset for use in the loss function during training.

```bash
uv run python -m src.preprocess.count_quality_freq --data_folder <data_folder> --quality_definition <quality_definition> --output <output>

```

* `--data_folder`: Directory containing normalized chords.
* **Default**: `./dataset/chords_normalize`


* `--quality_definition`: Definition file for chord qualities.
* **Default**: `./data/quality.json`


* `--output`: Path for the output JSON file containing frequency counts.
* **Default**: `./data/quality_freq_count.json`



---

# Training

### Step 1. First-Stage Model Training

```bash
uv run python -m src.train_transcription --config ./configs/train.yaml

```

### Step 2. Second-Stage Model Training (SegmentModel)

Specify the weights from the first-stage model in the checkpoint.

```bash
uv run python -m src.train_segment_transcription --config ./configs/train.yaml --checkpoint <base_transcription.pt> --training_backbone

```

# Inference

### Inference with the First-Stage Model

```bash
uv run python -m src.inference --config ./configs/train.yaml --checkpoint <base_transcription.pt> --audio <audio_path>

```

### Inference with the Second-Stage Model

```bash
uv run python -m src.inference --config ./configs/train.yaml --checkpoint <segment_model.pt> --audio <audio_path> --use_segment_model

```

# Pre-trained Models

Available for download [here](https://huggingface.co/anime-song/Chord-Transcription/tree/main).
