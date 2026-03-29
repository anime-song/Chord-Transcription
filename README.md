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

Generates a CSV file (training/validation pair list) that maps processed audio files to their corresponding chord and key labels.

```bash
uv run python -m src.preprocess.make_pairs_csv --chords_dir <chords_dir> --keys_dir <keys_dir> --songs_separated_dir <songs_separated_dir> --validation_ratio <validation_ratio>

```

* `--chords_dir`: Directory containing normalized chords.
* `--keys_dir`: Directory containing key information.
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

### Inference with a Base Model

```bash
uv run python -m src.chord_transcription.inference --checkpoint <base_transcription.pt> --audio <audio_path> --decode hmm

```

### Inference with a CRF Model

```bash
uv run python -m src.chord_transcription.inference --checkpoint <crf_model.pt> --audio <audio_path> --decode auto

```

Python library imports now live under `chord_transcription`, for example
`from chord_transcription import TranscriptionPredictor`.

Example:

```python
from chord_transcription import TranscriptionPredictor

predictor = TranscriptionPredictor.from_pretrained(
    "anime-song/Chord-Transcription",
    filename="model_epoch_150_public.pt",  # required when the repo contains multiple checkpoints
)
```

### Fine-tuning a Pre-trained Backbone

Use `build_model_from_pretrained()` when you want to continue training the same architecture.
If you want custom task heads, build your own model and load only `backbone.*` weights with `load_pretrained_backbone()`.

```python
import torch
from chord_transcription import build_model_from_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model_from_pretrained(
    "anime-song/Chord-Transcription",
    filename="model_epoch_150_public.pt",
    device=device,
)
model.train()  # the pretrained helper returns the model in eval mode

# root_chord / bass heads are detached from the backbone by default.
# Disable it if you want those losses to update the backbone as well.
model.set_label_head_detach(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

outputs = model(waveform)
loss = your_loss_fn(outputs, batch)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Initialize only the backbone if you want to attach new heads:

```python
from chord_transcription import build_model_from_config, load_pretrained_backbone

model = build_model_from_config(cfg).to(device)
load_pretrained_backbone(
    model.backbone,
    "anime-song/Chord-Transcription",
    filename="model_epoch_150_public.pt",
)
model.train()
```

### Linear Probe on a Frozen Backbone

Freeze the backbone and train only a linear head on top of the extracted frame-level features.

```python
import torch
import torch.nn as nn
from chord_transcription import build_backbone_from_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"

backbone = build_backbone_from_pretrained(
    "anime-song/Chord-Transcription",
    filename="model_epoch_150_public.pt",
    device=device,
)
for param in backbone.parameters():
    param.requires_grad = False
backbone.eval()

probe = nn.Linear(backbone.output_dim, num_labels).to(device)
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)

with torch.no_grad():
    features, _ = backbone(waveform)  # features: [B, T, D]

logits = probe(features)  # example: frame-wise labeling
loss = criterion(logits.transpose(1, 2), target)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

`Backbone.forward()` returns `(features, intermediates)`.
`features` is the final frame-level representation, and `intermediates` contains per-block axial-transformer outputs that can be used for analysis or auxiliary losses.

# Pre-trained Models

Available for download [here](https://huggingface.co/anime-song/Chord-Transcription/tree/main).
