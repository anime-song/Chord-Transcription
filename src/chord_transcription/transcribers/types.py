from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TranscriptionMetadata:
    checkpoint_path: str
    model_kind: str
    sample_rate: int
    seconds_per_frame: float
    num_root_classes: int
    num_quality_classes: int
    quality_labels: Tuple[str, ...]


@dataclass
class TranscriptionPrediction:
    metadata: TranscriptionMetadata
    decode_mode: str
    time_sec: np.ndarray
    frame_features: Optional[np.ndarray] = None
    pitch_chroma_scores: Optional[np.ndarray] = None
    boundary_scores: Optional[np.ndarray] = None
    beat_scores: Optional[np.ndarray] = None
    downbeat_scores: Optional[np.ndarray] = None
    key_boundary_scores: Optional[np.ndarray] = None
    root_chord_scores: Optional[np.ndarray] = None
    bass_scores: Optional[np.ndarray] = None
    key_scores: Optional[np.ndarray] = None
    root_chord_index: Optional[np.ndarray] = None
    bass_index: Optional[np.ndarray] = None
    key_index: Optional[np.ndarray] = None


@dataclass
class DecodedFramePrediction:
    time_sec: np.ndarray
    root_chord: Optional[List[str]] = None
    bass: Optional[List[str]] = None
    key: Optional[List[str]] = None
    chord: Optional[List[str]] = None
    beat_prob: Optional[np.ndarray] = None
    downbeat_prob: Optional[np.ndarray] = None


@dataclass
class TranscriptionEvents:
    chord_events: List[Dict[str, Any]]
    key_events: List[Dict[str, Any]]
    beat_events: List[Dict[str, Any]]
