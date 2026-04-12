from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ..hub import resolve_pretrained_checkpoint_path
from ..models.factory import (
    build_model_from_config,
    load_label_vocab_from_checkpoint,
    load_model_build_config_from_checkpoint,
    merge_model_build_config,
)
from ..models.semi_crf import (
    NeuralSemiCRFInterval,
    _build_interval_score,
    _zero_noise_score,
)
from .types import TranscriptionMetadata, TranscriptionPrediction


DecodeMode = Literal["auto", "none", "argmax", "hmm", "crf", "crf_pool"]


class TranscriptionPredictor:
    """
    checkpoint だけからモデルを復元し、音声からフレーム単位の出力を返す。

    ここでは「モデルを動かして index まで復号する」ことだけを担当し、
    文字列ラベル化やイベント化は decoder 側へ分離する。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: TranscriptionMetadata,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.config = config
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "TranscriptionPredictor":
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # checkpoint に埋め込まれた model 設定と語彙から、そのまま推論器を復元する。
        model_build_config = load_model_build_config_from_checkpoint(checkpoint)
        if model_build_config is None:
            raise KeyError("Checkpoint must contain 'model_build_config'.")
        label_vocab = load_label_vocab_from_checkpoint(checkpoint)

        config = merge_model_build_config({}, model_build_config)
        model_kind = str(model_build_config.get("model_kind", "base")).lower()
        use_crf_model = model_kind == "crf"

        torch_device = torch.device(device)
        model = build_model_from_config(config, use_crf_model=use_crf_model).to(torch_device)

        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint must contain 'ema_state_dict' or 'model_state_dict'.")
        model.load_state_dict(state_dict, strict=False)

        # 時間軸の復元に必要な最小メタ情報をまとめる。
        data_loader_cfg = config["data_loader"]
        if fps := data_loader_cfg.get("frames_per_second") or data_loader_cfg.get("fps"):
            seconds_per_frame = 1.0 / float(fps)
        elif hop_length := data_loader_cfg.get("hop_length"):
            seconds_per_frame = float(hop_length) / float(data_loader_cfg["sample_rate"])
        else:
            raise KeyError("Could not infer seconds_per_frame from checkpoint config.")

        metadata = TranscriptionMetadata(
            checkpoint_path=checkpoint_path,
            model_kind=model_kind,
            sample_rate=int(data_loader_cfg["sample_rate"]),
            seconds_per_frame=seconds_per_frame,
            num_root_classes=int(config["model"]["num_root_classes"]),
            num_quality_classes=int(config["model"]["num_quality_classes"]),
            quality_labels=tuple(label_vocab["quality"]),
        )
        return cls(model=model, metadata=metadata, config=config, device=torch_device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        local_files_only: bool = False,
        token: Optional[str] = None,
    ) -> "TranscriptionPredictor":
        checkpoint_path = resolve_pretrained_checkpoint_path(
            pretrained_model_name_or_path,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        return cls.from_checkpoint(checkpoint_path, device=device)

    def predict_file(
        self,
        audio_path: str | Path,
        decode_mode: DecodeMode = "auto",
    ) -> TranscriptionPrediction:
        waveform, sample_rate = self._load_audio_file(audio_path)
        return self.predict_waveform(waveform, sample_rate=sample_rate, decode_mode=decode_mode)

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        waveform, sample_rate = self._load_audio_file(audio_path)
        return self._prepare_waveform(waveform, sample_rate=sample_rate)

    @staticmethod
    def _load_audio_file(audio_path: str | Path) -> tuple[torch.Tensor, int]:
        path = str(Path(audio_path).expanduser())
        try:
            waveform, sample_rate = torchaudio.load(path)
            return waveform, int(sample_rate)
        except Exception as exc:
            message = str(exc).lower()
            if "torchcodec" not in message and "libtorchcodec" not in message:
                raise

            # TorchAudio 2.9+ routes file loading through TorchCodec. If its native
            # runtime deps are missing, fall back to libsndfile-based decoding.
            waveform_np, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(np.ascontiguousarray(waveform_np.T))
            return waveform, int(sample_rate)

    def predict_waveform(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int,
        decode_mode: DecodeMode = "auto",
    ) -> TranscriptionPrediction:
        # 1. 波形をモデルが受け取れる stereo / sample_rate に揃える。
        prepared_waveform = self._prepare_waveform(waveform, sample_rate=sample_rate)

        # 2. checkpoint の種類に応じて decode 方法を確定する。
        effective_decode_mode = decode_mode
        if effective_decode_mode == "auto":
            effective_decode_mode = "crf" if self.metadata.model_kind == "crf" else "argmax"

        if self.metadata.model_kind == "crf" and effective_decode_mode == "hmm":
            raise ValueError("HMM decode is only supported for base models.")
        if self.metadata.model_kind != "crf" and effective_decode_mode == "crf":
            raise ValueError("CRF decode is only supported for CRF checkpoints.")

        # 3. forward を実行する。ここでは保存やラベル化はまだしない。
        with torch.inference_mode():
            outputs = self.model(prepared_waveform)

        def to_numpy(tensor: torch.Tensor | None, *, squeeze_last: bool = False) -> np.ndarray | None:
            if tensor is None:
                return None
            array = tensor.detach().cpu().squeeze(0).numpy()
            if squeeze_last and array.ndim == 2 and array.shape[-1] == 1:
                array = array[..., 0]
            return np.asarray(array)

        root_scores = None
        bass_scores = None
        key_scores = None
        pitch_chroma_scores = None
        boundary_scores = None
        beat_scores = None
        downbeat_scores = None
        key_boundary_scores = None
        frame_features = None
        root_index = None
        bass_index = None
        key_index = None

        # 4. model 種別ごとに、必要な出力を numpy 化してまとめる。
        if self.metadata.model_kind == "crf":
            base_outputs = outputs["base_outputs"]
            root_scores = to_numpy(outputs.get("root_chord_emissions"))
            bass_scores = to_numpy(outputs.get("bass_emissions"))
            key_scores = to_numpy(outputs.get("key_emissions"))
            pitch_chroma_scores = to_numpy(base_outputs.get("pitch_chroma_logits"))
            if pitch_chroma_scores is not None:
                pitch_chroma_scores = 1.0 / (1.0 + np.exp(-pitch_chroma_scores))  # sigmoid
            boundary_scores = to_numpy(base_outputs.get("initial_boundary_logits"), squeeze_last=True)
            beat_scores = to_numpy(base_outputs.get("initial_beat_logits"), squeeze_last=True)
            downbeat_scores = to_numpy(base_outputs.get("initial_downbeat_logits"), squeeze_last=True)
            key_boundary_scores = to_numpy(base_outputs.get("initial_key_boundary_logits"), squeeze_last=True)
            frame_features = to_numpy(base_outputs.get("initial_features"))

            if effective_decode_mode == "crf":
                root_index = np.asarray(
                    self.model.crf_root_chord.decode(outputs["root_chord_emissions"])[0], dtype=np.int64
                )
                bass_index = np.asarray(self.model.crf_bass.decode(outputs["bass_emissions"])[0], dtype=np.int64)
                key_index = np.asarray(self.model.crf_key.decode(outputs["key_emissions"])[0], dtype=np.int64)
            elif effective_decode_mode == "argmax":
                root_index = np.asarray(
                    outputs["root_chord_emissions"].squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64
                )
                bass_index = np.asarray(
                    outputs["bass_emissions"].squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64
                )
                key_index = np.asarray(outputs["key_emissions"].squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
            elif effective_decode_mode != "none":
                raise ValueError(f"Unsupported decode_mode for CRF checkpoint: {effective_decode_mode}")
        else:
            root_scores = to_numpy(outputs.get("initial_root_chord_logits"))
            bass_scores = to_numpy(outputs.get("initial_bass_logits"))
            key_scores = to_numpy(outputs.get("initial_key_logits"))
            pitch_chroma_scores = to_numpy(outputs.get("pitch_chroma_logits"))
            if pitch_chroma_scores is not None:
                pitch_chroma_scores = 1.0 / (1.0 + np.exp(-pitch_chroma_scores))  # sigmoid
            boundary_scores = to_numpy(outputs.get("initial_boundary_logits"), squeeze_last=True)
            beat_scores = to_numpy(outputs.get("initial_beat_logits"), squeeze_last=True)
            downbeat_scores = to_numpy(outputs.get("initial_downbeat_logits"), squeeze_last=True)
            key_boundary_scores = to_numpy(outputs.get("initial_key_boundary_logits"), squeeze_last=True)
            frame_features = to_numpy(outputs.get("initial_features"))

            if effective_decode_mode != "none":
                root_logits = outputs.get("initial_root_chord_logits")
                bass_logits = outputs.get("initial_bass_logits")
                key_logits = outputs.get("initial_key_logits")

                if effective_decode_mode not in {"argmax", "hmm", "crf_pool"}:
                    raise ValueError(f"Unsupported decode_mode for base checkpoint: {effective_decode_mode}")

                if effective_decode_mode == "crf_pool":
                    interval_query = outputs.get("interval_query")
                    interval_key = outputs.get("interval_key")
                    interval_diag = outputs.get("interval_diag")

                    if interval_query is None or interval_key is None or interval_diag is None:
                        raise ValueError(
                            "crf_pool mode requires interval_query, interval_key, and interval_diag in outputs."
                        )

                    # まずベースとしてフレーム単位argmaxを入れておき、区間が存在する部分だけプーリング結果で上書きする
                    root_index = (
                        np.asarray(root_logits.squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
                        if root_logits is not None
                        else None
                    )
                    bass_index = (
                        np.asarray(bass_logits.squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
                        if bass_logits is not None
                        else None
                    )
                    key_index = (
                        np.asarray(key_logits.squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
                        if key_logits is not None
                        else None
                    )

                    chord_interval_cfg = self.config.get("model", {}).get("classifier", {}).get("chord_interval", {})
                    length_scaling = (
                        chord_interval_cfg.get("length_scaling", "sqrt")
                        if isinstance(chord_interval_cfg, dict)
                        else "sqrt"
                    )

                    time_steps = interval_query.shape[1]
                    query_single = interval_query[0:1].transpose(0, 1)  # [T, 1, D]
                    key_single = interval_key[0:1].transpose(0, 1)  # [T, 1, D]
                    diag_single = interval_diag[0:1].transpose(0, 1)  # [T, 1]

                    score = _build_interval_score(query_single, key_single, diag_single, length_scaling=length_scaling)
                    noise_score = _zero_noise_score(time_steps, batch_size=1, device=score.device)

                    semi_crf = NeuralSemiCRFInterval(score, noise_score)
                    decoded_batch = semi_crf.decode()
                    pred_intervals = decoded_batch[0] if decoded_batch else []

                    if not pred_intervals:
                        pred_intervals = [(0, time_steps - 1)]

                    # 予測された各区間に対して、区間内のlogitsの和(または平均)を取り、argmaxでラベルを決定する
                    for start, end in pred_intervals:
                        if root_logits is not None:
                            pooled = root_logits[0, start : end + 1].sum(dim=0)
                            cls_idx = int(pooled.argmax(dim=-1).item())
                            root_index[start : end + 1] = cls_idx
                        if bass_logits is not None:
                            pooled = bass_logits[0, start : end + 1].sum(dim=0)
                            cls_idx = int(pooled.argmax(dim=-1).item())
                            bass_index[start : end + 1] = cls_idx
                        if key_logits is not None:
                            pooled = key_logits[0, start : end + 1].sum(dim=0)
                            cls_idx = int(pooled.argmax(dim=-1).item())
                            key_index[start : end + 1] = cls_idx
                else:
                    if root_logits is not None:
                        if effective_decode_mode == "argmax":
                            root_index = np.asarray(root_logits.squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
                        else:
                            root_index = np.asarray(
                                _apply_hmm_smoothing(root_logits.squeeze(0), stay_prob=1.0), dtype=np.int64
                            )
                    if bass_logits is not None:
                        if effective_decode_mode == "argmax":
                            bass_index = np.asarray(bass_logits.squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
                        else:
                            bass_index = np.asarray(
                                _apply_hmm_smoothing(bass_logits.squeeze(0), stay_prob=1.0), dtype=np.int64
                            )
                    if key_logits is not None:
                        if effective_decode_mode == "argmax":
                            key_index = np.asarray(key_logits.squeeze(0).argmax(dim=-1).cpu().numpy(), dtype=np.int64)
                        else:
                            key_index = np.asarray(
                                _apply_hmm_smoothing(key_logits.squeeze(0), stay_prob=0.9), dtype=np.int64
                            )

        # 5. どの head を使った場合でも、共通の time axis を作る。
        num_frames = None
        for candidate in (
            root_scores,
            bass_scores,
            key_scores,
            pitch_chroma_scores,
            boundary_scores,
            beat_scores,
            downbeat_scores,
            key_boundary_scores,
            root_index,
            bass_index,
            key_index,
        ):
            if candidate is not None:
                num_frames = int(candidate.shape[0])
                break
        if num_frames is None:
            raise ValueError("No frame-level output was produced by the model.")
        time_sec = np.arange(num_frames, dtype=np.float32) * self.metadata.seconds_per_frame

        return TranscriptionPrediction(
            metadata=self.metadata,
            decode_mode=effective_decode_mode,
            time_sec=time_sec,
            frame_features=frame_features,
            pitch_chroma_scores=pitch_chroma_scores,
            boundary_scores=boundary_scores,
            beat_scores=beat_scores,
            downbeat_scores=downbeat_scores,
            key_boundary_scores=key_boundary_scores,
            root_chord_scores=root_scores,
            bass_scores=bass_scores,
            key_scores=key_scores,
            root_chord_index=root_index,
            bass_index=bass_index,
            key_index=key_index,
        )

    def _prepare_waveform(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform)
        else:
            waveform_tensor = waveform

        waveform_tensor = waveform_tensor.detach().cpu().to(torch.float32)

        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        elif waveform_tensor.ndim != 2:
            raise ValueError(f"waveform must be 1D or 2D, got shape={tuple(waveform_tensor.shape)}")

        # 波形は基本的に channel-first を期待するが、(T, C) が来た場合だけ補正する。
        if waveform_tensor.shape[0] > 4 and waveform_tensor.shape[1] <= 4:
            waveform_tensor = waveform_tensor.transpose(0, 1)

        # 学習時と同じ sample_rate に変換する。
        if sample_rate != self.metadata.sample_rate:
            waveform_tensor = torchaudio.functional.resample(
                waveform_tensor,
                orig_freq=sample_rate,
                new_freq=self.metadata.sample_rate,
            )

        # モデル入力は stereo 固定。mono は複製し、多チャンネルは先頭 2ch だけ使う。
        if waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor.repeat(2, 1)
        elif waveform_tensor.shape[0] > 2:
            waveform_tensor = waveform_tensor[:2]

        return waveform_tensor.unsqueeze(0).to(self.device)


def _apply_hmm_smoothing(logits: torch.Tensor, stay_prob: float = 0.9) -> list[int]:
    """
    base model 用の簡易デコード。
    HMM が使えない環境では argmax にフォールバックする。
    """
    try:
        from ..hmm import decode_viterbi_from_probs, make_sticky_transition
    except ImportError:
        return logits.argmax(dim=-1).tolist()

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    _, num_classes = probs.shape
    transition_matrix = make_sticky_transition(num_classes, stay_prob=stay_prob)
    path = decode_viterbi_from_probs(probs, transition_matrix=transition_matrix, init_probs=None, method="jit")
    return path.tolist()
