import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torchaudio
import numpy as np


from src.utils import build_model_from_config
from src.models.segment_model import BatchBoundarySegmenter, resolve_segment_decode_params
from stem_splitter.inference import separate_stems

try:
    import yaml
except ImportError:
    yaml = None

# 定数
PITCH_CLASS_LABELS_13: List[str] = ["N", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def load_config(config_path: Path) -> Dict[str, Any]:
    """YAML または JSON 形式の設定ファイルを読み込む。"""
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise ImportError(
                    "YAML ファイルを読み込むには PyYAML が必要です。`pip install pyyaml` を実行してください。"
                )
            return yaml.safe_load(f)
        else:
            return json.load(f)


class AudioTranscriber:
    """PyTorchモデルを使用して音声の楽譜起こしを行うクラス。"""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Optional[str],
        device: str,
        quality_json_path: Path,
        use_segment_model: bool = False,
    ):
        """
        モデルを初期化し、設定を読み込み、デバイスをセットアップする。
        """
        self.config = config
        self.device = torch.device(device)
        self.model = build_model_from_config(self.config, use_segment_model=use_segment_model).to(self.device)

        if not checkpoint_path:
            raise ValueError("checkpoint_path は必須です。")
        self._load_checkpoint(self.model, checkpoint_path)

        self.segment_decode_cfg = resolve_segment_decode_params(self.config.get("segment_decode", {}))
        self.use_boundary_segment_decode = bool(self.segment_decode_cfg["enabled"])
        self.segment_decode_heads = self.segment_decode_cfg["heads"]
        self.boundary_segmenter: Optional[BatchBoundarySegmenter] = None
        if self.use_boundary_segment_decode:
            threshold = self.segment_decode_cfg["threshold"]
            nms_window_radius = self.segment_decode_cfg["nms_window_radius"]
            min_segment_length = self.segment_decode_cfg["min_segment_length"]
            max_segments = self.segment_decode_cfg["max_segments"]
            self.boundary_segmenter = BatchBoundarySegmenter(
                threshold=threshold,
                nms_window_radius=nms_window_radius,
                min_segment_length=min_segment_length,
                max_segments=max_segments,
            )
            print(
                "[INFO] boundary-segment decode を有効化しました "
                f"(heads={self.segment_decode_heads}, threshold={threshold}, "
                f"nms_window_radius={nms_window_radius}, min_segment_length={min_segment_length}, "
                f"max_segments={max_segments})"
            )

        self.model.eval()
        torch.set_grad_enabled(False)

        self.quality_labels = self._load_quality_labels(quality_json_path)
        self.root_chord_labels = self._build_root_chord_labels()
        self.tempo_decode_cfg = self.config.get("tempo_decode", {}) or {}

    def _load_quality_labels(self, quality_json_path: Path) -> Dict[str, str]:
        with Path(quality_json_path).expanduser().resolve().open("r", encoding="utf-8") as f:
            idx_to_quality = json.load(f)  # {"0": "5", "1": "", ...}
        return idx_to_quality

    def _build_root_chord_labels(self) -> List[str]:
        """quality語彙を根音と組み合わせてroot_chordのラベル順序を再現する"""
        model_cfg = self.config.get("model", {})
        num_root_classes = model_cfg.get("num_root_classes")
        num_quality_classes = model_cfg.get("num_quality_classes")
        if num_root_classes is None or num_quality_classes is None:
            raise ValueError("model.num_root_classes と model.num_quality_classes が設定されている必要があります")

        quality_list: List[str] = []
        for i in range(num_quality_classes):
            key = str(i)
            if key not in self.quality_labels:
                raise ValueError(f"quality語彙にインデックス {i} が存在しません")
            quality_list.append(self.quality_labels[key])
        try:
            non_chord_idx = next(i for i, label in enumerate(quality_list) if label == "N")
        except StopIteration as exc:
            raise ValueError("quality語彙に 'N' が存在しません") from exc

        quality_slots = [label for i, label in enumerate(quality_list) if i != non_chord_idx]

        root_chord_labels: List[str] = []
        for root_idx in range(1, num_root_classes):
            root_name = PITCH_CLASS_LABELS_13[root_idx]
            for quality in quality_slots:
                root_chord_labels.append(f"{root_name}{quality}")

        root_chord_labels.append("N")
        return root_chord_labels

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str) -> None:
        """
        指定されたモデルにチェックポイントを読み込む。
        - 一般的なキー名（'state_dict', 'model'など）を自動で探索。
        - DataParallel/DDPによる 'module.' 接頭辞を自動で削除。
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = None

            if "ema_state_dict" in checkpoint:
                state_dict = checkpoint["ema_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                raise KeyError("チェックポイントファイルから有効な state_dict を見つけられませんでした。")

            model.load_state_dict(state_dict, strict=True)

            print(f"[INFO] {model.__class__.__name__} のチェックポイントを正常に読み込みました: {checkpoint_path}")

        except FileNotFoundError:
            print(f"[ERROR] チェックポイントファイルが見つかりません: {checkpoint_path}")
            raise
        except Exception as e:
            print(f"[ERROR] {model.__class__.__name__} のチェックポイント読み込みに失敗しました: {e}")
            raise

    def _prepare_stems(self, audio_path: Path, stems_dir: Path, reuse_stems: bool) -> torch.Tensor:
        """
        音声をステムに分離し、読み込み、リサンプリング、パディング、結合を行う。
        """
        stem_order = self.config["data_loader"]["stem_order"]
        target_sr = self.config["data_loader"]["sample_rate"]
        mixdown_to_mono = self.config["data_loader"].get("mixdown_to_mono", False)
        channels_per_stem = 1 if mixdown_to_mono else 2
        target_channels = max(1, len(stem_order) * channels_per_stem)

        # 単一のミックスのみを扱う場合は分離処理をスキップしてそのまま読み込む
        if len(stem_order) == 1 and stem_order[0].lower() == "mix":
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] < target_channels:
                repeat = (target_channels + waveform.shape[0] - 1) // waveform.shape[0]
                waveform = waveform.repeat(repeat, 1)[:target_channels]
            elif waveform.shape[0] > target_channels:
                waveform = waveform[:target_channels]

            return waveform.unsqueeze(0).to(self.device)

        base_name = audio_path.stem
        output_stem_dir = stems_dir / base_name
        stem_paths = [output_stem_dir / f"{base_name}_{stem_name}.wav" for stem_name in stem_order]

        # ステムが存在しない、または再利用が無効な場合に分離を実行
        if not (reuse_stems and all(p.exists() for p in stem_paths)):
            print(f"[INFO] ステムを分離しています: {audio_path.name} -> {output_stem_dir}")
            output_stem_dir.mkdir(parents=True, exist_ok=True)
            separate_stems(str(audio_path), str(output_stem_dir.parent))
        else:
            print("[INFO] 既存のステムを再利用します。")

        # ステムの波形を読み込み、リサンプリング
        waveforms = []
        for path in stem_paths:
            if not path.exists():
                raise FileNotFoundError(f"必要なステムファイルが見つかりません: {path}")

            waveform, sr = torchaudio.load(path)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            waveforms.append(waveform)

        # 最長のステムに合わせてゼロパディング
        max_len = max(w.shape[-1] for w in waveforms)
        padded_waveforms = [torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in waveforms]

        # チャンネル次元で結合し、バッチ次元を追加
        concatenated = torch.cat(padded_waveforms, dim=0).unsqueeze(0)
        return concatenated.to(self.device)

    def _segmentwise_decode_indices(
        self,
        logits: torch.Tensor,
        segment_ids_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        各セグメント内でlogitsを平均し、セグメントごとに1クラスを決定してフレームへ展開する。
        logits: (B, T, C), segment_ids_batch: (B, T)
        returns: (B, T) long
        """
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3D (B, T, C), but got shape={tuple(logits.shape)}")
        if segment_ids_batch.ndim != 2:
            raise ValueError(f"segment_ids_batch must be 2D (B, T), but got shape={tuple(segment_ids_batch.shape)}")
        if logits.shape[:2] != segment_ids_batch.shape:
            raise ValueError(
                "shape mismatch: logits and segment_ids_batch must share (B, T), "
                f"but got {tuple(logits.shape[:2])} and {tuple(segment_ids_batch.shape)}"
            )

        batch_size, total_frames, num_classes = logits.shape
        decoded = torch.empty((batch_size, total_frames), device=logits.device, dtype=torch.long)

        for b in range(batch_size):
            seg_ids = segment_ids_batch[b].long()
            if seg_ids.numel() == 0:
                continue

            num_segments = int(seg_ids.max().item()) + 1
            segment_sums = torch.zeros((num_segments, num_classes), device=logits.device, dtype=logits.dtype)
            segment_counts = torch.zeros((num_segments, 1), device=logits.device, dtype=logits.dtype)

            segment_sums.index_add_(0, seg_ids, logits[b])
            ones = torch.ones((total_frames, 1), device=logits.device, dtype=logits.dtype)
            segment_counts.index_add_(0, seg_ids, ones)

            segment_means = segment_sums / segment_counts.clamp_min(1.0)
            segment_labels = segment_means.argmax(dim=-1)
            decoded[b] = segment_labels.index_select(0, seg_ids)

        return decoded

    def _decode_with_boundary_segments(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        if not self.use_boundary_segment_decode or self.boundary_segmenter is None:
            return {}

        boundary_logits = model_outputs.get("initial_boundary_logits")
        if boundary_logits is None:
            print("[WARN] 'initial_boundary_logits' がないため boundary-segment decode をスキップします。")
            return {}
        if boundary_logits.ndim not in (2, 3):
            print(
                "[WARN] 'initial_boundary_logits' の次元が不正です "
                f"(shape={tuple(boundary_logits.shape)})。boundary-segment decode をスキップします。"
            )
            return {}

        if boundary_logits.ndim == 2:
            boundary_logits = boundary_logits.unsqueeze(-1)
        batch_size, total_frames, _ = boundary_logits.shape
        dummy_features = torch.zeros(
            (batch_size, total_frames, 1),
            device=boundary_logits.device,
            dtype=boundary_logits.dtype,
        )
        _, _, segment_ids_batch, segments_info = self.boundary_segmenter.process_batch(
            frame_features=dummy_features,
            boundary_logits=boundary_logits,
            detach_boundary=True,
        )

        if batch_size > 0:
            print(f"[INFO] boundary-segment decode: segments={len(segments_info[0])}, frames={total_frames}")

        decoded_indices: Dict[str, np.ndarray] = {}
        for head in self.segment_decode_heads:
            logits_key = f"initial_{head}_logits"
            logits = model_outputs.get(logits_key)
            if logits is None:
                print(f"[WARN] 出力 '{logits_key}' がないため、{head} のsegment decodeをスキップします。")
                continue
            try:
                decoded = self._segmentwise_decode_indices(logits, segment_ids_batch)
            except ValueError as exc:
                print(f"[WARN] {head} のsegment decodeに失敗しました: {exc}")
                continue

            decoded_indices[head] = decoded.detach().cpu().numpy()

        return decoded_indices

    def _decode_tempo(self, model_outputs: Dict[str, torch.Tensor], predictions: Dict[str, Any]) -> None:
        """
        initial_tempo を tempo_bpm（または tempo_class）へ変換して predictions に追加する。
        - 回帰: regression_scale が "log" の場合は exp で BPM に戻す。
               "linear" の場合は出力をそのまま BPM として扱う。
        - 分類: argmax を tempo_class として返し、tempo_class_values があれば tempo_bpm に変換
        """
        tempo_logits = model_outputs.get("initial_tempo")
        if tempo_logits is None:
            return

        regression_scale = str(self.tempo_decode_cfg.get("regression_scale", "log")).lower()
        if regression_scale not in {"log", "linear"}:
            print(f"[WARN] tempo_decode.regression_scale が不正です: {regression_scale}. 'log'として扱います。")
            regression_scale = "log"

        bpm_min = float(self.tempo_decode_cfg.get("bpm_min", 30.0))
        bpm_max = float(self.tempo_decode_cfg.get("bpm_max", 300.0))

        def decode_regression_to_bpm(pred_values: np.ndarray) -> np.ndarray:
            if regression_scale == "log":
                bpm = np.exp(pred_values)
            else:
                bpm = pred_values
            return np.clip(bpm, bpm_min, bpm_max)

        tempo_tensor = tempo_logits.squeeze(0).detach().cpu()
        if tempo_tensor.ndim == 1:
            # (T,) を回帰として解釈
            pred_values = tempo_tensor.numpy().astype(np.float32)
            bpm = decode_regression_to_bpm(pred_values)
            predictions["tempo_bpm"] = bpm.astype(np.float32).tolist()
            return

        if tempo_tensor.ndim != 2:
            print(f"[WARN] initial_tempo の形状が想定外です: {tuple(tempo_tensor.shape)}")
            return

        if tempo_tensor.size(-1) == 1:
            pred_values = tempo_tensor.squeeze(-1).numpy().astype(np.float32)
            bpm = decode_regression_to_bpm(pred_values)
            predictions["tempo_bpm"] = bpm.astype(np.float32).tolist()
            return

        # 分類モード
        class_indices = tempo_tensor.argmax(dim=-1).numpy().astype(np.int64)
        predictions["tempo_class"] = class_indices.tolist()

        class_values = self.tempo_decode_cfg.get("tempo_class_values")
        if isinstance(class_values, list) and len(class_values) == int(tempo_tensor.size(-1)):
            mapped = []
            for idx in class_indices:
                value = class_values[int(idx)]
                try:
                    mapped.append(float(value))
                except (TypeError, ValueError):
                    mapped.append(math.nan)
            predictions["tempo_bpm"] = mapped

    def _apply_hmm_smoothing(self, logits: torch.Tensor) -> List[int]:
        """
        HMM (Viterbi) を適用して、出力を平滑化する。
        logits: (T, C)
        """
        try:
            from src.hmm import decode_viterbi_from_probs, make_sticky_transition
        except ImportError:
            print("[WARN] src.hmm が見つからないため、HMM適用をスキップします。")
            return logits.argmax(dim=-1).tolist()

        # log_softmax -> exp -> probability
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        T, C = probs.shape

        # 手動遷移確率 (自己遷移0.9)
        transition_matrix = make_sticky_transition(C, stay_prob=0.9)

        # 初期確率は一様分布 (Noneで自動)
        path = decode_viterbi_from_probs(probs, transition_matrix=transition_matrix, init_probs=None, method="jit")
        return path.tolist()

    def predict(self, audio_path: str, stems_dir: str, reuse_stems: bool, use_hmm: bool = False) -> Dict[str, Any]:
        """
        音声ファイルに対して推論を実行し、ラベル付けされた予測結果を返す。
        """
        # ステムを準備し、波形を取得
        waveform = self._prepare_stems(Path(audio_path), Path(stems_dir), reuse_stems)
        print(f"[INFO] 入力波形 shape: {waveform.shape}")

        chord25_logits_np: Optional[np.ndarray] = None
        predictions = {}
        # モデルから生のlogit出力を取得
        model_outputs: Dict[str, torch.Tensor] = self.model(waveform, max_segments=None)
        boundary_segment_indices = self._decode_with_boundary_segments(model_outputs)

        if "initial_smooth_chord25_original" in model_outputs:
            chord25_logits_np = model_outputs["initial_smooth_chord25_original"].squeeze(0).detach().cpu().numpy()

        # 対象のヘッド（root_chord, bass, key）ごとに処理
        for head in ["root_chord", "bass", "key"]:
            if head in boundary_segment_indices:
                indices = boundary_segment_indices[head][0]
            else:
                logits_key = f"initial_{head}_logits"
                if logits_key not in model_outputs:
                    print(f"[WARN] 出力 '{logits_key}' がモデルの出力に含まれていません。")
                    continue

                # Shape: (T, C)
                logits = model_outputs[logits_key].squeeze(0)

                if use_hmm:
                    # HMM (Viterbi) 適用 (List[int]が返る)
                    indices = self._apply_hmm_smoothing(logits)
                else:
                    # 通常のArgmax
                    indices = logits.argmax(dim=-1).cpu().numpy()

            if head in ["bass", "key"]:
                labels = [PITCH_CLASS_LABELS_13[i] for i in indices]
            elif head == "root_chord":
                labels = [self.root_chord_labels[i] if 0 <= i < len(self.root_chord_labels) else "N/A" for i in indices]
            else:
                continue

            predictions[head] = labels

        self._decode_tempo(model_outputs, predictions)

        # タイムスタンプを計算
        sample_rate = self.config["data_loader"]["sample_rate"]
        seconds_per_frame = self._infer_seconds_per_frame(sample_rate)
        num_frames = None
        if predictions:
            num_frames = len(next(iter(predictions.values())))
        elif chord25_logits_np is not None:
            num_frames = chord25_logits_np.shape[0]

        if seconds_per_frame and num_frames:
            predictions["time_sec"] = np.arange(num_frames) * seconds_per_frame

        if chord25_logits_np is not None:
            predictions["final_chord25_logits"] = chord25_logits_np

        return predictions

    def _infer_seconds_per_frame(self, sample_rate: int) -> Optional[float]:
        """設定から1フレームあたりの秒数を推定する。"""
        conf = self.config.get("data_loader", self.config)
        if fps := conf.get("frames_per_second") or conf.get("fps"):
            return 1.0 / float(fps)
        if hop_length := conf.get("hop_length"):
            return float(hop_length) / float(sample_rate)
        return None
