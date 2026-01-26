import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torchaudio
import numpy as np


from src.utils import build_model_from_config
from src.models.crf_wrapper import TranscriptionCRFModel
from stem_splitter.inference import separate_stems

try:
    import yaml
except ImportError:
    yaml = None

# 定数
PITCH_CLASS_LABELS_13: List[str] = ["N", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
        crf_checkpoint_path: Optional[str] = None,
        use_segment_model: bool = False,
    ):
        """
        モデルを初期化し、設定を読み込み、デバイスをセットアップする。
        """
        self.config = config
        self.device = torch.device(device)
        self.model = build_model_from_config(self.config, use_segment_model=use_segment_model).to(self.device)

        if checkpoint_path:
            self._load_checkpoint(self.model, checkpoint_path)

        self.use_crf = False
        if crf_checkpoint_path:
            print(f"[INFO] CRFチェックポイントを読み込みます: {crf_checkpoint_path}")
            # Base modelをラップする
            self.model = TranscriptionCRFModel(self.model).to(self.device)
            self._load_checkpoint(self.model, crf_checkpoint_path)
            self.use_crf = True

        self.model.eval()
        torch.set_grad_enabled(False)

        self.quality_labels = self._load_quality_labels(quality_json_path)
        self.root_chord_labels = self._build_root_chord_labels()

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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if "ema_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            else:
                raise KeyError("チェックポイントファイルから有効な state_dict を見つけられませんでした。")
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

    def predict(self, audio_path: str, stems_dir: str, reuse_stems: bool) -> Dict[str, Any]:
        """
        音声ファイルに対して推論を実行し、ラベル付けされた予測結果を返す。
        """
        # ステムを準備し、波形を取得
        waveform = self._prepare_stems(Path(audio_path), Path(stems_dir), reuse_stems)
        print(f"[INFO] 入力波形 shape: {waveform.shape}")

        chord25_logits_np: Optional[np.ndarray] = None
        if self.use_crf:
            # CRFの場合、modelは indices (List[List[int]]) または tensor を返す
            # TranscriptionCRFModel.forward(..., labels=None) -> Dict[str, List[List[int]]]
            predictions_map = self.model(waveform)  # Dict[str, List[List[int]]]

            predictions = {}
            for head, indices_list in predictions_map.items():
                if not indices_list:
                    continue
                indices = indices_list[0]

                if head == "bass" or head == "key":
                    predictions[head] = [PITCH_CLASS_LABELS_13[i] for i in indices]
                elif head == "root":
                    predictions["root_chord"] = [
                        self.root_chord_labels[i] if 0 <= i < len(self.root_chord_labels) else "N/A" for i in indices
                    ]
        else:
            predictions = {}
            # モデルから生のlogit出力を取得
            model_outputs: Dict[str, torch.Tensor] = self.model(waveform, max_segments=None)

            if "initial_smooth_chord25_original" in model_outputs:
                chord25_logits_np = model_outputs["initial_smooth_chord25_original"].squeeze(0).detach().cpu().numpy()

            # 対象のヘッド（root_chord, bass, key）ごとに処理
            for head in ["root_chord", "bass", "key"]:
                logits_key = f"initial_{head}_logits"

                if logits_key in model_outputs:
                    logits = model_outputs[logits_key].squeeze(0).cpu()  # Shape: (T, C)
                    # 最も確率の高いクラスのインデックスを取得
                    indices = logits.argmax(dim=-1).numpy()

                    if head in ["bass", "key"]:
                        labels = [PITCH_CLASS_LABELS_13[i] for i in indices]
                    elif head == "root_chord":
                        labels = [
                            self.root_chord_labels[i] if 0 <= i < len(self.root_chord_labels) else "N/A"
                            for i in indices
                        ]
                    else:
                        continue  # 他のヘッドは無視

                    predictions[head] = labels
                else:
                    print(f"[WARN] 出力 '{logits_key}' がモデルの出力に含まれていません。")

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
