import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# src/utils.py の関数を利用
from src.utils import build_model_from_config, load_config
from src.models.segment_model import BatchBoundarySegmenter


class AudioProcessor:
    """
    音声ファイルの読み込みとモデル入力用のテンソル加工を担当するクラス。
    """

    @staticmethod
    def load_and_preprocess(
        audio_path: Path, target_sample_rate: int, target_channels: int, segment_seconds: float
    ) -> torch.Tensor:
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("音声ファイルの読み込みには torchaudio が必要です。") from exc

        waveform, sample_rate = torchaudio.load(str(audio_path))

        # リサンプリング
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # 時間方向の切り出しまたはパディング
        # 可視化なので、全尺処理したい場合はここを調整するが、
        # メモリ等の制約で固定長にするならsegment_secondsに従う
        if segment_seconds:
            max_samples = int(segment_seconds * target_sample_rate)
            current_samples = waveform.shape[1]
            if current_samples < max_samples:
                # 足りない場合はゼロパディング
                padding = torch.zeros((waveform.shape[0], max_samples - current_samples), dtype=waveform.dtype)
                waveform = torch.cat([waveform, padding], dim=1)
            else:
                # 長すぎる場合は切り出し
                waveform = waveform[:, :max_samples]

        # チャンネル数の調整
        current_channels = waveform.shape[0]
        if current_channels < target_channels:
            repeat_count = (target_channels + current_channels - 1) // current_channels
            waveform = waveform.repeat(repeat_count, 1)[:target_channels]
        elif current_channels > target_channels:
            waveform = waveform[:target_channels]

        # バッチ次元を追加 (1, Channels, Samples)
        return waveform.unsqueeze(0)


class MusicStructureVisualizer:
    def __init__(self, output_dir: Path, show_plot: bool = False):
        self.output_dir = output_dir
        self.show_plot = show_plot
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # クラスIDと名前のマッピングをロード (data/music_structures.json)
        self.id_to_label = {}
        structure_json = Path("data/music_structures.json")
        if structure_json.exists():
            with open(structure_json, "r", encoding="utf-8") as f:
                # json: {"Intro": 0, "Verse": 1, ...} -> {0: "Intro", ...}
                self.id_to_label = {v: k for k, v in json.load(f).items()}
        else:
            print(f"Warning: {structure_json} not found. Using numeric labels.")

    def process_frame_predictions(
        self, function_logits: np.ndarray, boundary_logits: np.ndarray, hop_length: int, sample_rate: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        SegmenterのNMS+Greedyロジックを用いて境界を検出し、
        セグメントごとにFunction LogitsをArgmaxしてラベルを決定します。
        """
        num_frames = function_logits.shape[0]

        # Segmenterの初期化 (パラメータは適宜調整、学習時と同じ設定が望ましいがここでは固定値)
        # min_segment_lengthはフレーム単位。1秒=SampleRate/HopLengthフレーム
        frames_per_sec = sample_rate / hop_length
        min_seg_len = int(frames_per_sec * 1.0)  # 1秒

        segmenter = BatchBoundarySegmenter(
            threshold=0.5,
            nms_window_radius=int(frames_per_sec * 0.5),  # 0.5秒窓
            min_segment_length=min_seg_len,
            max_segments=256,
        )

        # 境界検出 (1D)
        # boundary_logits: (T,) -> tensor
        boundary_tensor = torch.from_numpy(boundary_logits)
        probabilities = torch.sigmoid(boundary_tensor)

        # 内部メソッドを利用 (BatchBoundarySegmenterのインスタンスメソッド)
        # _detect_boundaries_1d は public method ではないが、ここでは再利用のために呼び出す
        # 本来は process_batch を使う設計だが、feature poolingまでは不要なため
        boundary_indices = segmenter._detect_boundaries_1d(probabilities)

        # セグメント区間に変換
        segments, _ = segmenter._convert_to_segments(boundary_indices, num_frames)

        # 各セグメントのラベル決定 (Argmax within segment)
        final_segments = []  # (start, end, label)
        processed_map = np.zeros(num_frames, dtype=int)

        for start, end in segments:
            if start >= end:
                continue

            # 区間内のlogitsを取得
            # (End - Start, NumClasses)
            seg_logits = function_logits[start:end]

            # 平均をとってArgmax (Sumでも同じ)
            # Softmaxしてから平均の方が確率的には正しいかもしれないが、Logits平均でも概ね機能する
            # ここではシンプルにLogitsの合計のArgmaxをとる
            avg_logits = np.mean(seg_logits, axis=0)  # (NumClasses,)
            label = np.argmax(avg_logits)

            final_segments.append((start, end, int(label)))
            processed_map[start:end] = int(label)

        return processed_map, final_segments

    def plot_results(
        self,
        function_logits: np.ndarray,
        boundary_logits: np.ndarray,
        filename: str,
        hop_length: int,
        sample_rate: int,
    ):
        """
        function_logits: (T, NumClasses)
        boundary_logits: (T,)
        """
        num_frames, num_classes = function_logits.shape
        duration = num_frames * hop_length / sample_rate
        time_axis = np.linspace(0, duration, num_frames)

        frames_per_sec = sample_rate / hop_length

        # NMS & Segmentation using BatchBoundarySegmenter logic
        segment_map, segments = self.process_frame_predictions(
            function_logits, boundary_logits, hop_length, sample_rate
        )

        # Boundary Probabilities (Sigmoid)
        boundary_probs = 1.0 / (1.0 + np.exp(-boundary_logits))

        # Plot Setup
        fig, (ax_func, ax_bound) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        # --- Plot 1: Structure Segments (Color Bar) ---
        # tab20: 20色。クラス数がこれを超える場合は繰り返し等が必要だが今回は11クラスなのでOK
        cmap = plt.get_cmap("tab20", num_classes) if num_classes <= 20 else plt.get_cmap("nipy_spectral", num_classes)

        # imshowで1次元配列を表示 (アスペクト比を自動調整)
        # extent=[0, duration, 0, 1] でY軸を0-1にする
        im = ax_func.imshow(
            segment_map[None, :],
            aspect="auto",
            origin="lower",
            extent=[0, duration, 0, 1],
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
            interpolation="nearest",
        )

        # セグメント境界線とラベル描画
        # 背景色等によっては見にくいので、境界線は黒か白
        ax_func.set_yticks([])
        ax_func.set_title("Predicted Music Structure (Segmented)")

        # 凡例 (Legend) 用のダミープロット
        # 各クラスの色と名前を表示したい
        # 実際に存在するクラスだけ凡例に出す
        present_labels = sorted(list(set([s[2] for s in segments])))
        patches = []
        for lbl_idx in present_labels:
            color = cmap(lbl_idx / (num_classes - 1) if num_classes > 1 else 0)
            label_name = self.id_to_label.get(lbl_idx, str(lbl_idx))
            patches.append(Line2D([0], [0], color=color, lw=4, label=label_name))

        if patches:
            ax_func.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize="small")

        # セグメントの上にテキストラベルを置く
        for start, end, label in segments:
            mid_time = (start + end) / 2 * hop_length / sample_rate
            label_name = self.id_to_label.get(label, str(label))
            # 短すぎるセグメントには文字を表示しない
            if (end - start) > frames_per_sec * 2:
                ax_func.text(
                    mid_time,
                    0.5,
                    label_name,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90 if (end - start) < frames_per_sec * 5 else 0,
                )

            # 境界線
            boundary_time = start * hop_length / sample_rate
            if start > 0:
                ax_func.axvline(boundary_time, color="white", linewidth=1, alpha=0.7)

        # --- Plot 2: Boundary Probability ---
        ax_bound.plot(time_axis, boundary_probs, color="tab:red", label="Boundary Prob")
        ax_bound.set_ylim(0, 1.05)
        ax_bound.set_ylabel("Boundary Probability")
        ax_bound.set_xlabel("Time (s)")
        ax_bound.grid(True, linestyle="--", alpha=0.6)

        # 閾値線
        ax_bound.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

        # 上のセグメントの切り替わり位置を下のグラフにも点線で表示
        for start, end, label in segments:
            if start > 0:
                boundary_time = start * hop_length / sample_rate
                ax_bound.axvline(boundary_time, color="black", linestyle="--", alpha=0.3, linewidth=0.8)

        self._save_and_show(fig, filename)

    def _save_and_show(self, fig, filename: str):
        fig.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=200)
        print(f"Saved: {output_path}")

        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MusicStructureTranscriptionModelの可視化スクリプト")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="学習用設定ファイルのパス")
    parser.add_argument("--checkpoint", type=Path, required=True, help="学習済みモデルのチェックポイント (.pt)")
    parser.add_argument("--audio", type=Path, required=True, help="入力音声ファイル")
    parser.add_argument(
        "--segment-seconds", type=float, default=None, help="解析する秒数（指定がなければ全尺またはconfig）"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("./results/structure_visualization"))
    parser.add_argument("--show", action="store_true", help="プロットを表示する")

    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # 1. モデルロード
    # use_structure_model=Trueを指定してロード
    model = build_model_from_config(config, use_structure_model=True).to(device)
    model.eval()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)  # Trainer保存形式に対応
    model.load_state_dict(state_dict, strict=False)

    # 2. データ準備
    sample_rate = int(config["data_loader"]["sample_rate"])
    hop_length = int(config["data_loader"]["hop_length"])

    num_stems = len(config["data_loader"]["stem_order"])
    channels_per_stem = 1 if config["data_loader"].get("mixdown_to_mono", False) else 2
    target_channels = num_stems * channels_per_stem

    # セグメント長: 指定があればそれ、なければconfig、それもなければ適当な長さ(例: 180秒)
    segment_seconds = args.segment_seconds
    waveform = AudioProcessor.load_and_preprocess(args.audio, sample_rate, target_channels, segment_seconds)

    # 3. 推論
    print("Running inference...")
    with torch.no_grad():
        outputs = model(waveform.to(device))

    # 4. 可視化
    # outputs keys: "initial_structure_function_logits", "initial_structure_boundary_logits"

    if "initial_structure_function_logits" in outputs:
        # (B, T, C) -> (T, C)
        func_logits = outputs["initial_structure_function_logits"][0].cpu().numpy()
        bound_logits = outputs["initial_structure_boundary_logits"][0].squeeze(-1).cpu().numpy()

        visualizer = MusicStructureVisualizer(args.output_dir, show_plot=args.show)
        visualizer.plot_results(
            function_logits=func_logits,
            boundary_logits=bound_logits,
            filename=f"{args.audio.stem}_structure.png",
            hop_length=hop_length,
            sample_rate=sample_rate,
        )
    else:
        print("Error: Model output does not contain structure logits.")


if __name__ == "__main__":
    main()
