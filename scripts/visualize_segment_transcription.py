import argparse
import math
import types
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import einops
import numpy as np
import torch
import torch.nn as nn
from matplotlib import colors, pyplot as plt

# 外部依存モジュールのインポート（インターフェースは維持）
from src.utils import build_model_from_config, load_base_model_checkpoint, load_config


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


class RuntimeTracer:
    """
    推論中のモデル内部状態（セグメント境界、Attention重み）をキャプチャするクラス。
    モンキーパッチなどのハック的な処理はこのクラス内に閉じ込める。
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.captured_segments: Optional[List[Tuple[int, int]]] = None
        self.captured_segment_ids: Optional[torch.Tensor] = None
        self.captured_attentions: Dict[int, List[torch.Tensor]] = {}
        self._exit_stack = ExitStack()

    def __enter__(self):
        self._register_segment_recorder()
        self._register_attention_recorder()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_stack.close()

    def _register_segment_recorder(self):
        """Segmenterのprocess_batchメソッドをフックして中間出力を取得する"""
        segmenter = self.model.segmenter
        original_process_batch = segmenter.process_batch

        def patched_process_batch(module_self, frame_features, boundary_logits, detach_boundary=True):
            seg_pad, seg_mask, seg_ids, segments = original_process_batch(
                frame_features, boundary_logits, detach_boundary=detach_boundary
            )
            self.captured_segment_ids = seg_ids.detach().cpu()
            self.captured_segments = segments
            return seg_pad, seg_mask, seg_ids, segments

        segmenter.process_batch = types.MethodType(patched_process_batch, segmenter)
        self._exit_stack.callback(lambda: setattr(segmenter, "process_batch", original_process_batch))

    def _register_attention_recorder(self):
        """TransformerのMultiHeadAttentionのforwardをフックしてAttention重みを取得する"""
        transformer = self.model.seg_transformer

        for layer_index, blocks in enumerate(transformer.layers):
            multi_head_attention = blocks[0]
            original_forward = multi_head_attention.forward

            def create_patched_forward(idx, orig_fn):
                def patched_forward(module_self, x, context=None, is_causal=False, attention_mask=None):
                    with torch.no_grad():
                        ctx = x if context is None else context

                        # モデル内部のAttention計算を再現
                        query = module_self.to_q(module_self.norm_q(x))
                        key = module_self.to_k(module_self.norm_context(ctx))

                        query = einops.rearrange(query, "b tq (h d) -> b h tq d", h=module_self.num_heads)
                        key = einops.rearrange(key, "b tk (h d) -> b h tk d", h=module_self.num_heads)

                        query, key = module_self.rope(query, key)

                        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
                        scores = scores / math.sqrt(module_self.head_dim)

                        if attention_mask is not None:
                            attn_mask = attention_mask.to(device=scores.device)
                            if attn_mask.dtype != torch.bool:
                                attn_mask = attn_mask != 0
                            while attn_mask.dim() < scores.dim() - 2:
                                attn_mask = attn_mask.unsqueeze(1)
                            scores = scores.masked_fill(attn_mask, float("-inf"))

                        if is_causal:
                            t_q, t_k = scores.shape[-2], scores.shape[-1]
                            causal_mask = torch.triu(
                                torch.ones((t_q, t_k), dtype=torch.bool, device=scores.device), diagonal=1
                            )
                            scores = scores.masked_fill(causal_mask, float("-inf"))

                        weights = torch.softmax(scores, dim=-1)
                        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

                        self.captured_attentions.setdefault(idx, []).append(weights.cpu())

                    return orig_fn(x, context=context, is_causal=is_causal, attention_mask=attention_mask)

                return patched_forward

            patched = create_patched_forward(layer_index, original_forward)
            multi_head_attention.forward = types.MethodType(patched, multi_head_attention)
            self._exit_stack.callback(lambda m=multi_head_attention, o=original_forward: setattr(m, "forward", o))


class ModelInference:
    """モデルのセットアップと推論実行を管理するクラス"""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = build_model_from_config(self.config, use_segment_model=True).to(self.device)
        model.eval()
        return model

    def load_weights(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("ema_state_dict", checkpoint)

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        print(f"モデルの重みを読み込みました: {len(state_dict)} tensors")

        if missing_keys:
            print(f"[警告] 不足しているキー: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[警告] 予期しないキー: {unexpected_keys[:5]}...")

    def run_inference_with_trace(self, waveform: torch.Tensor) -> Tuple[Dict[str, Any], RuntimeTracer]:
        tracer = RuntimeTracer(self.model)
        with tracer:
            with torch.inference_mode():
                outputs = self.model(waveform.to(self.device))

        if tracer.captured_segments is None:
            raise RuntimeError("セグメント情報のキャプチャに失敗しました。")

        return outputs, tracer

    def get_base_boundary_logits(
        self, base_checkpoint_path: Path, waveform: torch.Tensor, batch_index: int
    ) -> np.ndarray:
        base_model = build_model_from_config(self.config, use_segment_model=False).to(self.device)
        base_model.eval()
        load_base_model_checkpoint(base_model, str(base_checkpoint_path), self.device)

        with torch.inference_mode():
            outputs = base_model(waveform.to(self.device))

        if "initial_boundary_logits" not in outputs:
            raise RuntimeError("Baseモデルの出力に 'initial_boundary_logits' が含まれていません。")

        return outputs["initial_boundary_logits"][batch_index].squeeze(-1).detach().cpu().numpy()


class Visualizer:
    """可視化結果の描画と保存を担当するクラス"""

    def __init__(self, output_dir: Path, show_plot: bool = False):
        self.output_dir = output_dir
        self.show_plot = show_plot
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_segments(
        self,
        segment_ids: np.ndarray,
        segments: List[Tuple[int, int]],
        boundary_logits: Optional[np.ndarray],
        boundary_label: str,
        hop_length: int,
        sample_rate: int,
        filename: str,
    ):
        total_frames = segment_ids.shape[0]
        duration = (total_frames - 1) * hop_length / sample_rate if total_frames > 1 else 1.0
        extent = [0.0, duration, 0.0, 1.0]

        has_logits = boundary_logits is not None
        fig_height = 3.6 if has_logits else 2.0

        if has_logits:
            fig, (ax_seg, ax_prob) = plt.subplots(2, 1, figsize=(10.0, fig_height), sharex=True)
        else:
            fig, ax_seg = plt.subplots(figsize=(10.0, fig_height))
            ax_prob = None

        ax_seg.imshow(
            segment_ids[None, :], aspect="auto", origin="lower", interpolation="nearest", cmap="tab20", extent=extent
        )

        for start_idx, _ in segments[1:]:
            boundary_time = start_idx * hop_length / sample_rate
            ax_seg.axvline(boundary_time, color="white", linewidth=0.8, alpha=0.8)

        ax_seg.set_yticks([])
        ax_seg.set_ylabel("Segment ID")
        ax_seg.set_title(f"Segments (count={len(segments)})")

        if ax_prob is not None and has_logits:
            probabilities = 1.0 / (1.0 + np.exp(-boundary_logits))
            time_axis = np.linspace(0.0, duration, num=total_frames)
            ax_prob.plot(time_axis, probabilities, color="tab:red", linewidth=1.0)
            ax_prob.set_ylim(0.0, 1.0)
            ax_prob.set_ylabel(boundary_label)
            ax_prob.set_xlabel("Time (s)")
        else:
            ax_seg.set_xlabel("Time (s)")

        self._save_and_show(fig, filename)

    def plot_attention(
        self,
        attention_matrix: np.ndarray,
        title: str,
        filename: str,
        color_scale: str = "power",
        gamma: float = 0.5,
        vmax_percentile: float = 99.0,
    ):
        vmax = float(np.percentile(attention_matrix, vmax_percentile))
        if vmax <= 0.0:
            vmax = float(attention_matrix.max()) if float(attention_matrix.max()) > 0 else 1.0
        vmin = 0.0

        if color_scale == "log":
            epsilon = max(
                1e-6, float(attention_matrix[attention_matrix > 0].min()) if np.any(attention_matrix > 0) else 1e-6
            )
            norm = colors.LogNorm(vmin=epsilon, vmax=vmax)
        elif color_scale == "power":
            norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        clipped_data = np.clip(attention_matrix, vmin, vmax)

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        img = ax.imshow(clipped_data, aspect="auto", origin="upper", interpolation="nearest", norm=norm)
        fig.colorbar(img, ax=ax, shrink=0.8, label="Attention weight")

        ax.set_xlabel("Query segment")
        ax.set_ylabel("Key segment")
        ax.set_title(title)

        self._save_and_show(fig, filename)

    def _save_and_show(self, fig, filename: str):
        fig.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=200)
        # 大量に出力されるため、ログ出力を簡略化または抑制してもよいが、ここでは保持
        print(f"Saved: {filename}")

        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SegmentTranscriptionModelのセグメント構造と全Attention Headの可視化を行います。"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="学習用設定ファイルのパス")
    parser.add_argument("--checkpoint", type=Path, required=True, help="ベースモデルのチェックポイント (.pt)")
    parser.add_argument("--audio", type=Path, help="入力音声ファイル")
    parser.add_argument("--segment-seconds", type=float, default=None, help="解析する秒数（デフォルトはconfig準拠）")
    parser.add_argument("--device", type=str, default="cpu", help="使用デバイス (例: cuda, cpu)")
    parser.add_argument("--batch-index", type=int, default=0, help="可視化対象のバッチインデックス")

    # セグメント可視化設定
    parser.add_argument("--boundary-source", choices=["segment", "base"], default="segment", help="境界確率のソース")
    parser.add_argument(
        "--base-checkpoint", type=Path, default=None, help="Baseモデルのチェックポイント（boundary-source=base用）"
    )
    parser.add_argument("--max-segments", type=int, default=None, help="可視化するセグメント数の上限")
    parser.add_argument("--no-boundary-prob", action="store_true", help="境界確率のプロットを無効化")

    # 出力設定
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./results/segment_visualization"), help="保存先ディレクトリ"
    )
    parser.add_argument("--show", action="store_true", help="保存後に画像を表示")
    parser.add_argument(
        "--color-scale", choices=["linear", "power", "log"], default="power", help="Attentionマップの色スケール"
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="PowerNorm用のガンマ値")
    parser.add_argument("--vmax-percentile", type=float, default=99.0, help="色スケールの最大値のパーセンタイル")

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config)
    device = torch.device(args.device)

    # 1. データ準備
    sample_rate = int(config["data_loader"]["sample_rate"])
    hop_length = int(config["data_loader"]["hop_length"])

    num_stems = len(config["data_loader"]["stem_order"])
    channels_per_stem = 1 if config["data_loader"].get("mixdown_to_mono", False) else 2
    target_channels = num_stems * channels_per_stem

    segment_seconds = args.segment_seconds or float(config.get("base_model_training", {}).get("segment_seconds", 60.0))

    if args.audio:
        waveform = AudioProcessor.load_and_preprocess(args.audio, sample_rate, target_channels, segment_seconds)
    else:
        num_samples = int(sample_rate * segment_seconds)
        waveform = torch.randn(1, target_channels, num_samples)
        print("[Info] 音声ファイルが指定されていないため、ランダムノイズを使用します。")

    # 2. 推論実行
    inference = ModelInference(config, device)
    inference.load_weights(args.checkpoint)

    print("推論を開始します...")
    outputs, tracer = inference.run_inference_with_trace(waveform)

    captured_seg_ids = tracer.captured_segment_ids
    if args.batch_index < 0 or args.batch_index >= captured_seg_ids.shape[0]:
        raise ValueError(f"batch-index は [0, {captured_seg_ids.shape[0] - 1}] の範囲である必要があります。")

    # 3. 可視化データの抽出
    target_seg_ids = captured_seg_ids[args.batch_index].numpy()
    target_segments = tracer.captured_segments[args.batch_index]

    boundary_logits = None
    boundary_label = "Boundary p (segment)"

    if not args.no_boundary_prob:
        if args.boundary_source == "base":
            base_ckpt = args.base_checkpoint or args.checkpoint
            boundary_logits = inference.get_base_boundary_logits(base_ckpt, waveform, args.batch_index)
            boundary_label = "Boundary p (base)"
        elif "initial_boundary_logits" in outputs:
            boundary_logits = outputs["initial_boundary_logits"][args.batch_index].squeeze(-1).detach().cpu().numpy()

    # 4. 描画処理
    visualizer = Visualizer(args.output_dir, show_plot=args.show)

    visualizer.plot_segments(
        segment_ids=target_seg_ids,
        segments=target_segments,
        boundary_logits=boundary_logits,
        boundary_label=boundary_label,
        hop_length=hop_length,
        sample_rate=sample_rate,
        filename=f"segments_batch{args.batch_index}.png",
    )

    # Attentionの全レイヤー・全ヘッドプロット
    num_segments_to_plot = len(target_segments)
    if args.max_segments is not None and args.max_segments > 0:
        num_segments_to_plot = min(num_segments_to_plot, args.max_segments)

    # 取得された全てのレイヤーに対してループ
    for layer_idx, layer_attentions in sorted(tracer.captured_attentions.items()):
        # (Batch, Heads, Segments, Segments)
        layer_attention = layer_attentions[0][args.batch_index]
        num_heads = layer_attention.shape[0]

        # そのレイヤーの全てのヘッドに対してループ
        for head_idx in range(num_heads):
            matrix = layer_attention[head_idx, :num_segments_to_plot, :num_segments_to_plot].numpy()
            visualizer.plot_attention(
                attention_matrix=matrix,
                title=f"seg_transformer L{layer_idx} Head{head_idx}",
                filename=f"attention_layer{layer_idx}_head{head_idx}.png",
                color_scale=args.color_scale,
                gamma=args.gamma,
                vmax_percentile=args.vmax_percentile,
            )


if __name__ == "__main__":
    main()
