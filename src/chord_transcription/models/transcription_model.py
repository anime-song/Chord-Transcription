import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any


from .spec_augment import SpecAugment
from .cqt import RecursiveCQT
from .transformer import RMSNorm, Transformer
from .semi_crf import (
    _build_interval_score,
)


class AudioFeatureExtractor(nn.Module):
    """
    音声波形から特徴量(CQTスペクトログラム)を抽出するクラス。
    SpecAugmentや標準化も行う。
    """

    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        bins_per_octave: int = 12 * 3,
        n_bins: int = 12 * 3 * 8,
        spec_augment_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_audio_channels = 2

        self.bins_per_octave = int(bins_per_octave)
        self.n_bins = int(n_bins)
        if self.bins_per_octave <= 0 or self.bins_per_octave % 12 != 0:
            raise ValueError("bins_per_octave must be a positive multiple of 12")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self.cqt = RecursiveCQT(
            sr=sampling_rate,
            hop_length=hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            filter_scale=0.4375,
            fmin=27.5,
        )

        self.spec_augment = SpecAugment(**spec_augment_params) if spec_augment_params else None

    def forward(self, waveform: torch.Tensor) -> "BackboneContext":
        # center=True の STFT を想定しているため、ラベルと一致するようにクロップする
        crop_length = (waveform.shape[-1] - self.n_fft) // self.hop_length + 1

        batch_size = waveform.shape[0]
        waveform_flat = einops.rearrange(waveform, "b c t -> (b c) t")
        spec = self.cqt(waveform_flat.float(), return_complex=False)
        spec = einops.rearrange(spec, "(b c) f t -> b c f t", b=batch_size, c=self.num_audio_channels).contiguous()

        # 標準化(バッチ単位の全体平均/分散)
        mean = spec.mean(dim=(1, 2, 3), keepdim=True)
        std = spec.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        spec = (spec - mean) / std

        if self.training and self.spec_augment is not None:
            spec, _ = self.spec_augment(spec)

        spec = spec.to(waveform.dtype)

        spec = einops.rearrange(spec, "b c f t -> b c t f").contiguous()

        return BackboneContext(
            spec=spec,
            crop_length=crop_length,
        )


@dataclass
class BackboneContext:
    spec: torch.Tensor
    crop_length: int


@dataclass
class BackboneOutput:
    """Backbone の出力を band / interval query に分離して保持する。"""

    band_features: torch.Tensor  # [B, T, D] — band tokens
    interval_query_features: torch.Tensor  # [B, T, num_interval_queries, D]


def checkpoint(
    module: nn.Module,
    x: torch.Tensor,
    *,
    use_checkpoint: bool,
    **kwargs: Any,
) -> torch.Tensor:
    if not use_checkpoint:
        return module(x, **kwargs)

    def forward_fn(tensor: torch.Tensor) -> torch.Tensor:
        return module(tensor, **kwargs)

    return torch.utils.checkpoint.checkpoint(
        forward_fn,
        x,
        use_reentrant=False,
    )


class StemConv(nn.Module):
    """
    入力:
        x: [B, in_ch, T, F]

    出力:
        y: [B, 4 * base_ch, T/8, F/4]
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.block1 = nn.Sequential(
            nn.ConstantPad2d((2, 1, 4, 3), value=0.0),
            nn.Conv2d(
                in_channels=base_ch,
                out_channels=base_ch * 2,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 1),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 2),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=base_ch * 2,
                out_channels=base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 2),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=base_ch * 4,
                out_channels=base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 2),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
        )

        self.out_ch = base_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Backbone(nn.Module):
    def __init__(
        self,
        feature_extractor: AudioFeatureExtractor,
        hidden_size: int,
        base_ch: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 8,
        num_interval_queries: int = 4,
        low_bands: int = 32,
        dropout: float = 0.0,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if num_interval_queries <= 0:
            raise ValueError("num_interval_queries must be positive")

        self.feature_extractor = feature_extractor
        self.hidden_size = hidden_size
        self.base_ch = base_ch
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.num_interval_queries = num_interval_queries

        self.num_audio_channels = feature_extractor.num_audio_channels
        self.n_bins = feature_extractor.n_bins
        self.model_dim = hidden_size

        # Stem は時間方向だけ 1/8 に圧縮し、周波数方向は保持する。
        self.stem = StemConv(
            in_ch=self.num_audio_channels,
            base_ch=base_ch,
            dropout=dropout,
        )

        # コード区間クエリ: semi-CRF のスコア計算に使う学習可能トークン
        self.interval_query_embed = nn.Embedding(num_interval_queries, base_ch * 4)
        self.register_buffer(
            "interval_ids",
            torch.arange(num_interval_queries, dtype=torch.long),
            persistent=False,
        )

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            time_roformer = Transformer(
                input_dim=base_ch * 4,
                head_dim=hidden_size // num_heads,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=4,
                dropout=dropout,
            )
            band_roformer = Transformer(
                input_dim=base_ch * 4,
                head_dim=hidden_size // num_heads,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=4,
                dropout=dropout,
            )
            self.layers.append(nn.ModuleList([time_roformer, band_roformer]))

        self.stem_dim = base_ch * 4  # Transformer の入出力次元
        self.query_feature_dim = self.stem_dim
        self.final_norm = RMSNorm(self.stem_dim)

        self.up_conv = nn.ConvTranspose1d(
            self.stem_dim,
            self.stem_dim,
            kernel_size=8,
            stride=8,
        )

        # band tokens の帯域圧縮: num_bands → low_bands に縮小してから flatten + Linear
        self.low_bands = low_bands
        # StemConvによる周波数方向のダウンサンプリング結果である正確なサイズを求める
        with torch.no_grad():
            dummy = torch.zeros(1, self.num_audio_channels, 64, self.n_bins)
            dummy_out = self.stem(dummy)
            num_bands = dummy_out.shape[-1]  # 厳密な周波数ビンの数を取得

        self.band_pool = nn.Sequential(
            nn.Linear(num_bands, low_bands),
            nn.RMSNorm(low_bands),
            nn.GELU())
        self.band_proj = nn.Linear(low_bands * self.stem_dim, self.output_dim)

    @staticmethod
    def _match_time_length(x: torch.Tensor, target_T: int) -> torch.Tensor:
        """x の時間次元（dim=2）を target_T に合わせる。"""
        if x.shape[2] < target_T:
            return F.pad(x, (0, 0, 0, target_T - x.shape[2]))
        return x[:, :, :target_T]

    def forward(
        self,
        waveform: torch.Tensor,
        context: Optional[BackboneContext] = None,
    ) -> BackboneOutput:
        if context is None:
            context = self.feature_extractor(waveform)

        use_checkpoint = self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()

        # CQT を stem で時間圧縮し、周波数方向は band token に切る。
        # stem 出力: [B, D_stem, T/8, F'] → rearrange → [B, T/8, F', D_stem]
        stem_features = self.stem(context.spec)
        x = einops.rearrange(stem_features, "b d t f -> b t f d")

        B, T, num_bands, D = x.shape

        interval_query = self.interval_query_embed(self.interval_ids)  # [N_iq, D]
        interval_query = interval_query.unsqueeze(0).unsqueeze(0)  # [1, 1, N_iq, D]
        interval_query = interval_query.expand(B, T, -1, -1)  # [B, T, N_iq, D]

        # [B, T, num_bands + N_iq, D]
        x = torch.cat([x, interval_query], dim=2)

        for time_roformer, band_roformer in self.layers:
            B, T, K, D = x.shape
            # バンド軸Transformer
            x = x.reshape(B * T, K, D)
            x = checkpoint(band_roformer, x, use_checkpoint=use_checkpoint)
            x = x.reshape(B, T, K, D)

            # 時間軸Transformer
            x = einops.rearrange(x, "b t k d -> (b k) t d")
            x = checkpoint(time_roformer, x, use_checkpoint=use_checkpoint)
            x = einops.rearrange(x, "(b k) t d -> b t k d", k=K)

        x = self.final_norm(x)

        # アップサンプリング
        B, T, K, D = x.shape
        x = einops.rearrange(x, "b t k d -> (b k) d t")
        x = self.up_conv(x)
        frame_features = einops.rearrange(x, "(b k) d t -> b k t d", k=K)

        target_T = context.crop_length
        # STFT/CQT の center 処理に合わせて最終長をラベル側に揃える。
        frame_features = self._match_time_length(frame_features, target_T)
        # frame_features: [B, K, T_out, D]

        # band / interval query を分離
        band_part = frame_features[:, :num_bands, :, :]  # [B, num_bands, T, D]
        interval_part = frame_features[:, num_bands:, :, :]  # [B, N_iq, T, D]

        # band tokens: num_bands → low_bands に圧縮してから flatten + Linear
        # [B, num_bands, T, D] → [B*T, D, num_bands] (channels=D, length=num_bands)
        B_b, nb, T_b, D_b = band_part.shape
        band_flat = band_part.permute(0, 2, 3, 1).reshape(B_b * T_b, D_b, nb)
        band_pooled = self.band_pool(band_flat)  # [B*T, D, low_bands]
        band_features = band_pooled.reshape(B_b, T_b, D_b * self.low_bands)  # [B, T, D * low_bands]
        band_features = self.band_proj(band_features)  # [B, T, output_dim]

        interval_query_features = interval_part.permute(0, 2, 1, 3).contiguous()  # [B, T, N_iq, stem_dim]

        return BackboneOutput(
            band_features=band_features.contiguous(),
            interval_query_features=interval_query_features.contiguous(),
        )


class ChordIntervalScorer(nn.Module):
    """
    Interval query 特徴量から semi-CRF 用のスコアテンソルを構築する。

    Transkun v2 と同様に scaled inner product で interval score を計算し、
    既存の NeuralSemiCRFInterval で decode / loss を行う。
    """

    def __init__(
        self,
        input_dim: int,
        interval_dim: int = 32,
        num_interval_queries: int = 4,
        length_scaling: str = "sqrt",
    ) -> None:
        super().__init__()
        self.interval_dim = interval_dim
        self.length_scaling = length_scaling

        # interval query 全体を集約してから query/key を射影
        aggregated_dim = input_dim * num_interval_queries
        self.query_proj = nn.Linear(aggregated_dim, interval_dim)
        self.key_proj = nn.Linear(aggregated_dim, interval_dim)
        self.diag_proj = nn.Linear(aggregated_dim, 1)

    def forward(
        self,
        interval_query_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            interval_query_features: [B, T, num_interval_queries, D]

        Returns:
            interval_query: [B, T, interval_dim]
            interval_key: [B, T, interval_dim]
            interval_diag: [B, T]  — singleton (対角) スコア
        """
        B, T, num_queries, D = interval_query_features.shape
        # 全 interval query を結合して 1 つの特徴ベクトルにする
        aggregated = interval_query_features.reshape(B, T, num_queries * D)

        interval_query = self.query_proj(aggregated)  # [B, T, interval_dim]
        interval_key = self.key_proj(aggregated)  # [B, T, interval_dim]
        interval_diag = self.diag_proj(aggregated).squeeze(-1)  # [B, T]

        return {
            "interval_query": interval_query,
            "interval_key": interval_key,
            "interval_diag": interval_diag,
        }

    def build_score_tensor(
        self,
        interval_query: torch.Tensor,
        interval_key: torch.Tensor,
        interval_diag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score テンソルを構築する。

        Args:
            interval_query: [B, T, interval_dim]
            interval_key: [B, T, interval_dim]
            interval_diag: [B, T]

        Returns:
            score: [T, T, B] — semi-CRF の入力形式
        """
        # [T, B, D] に転置して _build_interval_score に渡す
        query_tbf = interval_query.permute(1, 0, 2)  # [T, B, D]
        key_tbf = interval_key.permute(1, 0, 2)  # [T, B, D]
        diag_tb = interval_diag.permute(1, 0)  # [T, B]

        score = _build_interval_score(
            query_tbf,
            key_tbf,
            diag_tb,
            length_scaling=self.length_scaling,
        )
        return score


class BaseTranscriptionModel(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        hidden_size: int,
        num_quality_classes: int,
        num_bass_classes: int,
        num_key_classes: int,
        dropout_probability: float = 0.0,
        use_layer_norm: bool = True,
        chord_interval_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.detach_label_heads = True
        self.norm = RMSNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_probability) if dropout_probability > 0.0 else nn.Identity()

        self.num_root_quality_classes = (num_quality_classes - 1) * 12 + 1

        # Stage 1: backbone 特徴量から各属性を予測するヘッド
        self.boundary_head = nn.Linear(hidden_size, 1)
        self.beat_head = nn.Linear(hidden_size, 1)
        self.downbeat_head = nn.Linear(hidden_size, 1)
        self.key_boundary_head = nn.Linear(hidden_size, 1)
        self.key_head = nn.Linear(hidden_size, num_key_classes)
        self.root_chord_head = nn.Linear(hidden_size, self.num_root_quality_classes)
        self.bass_head = nn.Linear(hidden_size, num_bass_classes)
        self.pitch_chroma_head = nn.Linear(hidden_size, 12)

        # Chord Interval Scorer: semi-CRF 用
        self.interval_feature_dim = backbone.query_feature_dim
        chord_interval_cfg = chord_interval_config or {}
        self.chord_interval_enabled = bool(chord_interval_cfg.get("enabled", True))
        if self.chord_interval_enabled:
            self.chord_interval_scorer = ChordIntervalScorer(
                input_dim=self.interval_feature_dim,
                interval_dim=int(chord_interval_cfg.get("interval_dim", 32)),
                num_interval_queries=backbone.num_interval_queries,
                length_scaling=str(chord_interval_cfg.get("length_scaling", "sqrt")),
            )

    def _compute_heads(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stage 1: backbone の band 特徴量から各属性を予測する。"""
        normed = self.norm(features)
        features = self.dropout(normed)

        boundary_logits = self.boundary_head(features)
        downbeat_logits = self.downbeat_head(features)
        beat_logits = self.beat_head(features) + downbeat_logits
        key_logits = self.key_head(features)
        key_boundary_logits = self.key_boundary_head(features)

        return {
            "initial_key_logits": key_logits,
            "initial_boundary_logits": boundary_logits,
            "initial_beat_logits": beat_logits,
            "initial_downbeat_logits": downbeat_logits,
            "initial_key_boundary_logits": key_boundary_logits,
            "initial_features": features,  # CRF 等の外部モジュールが使用
            "initial_root_chord_logits": self.root_chord_head(features),
            "initial_bass_logits": self.bass_head(features),
            "pitch_chroma_logits": self.pitch_chroma_head(features)
        }

    def _compute_interval_scores(
        self,
        interval_query_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Interval query 特徴量から semi-CRF 用のスコアを計算する。

        Returns:
            interval_query, interval_key, interval_diag の辞書
        """
        if not self.chord_interval_enabled:
            return {}
        return self.chord_interval_scorer(interval_query_features)

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> Dict[str, Any]:
        # --- Backbone ---
        backbone_output = self.backbone(waveform)

        # --- Band features → 各属性ヘッド ---
        outputs = self._compute_heads(backbone_output.band_features)

        # --- Interval Query → Semi-CRF スコア ---
        interval_scores = self._compute_interval_scores(
            backbone_output.interval_query_features,
        )
        outputs.update(interval_scores)
        outputs["interval_query_features"] = backbone_output.interval_query_features

        return outputs
