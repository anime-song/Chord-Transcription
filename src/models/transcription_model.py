import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union


from .spec_augment import SpecAugment
from .cqt import RecursiveCQT
from .transformer import RMSNorm, Transformer


def checkpoint_bypass(func, *args, **kwargs):
    """チェックポイントを使用しない場合のバイパス関数"""
    return func(*args)


class OctaveSharedAggregate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        f_total: int,
        hidden_size: int,
        bins_per_octave: int = 36,
        octave_emb_dim: int = 32,
        conv_kernel_size: int = 3,
        dropout: float = 0.0,
        gate_hidden_factor: float = 0.5,
        use_film: bool = True,
        return_weights: bool = False,
    ):
        super().__init__()
        assert f_total % bins_per_octave == 0, "F_total must be divisible by bins_per_octave"
        self.in_channels = in_channels
        self.f_total = f_total
        self.hidden = hidden_size
        self.bins_per_octave = bins_per_octave
        self.num_octaves = f_total // bins_per_octave
        self.use_film = use_film
        self.return_weights = return_weights

        pad = conv_kernel_size // 2

        def _choose_gn_groups(channels: int, max_groups: int = 8) -> int:
            # GroupNormのgroupsは channels を割り切る必要あり
            for g in reversed(range(1, max_groups + 1)):
                if channels % g == 0:
                    return g
            return 1

        gn_groups = _choose_gn_groups(hidden_size, max_groups=8)

        # Shared conv applied per octave: (B*O, C, T, bins) -> (B*O, hidden, T, bins)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=conv_kernel_size, padding=pad),
            nn.GroupNorm(gn_groups, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Octave index embedding -> FiLM params (gamma, beta) per octave
        #    gamma/beta are channel-wise (hidden) and broadcast over (T, bins).
        self.oct_emb = nn.Embedding(self.num_octaves, octave_emb_dim)

        if use_film:
            self.film = nn.Sequential(
                nn.Linear(octave_emb_dim, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
            )
        else:
            # additive bias (weaker but simpler)
            self.oct_add = nn.Linear(octave_emb_dim, hidden_size)

        # Data-dependent octave gate: softmax over octaves
        gate_h = max(8, int(hidden_size * gate_hidden_factor))
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, gate_h),
            nn.GELU(),
            nn.Linear(gate_h, 1),
        )

    def forward(self, x: torch.Tensor):
        B, C, T, F = x.shape

        # (B, C, T, (O*bins)) -> (B, O, C, T, bins)
        x = einops.rearrange(x, "b c t (o bins) -> b o c t bins", o=self.num_octaves, bins=self.bins_per_octave)

        # merge (B,O) for shared conv
        octave_features = []
        for i in range(self.num_octaves):
            octave_features.append(self.shared(x[:, i]))
        x = torch.stack(octave_features, dim=1)

        # octave index embedding injection
        octave_ids = torch.arange(self.num_octaves, device=x.device)  # (O,)
        emb = self.oct_emb(octave_ids)  # (O, E)
        emb = emb.unsqueeze(0).expand(B, -1, -1)  # (B, O, E)

        if self.use_film:
            film = self.film(emb)  # (B, O, 2H)
            gamma, beta = film.chunk(2, dim=-1)  # (B, O, H), (B, O, H)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, O, H, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, O, H, 1, 1)
            x = x * (1.0 + torch.tanh(gamma)) + beta
        else:
            add = self.oct_add(emb).unsqueeze(-1).unsqueeze(-1)  # (B, O, H, 1, 1)
            x = x + add

        # gate weights from per-octave summary
        # summary: (B, O, H)
        summary = x.mean(dim=(-1, -2))  # average over (T,bins) -> (B, O, H)
        w = self.gate(summary)  # (B, O, 1)
        w = torch.softmax(w, dim=1)  # softmax over octaves
        w5 = w.unsqueeze(-1).unsqueeze(-1)  # (B, O, 1, 1, 1)

        # weighted sum over octave dim -> (B, H, T, bins)
        y = (x * w5).sum(dim=1)

        if self.return_weights:
            # return weights as (B, O)
            return y, w.squeeze(-1)
        return y


class AudioFeatureExtractor(nn.Module):
    """
    音声波形から特徴量（CQTスペクトログラム）を抽出するクラス。
    SpecAugmentや標準化も行う。
    """

    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        num_audio_channels: int = 12,
        num_stems: int = 6,
        spec_augment_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_audio_channels = num_audio_channels
        self.num_stems = num_stems

        if self.num_stems <= 0:
            raise ValueError("num_stems must be a positive integer")
        if self.num_audio_channels % self.num_stems != 0:
            raise ValueError(
                f"num_audio_channels ({self.num_audio_channels}) must be divisible by num_stems ({self.num_stems})"
            )
        self.channels_per_stem = self.num_audio_channels // self.num_stems

        self.bins_per_octave = 12 * 3
        self.n_bins = 12 * 3 * 7  # 252 bins
        self.cqt = RecursiveCQT(
            sr=sampling_rate,
            hop_length=hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            filter_scale=0.4375,
        )

        self.spec_augment = SpecAugment(**spec_augment_params) if spec_augment_params else None

    def forward(self, waveform: torch.Tensor) -> "BackboneContext":
        # center=True の STFT を想定しているため、ラベルと一致するようにクロップする
        crop_length = (waveform.shape[-1] - self.n_fft) // self.hop_length + 1

        waveform_per_stem = einops.rearrange(
            waveform, "b (s c) t -> b s c t", s=self.num_stems, c=self.channels_per_stem
        )

        if self.training:
            batch_size = waveform.shape[0]
            stem_specs: List[torch.Tensor] = []
            for stem_idx in range(self.num_stems):
                stem_waveform = waveform_per_stem[:, stem_idx]  # [B, C_per_stem, T]
                stem_flat = einops.rearrange(stem_waveform, "b c t -> (b c) t")
                stem_cqt = self.cqt(stem_flat.float(), return_complex=False)
                stem_cqt = einops.rearrange(
                    stem_cqt, "(b c) f t -> b c f t", b=batch_size, c=self.channels_per_stem
                ).contiguous()

                if self.spec_augment is not None:
                    stem_cqt_aug, _ = self.spec_augment(stem_cqt)
                    stem_specs.append(stem_cqt_aug)
                    del stem_cqt, stem_cqt_aug
                else:
                    stem_specs.append(stem_cqt)
                    del stem_cqt
            spec = torch.cat(stem_specs, dim=1)
        else:
            waveform_flat = einops.rearrange(waveform, "b c t -> (b c) t")
            spec = self.cqt(waveform_flat.float(), return_complex=False)
            spec = einops.rearrange(spec, "(b c) f t -> b c f t", c=self.num_audio_channels)

        # 標準化（バッチ単位の全体平均/分散）
        mean = spec.mean(dim=(2, 3), keepdim=True)
        std = spec.std(dim=(2, 3), keepdim=True) + 1e-8
        spec = (spec - mean) / std
        spec = spec.to(waveform.dtype)

        spec = einops.rearrange(spec, "b c f t -> b c t f").contiguous()

        original_time_steps = spec.shape[-2]

        return BackboneContext(
            spec=spec,
            crop_length=crop_length,
            original_time_steps=original_time_steps,
        )


@dataclass
class BackboneContext:
    spec: torch.Tensor
    crop_length: int
    original_time_steps: int


class Backbone(nn.Module):
    def __init__(
        self,
        feature_extractor: AudioFeatureExtractor,
        hidden_size: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.hidden_size = hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.num_audio_channels = feature_extractor.num_audio_channels
        self.n_bins = feature_extractor.n_bins

        self.oct_frontend = OctaveSharedAggregate(
            in_channels=self.num_audio_channels,  # 12
            f_total=self.n_bins,  # 252
            hidden_size=hidden_size,
            bins_per_octave=feature_extractor.bins_per_octave,
            octave_emb_dim=32,
            dropout=dropout,
            use_film=True,
        )

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1, stride=(2, 1)),
            nn.GroupNorm(4, hidden_size * 2),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 2,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 4,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_size * 4),
        )

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            time_roformer = Transformer(
                input_dim=hidden_size * 4,
                head_dim=hidden_size * 4 // 8,
                num_layers=1,
                num_heads=8,
                ffn_hidden_size_factor=2,
                dropout=dropout,
            )
            band_roformer = Transformer(
                input_dim=hidden_size * 4,
                head_dim=hidden_size * 4 // 8,
                num_layers=1,
                num_heads=8,
                ffn_hidden_size_factor=2,
                dropout=dropout,
            )
            self.layers.append(nn.ModuleList([time_roformer, band_roformer]))
        self.final_norm = RMSNorm(hidden_size * 4)

        freq_downsample_factor = 1
        input_freq_bins = feature_extractor.bins_per_octave
        output_freq_bins = input_freq_bins // freq_downsample_factor

        final_channels = hidden_size * 4
        flattened_dim = output_freq_bins * final_channels

        self.to_time_features = nn.Conv1d(flattened_dim, self.output_dim, kernel_size=1)

        self.up_time = nn.ConvTranspose1d(
            self.output_dim,
            self.output_dim,
            kernel_size=8,
            stride=8,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        context: Optional[BackboneContext] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if context is None:
            context = self.feature_extractor(waveform)

        use_checkpoint = self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()
        checkpoint_fn = torch.utils.checkpoint.checkpoint if use_checkpoint else checkpoint_bypass

        x = self.oct_frontend(context.spec)  # (B, hidden_size, T, 36)
        x = self.conv1(x)
        # ダウンサンプリング
        x = self.down_conv(x)
        x = einops.rearrange(x, "b c t f -> b t f c")  # (B, downT, F, C)

        intermediate_features = []

        for time_roformer, band_roformer in self.layers:
            B, T, freq, C = x.shape
            # 周波数軸Transformer
            x = x.reshape(B * T, freq, C)  # [B*T, F, C]
            x = checkpoint_fn(band_roformer, x, use_reentrant=False)
            x = x.reshape(B, T, freq, C)

            # 時間軸Transformer
            x = einops.rearrange(x, "b t f c -> (b f) t c")  # [B*F, T, C]
            x = checkpoint_fn(time_roformer, x, use_reentrant=False)
            x = einops.rearrange(x, "(b f) t c -> b t f c", f=freq)  # [B, T, F, C]

            if return_intermediate:
                intermediate_features.append(x)

        x = self.final_norm(x)  # [B, T, F, D]

        x = einops.rearrange(x, "b t f d -> b (f d) t")  # [B, F*D, downT]
        x = self.to_time_features(x)  # [B, output_dim, downT]

        x = self.up_time(x)  # [B, output_dim, downT*8]

        target_T = context.crop_length
        if x.shape[-1] < target_T:
            x = F.pad(x, (0, target_T - x.shape[-1]))
        else:
            x = x[..., :target_T]

        x = einops.rearrange(x, "b d t -> b t d")  # [B, T, output_dim]

        if return_intermediate:
            return x, intermediate_features

        return x


class BaseTranscriptionModel(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        hidden_size: int,
        num_quality_classes: int,
        num_root_classes: int,
        num_bass_classes: int,
        num_key_classes: int,
        num_tempo_classes: Optional[int] = None,
        dropout_probability: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_quality_classes = num_quality_classes
        self.num_root_classes = num_root_classes

        self.norm = RMSNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_probability) if dropout_probability > 0.0 else nn.Identity()

        self.num_root_quality_classes = (num_quality_classes - 1) * 12 + 1
        self.num_bass_classes = num_bass_classes

        # Main Branch Heads (backbone特徴量から予測)
        chord25_hidden = 25 + 12
        self.chord25_head = nn.Sequential(nn.Linear(hidden_size, chord25_hidden))
        self.boundary_head = nn.Linear(hidden_size, 1)
        self.key_boundary_head = nn.Linear(hidden_size, 1)
        self.key_head = nn.Linear(hidden_size, num_key_classes)
        self.bass_head = nn.Linear(hidden_size, num_bass_classes)

        if num_tempo_classes is None:
            self.tempo_head = nn.Linear(hidden_size, 1)
            self.tempo_is_regression = True
        else:
            self.tempo_head = nn.Linear(hidden_size, num_tempo_classes)
            self.tempo_is_regression = False

        # --- Refiner Branch ---
        smooth_hidden = 256
        self.smooth_stage1 = nn.Sequential(
            nn.Linear(chord25_hidden, smooth_hidden),
            RMSNorm(smooth_hidden),
            nn.Tanh(),  # stage_1_activation
        )

        self.smooth_rnn = nn.LSTM(smooth_hidden, smooth_hidden // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.smooth_net_proj = nn.Sequential(
            nn.Linear(smooth_hidden * 2, smooth_hidden),
            RMSNorm(smooth_hidden),
            nn.ReLU(inplace=True),
        )

        self.smooth_head = nn.Sequential(nn.Linear(smooth_hidden, chord25_hidden))
        self.root_chord_head = nn.Linear(chord25_hidden, self.num_root_quality_classes)

    def _compute_heads_and_answer(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        checkpoint_fn = (
            torch.utils.checkpoint.checkpoint if self.training and torch.is_grad_enabled() else checkpoint_bypass
        )
        normed = self.norm(features)  # RMSNorm or Identity
        features = self.dropout(normed)

        # Primary heads
        chord25_logits = self.chord25_head(features)
        chord25_logits = torch.clamp(torch.relu(chord25_logits), max=1.0)

        outputs = {
            "initial_chord25_logits": chord25_logits[..., :25],
            "initial_chord25_original": chord25_logits,
            "initial_key_logits": self.key_head(features),
            "initial_tempo": self.tempo_head(features),
            "initial_boundary_logits": self.boundary_head(features),
            "initial_key_boundary_logits": self.key_boundary_head(features),
            "initial_bass_logits": self.bass_head(features),
        }

        # Smooth Branch
        # chord25_logits: (B, T, 25)

        # 1. Stage 1 (Dense Tanh)
        # Linear: (B, T, 25) -> (B, T, 256)
        x = self.smooth_stage1(chord25_logits)

        # 2. RNN
        skip = x
        self.smooth_rnn.flatten_parameters()
        x, _ = checkpoint_fn(self.smooth_rnn, x, use_reentrant=True)  # (B, T, 256)
        x = torch.concat([skip, x], dim=-1)

        # 3. Stage 2 (Dense ReLU)
        # Linear: (B, T, 256) -> (B, T, 256)
        x = self.smooth_net_proj(x)

        # 4. Smooth Head (Dense BN ReLU 25)
        # Linear: (B, T, 256) -> (B, T, 25)
        smooth_features = self.smooth_head(x)
        smooth_features = torch.clamp(torch.relu(smooth_features), max=1.0)

        # Root Chord from Smooth Features
        outputs["initial_root_chord_logits"] = self.root_chord_head(smooth_features)

        # Smooth Chord25 Logits (for loss calculation)
        outputs["initial_smooth_chord25_logits"] = smooth_features[..., :25]
        outputs["initial_smooth_chord25_original"] = smooth_features

        return outputs

    def forward(
        self, waveform: torch.Tensor, global_step: Optional[int] = None, max_segments: int = 256
    ) -> Dict[str, torch.Tensor]:
        # Backbone Feature Extraction
        features = self.backbone(waveform)

        # Heads & Segment Logic
        outputs = self._compute_heads_and_answer(features)
        return outputs


class MusicStructureTranscriptionModel(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        hidden_size: int,
        num_structure_classes: int,
        dropout_probability: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_structure_classes = num_structure_classes

        self.norm = RMSNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_probability) if dropout_probability > 0.0 else nn.Identity()

        # 中間特徴量の処理用
        # Backboneの内部チャネル数: backbone.hidden_size * 4
        # 周波数ビン数: bins_per_octave (default 36)
        freq_bins = backbone.oct_frontend.bins_per_octave
        inter_channels = backbone.hidden_size * 4 * freq_bins
        num_backbone_layers = len(backbone.layers)

        self.inter_projections = nn.ModuleList()
        for _ in range(num_backbone_layers):
            self.inter_projections.append(
                nn.Sequential(
                    nn.Linear(inter_channels, hidden_size),
                    nn.GELU(),
                )
            )

        # 時間アップサンプリング (Backboneと同じ8倍)
        self.up_sample = nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=8, stride=8)

        # LSTM
        # 入力次元 = (Backbone最終出力) + (中間層数 * 中間層出力)
        lstm_input_dim = hidden_size + (num_backbone_layers * hidden_size)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True
        )

        # Heads (LSTM is bidirectional -> hidden_size * 2)
        self.structure_function_head = nn.Linear(hidden_size * 2, num_structure_classes)
        self.structure_boundary_head = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        waveform: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        # Backbone Feature Extraction with intermediates
        # final_features: (B, T, D)
        # inter_features: List[(B, T_down, F, C)]
        final_features, inter_features = self.backbone(waveform, return_intermediate=True)

        normed = self.norm(final_features)
        final_features = self.dropout(normed)

        # 中間特徴量の処理
        processed_inters = []
        target_T = final_features.shape[1]

        for feat, proj in zip(inter_features, self.inter_projections):
            # feat: (B, T_down, F, C)
            # 周波数方向をフラット化 -> (B, T_down, F*C)
            feat = einops.rearrange(feat, "b t f c -> b t (f c)")

            # 線形射影 -> (B, T_down, H)
            feat = proj(feat)

            # アップサンプリングのために転置 -> (B, H, T_down)
            feat = einops.rearrange(feat, "b t h -> b h t")

            # Upsample -> (B, H, T_up)
            feat = self.up_sample(feat)

            # クロップまたはパディングしてfinal_featuresに合わせる
            if feat.shape[-1] < target_T:
                feat = F.pad(feat, (0, target_T - feat.shape[-1]))
            else:
                feat = feat[..., :target_T]

            # (B, T, H)に戻す
            feat = einops.rearrange(feat, "b h t -> b t h")
            processed_inters.append(feat)

        # 全特徴量を結合 -> (B, T, D + N*H)
        concat_features = torch.cat([final_features] + processed_inters, dim=-1)

        # LSTM処理
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(concat_features)  # (B, T, 2*H)

        outputs = {
            "initial_structure_function_logits": self.structure_function_head(lstm_out),
            "initial_structure_boundary_logits": self.structure_boundary_head(lstm_out),
        }
        return outputs
