import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, Dict, Tuple, List


from .transformer import RMSNorm, Transformer
from .transcription_model import Backbone


def checkpoint_bypass(func, *args, **kwargs):
    """チェックポイントを使用しない場合のバイパス関数"""
    return func(*args)


class SegmentFeatureProcessor:
    """
    セグメントに基づいた特徴量の集約（プーリング）、パディング、復元（ブロードキャスト）を行うクラス。
    計算ロジック（どう計算するか）のみに責務を持つ。
    """

    def aggregate_features(
        self,
        frame_features: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        1サンプル分の特徴量をセグメント単位で平均プーリングする。

        Args:
            frame_features: (TotalFrames, FeatureDim)
            segment_ids: (TotalFrames,) 各フレームの所属セグメントID
            num_segments: セグメント総数

        Returns:
            aggregated_features: (NumSegments, FeatureDim)
        """
        # Tensorの形状とデバイス情報を取得
        total_frames, feature_dim = frame_features.shape
        device = frame_features.device
        dtype = frame_features.dtype

        # 集計用バッファの初期化
        sum_features = torch.zeros((num_segments, feature_dim), device=device, dtype=dtype)
        frame_counts = torch.zeros((num_segments, 1), device=device, dtype=dtype)

        # IDに基づいて特徴量を加算 (Scatter Add)
        sum_features.index_add_(0, segment_ids, frame_features)

        # IDごとのフレーム数をカウント
        ones = torch.ones((total_frames, 1), device=device, dtype=dtype)
        frame_counts.index_add_(0, segment_ids, ones)

        # 平均を計算 (ゼロ除算防止)
        return sum_features / (frame_counts + epsilon)

    def collate_segments(
        self,
        segment_feature_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        可変長のセグメント特徴量リストをパディングしてバッチ化する。

        Returns:
            padded_features: (BatchSize, MaxSegments, FeatureDim)
            padding_mask: (BatchSize, MaxSegments) 有効な場所がTrue
        """
        batch_size = len(segment_feature_list)
        if batch_size == 0:
            return torch.empty(0), torch.empty(0)

        feature_dim = segment_feature_list[0].shape[-1]
        # バッチ内で最大のセグメント数を取得
        max_segments = max(feat.shape[0] for feat in segment_feature_list)

        device = segment_feature_list[0].device
        dtype = segment_feature_list[0].dtype

        padded_features = torch.zeros((batch_size, max_segments, feature_dim), device=device, dtype=dtype)
        padding_mask = torch.zeros((batch_size, max_segments), device=device, dtype=torch.bool)

        for i, features in enumerate(segment_feature_list):
            num_seg = features.shape[0]
            padded_features[i, :num_seg] = features
            padding_mask[i, :num_seg] = True

        return padded_features, padding_mask

    def broadcast_to_frames(
        self,
        padded_segment_features: torch.Tensor,
        segment_ids_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        セグメント特徴量をフレーム単位に引き伸ばす（アップサンプリング）。

        Args:
            padded_segment_features: (BatchSize, MaxSegments, FeatureDim)
            segment_ids_batch: (BatchSize, TotalFrames) フレームごとの所属ID

        Returns:
            frame_features: (BatchSize, TotalFrames, FeatureDim)
        """
        batch_size, total_frames = segment_ids_batch.shape
        feature_dim = padded_segment_features.shape[-1]

        output_frame_features = torch.zeros(
            (batch_size, total_frames, feature_dim),
            device=padded_segment_features.device,
            dtype=padded_segment_features.dtype,
        )

        # バッチごとのループ処理（可読性と安全性を優先）
        # ※ gatherを使ったベクトル化も可能だが、メモリ消費と可読性のトレードオフを考慮しループを採用
        for i in range(batch_size):
            # (TotalFrames,)
            ids = segment_ids_batch[i]
            # IDに対応する特徴量をコピー: (TotalFrames, FeatureDim)
            output_frame_features[i] = padded_segment_features[i].index_select(0, ids)

        return output_frame_features


class BatchBoundarySegmenter:
    """
    バッチデータに対する「境界検出」から「特徴量プーリング」までを一括管理するクラス。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        nms_window_radius: int = 3,
        min_segment_length: int = 8,
        max_segments: Optional[int] = None,
        feature_processor: Optional[SegmentFeatureProcessor] = None,
    ):
        # 境界検出のパラメータ
        self.threshold = threshold
        self.nms_window_radius = nms_window_radius
        self.min_segment_length = min_segment_length
        self.max_segments = max_segments

        # 計算ロジック担当のクラス（DI: Dependency Injection）
        self.processor = feature_processor or SegmentFeatureProcessor()

    def set_max_segments(self, max_segments: int):
        self.max_segments = max_segments

    @torch.no_grad()
    def process_batch(
        self,
        frame_features: torch.Tensor,  # (B, T, D)
        boundary_logits: torch.Tensor,  # (B, T) or (B, T, 1)
        detach_boundary: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]]]:
        """
        バッチ処理の実行メソッド。

        Returns:
            padded_features: (B, S_max, D) セグメントごとの特徴量（パディング済み）
            padding_mask: (B, S_max) マスク
            segment_ids_batch: (B, T) 各フレームのセグメントID
            segments_info_batch: List[List[(start, end)]] 区間情報のリスト
        """
        # 入力形状の正規化
        if boundary_logits.dim() == 3:
            boundary_logits = boundary_logits.squeeze(-1)

        batch_size, total_frames, feature_dim = frame_features.shape

        # 確率値への変換
        probabilities = torch.sigmoid(boundary_logits)
        if detach_boundary:
            probabilities = probabilities.detach()

        # 結果格納用
        segment_features_list: List[torch.Tensor] = []
        segment_ids_list: List[torch.Tensor] = []
        segments_info_batch: List[List[Tuple[int, int]]] = []

        for i in range(batch_size):
            # 1. 境界検出
            boundary_indices = self._detect_boundaries_1d(probabilities[i])

            # 2. セグメント情報への変換
            segments, seg_ids = self._convert_to_segments(boundary_indices, total_frames)

            # デバイス転送
            seg_ids = seg_ids.to(frame_features.device)
            num_segments = len(segments)

            # 3. 特徴量プーリング (Processorに委譲)
            pooled_feat = self.processor.aggregate_features(frame_features[i], seg_ids, num_segments)

            segment_features_list.append(pooled_feat)
            segment_ids_list.append(seg_ids)
            segments_info_batch.append(segments)

        # 4. パディング処理 (Processorに委譲)
        padded_features, padding_mask = self.processor.collate_segments(segment_features_list)

        # 5. IDリストをTensorにスタック (フレーム数固定の前提により可能)
        segment_ids_batch = torch.stack(segment_ids_list)

        return padded_features, padding_mask, segment_ids_batch, segments_info_batch

    def expand_to_frames(self, padded_segment_features: torch.Tensor, segment_ids_batch: torch.Tensor) -> torch.Tensor:
        """
        セグメント特徴量をフレーム単位に復元する。
        実際の処理は internal processor に委譲する。
        """
        return self.processor.broadcast_to_frames(padded_segment_features, segment_ids_batch)

    # --- 内部メソッド: 境界検出ロジック ---

    def _detect_boundaries_1d(self, probabilities: torch.Tensor) -> List[int]:
        """1サンプル分の境界検出（NMS + Greedy）"""
        sequence_length = probabilities.numel()
        if sequence_length <= 1:
            return []

        # NMSによるピーク抽出
        win_size = 2 * self.nms_window_radius + 1
        max_pooled = F.max_pool1d(
            probabilities.view(1, 1, sequence_length), kernel_size=win_size, stride=1, padding=self.nms_window_radius
        ).view(sequence_length)

        is_peak = (probabilities >= max_pooled) & (probabilities > self.threshold)
        candidates = torch.nonzero(is_peak).flatten()
        candidates = candidates[(candidates > 0) & (candidates < sequence_length)]  # 始点終点除外

        if candidates.numel() == 0:
            return []

        # 距離制約によるフィルタリング
        # スコア順にソート
        scores = probabilities[candidates]
        sorted_indices = candidates[torch.argsort(scores, descending=True)].tolist()

        selected: List[int] = []
        for t in sorted_indices:
            if self.max_segments and len(selected) >= (self.max_segments - 1):
                break

            # 制約チェック
            if t < self.min_segment_length:
                continue
            if (sequence_length - t) < self.min_segment_length:
                continue

            is_far_enough = True
            for s in selected:
                if abs(t - s) < self.min_segment_length:
                    is_far_enough = False
                    break

            if is_far_enough:
                selected.append(t)

        selected.sort()
        return selected

    def _convert_to_segments(
        self, boundary_indices: List[int], total_frames: int
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """境界位置リストからセグメント区間とIDマップを生成"""
        cut_points = [0] + boundary_indices + [total_frames]
        segments = []
        for i in range(len(cut_points) - 1):
            s, e = int(cut_points[i]), int(cut_points[i + 1])
            if e > s:
                segments.append((s, e))

        seg_ids = torch.empty((total_frames,), dtype=torch.long)
        for idx, (s, e) in enumerate(segments):
            seg_ids[s:e] = idx

        return segments, seg_ids


class SegmentTranscriptionModel(nn.Module):
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
        transformer_hidden_size: int = 256,
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 3,
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
        self.boundary_head = nn.Linear(hidden_size, 1)

        if num_tempo_classes is None:
            self.tempo_head = nn.Linear(hidden_size, 1)
            self.tempo_is_regression = True
        else:
            self.tempo_head = nn.Linear(hidden_size, num_tempo_classes)
            self.tempo_is_regression = False

        # --- Segment Branch ---
        self.segmenter = BatchBoundarySegmenter(
            threshold=0.5, nms_window_radius=3, min_segment_length=8, max_segments=256
        )
        self.seg_transformer = Transformer(
            input_dim=hidden_size,
            head_dim=transformer_hidden_size // transformer_num_heads,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            ffn_hidden_size_factor=2,
        )
        self.seg_out_proj = nn.Linear(hidden_size, hidden_size)

        self.key_head = nn.Linear(hidden_size, num_key_classes)
        self.smooth_head = nn.Sequential(nn.Linear(hidden_size, 25 + 12))
        self.root_chord_head = nn.Linear(25 + 12, self.num_root_quality_classes)
        self.bass_head = nn.Linear(25 + 12, num_bass_classes)

    def _ste_broadcast(self, padded_features, seg_ids, boundary_logits):
        # 1. 形状調整
        if boundary_logits.dim() == 3:
            boundary_logits = boundary_logits.squeeze(-1)

        max_segments = padded_features.shape[1]
        device = padded_features.device

        # 2. 位置情報の準備
        probs = torch.sigmoid(boundary_logits)
        # pos_raw: 勾配の源 (float)
        pos_raw = torch.cumsum(probs, dim=1)  # (B, T)

        # pos_anchor: 値の固定先 (int -> float)
        pos_anchor = seg_ids.float()  # (B, T)

        # 値は pos_anchor (整数)、勾配は pos_raw (確率) に流す
        pos_aligned = pos_anchor + (pos_raw - pos_raw.detach())

        # 3. Soft重みの計算 (Gaussian / Softmax Kernel)
        # (B, T, 1)
        current_pos = pos_aligned.unsqueeze(-1)
        # (1, 1, S)
        target_idx = torch.arange(max_segments, device=device).view(1, 1, -1)

        # 距離の二乗 (B, T, S)
        dist_sq = (current_pos - target_idx) ** 2

        # sigma相当の係数を掛けて鋭さを調整 (大きいほど鋭い)
        sharpness = 10.0
        weight_soft = F.softmax(-sharpness * dist_sq, dim=-1)

        # 4. Hard重みの計算 (Forward用)
        weight_hard = F.one_hot(seg_ids, num_classes=max_segments).float()

        # 5. 重みレベルでのSTE
        weight_ste = weight_hard + (weight_soft - weight_soft.detach())

        # 6. 復元
        return torch.matmul(weight_ste, padded_features)

    def _compute_heads_and_answer(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        use_checkpoint = self.training and torch.is_grad_enabled()
        checkpoint_fn = torch.utils.checkpoint.checkpoint if use_checkpoint else checkpoint_bypass

        normed = self.norm(features)  # RMSNorm or Identity
        features = self.dropout(normed)

        boundary_logits = self.boundary_head(features)
        outputs = {
            "initial_tempo": self.tempo_head(features),
            "initial_boundary_logits": boundary_logits,
        }

        seg_pad, seg_mask, seg_ids, segments = self.segmenter.process_batch(
            frame_features=features, boundary_logits=boundary_logits, detach_boundary=True
        )

        x = seg_pad * seg_mask.unsqueeze(-1).to(seg_pad.dtype)
        x = checkpoint_fn(self.seg_transformer, x, use_reentrant=False)
        x = self.seg_out_proj(x)

        if self.training and torch.is_grad_enabled():
            frame_ctx = self._ste_broadcast(x, seg_ids, boundary_logits)
        else:
            frame_ctx = self.segmenter.expand_to_frames(x, seg_ids)

        refined_features = features + frame_ctx

        outputs["initial_key_logits"] = self.key_head(refined_features)

        smooth_features = self.smooth_head(refined_features)
        smooth_features = torch.clamp(torch.relu(smooth_features), max=1.0)

        # Root Chord from Smooth Features
        outputs["initial_root_chord_logits"] = self.root_chord_head(smooth_features)
        outputs["initial_bass_logits"] = self.bass_head(smooth_features)

        # Smooth Chord25 Logits (for loss calculation)
        outputs["initial_smooth_chord25_logits"] = smooth_features[..., :25]
        outputs["initial_smooth_chord25_original"] = smooth_features

        return outputs

    def forward(
        self, waveform: torch.Tensor, global_step: Optional[int] = None, max_segments: int = 256
    ) -> Dict[str, torch.Tensor]:
        self.segmenter.set_max_segments(max_segments)

        # Backbone Feature Extraction
        features = self.backbone(waveform)

        # Heads & Segment Logic
        outputs = self._compute_heads_and_answer(features)
        return outputs
