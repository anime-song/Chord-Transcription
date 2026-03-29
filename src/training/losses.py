import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List, Optional, Tuple


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: Union[List[int], torch.Tensor], tau: float = 1.0):
        """
        Args:
            class_counts (Union[List[int], torch.Tensor]):
                各クラスの出現回数のリストまたはテンソル。
                事前に Laplace 平滑化（全カウントに+1するなど）を推奨します。
            tau (float, optional): 補正のスケール係数. Defaults to 1.0.
        """
        super().__init__()

        class_counts = torch.tensor(class_counts, dtype=torch.float32)

        # log_prior を計算し、バッファとして登録
        # カウントが0のクラスは-infになるのを防ぐため、非常に小さい値にクリップ
        log_prior = torch.log(torch.clamp(class_counts, min=1e-9))

        self.register_buffer("log_prior", log_prior)
        self.tau = tau

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): モデルの出力ロジット (B, T, C)
            labels (torch.Tensor): 正解ラベル (B, T, 1)

        Returns:
            torch.Tensor: 計算された損失値 (スカラー)
        """
        # 形状を合わせる
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))  # (B*T, C)
            labels = labels.reshape(-1)  # (B*T,)

        # ロジット補正: z_k <- z_k + τ * log(n_k)
        adjusted_logits = logits + self.tau * self.log_prior

        loss = F.cross_entropy(adjusted_logits, labels)
        return loss


# https://github.com/CPJKU/beat_this/blob/main/beat_this/model/loss.py
class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss variant for sequence labeling that tolerates small shifts between
    predictions and targets. This is accomplished by max-pooling the
    predictions with a given tolerance and a stride of 1, so the gradient for a
    positive label affects the largest prediction in a window around it.
    Expects predictions to be given as logits, and accepts an optional mask
    with zeros indicating the entries to ignore. Note that the edges of the
    sequence will not receive a gradient, as it is assumed to be unknown
    whether there is a nearby positive annotation.

    Args:
        pos_weight (float): Weight for positive examples compared to negative
            examples (default: 1)
        tolerance (int): Tolerated shift in time steps in each direction
            (default: 1)
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 1):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # spread preds and crop targets to match
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # ignore around the positive targets
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:  # consider padding and no-downbeat mask
            look_at = look_at * self.crop(mask, factor=2)
        # compute loss
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )


def tversky_index(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Tversky Indexを計算するヘルパー関数。

    Args:
        y_true (torch.Tensor): 正解ラベル (0 or 1)
        y_pred (torch.Tensor): 予測確率 (0 to 1)
        alpha (float): FPとFNの重み付けを調整するパラメータ
        smooth (float): ゼロ除算を防ぐための平滑化係数
    """
    # バッチと時間次元をフラット化
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # True Positives, False Positives & False Negatives
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    return (tp + smooth) / (tp + alpha * fp + (1 - alpha) * fn + smooth)


def focal_tversky_loss(
    y_true: torch.Tensor, y_pred_logits: torch.Tensor, alpha: float = 0.3, gamma: float = 1.5, smooth: float = 1e-6
) -> torch.Tensor:
    """
    PyTorch版 Focal Tversky Loss。

    Args:
        y_true (torch.Tensor): 正解ラベル (B, T, 25)
        y_pred_logits (torch.Tensor): モデルからの生の出力 (B, T, 25)
        alpha (float): Tversky Indexのalpha
        gamma (float): フォーカス係数。1に近いほど損失が大きくなる。
    """
    ti = tversky_index(y_true, y_pred_logits, alpha=alpha, smooth=smooth)
    return torch.pow((1 - ti), gamma)


def accuracy_ignore_index_with_logits(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """ignore_indexを除外して分類精度を計算します。"""
    with torch.no_grad():
        predicted = logits.argmax(dim=-1)
        valid_mask = target != ignore_index
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=logits.device)
        correct = (predicted[valid_mask] == target[valid_mask]).sum()
        return correct.float() / valid_count.float()


def topk_accuracy_ignore_index_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    *,
    k: int,
) -> torch.Tensor:
    """ignore_indexを除外して top-k 精度を計算します。"""
    with torch.no_grad():
        valid_mask = target != ignore_index
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=logits.device)

        filtered_logits = logits[valid_mask]
        filtered_target = target[valid_mask]
        effective_k = min(max(int(k), 1), filtered_logits.shape[-1])
        topk_indices = filtered_logits.topk(k=effective_k, dim=-1).indices
        correct = (topk_indices == filtered_target.unsqueeze(-1)).any(dim=-1).sum()
        return correct.float() / valid_count.float()


def accuracy_ignore_index(pred: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """ignore_indexを除外して分類精度を計算します。"""
    with torch.no_grad():
        valid_mask = target != ignore_index
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device)
        correct = (pred[valid_mask] == target[valid_mask]).sum()
        return correct.float() / valid_count.float()


def _filter_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = labels != ignore_index

    if not valid_mask.any():
        empty_logits = logits.new_zeros((0, logits.shape[-1]))
        empty_labels = labels.new_zeros((0,), dtype=labels.dtype)
        return empty_logits, empty_labels

    return logits[valid_mask], labels[valid_mask]


def _resolve_repeat_sample_mask(
    batch: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    sample_mask = batch.get("repeat_loss_mask")
    if sample_mask is None:
        return torch.ones((batch_size,), dtype=torch.bool, device=device)
    return sample_mask.to(device=device, dtype=torch.bool)


def _pool_probabilities_by_segments(
    probabilities: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    """フレーム確率を GT セグメント単位へ平均プーリングする。"""
    if segment_count <= 0:
        return probabilities.new_zeros((0, probabilities.shape[-1]))

    pooled_segments = []
    for segment_idx in range(segment_count):
        start = int(segment_starts[segment_idx].item())
        end = int(segment_ends[segment_idx].item())
        if end <= start:
            pooled_segments.append(probabilities.new_zeros((probabilities.shape[-1],)))
            continue
        pooled_segments.append(probabilities[start:end].mean(dim=0))
    return torch.stack(pooled_segments, dim=0)


def _js_divergence_from_probabilities(
    probabilities_p: torch.Tensor,
    probabilities_q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """確率分布どうしの Jensen-Shannon divergence を要素ごとに返す。"""
    p = probabilities_p.clamp_min(eps)
    q = probabilities_q.clamp_min(eps)
    m = (0.5 * (p + q)).clamp_min(eps)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _compute_repeat_prediction_stats(
    logits: torch.Tensor,
    repeat_ssm_output,
    sample_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    repeat 候補区間どうしで予測分布がどれだけ近いかを集計する。

    `ChordWindowSSM` 側で作った sparse pair list を使い、
    まず GT セグメント単位に pooled してから aligned segment ごとに比較する。
    """
    probabilities = F.softmax(logits, dim=-1)
    batch_size = logits.shape[0]
    device = logits.device

    total_pair_count = logits.new_zeros(())
    total_aligned_segments = logits.new_zeros(())
    total_probability_cosine = logits.new_zeros(())
    total_argmax_match = logits.new_zeros(())
    total_audio_similarity = logits.new_zeros(())
    total_span_ratio = logits.new_zeros(())
    loss_terms = []
    if sample_mask is None:
        sample_mask = torch.ones((batch_size,), dtype=torch.bool, device=device)
    valid_batch_count = sample_mask.to(dtype=logits.dtype).sum().clamp_min(1.0)

    for batch_idx in range(batch_size):
        if not bool(sample_mask[batch_idx].item()):
            continue
        segment_count = int(repeat_ssm_output.segment_counts[batch_idx].item())
        pair_count = int(repeat_ssm_output.repeat_pair_counts[batch_idx].item())
        if segment_count == 0 or pair_count == 0:
            continue

        pooled = _pool_probabilities_by_segments(
            probabilities=probabilities[batch_idx],
            segment_starts=repeat_ssm_output.segment_starts[batch_idx],
            segment_ends=repeat_ssm_output.segment_ends[batch_idx],
            segment_count=segment_count,
        )
        pair_starts = repeat_ssm_output.repeat_pairs_segment_starts[batch_idx, :pair_count]
        pair_lengths = repeat_ssm_output.repeat_pairs_lengths[batch_idx, :pair_count]
        pair_similarity = getattr(repeat_ssm_output, "repeat_pairs_similarity", None)
        if pair_similarity is not None:
            pair_similarity = pair_similarity[batch_idx, :pair_count]
        pair_span_ratio = repeat_ssm_output.repeat_pairs_span_ratio[batch_idx, :pair_count]

        total_pair_count = total_pair_count + pair_count
        if pair_similarity is not None:
            total_audio_similarity = total_audio_similarity + pair_similarity.sum()
        total_span_ratio = total_span_ratio + pair_span_ratio.sum()

        for pair_idx in range(pair_count):
            start_i = int(pair_starts[pair_idx, 0].item())
            start_j = int(pair_starts[pair_idx, 1].item())
            length = int(pair_lengths[pair_idx].item())
            if length <= 0:
                continue

            seq_i = pooled[start_i : start_i + length]
            seq_j = pooled[start_j : start_j + length]
            if seq_i.shape[0] != length or seq_j.shape[0] != length:
                continue

            cosine = F.cosine_similarity(seq_i, seq_j, dim=-1, eps=1e-8)
            js_divergence = _js_divergence_from_probabilities(seq_i, seq_j)
            argmax_match = (seq_i.argmax(dim=-1) == seq_j.argmax(dim=-1)).to(probabilities.dtype)

            total_aligned_segments = total_aligned_segments + length
            total_probability_cosine = total_probability_cosine + cosine.sum()
            total_argmax_match = total_argmax_match + argmax_match.sum()
            loss_terms.append(js_divergence)

    if loss_terms:
        consistency_loss = torch.cat(loss_terms, dim=0).mean()
    else:
        consistency_loss = logits.sum() * 0.0

    safe_pair_count = total_pair_count.clamp_min(1.0)
    safe_segment_count = total_aligned_segments.clamp_min(1.0)

    return {
        "loss": consistency_loss,
        "pair_count": total_pair_count / valid_batch_count,
        "aligned_segments": total_aligned_segments / valid_batch_count,
        "probability_cosine": total_probability_cosine / safe_segment_count,
        "argmax_match": total_argmax_match / safe_segment_count,
        "audio_similarity": total_audio_similarity / safe_pair_count,
        "span_ratio": total_span_ratio / safe_pair_count,
    }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    root_chord_loss_fn: Optional[nn.Module] = None,
) -> Dict[str, torch.Tensor]:
    """モデル出力とバッチから、各タスクの損失を計算して辞書で返します。"""
    ce_ignore_index = loss_cfg.get("ce_ignore_index", -100)
    key_label_smoothing = float(loss_cfg.get("key_label_smoothing", 0.0))
    if not 0.0 <= key_label_smoothing < 1.0:
        raise ValueError("loss.key_label_smoothing must be in [0.0, 1.0).")
    weights = loss_cfg.get("weights", {})
    all_losses = {}

    # 分類タスクの損失 (CrossEntropy)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)
    key_ce_loss_fn = nn.CrossEntropyLoss(
        ignore_index=ce_ignore_index,
        label_smoothing=key_label_smoothing,
    )
    ftl_cfg = loss_cfg.get("focal_tversky", {})
    stage = "initial"

    # Root-chord
    if f"{stage}_root_chord_logits" in outputs:
        x = outputs[f"{stage}_root_chord_logits"]
        if root_chord_loss_fn is not None:
            filtered_x, filtered_y = _filter_logits_and_labels(
                x,
                batch["root_chord_index"],
                ce_ignore_index,
            )
            if filtered_y.numel() == 0:
                loss = x.sum() * 0.0
            else:
                loss = root_chord_loss_fn(filtered_x, filtered_y)
        else:
            loss = ce_loss_fn(x.transpose(1, 2), batch["root_chord_index"])
        all_losses[f"{stage}/root_chord"] = loss * weights.get("root_chord", 1.0)

    # Bass
    if f"{stage}_bass_logits" in outputs:
        x = outputs[f"{stage}_bass_logits"]
        loss = ce_loss_fn(x.transpose(1, 2), batch["bass_index"])
        all_losses[f"{stage}/bass"] = loss * weights.get("bass", 1.0)

    # Key
    if f"{stage}_key_logits" in outputs:
        x = outputs[f"{stage}_key_logits"]
        loss = key_ce_loss_fn(x.transpose(1, 2), batch["key_index"])
        all_losses[f"{stage}/key"] = loss * weights.get("key", 1.0)

    # --- 境界検出タスク ---
    if f"{stage}_boundary_logits" in outputs:
        # 境界検出タスクの損失 (ShiftTolerantBCELoss)
        st_bce_loss_fn = ShiftTolerantBCELoss(
            pos_weight=loss_cfg.get("boundary_pos_weight", 1.0), tolerance=loss_cfg.get("boundary_tolerance", 1)
        )

        loss = st_bce_loss_fn(
            preds=outputs[f"{stage}_boundary_logits"][..., 0],
            targets=batch["boundary"][..., 0],
        )
        all_losses[f"{stage}/boundary"] = loss * weights.get("boundary", 1.0)

    # --- キーバウンダリ検出タスク ---
    if f"{stage}_key_boundary_logits" in outputs and "key_boundary" in batch:
        key_bce_loss_fn = ShiftTolerantBCELoss(
            pos_weight=loss_cfg.get("key_boundary_pos_weight", 30.0),
            tolerance=loss_cfg.get("key_boundary_tolerance", 1),
        )
        loss = key_bce_loss_fn(
            preds=outputs[f"{stage}_key_boundary_logits"][..., 0],
            targets=batch["key_boundary"][..., 0],
        )
        all_losses[f"{stage}/key_boundary"] = loss * weights.get("key_boundary", 1.0)

    # --- Beat 検出タスク ---
    if f"{stage}_beat_logits" in outputs and "beat" in batch:
        beat_bce_loss_fn = ShiftTolerantBCELoss(
            pos_weight=loss_cfg.get("beat_pos_weight", 5.0),
            tolerance=loss_cfg.get("beat_tolerance", 1),
        )
        loss = beat_bce_loss_fn(
            preds=outputs[f"{stage}_beat_logits"][..., 0],
            targets=batch["beat"][..., 0],
        )
        all_losses[f"{stage}/beat"] = loss * weights.get("beat", 1.0)

    # --- Downbeat 検出タスク ---
    if f"{stage}_downbeat_logits" in outputs and "downbeat" in batch:
        downbeat_bce_loss_fn = ShiftTolerantBCELoss(
            pos_weight=loss_cfg.get("downbeat_pos_weight", 20.0),
            tolerance=loss_cfg.get("downbeat_tolerance", 1),
        )
        loss = downbeat_bce_loss_fn(
            preds=outputs[f"{stage}_downbeat_logits"][..., 0],
            targets=batch["downbeat"][..., 0],
        )
        all_losses[f"{stage}/downbeat"] = loss * weights.get("downbeat", 1.0)

    # --- Chord25 タスク (Standard) ---
    if f"{stage}_chord25_logits" in outputs:
        loss = focal_tversky_loss(
            y_true=batch["chord25"],
            y_pred_logits=outputs[f"{stage}_chord25_logits"],
            alpha=ftl_cfg.get("alpha", 0.3),
            gamma=ftl_cfg.get("gamma", 1.5),
        )
        all_losses[f"{stage}/chord25"] = loss * weights.get("chord25", 1.0)

    # --- BiLSTM 深層監視 (Tversky) ---
    # bilstm_intermediates: List[(B, T, chord_dim)] 各イテレーション後の出力
    if "bilstm_intermediates" in outputs and "chord25" in batch:
        bilstm_weight = weights.get("bilstm_chord25", 0.5)
        for i, intermediate in enumerate(outputs["bilstm_intermediates"]):
            loss = focal_tversky_loss(
                y_true=batch["chord25"],
                y_pred_logits=intermediate[..., :25],
                alpha=ftl_cfg.get("alpha", 0.3),
                gamma=ftl_cfg.get("gamma", 1.5),
            )
            all_losses[f"bilstm/chord25_iter{i}"] = loss * bilstm_weight

    repeat_ssm_output = outputs.get("repeat_ssm_output")
    if repeat_ssm_output is not None:
        repeat_sample_mask = _resolve_repeat_sample_mask(
            batch,
            batch_size=repeat_ssm_output.repeat_pair_counts.shape[0],
            device=repeat_ssm_output.repeat_pair_counts.device,
        )
        if f"{stage}_root_chord_logits" in outputs:
            repeat_root_stats = _compute_repeat_prediction_stats(
                outputs[f"{stage}_root_chord_logits"],
                repeat_ssm_output,
                sample_mask=repeat_sample_mask,
            )
            all_losses[f"{stage}/repeat_root_consistency"] = repeat_root_stats["loss"] * weights.get(
                "repeat_root_consistency", 0.0
            )
        if f"{stage}_bass_logits" in outputs:
            repeat_bass_stats = _compute_repeat_prediction_stats(
                outputs[f"{stage}_bass_logits"],
                repeat_ssm_output,
                sample_mask=repeat_sample_mask,
            )
            all_losses[f"{stage}/repeat_bass_consistency"] = repeat_bass_stats["loss"] * weights.get(
                "repeat_bass_consistency", 0.0
            )

    return all_losses


def _try_resolve_head(
    outputs: Dict[str, torch.Tensor], head: str, *, prefer: Tuple[str, ...] = ("final", "initial")
) -> Optional[torch.Tensor]:
    for stage in prefer:
        key = f"{stage}_{head}"
        if key in outputs:
            return outputs[key]
    return None


def compute_metrics(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ce_ignore_index: int
) -> Dict[str, torch.Tensor]:
    """
    モデル出力とバッチから、各タスクのメトリクス（精度など）を計算します。
    """
    metrics = {}

    # Root Chord
    logits = _try_resolve_head(outputs, "root_chord_logits")
    if logits is not None and "root_chord_index" in batch:
        metrics["acc/root_chord"] = accuracy_ignore_index_with_logits(
            logits, batch["root_chord_index"], ce_ignore_index
        )
        metrics["acc/root_chord_top3"] = topk_accuracy_ignore_index_with_logits(
            logits,
            batch["root_chord_index"],
            ce_ignore_index,
            k=3,
        )

    # Bass
    logits = _try_resolve_head(outputs, "bass_logits")
    if logits is not None and "bass_index" in batch:
        metrics["acc/bass"] = accuracy_ignore_index_with_logits(logits, batch["bass_index"], ce_ignore_index)

    # Key
    logits = _try_resolve_head(outputs, "key_logits")
    if logits is not None and "key_index" in batch:
        metrics["acc/key"] = accuracy_ignore_index_with_logits(logits, batch["key_index"], ce_ignore_index)

    repeat_ssm_output = outputs.get("repeat_ssm_output")
    if repeat_ssm_output is not None:
        repeat_sample_mask = _resolve_repeat_sample_mask(
            batch,
            batch_size=repeat_ssm_output.repeat_pair_counts.shape[0],
            device=repeat_ssm_output.repeat_pair_counts.device,
        )
        valid_sample_count = repeat_sample_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
        metrics["repeat/pair_count"] = (
            repeat_ssm_output.repeat_pair_counts[repeat_sample_mask].to(dtype=torch.float32).sum() / valid_sample_count
        )
        pair_similarity = getattr(repeat_ssm_output, "repeat_pairs_similarity", None)
        max_pairs = repeat_ssm_output.repeat_pairs_lengths.shape[1]
        pair_indices = torch.arange(max_pairs, device=repeat_ssm_output.repeat_pair_counts.device)
        valid_pair_mask = pair_indices.unsqueeze(0) < repeat_ssm_output.repeat_pair_counts.unsqueeze(1)
        valid_pair_mask &= repeat_sample_mask.unsqueeze(1)
        if bool(valid_pair_mask.any().item()):
            if pair_similarity is not None:
                metrics["repeat/audio_similarity"] = pair_similarity[valid_pair_mask].mean()
            metrics["repeat/span_ratio"] = repeat_ssm_output.repeat_pairs_span_ratio[valid_pair_mask].mean()
        else:
            zero = repeat_ssm_output.repeat_pairs_span_ratio.new_zeros(())
            if pair_similarity is not None:
                metrics["repeat/audio_similarity"] = zero
            metrics["repeat/span_ratio"] = zero

        logits = _try_resolve_head(outputs, "root_chord_logits")
        if logits is not None:
            root_stats = _compute_repeat_prediction_stats(logits, repeat_ssm_output, sample_mask=repeat_sample_mask)
            metrics["repeat/root_prob_cosine"] = root_stats["probability_cosine"]
            metrics["repeat/root_argmax_match"] = root_stats["argmax_match"]
            metrics["repeat/aligned_segments"] = root_stats["aligned_segments"]

        logits = _try_resolve_head(outputs, "bass_logits")
        if logits is not None:
            bass_stats = _compute_repeat_prediction_stats(logits, repeat_ssm_output, sample_mask=repeat_sample_mask)
            metrics["repeat/bass_prob_cosine"] = bass_stats["probability_cosine"]
            metrics["repeat/bass_argmax_match"] = bass_stats["argmax_match"]

    return metrics
