import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List, Optional, Tuple

from ..chord_transcription.metrics import segment_macro_f1_score
from ..chord_transcription.models.semi_crf import (
    NeuralSemiCRFInterval,
    _build_interval_score,
    _build_length_scale,
    _zero_noise_score,
    _sanitize_interval_batch,
)


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
    クラスごとの Tversky Index を計算し、その平均を返す。

    Args:
        y_true (torch.Tensor): 正解ラベル (B, T, C) — 0 or 1
        y_pred (torch.Tensor): 予測確率 (B, T, C) — 0 to 1
        alpha (float): FPとFNの重み付けを調整するパラメータ
        smooth (float): ゼロ除算を防ぐための平滑化係数

    Returns:
        torch.Tensor: クラス平均の Tversky Index (スカラー)
    """
    # クラスごとに TP / FP / FN を集計 → [C]
    tp = (y_true * y_pred).sum(dim=(0, 1))
    fp = ((1 - y_true) * y_pred).sum(dim=(0, 1))
    fn = (y_true * (1 - y_pred)).sum(dim=(0, 1))

    per_class_ti = (tp + smooth) / (tp + alpha * fp + (1 - alpha) * fn + smooth)
    return per_class_ti.mean()


def focal_tversky_loss(
    y_true: torch.Tensor, y_pred_logits: torch.Tensor, alpha: float = 0.3, gamma: float = 1.5, smooth: float = 1e-6
) -> torch.Tensor:
    """
    PyTorch版 Focal Tversky Loss（クラスごとに計算）。

    Args:
        y_true (torch.Tensor): 正解ラベル (B, T, 12)
        y_pred_logits (torch.Tensor): モデルからの生の出力 (B, T, 12)
        alpha (float): Tversky Indexのalpha
        gamma (float): フォーカス係数。1に近いほど損失が大きくなる。
    """
    y_pred = torch.sigmoid(y_pred_logits)
    ti = tversky_index(y_true, y_pred, alpha=alpha, smooth=smooth)
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


def extract_chord_intervals_from_boundary(
    boundary: torch.Tensor,
    threshold: float = 0.8,
) -> List[List[Tuple[int, int]]]:
    """
    Boundary ラベルからコード区間（開始/終了のペア）を抽出する。

    boundary の 1.0 はコード変化点（新しいコードの先頭フレーム）を意味する。
    連続する同一コード区間を closed interval [begin, end] として返す。

    Args:
        boundary: [B, T, 1] — 0.0/0.5/1.0 の soft boundary ラベル
        threshold: 境界とみなす閾値（0.8 なら 1.0 のフレームのみ抽出）

    Returns:
        batch ごとの interval リスト: [[(begin, end), ...], ...]
    """
    if boundary.dim() == 3:
        boundary = boundary.squeeze(-1)  # [B, T]

    batch_size, time_steps = boundary.shape
    result: List[List[Tuple[int, int]]] = []

    for batch_idx in range(batch_size):
        boundary_sample = boundary[batch_idx]  # [T]
        # 閾値以上のフレームを境界（新コードの先頭）とみなす
        change_points = (boundary_sample >= threshold).nonzero(as_tuple=False).squeeze(-1)

        intervals: List[Tuple[int, int]] = []
        if change_points.numel() == 0:
            # 境界がない = 全体が1区間
            if time_steps > 0:
                intervals.append((0, time_steps - 1))
        else:
            change_list = change_points.tolist()
            # 最初の境界より前の区間
            if change_list[0] > 0:
                intervals.append((0, change_list[0] - 1))
            # 各境界間の区間
            for i in range(len(change_list)):
                begin = change_list[i]
                end = change_list[i + 1] - 1 if i + 1 < len(change_list) else time_steps - 1
                if end >= begin:
                    intervals.append((begin, end))

        result.append(intervals)

    return result


def compute_segment_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gt_intervals: List[List[Tuple[int, int]]],
    loss_fn: nn.Module,
    ignore_index: int,
) -> torch.Tensor:
    """
    GT区間でロジットをプーリング（平均）し、区間中央のラベルに対してロスを計算する。
    推論時の CRF pool デコードと一貫性を持たせる働きをします。

    Args:
        logits: (B, T, C)
        labels: (B, T)
        gt_intervals: extract_chord_intervals_from_boundary で抽出した区間リスト
        loss_fn: 適用するロス関数 (CrossEntropyLoss, BalancedSoftmaxLoss など)
        ignore_index: 無視するラベルのインデックス
    """
    batch_size = logits.shape[0]
    pooled_logits_list = []
    target_labels_list = []

    for batch_idx in range(batch_size):
        intervals = gt_intervals[batch_idx]
        for start, end in intervals:
            # 区間内のロジットをプーリング（平均）
            pooled = logits[batch_idx, start : end + 1].mean(dim=0)

            # 区間中央のフレームのラベルを取得
            mid = (start + end) // 2
            target = labels[batch_idx, mid]

            pooled_logits_list.append(pooled)
            target_labels_list.append(target)

    if not pooled_logits_list:
        return logits.sum() * 0.0

    pooled_logits = torch.stack(pooled_logits_list)  # (N, C)
    target_labels = torch.stack(target_labels_list)  # (N,)

    # ignore_index のフィルタリング
    valid_mask = target_labels != ignore_index
    if not valid_mask.any():
        return logits.sum() * 0.0

    filtered_logits = pooled_logits[valid_mask]
    filtered_labels = target_labels[valid_mask]

    loss = loss_fn(filtered_logits, filtered_labels)
    return loss


def compute_chord_interval_loss(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    boundary: torch.Tensor,
    length_scaling: str = "sqrt",
    boundary_threshold: float = 0.8,
) -> Tuple[torch.Tensor, int]:
    """
    Semi-CRF によるコード区間の NLL loss を計算する。

    Args:
        interval_query: [B, T, D]
        interval_key: [B, T, D]
        interval_diag: [B, T]
        boundary: [B, T, 1] — GT boundary ラベル
        length_scaling: interval score のスケーリング方式
        boundary_threshold: 境界判定の閾値

    Returns:
        (loss, num_intervals): NLL loss と GT 区間の総数
    """
    # GT コード区間を抽出
    gt_intervals = extract_chord_intervals_from_boundary(
        boundary,
        threshold=boundary_threshold,
    )

    batch_size, time_steps, _ = interval_query.shape
    total_log_prob = interval_query.new_zeros(())
    total_samples = 0

    for batch_idx in range(batch_size):
        intervals = gt_intervals[batch_idx]
        if not intervals:
            continue

        # 1サンプル分のスコアテンソルを構築
        query_single = interval_query[batch_idx].unsqueeze(1)  # [T, 1, D]
        key_single = interval_key[batch_idx].unsqueeze(1)  # [T, 1, D]
        diag_single = interval_diag[batch_idx].unsqueeze(1)  # [T, 1]

        score = _build_interval_score(
            query_single,
            key_single,
            diag_single,
            length_scaling=length_scaling,
        )

        noise_score = _zero_noise_score(
            time_steps,
            batch_size=1,
            device=score.device,
        )

        sanitized = _sanitize_interval_batch([intervals], length=time_steps)
        semi_crf = NeuralSemiCRFInterval(score, noise_score)
        log_prob = semi_crf.logProb(sanitized)
        total_log_prob = total_log_prob + log_prob.sum()
        total_samples += 1

    if total_samples <= 0:
        return interval_query.sum() * 0.0, 0

    return -total_log_prob / float(total_samples), sum(len(iv) for iv in gt_intervals)


@torch.no_grad()
def decode_chord_intervals(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    *,
    length_scaling: str = "sqrt",
) -> List[List[Tuple[int, int]]]:
    if interval_query.shape != interval_key.shape:
        raise ValueError(
            "interval_query and interval_key must share the same shape, "
            f"got {tuple(interval_query.shape)} vs {tuple(interval_key.shape)}"
        )
    if interval_query.dim() != 3:
        raise ValueError("interval_query must have shape [B, T, D].")
    if interval_diag.shape != interval_query.shape[:2]:
        raise ValueError(
            "interval_diag must have shape [B, T], "
            f"got {tuple(interval_diag.shape)} for interval_query {tuple(interval_query.shape)}"
        )

    batch_size, time_steps, _ = interval_query.shape
    decoded: List[List[Tuple[int, int]]] = []
    for batch_idx in range(batch_size):
        query_single = interval_query[batch_idx].unsqueeze(1)  # [T, 1, D]
        key_single = interval_key[batch_idx].unsqueeze(1)  # [T, 1, D]
        diag_single = interval_diag[batch_idx].unsqueeze(1)  # [T, 1]

        score = _build_interval_score(
            query_single,
            key_single,
            diag_single,
            length_scaling=length_scaling,
        )
        semi_crf = NeuralSemiCRFInterval(
            score,
            _zero_noise_score(
                time_steps,
                batch_size=1,
                device=score.device,
            ),
        )
        decoded_batch = semi_crf.decode()
        decoded.append(decoded_batch[0] if decoded_batch else [])
    return decoded


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    root_chord_loss_fn: Optional[nn.Module] = None,
    *,
    chord_interval_length_scaling: str = "sqrt",
    chord_interval_threshold: float = 0.8,
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

    # --- Pitch Chroma (per-pitch on/off) ---
    if "pitch_chroma_logits" in outputs and "chord25" in batch:
        loss = F.binary_cross_entropy_with_logits(
            outputs["pitch_chroma_logits"],
            batch["chord25"][..., :12],  # [B, T, 12] — ピッチ部分のみ
        )
        all_losses["pitch_chroma"] = loss * weights.get("pitch_chroma", 5.0)

    # --- GT 区間プーリング分類ロス (Segment-level Classification) ---
    segment_weight = weights.get("segment_classifier", 0.0)
    if segment_weight > 0.0 and "boundary" in batch:
        gt_intervals = extract_chord_intervals_from_boundary(batch["boundary"], threshold=chord_interval_threshold)

        # Root-chord segment loss
        if f"{stage}_root_chord_logits" in outputs and "root_chord_index" in batch:
            loss = compute_segment_classification_loss(
                logits=outputs[f"{stage}_root_chord_logits"],
                labels=batch["root_chord_index"],
                gt_intervals=gt_intervals,
                loss_fn=root_chord_loss_fn if root_chord_loss_fn is not None else ce_loss_fn,
                ignore_index=ce_ignore_index,
            )
            all_losses[f"{stage}/segment_root_chord"] = loss * segment_weight * weights.get("root_chord", 1.0)

        # Bass segment loss
        if f"{stage}_bass_logits" in outputs and "bass_index" in batch:
            loss = compute_segment_classification_loss(
                logits=outputs[f"{stage}_bass_logits"],
                labels=batch["bass_index"],
                gt_intervals=gt_intervals,
                loss_fn=ce_loss_fn,
                ignore_index=ce_ignore_index,
            )
            all_losses[f"{stage}/segment_bass"] = loss * segment_weight * weights.get("bass", 1.0)

        # Key segment loss
        if f"{stage}_key_logits" in outputs and "key_index" in batch:
            loss = compute_segment_classification_loss(
                logits=outputs[f"{stage}_key_logits"],
                labels=batch["key_index"],
                gt_intervals=gt_intervals,
                loss_fn=key_ce_loss_fn,
                ignore_index=ce_ignore_index,
            )
            all_losses[f"{stage}/segment_key"] = loss * segment_weight * weights.get("key", 1.0)

    # --- Chord Interval (Semi-CRF) ---
    if "interval_query" in outputs and "boundary" in batch:
        chord_interval_loss, num_intervals = compute_chord_interval_loss(
            interval_query=outputs["interval_query"],
            interval_key=outputs["interval_key"],
            interval_diag=outputs["interval_diag"],
            boundary=batch["boundary"],
            length_scaling=chord_interval_length_scaling,
            boundary_threshold=chord_interval_threshold,
        )
        all_losses["chord_interval/nll"] = chord_interval_loss * weights.get("chord_interval", 1.0)

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
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    ce_ignore_index: int,
    *,
    chord_interval_length_scaling: str = "sqrt",
    chord_interval_iou_threshold: float = 0.5,
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

    if "interval_query" in outputs and "interval_key" in outputs and "interval_diag" in outputs and "boundary" in batch:
        gt_intervals = extract_chord_intervals_from_boundary(batch["boundary"])
        pred_intervals = decode_chord_intervals(
            outputs["interval_query"],
            outputs["interval_key"],
            outputs["interval_diag"],
            length_scaling=chord_interval_length_scaling,
        )
        segment_macro_f1 = segment_macro_f1_score(
            gt_intervals,
            pred_intervals,
            iou_threshold=chord_interval_iou_threshold,
            inclusive_end=True,
        )
        metrics["segment/chord_macro_f1_iou_50"] = outputs["interval_query"].new_tensor(segment_macro_f1)

    return metrics
