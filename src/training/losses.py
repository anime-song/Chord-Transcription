import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List, Optional, Tuple
from src.models.segment_model import BatchBoundarySegmenter, resolve_segment_decode_params


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


def accuracy_ignore_index(pred: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """ignore_indexを除外して分類精度を計算します。"""
    with torch.no_grad():
        valid_mask = target != ignore_index
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device)
        correct = (pred[valid_mask] == target[valid_mask]).sum()
        return correct.float() / valid_count.float()


def tempo_regression_loss(
    pred_logits: torch.Tensor,
    target_bpm: torch.Tensor,
    ignore_value: float = -100.0,
    scale: str = "log",
) -> torch.Tensor:
    """テンポ（回帰）の損失を計算します。"""
    pred = pred_logits.squeeze(-1)
    valid_mask = target_bpm != ignore_value
    if valid_mask.sum() == 0:
        return pred.sum() * 0.0

    scale = str(scale).lower()
    if scale == "linear":
        true_values = target_bpm[valid_mask]
        pred_values = pred[valid_mask]
        return F.smooth_l1_loss(pred_values, true_values, beta=2.0, reduction="mean")
    if scale == "log":
        # log(BPM)ドメインでSmooth L1 Lossを計算
        true_values = torch.log(torch.clamp(target_bpm[valid_mask], min=1.0))
        pred_values = pred[valid_mask]
        return F.smooth_l1_loss(pred_values, true_values, beta=0.2, reduction="mean")

    raise ValueError(f"Unsupported tempo regression scale: {scale}")


def key_transition_weighted_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    *,
    transition_weight: float = 4.0,
    transition_tolerance_frames: int = 32,
    base_weight: float = 1.0,
    non_key_index: int = 0,
    ignore_non_key_to_n: bool = True,
) -> torch.Tensor:
    """
    キー遷移近傍フレームの重みを上げる加重CE。
    - 遷移は target の隣接フレーム比較で判定
    - 遷移フレームは tolerance_frames ぶん前後へ拡張
    """
    if logits.ndim != 3 or target.ndim != 2:
        raise ValueError(f"Expected logits:(B,T,C), target:(B,T), got {tuple(logits.shape)} and {tuple(target.shape)}")
    if logits.shape[:2] != target.shape:
        raise ValueError(f"logits and target must share (B,T), got {tuple(logits.shape[:2])} and {tuple(target.shape)}")

    batch_size, num_frames, num_classes = logits.shape
    valid_mask = target != ignore_index

    # 境界検出: i-1 -> i でクラスが変わるフレーム i を1とする
    transitions = torch.zeros((batch_size, num_frames), dtype=torch.bool, device=target.device)
    if num_frames >= 2:
        prev_target = target[:, :-1]
        curr_target = target[:, 1:]
        pair_valid = (prev_target != ignore_index) & (curr_target != ignore_index)
        changed = pair_valid & (prev_target != curr_target)
        if ignore_non_key_to_n:
            changed = changed & ~((prev_target != non_key_index) & (curr_target == non_key_index))
        transitions[:, 1:] = changed

    tol = max(0, int(transition_tolerance_frames))
    if tol > 0:
        transitions_f = transitions.float().unsqueeze(1)  # (B,1,T)
        transitions_f = F.max_pool1d(transitions_f, kernel_size=2 * tol + 1, stride=1, padding=tol)
        transitions = transitions_f.squeeze(1) > 0.0

    frame_weights = torch.full_like(target, fill_value=float(base_weight), dtype=logits.dtype)
    frame_weights = torch.where(
        transitions,
        torch.full_like(frame_weights, float(transition_weight)),
        frame_weights,
    )
    frame_weights = frame_weights * valid_mask.to(dtype=logits.dtype)

    flat_logits = logits.reshape(-1, num_classes)
    flat_target = target.reshape(-1)
    per_frame_ce = F.cross_entropy(flat_logits, flat_target, ignore_index=ignore_index, reduction="none").reshape(
        batch_size, num_frames
    )

    denom = frame_weights.sum().clamp_min(1.0)
    return (per_frame_ce * frame_weights).sum() / denom


def build_y_joint(
    y_rq: torch.Tensor,
    y_bs: torch.Tensor,
    *,
    BS: int,
    RQ: int,
    ignore_index: int,
    no_chord_index: Optional[int] = None,
    no_bass_index: int = 0,
) -> torch.Tensor:
    if no_chord_index is None:
        no_chord_index = RQ - 1
    y_joint = y_rq * BS + y_bs

    mask = (y_rq == ignore_index) | (y_bs == ignore_index)
    mask = mask | (y_rq == no_chord_index) | (y_bs == no_bass_index)

    return y_joint.masked_fill(mask, ignore_index)


def _segment_majority_targets(
    labels: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    セグメントごとの多数決ラベルを計算する。
    Returns:
      targets: (num_segments,) ignore_index込み
      valid_counts: (num_segments,) セグメント内の有効ラベル数
    """
    targets = torch.full((num_segments,), ignore_index, dtype=labels.dtype, device=labels.device)
    valid_counts = torch.zeros((num_segments,), dtype=torch.long, device=labels.device)

    for seg_idx in range(num_segments):
        seg_mask = segment_ids == seg_idx
        seg_labels = labels[seg_mask]
        seg_labels = seg_labels[seg_labels != ignore_index]
        if seg_labels.numel() == 0:
            continue

        uniq, counts = torch.unique(seg_labels, return_counts=True)
        best_idx = counts.argmax()
        targets[seg_idx] = uniq[best_idx]
        valid_counts[seg_idx] = seg_labels.numel()

    return targets, valid_counts


def _segment_consistency_loss_for_head(
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_ids_batch: torch.Tensor,
    ignore_index: int,
    length_weighted: bool,
) -> Optional[torch.Tensor]:
    """
    セグメント単位でlogitsを平均し、セグメント多数決ラベルへのCEを計算する。
    logits: (B, T, C), labels: (B, T), segment_ids_batch: (B, T)
    """
    if logits.ndim != 3 or labels.ndim != 2 or segment_ids_batch.ndim != 2:
        return None
    if logits.shape[:2] != labels.shape or labels.shape != segment_ids_batch.shape:
        return None

    total_loss = logits.new_tensor(0.0)
    total_weight = logits.new_tensor(0.0)

    batch_size, _, num_classes = logits.shape
    for b in range(batch_size):
        seg_ids = segment_ids_batch[b].long()
        if seg_ids.numel() == 0:
            continue

        num_segments = int(seg_ids.max().item()) + 1
        seg_sum = torch.zeros((num_segments, num_classes), device=logits.device, dtype=logits.dtype)
        seg_count = torch.zeros((num_segments, 1), device=logits.device, dtype=logits.dtype)

        seg_sum.index_add_(0, seg_ids, logits[b])
        ones = torch.ones((seg_ids.shape[0], 1), device=logits.device, dtype=logits.dtype)
        seg_count.index_add_(0, seg_ids, ones)
        seg_mean = seg_sum / seg_count.clamp_min(1.0)

        seg_targets, seg_valid_counts = _segment_majority_targets(
            labels[b],
            seg_ids,
            num_segments=num_segments,
            ignore_index=ignore_index,
        )
        valid_mask = seg_targets != ignore_index
        if not valid_mask.any():
            continue

        ce = F.cross_entropy(seg_mean[valid_mask], seg_targets[valid_mask].long(), reduction="none")
        if length_weighted:
            weights = seg_valid_counts[valid_mask].to(ce.dtype).clamp_min(1.0)
            total_loss = total_loss + (ce * weights).sum()
            total_weight = total_weight + weights.sum()
        else:
            total_loss = total_loss + ce.sum()
            total_weight = total_weight + ce.new_tensor(float(ce.numel()))

    if total_weight.item() <= 0.0:
        return None
    return total_loss / total_weight


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    root_chord_loss_fn: Optional[nn.Module] = None,
    segment_decode_cfg: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """モデル出力とバッチから、各タスクの損失を計算して辞書で返します。"""
    ce_ignore_index = loss_cfg.get("ce_ignore_index", -100)
    weights = loss_cfg.get("weights", {})
    all_losses = {}

    # 分類タスクの損失 (CrossEntropy)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)
    ftl_cfg = loss_cfg.get("focal_tversky", {})
    stage = "initial"

    # Root-chord
    if f"{stage}_root_chord_logits" in outputs:
        x = outputs[f"{stage}_root_chord_logits"]
        if root_chord_loss_fn is not None:
            loss = root_chord_loss_fn(x, batch["root_chord_index"])
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
        key_emphasis_cfg = loss_cfg.get("key_transition_emphasis", {}) or {}
        if bool(key_emphasis_cfg.get("enabled", False)):
            loss = key_transition_weighted_ce_loss(
                x,
                batch["key_index"],
                ignore_index=ce_ignore_index,
                transition_weight=float(key_emphasis_cfg.get("transition_weight", 4.0)),
                transition_tolerance_frames=int(key_emphasis_cfg.get("transition_tolerance_frames", 32)),
                base_weight=float(key_emphasis_cfg.get("base_weight", 1.0)),
                non_key_index=int(key_emphasis_cfg.get("non_key_index", 0)),
                ignore_non_key_to_n=bool(key_emphasis_cfg.get("ignore_non_key_to_n", True)),
            )
        else:
            loss = ce_loss_fn(x.transpose(1, 2), batch["key_index"])
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

    # --- テンポタスク ---
    if f"{stage}_tempo" in outputs:
        tempo_output = outputs[f"{stage}_tempo"]
        if tempo_output.size(-1) == 1:  # 回帰
            tempo_regression_scale = loss_cfg.get("tempo_regression_scale", "log")
            loss = tempo_regression_loss(tempo_output, batch["tempo_value"], scale=tempo_regression_scale)
        else:  # 分類
            loss = ce_loss_fn(tempo_output.transpose(1, 2), batch["tempo_index"])
        all_losses[f"{stage}/tempo"] = loss * weights.get("tempo", 1.0)

    # --- Chord25 タスク (Standard) ---
    if f"{stage}_chord25_logits" in outputs:
        loss = focal_tversky_loss(
            y_true=batch["chord25"],
            y_pred_logits=outputs[f"{stage}_chord25_logits"],
            alpha=ftl_cfg.get("alpha", 0.3),
            gamma=ftl_cfg.get("gamma", 1.5),
        )
        all_losses[f"{stage}/chord25"] = loss * weights.get("chord25", 1.0)

    # --- Smooth Chord25 タスク (Single) ---
    if "initial_smooth_chord25_logits" in outputs:
        # smooth_chord25 も chord25 と同様のロス計算を行う
        loss = focal_tversky_loss(
            y_true=batch["chord25"],
            y_pred_logits=outputs["initial_smooth_chord25_logits"],
            alpha=ftl_cfg.get("alpha", 0.3),
            gamma=ftl_cfg.get("gamma", 1.5),
        )
        all_losses["initial/smooth_chord25"] = loss * weights.get("smooth_chord25", 1.0)

    # --- Segment Consistency Loss ---
    seg_cons_cfg = loss_cfg.get("segment_consistency", {}) or {}
    if bool(seg_cons_cfg.get("enabled", False)) and "initial_boundary_logits" in outputs:
        seg_params = resolve_segment_decode_params(segment_decode_cfg)
        seg_heads = list(seg_cons_cfg.get("heads", ["root_chord", "bass"]))
        seg_weights = seg_cons_cfg.get("weights", {}) or {}
        length_weighted = bool(seg_cons_cfg.get("length_weighted", True))

        segmenter = BatchBoundarySegmenter(
            threshold=seg_params["threshold"],
            nms_window_radius=seg_params["nms_window_radius"],
            min_segment_length=seg_params["min_segment_length"],
            max_segments=seg_params["max_segments"],
        )
        with torch.no_grad():
            boundary_logits = outputs["initial_boundary_logits"]
            if boundary_logits.ndim == 2:
                boundary_logits = boundary_logits.unsqueeze(-1)
            batch_size, total_frames, _ = boundary_logits.shape
            dummy_features = torch.zeros(
                (batch_size, total_frames, 1),
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            _, _, seg_ids_batch, _ = segmenter.process_batch(
                frame_features=dummy_features,
                boundary_logits=boundary_logits,
                detach_boundary=True,
            )

        head_to_target = {
            "root_chord": "root_chord_index",
            "bass": "bass_index",
            "key": "key_index",
        }
        for head in seg_heads:
            logits_key = f"{stage}_{head}_logits"
            target_key = head_to_target.get(head)
            if target_key is None or logits_key not in outputs or target_key not in batch:
                continue

            seg_loss = _segment_consistency_loss_for_head(
                outputs[logits_key],
                batch[target_key],
                seg_ids_batch,
                ignore_index=ce_ignore_index,
                length_weighted=length_weighted,
            )
            if seg_loss is None:
                continue

            head_weight = float(seg_weights.get(head, 1.0))
            all_losses[f"{stage}/segment_consistency_{head}"] = seg_loss * head_weight

    # --- Structure Function Task ---
    if f"{stage}_structure_function_logits" in outputs:
        x = outputs[f"{stage}_structure_function_logits"]
        loss = ce_loss_fn(x.transpose(1, 2), batch["structure_function_index"])
        all_losses[f"{stage}/structure_function"] = loss * weights.get("structure_function", 1.0)

    # --- Structure Boundary Task ---
    if f"{stage}_structure_boundary_logits" in outputs:
        # 構造境界検出タスクの損失 (ShiftTolerantBCELoss) - 専用の重み設定を使用
        structure_bce_loss_fn = ShiftTolerantBCELoss(
            pos_weight=loss_cfg.get("structure_boundary_pos_weight", 1.0),
            tolerance=loss_cfg.get("structure_boundary_tolerance", 1),
        )
        loss = structure_bce_loss_fn(
            preds=outputs[f"{stage}_structure_boundary_logits"][..., 0],
            targets=batch["structure_boundary"][..., 0],
        )
        all_losses[f"{stage}/structure_boundary"] = loss * weights.get("structure_boundary", 1.0)

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

    # Bass
    logits = _try_resolve_head(outputs, "bass_logits")
    if logits is not None and "bass_index" in batch:
        metrics["acc/bass"] = accuracy_ignore_index_with_logits(
            logits, batch["bass_index"], ce_ignore_index
        )

    # Key
    logits = _try_resolve_head(outputs, "key_logits")
    if logits is not None and "key_index" in batch:
        metrics["acc/key"] = accuracy_ignore_index_with_logits(
            logits, batch["key_index"], ce_ignore_index
        )

    # Structure Function
    logits = _try_resolve_head(outputs, "structure_function_logits")
    if logits is not None and "structure_function_index" in batch:
        metrics["acc/structure_function"] = accuracy_ignore_index_with_logits(
            logits, batch["structure_function_index"], ce_ignore_index
        )

    return metrics
