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
) -> torch.Tensor:
    """テンポ（回帰）の損失を計算します。"""
    pred = pred_logits.squeeze(-1)
    valid_mask = target_bpm != ignore_value
    if valid_mask.sum() == 0:
        return pred.sum() * 0.0

    # log(BPM)ドメインでSmooth L1 Lossを計算
    true_log_bpm = torch.log(torch.clamp(target_bpm[valid_mask], min=1.0))
    pred_log_bpm = pred[valid_mask]

    return F.smooth_l1_loss(pred_log_bpm, true_log_bpm, beta=0.2, reduction="mean")


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


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    root_chord_loss_fn: Optional[nn.Module] = None,
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
            loss = tempo_regression_loss(tempo_output, batch["tempo_value"])
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
