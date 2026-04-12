import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union

from .losses import extract_chord_intervals_from_boundary
from ..chord_transcription.models.semi_crf import (
    NeuralSemiCRFInterval,
    _build_interval_score,
    _zero_noise_score,
)


def save_semi_crf_visualization(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    boundary: torch.Tensor,
    output_path: Union[str, Path],
    length_scaling: str = "sqrt",
    boundary_threshold: float = 0.8,
    boundary_logits: Optional[torch.Tensor] = None,
) -> None:
    """
    Validation step などで、正解のコード区間と Semi-CRF の予測区間を比較して画像保存する。
    バッチ内の最初のサンプルのみ描画する。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # バッチ内の最初のサンプルを対象とする
    batch_idx = 0
    with torch.no_grad():
        # GT 区間抽出
        gt_intervals_batch = extract_chord_intervals_from_boundary(
            boundary[batch_idx : min(batch_idx + 1, boundary.shape[0])], threshold=boundary_threshold
        )
        if not gt_intervals_batch:
            return
        gt_intervals = gt_intervals_batch[0]

        # 予測区間を Semi-CRF Decode で取得
        time_steps = interval_query.shape[1]
        query_single = interval_query[batch_idx : batch_idx + 1].transpose(0, 1)  # [T, 1, D]
        key_single = interval_key[batch_idx : batch_idx + 1].transpose(0, 1)  # [T, 1, D]
        diag_single = interval_diag[batch_idx : batch_idx + 1].transpose(0, 1)  # [T, 1]

        score = _build_interval_score(query_single, key_single, diag_single, length_scaling=length_scaling)
        noise_score = _zero_noise_score(time_steps, batch_size=1, device=score.device)

        semi_crf = NeuralSemiCRFInterval(score, noise_score)
        decoded_batch = semi_crf.decode()
        pred_intervals = decoded_batch[0] if decoded_batch else []

    # 描画
    fig, ax = plt.subplots(figsize=(15, 3))

    # タイムステップ配列
    t_max = time_steps

    # GT描画 (y=0.7)
    for start, end in gt_intervals:
        ax.plot([start, end], [0.7, 0.7], color="tab:blue", linewidth=6, alpha=0.8, solid_capstyle="butt")
        # 背景として少し太めの半透明な実線で描画
        ax.axvline(start, color="tab:blue", alpha=0.25, linestyle="-", linewidth=3)
        ax.axvline(end, color="tab:blue", alpha=0.25, linestyle="-", linewidth=3)

    # Pred描画 (y=0.3)
    for start, end in pred_intervals:
        ax.plot([start, end], [0.3, 0.3], color="tab:orange", linewidth=6, alpha=0.8, solid_capstyle="butt")
        # 前景としてはっきり見える破線で描画
        ax.axvline(start, color="tab:orange", alpha=0.9, linestyle="--", linewidth=1.5)
        ax.axvline(end, color="tab:orange", alpha=0.9, linestyle="--", linewidth=1.5)

    ax.text(-t_max * 0.02, 0.7, "Ground Truth", va="center", ha="right", color="tab:blue", fontweight="bold")
    ax.text(-t_max * 0.02, 0.3, "Semi-CRF Pred", va="center", ha="right", color="tab:orange", fontweight="bold")

    # Boundary Probabilities の描画 (与えられた場合)
    if boundary_logits is not None:
        with torch.no_grad():
            # boundary_logits は [B, T, 1] または [B, T]
            b_logits = boundary_logits[batch_idx]
            if b_logits.dim() == 2:
                b_logits = b_logits.squeeze(-1)
            b_probs = torch.sigmoid(b_logits).cpu().numpy()
        time_axis = np.arange(time_steps)
        ax.plot(time_axis, b_probs, color="tab:green", alpha=0.5, linewidth=1.5, label="Boundary Prob")

    ax.set_ylim(0, 1)
    ax.set_xlim(0, t_max)
    ax.set_yticks([])
    ax.set_xlabel("Time (frames)")
    ax.set_title("Chord Intervals: Ground Truth vs Semi-CRF Prediction")
    if boundary_logits is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_pitch_chroma_visualization(
    pitch_logits: torch.Tensor,
    chroma_target: torch.Tensor,
    output_path: Union[str, Path],
) -> None:
    """
    Validation step などで、正解の Pitch Chroma と予測確率を比較して画像保存する。
    バッチ内の最初のサンプルのみ描画する。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        pred_probs = torch.sigmoid(pitch_logits[0]).cpu().numpy().T
        gt_labels = chroma_target[0, :, :12].cpu().numpy().T

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

    # 描画パラメータ
    cmap = "viridis"
    aspect = "auto"
    origin = "lower"

    # GT描画
    im0 = axes[0].imshow(gt_labels, aspect=aspect, origin=origin, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Pitch Chroma Ground Truth")
    axes[0].set_ylabel("Pitch Class (0-11)")

    # Pred描画
    im1 = axes[1].imshow(pred_probs, aspect=aspect, origin=origin, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Pitch Chroma Predicted Probabilities")
    axes[1].set_ylabel("Pitch Class (0-11)")
    axes[1].set_xlabel("Time (frames)")

    fig.colorbar(im1, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

