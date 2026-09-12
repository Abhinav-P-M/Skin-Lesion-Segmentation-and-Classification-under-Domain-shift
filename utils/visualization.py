"""
Visualization Utilities for Skin Lesion Segmentation, Classification, and Domain Shift.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import Dict, List, Optional, Tuple


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (C, H, W) or (1, C, H, W) tensor in [0, 1] to a (H, W, C) numpy uint8 array."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def create_mask_overlay(
    img_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    boundary_color: Tuple[int, int, int] = (0, 255, 128),
    fill_color: Tuple[int, int, int] = (0, 200, 255),
    fill_alpha: float = 0.35
) -> np.ndarray:
    """
    Overlays predicted lesion mask onto the original dermoscopic image with a prominent boundary contour.
    """
    img_np = tensor_to_numpy_img(img_tensor).astype(np.float32)
    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor.squeeze(0)
    if mask_tensor.dim() == 3:
        mask_np = mask_tensor.squeeze(0).detach().cpu().numpy()
    else:
        mask_np = mask_tensor.detach().cpu().numpy()

    binary_mask = (mask_np >= 0.5).astype(np.uint8)

    # Detect boundary using simple gradient / erosion difference
    pad = np.pad(binary_mask, 1, mode="constant")
    eroded = (
        pad[1:-1, 1:-1] & pad[0:-2, 1:-1] & pad[2:, 1:-1] &
        pad[1:-1, 0:-2] & pad[1:-1, 2:]
    )
    boundary = (binary_mask - eroded) > 0

    overlay = img_np.copy()
    # Semi-transparent lesion fill
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            overlay[:, :, c] * (1.0 - fill_alpha) + fill_color[c] * fill_alpha,
            overlay[:, :, c]
        )
    # Bright boundary contour
    for c in range(3):
        overlay[:, :, c] = np.where(boundary, boundary_color[c], overlay[:, :, c])

    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_confusion_matrix_fig(
    cm: np.ndarray,
    class_codes: List[str],
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """Plots a clean, dark-themed publication-quality confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#060d1f")
    ax.set_facecolor("#0a1628")
    im = ax.imshow(cm, interpolation="nearest", cmap="mako")
    cb = ax.figure.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#94a3b8')

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_codes,
        yticklabels=class_codes,
        title=title,
        ylabel="True Diagnosis",
        xlabel="Predicted Diagnosis"
    )
    ax.set_title(title, color="#f1f5f9", fontweight="bold", fontsize=12, pad=12)
    ax.set_ylabel("True Diagnosis", color="#94a3b8", fontweight="600")
    ax.set_xlabel("Predicted Diagnosis", color="#94a3b8", fontweight="600")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.1))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color="#cbd5e1")
    plt.setp(ax.get_yticklabels(), color="#cbd5e1")

    # Annotate numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="#ffffff" if cm[i, j] > thresh else "#94a3b8",
                fontweight="bold"
            )
    fig.tight_layout()
    return fig


def plot_domain_shift_comparison(
    cnn_src_auc: float,
    cnn_tgt_auc: float,
    dann_src_auc: float,
    dann_tgt_auc: float
) -> plt.Figure:
    """
    Plots the central research experiment: CNN vs DANN performance drop under domain shift in dark theme.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#060d1f")
    ax.set_facecolor("#0a1628")

    models = ["Baseline CNN", "Proposed DANN"]
    src_aucs = [cnn_src_auc * 100, dann_src_auc * 100]
    tgt_aucs = [cnn_tgt_auc * 100, dann_tgt_auc * 100]
    drops = [(cnn_src_auc - cnn_tgt_auc) * 100, (dann_src_auc - dann_tgt_auc) * 100]

    x = np.arange(len(models))
    width = 0.32

    rects1 = ax.bar(x - width/2, src_aucs, width, label="Source Domain AUROC", color="#6366f1", alpha=0.9, edgecolor="#818cf8")
    rects2 = ax.bar(x + width/2, tgt_aucs, width, label="Unseen Target Domain AUROC", color="#14b8a6", alpha=0.9, edgecolor="#2dd4bf")

    ax.set_ylabel("Macro-AUROC (%)", fontsize=11, fontweight="bold", color="#94a3b8")
    ax.set_title("Cross-Dataset Domain Shift: CNN vs DANN Generalization", fontsize=12, fontweight="bold", pad=12, color="#f1f5f9")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold", color="#e2e8f0")
    ax.tick_params(colors="#cbd5e1")
    leg = ax.legend(loc="lower left", frameon=True, facecolor="#070e22", edgecolor=(1.0, 1.0, 1.0, 0.15), labelcolor="#e2e8f0")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.1))

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#818cf8", fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#2dd4bf", fontweight="bold")

    # Add performance drop indicators
    for i, drop in enumerate(drops):
        bg_col = "#7f1d1d" if i == 0 else "#064e3b"
        txt_col = "#fecdd3" if i == 0 else "#a7f3d0"
        border_col = "#f43f5e" if i == 0 else "#10b981"
        ax.text(i, 10, f"Shift Drop: {'+' if drop>0 else ''}{drop:.1f}%", ha="center", fontsize=9.5, fontweight="bold",
                color=txt_col,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=bg_col, edgecolor=border_col, alpha=0.9))

    fig.tight_layout()
    return fig
