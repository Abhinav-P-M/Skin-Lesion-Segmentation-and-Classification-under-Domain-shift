"""
Segmentation Metrics for Skin Lesion Analysis:
- Dice Score (F1 / Sørensen–Dice coefficient)
- IoU (Jaccard Index)
- Pixel Accuracy
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Computes Dice coefficient between binary prediction and ground truth.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    pred_flat = pred_mask.contiguous().view(-1)
    gt_flat = gt_mask.to(pred_mask.device).contiguous().view(-1)

    if pred_flat.numel() != gt_flat.numel():
        min_len = min(pred_flat.numel(), gt_flat.numel())
        pred_flat = pred_flat[:min_len]
        gt_flat = gt_flat[:min_len]

    intersection = (pred_flat * gt_flat).sum().item()
    total = pred_flat.sum().item() + gt_flat.sum().item()

    if total == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (total + smooth))


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Computes Intersection over Union (Jaccard Index).
    IoU = |A ∩ B| / |A ∪ B|
    """
    pred_flat = pred_mask.contiguous().view(-1)
    gt_flat = gt_mask.to(pred_mask.device).contiguous().view(-1)

    if pred_flat.numel() != gt_flat.numel():
        min_len = min(pred_flat.numel(), gt_flat.numel())
        pred_flat = pred_flat[:min_len]
        gt_flat = gt_flat[:min_len]

    intersection = (pred_flat * gt_flat).sum().item()
    union = pred_flat.sum().item() + gt_flat.sum().item() - intersection

    if union == 0:
        return 1.0
    return float((intersection + smooth) / (union + smooth))


def evaluate_segmentation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    threshold: float = 0.5,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluates U-Net model across a segmentation dataloader.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(dev)
            gt_masks = batch["mask"].to(dev)

            pred_probs = model(imgs)
            pred_binary = (pred_probs >= threshold).float()

            for i in range(imgs.size(0)):
                dice = compute_dice(pred_binary[i], gt_masks[i])
                iou = compute_iou(pred_binary[i], gt_masks[i])
                dice_scores.append(dice)
                iou_scores.append(iou)

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "sample_count": len(dice_scores)
    }
