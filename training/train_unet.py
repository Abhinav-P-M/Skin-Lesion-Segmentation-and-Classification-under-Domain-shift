"""
Training Script for U-Net Lesion Segmentation.
Loss Function: Combined Binary Cross-Entropy (BCE) + Soft Dice Loss.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet import UNet
from evaluation.segmentation_metrics import evaluate_segmentation, compute_dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def train_unet(
    dataset,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 0.002,
    device: str = "cpu",
    save_path: str = "models/checkpoints/unet/best_model.pth"
) -> UNet:
    print("\n--- Training U-Net Lesion Segmentation Model ---")
    dev = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=3, out_channels=1, base_channels=16).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss()

    best_dice = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in loader:
            imgs = batch["image"].to(dev)
            masks = batch["mask"].to(dev)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        # Evaluate epoch dice
        metrics = evaluate_segmentation(model, loader, device=device)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {running_loss/max(1, batches):.4f} | Train Dice: {metrics['mean_dice']*100:.2f}% | IoU: {metrics['mean_iou']*100:.2f}%")

        if metrics["mean_dice"] >= best_dice:
            best_dice = metrics["mean_dice"]
            torch.save(model.state_dict(), save_path)

    print(f"U-Net Training Complete. Best Dice: {best_dice*100:.2f}%. Checkpoint saved to {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=dev))
    return model
