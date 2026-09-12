"""
Training Script for Baseline Conventional CNN on Source Domain.
Pipeline: Source Images -> CNN Feature Extractor -> Fully Connected -> 7 HAM10000 Classes.
No domain adaptation mechanisms applied.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn import BaselineCNN
from evaluation.classification_metrics import evaluate_classifier


def train_baseline_cnn(
    src_train_ds,
    src_test_ds,
    num_classes: int = 7,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 0.001,
    device: str = "cpu",
    save_path: str = "models/checkpoints/cnn/best_model.pth"
) -> BaselineCNN:
    print("\n--- Training Baseline Conventional CNN (Source Domain Only) ---")
    dev = torch.device(device)
    train_loader = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(src_test_ds, batch_size=batch_size, shuffle=False)

    model = BaselineCNN(num_classes=num_classes, feature_dim=256).to(dev)

    # Class-weighted Cross-Entropy to handle HAM10000 imbalance
    labels = src_train_ds.labels.numpy()
    class_counts = [max(1, int((labels == c).sum())) for c in range(num_classes)]
    weights = [sum(class_counts) / (num_classes * c_count) for c_count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(dev)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            x = batch["image"].to(dev)
            y = batch["label"].to(dev)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        val_metrics = evaluate_classifier(model, test_loader, is_dann=False, device=device)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {running_loss/max(1, batches):.4f} | Source Val AUROC: {val_metrics['macro_auroc']*100:.2f}% | Acc: {val_metrics['accuracy']*100:.2f}%")

        if val_metrics["macro_auroc"] >= best_auc:
            best_auc = val_metrics["macro_auroc"]
            torch.save(model.state_dict(), save_path)

    print(f"CNN Training Complete. Best Source AUROC: {best_auc*100:.2f}%. Saved to {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=dev))
    return model
