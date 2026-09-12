"""
Adversarial Training Script for DANN (Domain-Adversarial Neural Network).
Learns representations that are discriminative for disease diagnosis
while being invariant to domain shift (Source vs Unseen Target).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.dann import DANN
from models.gradient_reversal import calc_lambda
from evaluation.classification_metrics import evaluate_classifier


def train_dann(
    src_train_ds,
    tgt_adapt_ds,
    src_test_ds,
    num_classes: int = 7,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 0.001,
    gamma: float = 10.0,
    device: str = "cpu",
    save_path: str = "models/checkpoints/dann/best_model.pth"
) -> DANN:
    print("\n--- Training Domain-Adversarial Neural Network (DANN) ---")
    dev = torch.device(device)

    src_loader = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_adapt_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    src_test_loader = DataLoader(src_test_ds, batch_size=batch_size, shuffle=False)

    model = DANN(num_classes=num_classes, feature_dim=256).to(dev)

    # Class weights for HAM10000
    labels = src_train_ds.labels.numpy()
    class_counts = [max(1, int((labels == c).sum())) for c in range(num_classes)]
    weights = [sum(class_counts) / (num_classes * c_count) for c_count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(dev)

    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_steps = epochs * min(len(src_loader), len(tgt_loader))
    current_step = 0
    best_score = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_class_loss = 0.0
        running_domain_loss = 0.0
        batches = 0

        tgt_iter = iter(tgt_loader)

        for src_batch in src_loader:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            x_src = src_batch["image"].to(dev)
            y_src = src_batch["label"].to(dev)
            x_tgt = tgt_batch["image"].to(dev)

            B_src = x_src.size(0)
            B_tgt = x_tgt.size(0)

            # Gradual lambda scheduling
            current_step += 1
            lambd = calc_lambda(current_step, total_steps, gamma=gamma)

            optimizer.zero_grad()

            # 1. Forward Source data
            src_class_logits, src_domain_logits, _ = model(x_src, lambd=lambd)
            loss_class = class_criterion(src_class_logits, y_src)
            d_label_src = torch.zeros(B_src, dtype=torch.long, device=dev)
            loss_domain_src = domain_criterion(src_domain_logits, d_label_src)

            # 2. Forward Target data (strictly for domain discrimination - no disease labels used!)
            _, tgt_domain_logits, _ = model(x_tgt, lambd=lambd)
            d_label_tgt = torch.ones(B_tgt, dtype=torch.long, device=dev)
            loss_domain_tgt = domain_criterion(tgt_domain_logits, d_label_tgt)

            loss_domain = 0.5 * (loss_domain_src + loss_domain_tgt)
            total_loss = loss_class + loss_domain

            total_loss.backward()
            optimizer.step()

            running_class_loss += loss_class.item()
            running_domain_loss += loss_domain.item()
            batches += 1

        val_metrics = evaluate_classifier(model, src_test_loader, is_dann=True, device=device)
        print(f"Epoch [{epoch}/{epochs}] (lambda={lambd:.2f}) - Class Loss: {running_class_loss/max(1, batches):.4f} | Domain Loss: {running_domain_loss/max(1, batches):.4f} | Source Val AUROC: {val_metrics['macro_auroc']*100:.2f}%")

        if val_metrics["macro_auroc"] >= best_score:
            best_score = val_metrics["macro_auroc"]
            torch.save(model.state_dict(), save_path)

    print(f"DANN Training Complete. Best Score: {best_score*100:.2f}%. Saved to {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=dev))
    return model
