"""
Domain Generalization Evaluation & Cross-Dataset Benchmark Analysis.
Computes Source vs Unseen Target Domain Performance and the Performance Drop (Degradation).
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from typing import Dict

from evaluation.classification_metrics import evaluate_classifier


def run_cross_domain_comparison(
    cnn_model: torch.nn.Module,
    dann_model: torch.nn.Module,
    src_test_loader: DataLoader,
    tgt_eval_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, object]:
    """
    Evaluates CNN baseline and DANN on both Source Test and Unseen Target datasets,
    computing the empirical performance drop.
    """
    # 1. Baseline CNN evaluation
    cnn_src_metrics = evaluate_classifier(cnn_model, src_test_loader, is_dann=False, device=device)
    cnn_tgt_metrics = evaluate_classifier(cnn_model, tgt_eval_loader, is_dann=False, device=device)

    # 2. DANN evaluation
    dann_src_metrics = evaluate_classifier(dann_model, src_test_loader, is_dann=True, device=device)
    dann_tgt_metrics = evaluate_classifier(dann_model, tgt_eval_loader, is_dann=True, device=device)

    # 3. Calculate Performance Drop (Source - Target)
    cnn_drop = cnn_src_metrics["macro_auroc"] - cnn_tgt_metrics["macro_auroc"]
    dann_drop = dann_src_metrics["macro_auroc"] - dann_tgt_metrics["macro_auroc"]

    cnn_acc_drop = cnn_src_metrics["accuracy"] - cnn_tgt_metrics["accuracy"]
    dann_acc_drop = dann_src_metrics["accuracy"] - dann_tgt_metrics["accuracy"]

    results = {
        "cnn": {
            "source": cnn_src_metrics,
            "target": cnn_tgt_metrics,
            "auroc_drop": float(cnn_drop),
            "accuracy_drop": float(cnn_acc_drop)
        },
        "dann": {
            "source": dann_src_metrics,
            "target": dann_tgt_metrics,
            "auroc_drop": float(dann_drop),
            "accuracy_drop": float(dann_acc_drop)
        },
        "comparison_table": [
            {
                "Model": "Baseline CNN",
                "Source AUROC": f"{cnn_src_metrics['macro_auroc']*100:.2f}%",
                "Target AUROC": f"{cnn_tgt_metrics['macro_auroc']*100:.2f}%",
                "Performance Drop": f"{cnn_drop*100:.2f}%",
                "Target Accuracy": f"{cnn_tgt_metrics['accuracy']*100:.2f}%",
                "Target Sensitivity": f"{cnn_tgt_metrics['sensitivity']*100:.2f}%",
                "Target Specificity": f"{cnn_tgt_metrics['specificity']*100:.2f}%"
            },
            {
                "Model": "Proposed DANN",
                "Source AUROC": f"{dann_src_metrics['macro_auroc']*100:.2f}%",
                "Target AUROC": f"{dann_tgt_metrics['macro_auroc']*100:.2f}%",
                "Performance Drop": f"{dann_drop*100:.2f}%",
                "Target Accuracy": f"{dann_tgt_metrics['accuracy']*100:.2f}%",
                "Target Sensitivity": f"{dann_tgt_metrics['sensitivity']*100:.2f}%",
                "Target Specificity": f"{dann_tgt_metrics['specificity']*100:.2f}%"
            }
        ]
    }
    return results
