"""
Classification Metrics for 7-class HAM10000 Skin Lesion Diagnosis:
- Macro-AUROC
- Sensitivity (Macro-Recall)
- Specificity (True Negative Rate)
- Per-Class AUROC
- Confusion Matrix
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
from typing import Dict, List, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = 7
) -> Dict[str, object]:
    """
    Computes comprehensive classification metrics.
    Args:
        y_true: 1D array of true class indices (0..num_classes-1)
        y_probs: 2D array of predicted class probabilities (N, num_classes)
    """
    y_pred = np.argmax(y_probs, axis=-1)
    acc = float(np.mean(y_true == y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class sensitivity and specificity
    sensitivities = []
    specificities = []
    per_class_auroc = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

        # Per-class binary AUROC (One-vs-Rest)
        y_true_binary = (y_true == c).astype(int)
        if len(np.unique(y_true_binary)) > 1:
            try:
                auc_c = roc_auc_score(y_true_binary, y_probs[:, c])
                per_class_auroc[c] = float(auc_c)
            except Exception:
                per_class_auroc[c] = 0.5
        else:
            per_class_auroc[c] = 0.5

    macro_sensitivity = float(np.mean(sensitivities))
    macro_specificity = float(np.mean(specificities))

    # Macro AUROC
    valid_aurocs = list(per_class_auroc.values())
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5

    return {
        "accuracy": acc,
        "macro_auroc": macro_auroc,
        "sensitivity": macro_sensitivity,
        "specificity": macro_specificity,
        "per_class_auroc": per_class_auroc,
        "confusion_matrix": cm.tolist(),
        "sample_count": int(len(y_true))
    }


def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    is_dann: bool = False,
    device: str = "cpu"
) -> Dict[str, object]:
    """
    Evaluates CNN or DANN on a DataLoader.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(dev)
            y = batch["label"].cpu().numpy()

            if is_dann:
                probs = model.predict_class_proba(x).cpu().numpy()
            else:
                probs = model.predict_proba(x).cpu().numpy()

            all_targets.append(y)
            all_probs.append(probs)

    y_true = np.concatenate(all_targets, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_classification_metrics(y_true, y_probs)
    return metrics
