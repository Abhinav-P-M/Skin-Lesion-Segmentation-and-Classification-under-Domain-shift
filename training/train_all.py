"""
Master Training and Benchmarking Pipeline:
Trains U-Net, Baseline CNN, and DANN, evaluates cross-domain degradation,
and exports benchmark_results.json for Streamlit caching.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from data.dataset_loader import build_benchmark_suites
from training.train_unet import train_unet
from training.train_cnn import train_baseline_cnn
from training.train_dann import train_dann
from evaluation.domain_eval import run_cross_domain_comparison
from evaluation.segmentation_metrics import evaluate_segmentation


def run_full_pipeline(
    epochs: int = 3,
    samples_per_class: int = 50,
    img_size: int = 64,
    device: str = "cpu",
    results_path: str = "models/checkpoints/benchmark_results.json"
):
    print("=================================================================")
    print("  CROSS-DATASET SKIN LESION SEGMENTATION & DANN CLASSIFICATION  ")
    print(f"  Device: {device} | Epochs: {epochs} | Image Resolution: {img_size}x{img_size}")
    print("=================================================================\n")

    # 1. Build benchmark data suites
    print("[1/5] Constructing Benchmark Suites (Source, Target, Segmentation)...")
    suites = build_benchmark_suites(
        samples_per_class=samples_per_class,
        img_size=img_size,
        save_sample_gallery=True,
        sample_dir="data/sample_data"
    )

    src_train = suites["src_train"]
    src_test = suites["src_test"]
    tgt_adapt = suites["tgt_adapt"]
    tgt_eval = suites["tgt_eval"]
    seg_suite = suites["segmentation_suite"]

    # 2. Train U-Net Lesion Segmentation
    print("\n[2/5] Training Part A: U-Net Lesion Segmentation...")
    unet_model = train_unet(
        seg_suite,
        epochs=epochs,
        batch_size=16,
        device=device,
        save_path="models/checkpoints/unet/best_model.pth"
    )

    seg_loader = DataLoader(seg_suite, batch_size=16, shuffle=False)
    seg_metrics = evaluate_segmentation(unet_model, seg_loader, device=device)

    # 3. Train Baseline Conventional CNN
    print("\n[3/5] Training Part B (Baseline): Conventional CNN on Source Domain...")
    cnn_model = train_baseline_cnn(
        src_train,
        src_test,
        epochs=epochs,
        batch_size=16,
        device=device,
        save_path="models/checkpoints/cnn/best_model.pth"
    )

    # 4. Train Proposed DANN
    print("\n[4/5] Training Part B (Proposed): Domain-Adversarial Neural Network (DANN)...")
    dann_model = train_dann(
        src_train,
        tgt_adapt,
        src_test,
        epochs=epochs,
        batch_size=16,
        device=device,
        save_path="models/checkpoints/dann/best_model.pth"
    )

    # 5. Cross-Dataset Evaluation & Performance Drop Analysis
    print("\n[5/5] Conducting Cross-Dataset Evaluation (CNN vs DANN)...")
    src_test_loader = DataLoader(src_test, batch_size=32, shuffle=False)
    tgt_eval_loader = DataLoader(tgt_eval, batch_size=32, shuffle=False)

    cross_domain_res = run_cross_domain_comparison(
        cnn_model,
        dann_model,
        src_test_loader,
        tgt_eval_loader,
        device=device
    )

    # Print clean summary table
    print("\n======================= BENCHMARK RESULTS =======================")
    print(f"{'Model':<16} | {'Source AUROC':<14} | {'Target AUROC':<14} | {'Performance Drop':<16}")
    print("-" * 68)
    for row in cross_domain_res["comparison_table"]:
        print(f"{row['Model']:<16} | {row['Source AUROC']:<14} | {row['Target AUROC']:<14} | {row['Performance Drop']:<16}")
    print("-" * 68)
    print(f"U-Net Segmentation  | Mean Dice: {seg_metrics['mean_dice']*100:.2f}% | Mean IoU: {seg_metrics['mean_iou']*100:.2f}%\n")

    # Export results payload
    payload = {
        "segmentation": seg_metrics,
        "classification": cross_domain_res,
        "meta": {
            "samples_per_class": samples_per_class,
            "img_size": img_size,
            "device": device
        }
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Benchmark results successfully exported to {results_path}")
    return payload


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_full_pipeline(epochs=3, samples_per_class=45, img_size=64, device=device)
