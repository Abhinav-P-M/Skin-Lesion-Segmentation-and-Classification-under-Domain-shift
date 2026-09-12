https://drive.google.com/file/d/1EgMi2_ljL-mQyNkhSSkSE3gLBBBsWLVS/view?usp=drive_link  This is the model file ,please paste this inside a folder named pretrained inside the folder model,inside the main project folder.
# Cross-Dataset Skin Lesion Segmentation and Classification Under Domain Shift Using DANN

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, hackathon-quality medical AI system addressing the central bottleneck of clinical dermatology AI: **cross-dataset domain shift**. Built with **Python, PyTorch, and Streamlit**.

---

## 1. Problem Statement & Research Question

Deep learning models for skin cancer detection often report >90% accuracy on standard benchmarks. However, when deployed in external clinics or evaluated on images captured by different cameras, dermatoscopes, or mobile phones, their performance degrades precipitously.

### Sources of Real-World Clinical Domain Shift:
1. **Optical & Polarization Discrepancies:** High-grade polarized contact dermatoscopes eliminate surface reflection; non-contact cameras and mobile lenses capture heavy specular reflections and vignetting.
2. **Illumination & Color Temperature:** Hospital halogen examination lamps, daylight LEDs, and smartphone camera flashes produce severe spectral shifts.
3. **Patient Population Variance:** Distributional shifts across Fitzpatrick skin phototypes (e.g., Central European vs South American populations).
4. **Resolution & Sensor Noise:** Variations in sensor dynamic range, ISO noise, and compression artifacts.

### Central Research Question:
> **Can a Domain-Adversarial Neural Network (DANN) learn representations that generalize better to an unseen skin-lesion dataset than a conventional CNN?**

---

## 2. System Architecture

```
                              Input Dermoscopic Image
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
           Part A: Segmentation                     Part B: Classification
          (U-Net Pixel Localization)               (Cross-Domain Evaluation)
                    │                                         │
         ┌──────────┴──────────┐                              │
         ▼                     ▼                              │
    Binary Mask         Boundary Contour                      ▼
  (Where is lesion?)   (Overlay Visualization)     Feature Extractor G_f(x)
                                                              │
                                                      ┌───────┴───────┐
                                                      ▼               ▼
                                              Label Classifier  Gradient Reversal
                                                  G_y(f)           Layer (GRL)
                                                      │               │
                                                      ▼               ▼
                                                HAM10000 7    Domain Classifier
                                                Categories          G_d(f)
                                                               (Source vs Target)
```

---

## 3. Mathematical Formulation (DANN)

The Domain-Adversarial Neural Network (Ganin et al., JMLR 2016) optimizes an adversarial minimax objective:

$$\min_{\theta_f, \theta_y} \max_{\theta_d} \mathcal{L}_{class}(G_y(G_f(X_s)), Y_s) - \lambda(p) \cdot \mathcal{L}_{domain}(G_d(\mathcal{R}(G_f(X))), D)$$

### Dynamic Gradient Reversal Layer (GRL)
During backpropagation, the GRL reverses the sign of the domain classifier gradients:
$$\mathcal{R}(f) = f, \quad \frac{d\mathcal{R}}{df} = -\lambda(p) \cdot \mathbf{I}$$

### Gradual $\lambda$ Scheduling Strategy
Rather than applying an aggressive domain penalty immediately, $\lambda$ is dynamically scaled with training progress $p \in [0, 1]$:
$$\lambda(p) = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1 \quad (\gamma = 10)$$
This ensures the feature extractor first learns basic discriminative pathology features before domain alignment is enforced.

---

## 4. Supported Disease Categories (HAM10000)

The system classifies lesions into the 7 official categories of the HAM10000 dataset:

| Code | Disease Name | Clinical Category | Pathological Relevance |
| :--- | :--- | :--- | :--- |
| `akiec` | Actinic Keratoses & Intraepithelial Carcinoma | Potentially Malignant | Pre-cancerous / early non-melanoma lesion (Bowen's disease) |
| `bcc` | Basal Cell Carcinoma | Potentially Malignant | Locally destructive cutaneous malignancy |
| `bkl` | Benign Keratosis-like Lesions | Benign | Seborrheic keratoses and solar lentigines |
| `df` | Dermatofibroma | Benign | Benign cutaneous fibrohistiocytic nodule |
| `mel` | Melanoma | Potentially Malignant | Highly invasive malignant melanocytic lesion |
| `nv` | Melanocytic Nevi | Benign | Common mole (benign melanocyte proliferation) |
| `vasc` | Vascular Lesions | Benign | Angiomas, pyogenic granulomas, vascular malformations |

---

## 5. Project Directory Structure

```
skin_lesion_dann_domain_shift/
├── app.py                              # 7-page interactive Streamlit dashboard
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation & research context
│
├── models/
│   ├── unet.py                         # U-Net lesion segmentation architecture
│   ├── cnn.py                          # Baseline conventional CNN classifier
│   ├── dann.py                         # Domain-Adversarial Neural Network
│   ├── gradient_reversal.py            # Autograd GRL & lambda schedule
│   └── checkpoints/
│       ├── unet/best_model.pth         # Real trained U-Net checkpoint
│       ├── cnn/best_model.pth          # Real trained CNN baseline checkpoint
│       ├── dann/best_model.pth         # Real trained DANN checkpoint
│       └── benchmark_results.json      # Cached empirical evaluation results
│
├── data/
│   ├── dataset_loader.py               # HAM10000 & segmentation dataset manager
│   ├── domain_shift.py                 # Clinical domain shift simulator
│   └── sample_data/                    # Benchmark gallery (source, target, masks)
│
├── training/
│   ├── train_unet.py                   # BCE + Soft Dice training
│   ├── train_cnn.py                    # Class-weighted Cross-Entropy on Source
│   ├── train_dann.py                   # Adversarial training with gradual lambda
│   └── train_all.py                    # Master script to train all models
│
├── evaluation/
│   ├── segmentation_metrics.py         # Dice Score & IoU calculation
│   ├── classification_metrics.py       # Macro-AUROC, Sensitivity, Specificity
│   └── domain_eval.py                  # Cross-domain degradation analysis
│
├── utils/
│   ├── preprocessing.py                # Clinically sound transforms
│   ├── visualization.py                # Mask overlays & confusion matrices
│   └── gradcam.py                      # PyTorch Grad-CAM explainability
│
└── tests/
    └── test_pipeline.py                # Complete unit and integration test suite
```

---

## 6. Quick Start & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Run Full Training & Benchmarking Pipeline
Train U-Net, Baseline CNN, and DANN, evaluate cross-domain degradation, and export real checkpoints:
```bash
python training/train_all.py
```

### 4. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```

---

## 7. Streamlit Dashboard Features (7 Pages)

1. **Project Overview:** Clinical motivation, domain shift breakdown, interactive architecture diagram.
2. **Dataset Explorer:** HAM10000 7-class reference table, real-world class imbalance chart, separate segmentation dataset explorer.
3. **Live Image Analysis:** Upload custom image or select benchmark dermoscopy $\to$ U-Net segmentation mask & contour overlay $\to$ Side-by-side Baseline CNN vs Domain-Adaptive DANN prediction cards $\to$ Domain attribution $\to$ Medical disclaimer.
4. **CNN vs DANN Benchmark:** Empirical cross-dataset comparison table, $\Delta\text{AUROC}$ degradation bar charts, target domain confusion matrices.
5. **Segmentation Evaluation:** Quantitative Mean Dice & Mean IoU cards, qualitative gallery (raw image, ground truth, prediction, overlay), failure case analysis.
6. **Model Explainability (Grad-CAM):** Visual attention heatmaps showing CNN vs DANN focus areas under domain shift.
7. **Research Summary:** Executive pitch summary, hypothesis validation, clinical limitations.

---

## 8. Medical AI Safety Disclaimer

> **RESEARCH AND EDUCATIONAL USE ONLY**  
> This software is a research prototype developed for evaluating machine learning robustness under clinical domain shifts.  
> It does **NOT** constitute medical software, a diagnostic device, or clinical advice.  
> The system classifies skin lesions strictly into the categories represented in the dataset and does not detect all skin malignancies.  
> Any suspected skin lesion must be formally evaluated by a qualified board-certified dermatologist with histological examination.
