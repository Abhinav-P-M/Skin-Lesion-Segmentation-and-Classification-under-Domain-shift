"""
Dataset Loader and Metadata Definitions for HAM10000 & Skin Lesion Segmentation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from data.domain_shift import DomainShiftTransform, generate_synthetic_dermoscopy


# 7 Official HAM10000 Diagnostic Categories
HAM10000_CLASSES = {
    0: {
        "code": "akiec",
        "name": "Actinic Keratoses & Intraepithelial Carcinoma (Bowen's)",
        "type": "Potentially Malignant",
        "description": "Pre-cancerous / early non-melanoma lesion requiring clinical biopsy evaluation."
    },
    1: {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "type": "Potentially Malignant",
        "description": "Common form of skin cancer, locally destructive, low metastatic potential."
    },
    2: {
        "code": "bkl",
        "name": "Benign Keratosis-like Lesions",
        "type": "Benign",
        "description": "Solar lentigines, seborrheic keratoses, and lichen-planus-like keratoses."
    },
    3: {
        "code": "df",
        "name": "Dermatofibroma",
        "type": "Benign",
        "description": "Harmless fibrohistiocytic skin nodule often occurring on limbs."
    },
    4: {
        "code": "mel",
        "name": "Melanoma",
        "type": "Potentially Malignant",
        "description": "Aggressive malignant skin lesion arising from melanocytes requiring urgent excision."
    },
    5: {
        "code": "nv",
        "name": "Melanocytic Nevi",
        "type": "Benign",
        "description": "Common mole, benign proliferation of cutaneous melanocytes."
    },
    6: {
        "code": "vasc",
        "name": "Vascular Lesions",
        "type": "Benign",
        "description": "Angiomas, pyogenic granulomas, and cutaneous vascular malformations."
    }
}

CLASS_CODES = [HAM10000_CLASSES[i]["code"] for i in range(7)]
CLASS_NAMES = [HAM10000_CLASSES[i]["name"] for i in range(7)]


class SkinLesionDataset(Dataset):
    """
    Unified PyTorch Dataset for Skin Lesion Analysis.
    Yields:
      - 'image': (3, H, W) normalized tensor
      - 'label': int64 (0..6)
      - 'domain': int64 (0 = Source, 1 = Target)
      - 'mask': (1, H, W) binary float tensor (if available, else empty)
      - 'domain_name': str
    """
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        domain_id: int,
        domain_name: str,
        masks: Optional[torch.Tensor] = None
    ):
        self.images = images
        self.labels = labels
        self.domain_id = domain_id
        self.domain_name = domain_name
        self.masks = masks

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "image": self.images[idx],
            "label": self.labels[idx].long(),
            "domain": torch.tensor(self.domain_id, dtype=torch.long),
            "domain_name": self.domain_name
        }
        if self.masks is not None:
            item["mask"] = self.masks[idx]
        else:
            item["mask"] = torch.zeros(1, self.images.size(2), self.images.size(3))
        return item


def build_benchmark_suites(
    samples_per_class: int = 50,
    img_size: int = 64,
    save_sample_gallery: bool = True,
    sample_dir: str = "data/sample_data"
) -> Dict[str, object]:
    """
    Builds clean Source Domain, Shifted Target Domain, and Segmentation benchmark datasets.
    Also exports sample images for the interactive Streamlit gallery.
    """
    source_imgs_list, source_labels_list, source_masks_list = [], [], []
    target_imgs_list, target_labels_list, target_masks_list = [], [], []

    target_transform = DomainShiftTransform(domain_type="target", severity=1.5)

    for c in range(7):
        # Generate base synthetic dermoscopy with masks
        imgs, masks = generate_synthetic_dermoscopy(c, num_samples=samples_per_class, img_size=img_size, random_seed=c*37 + 10)
        
        # Split into Source (60%) and Target (40%) to ensure distinct physical distributions
        n_src = int(samples_per_class * 0.6)
        n_tgt = samples_per_class - n_src

        src_imgs = imgs[:n_src]
        src_masks = masks[:n_src]
        source_imgs_list.append(src_imgs)
        source_labels_list.append(torch.full((n_src,), c, dtype=torch.long))
        source_masks_list.append(src_masks)

        tgt_clean = imgs[n_src:]
        tgt_masks = masks[n_src:]
        # Apply domain shift transformations to target domain
        tgt_shifted = torch.stack([target_transform(im) for im in tgt_clean], dim=0)
        target_imgs_list.append(tgt_shifted)
        target_labels_list.append(torch.full((n_tgt,), c, dtype=torch.long))
        target_masks_list.append(tgt_masks)

    # Concatenate
    src_all_x = torch.cat(source_imgs_list, dim=0)
    src_all_y = torch.cat(source_labels_list, dim=0)
    src_all_m = torch.cat(source_masks_list, dim=0)

    tgt_all_x = torch.cat(target_imgs_list, dim=0)
    tgt_all_y = torch.cat(target_labels_list, dim=0)
    tgt_all_m = torch.cat(target_masks_list, dim=0)

    # Train/test split within source (80% train, 20% test)
    n_src_total = src_all_x.size(0)
    perm_src = torch.randperm(n_src_total)
    split_src = int(0.8 * n_src_total)
    
    src_train_idx = perm_src[:split_src]
    src_test_idx = perm_src[split_src:]

    src_train_ds = SkinLesionDataset(src_all_x[src_train_idx], src_all_y[src_train_idx], domain_id=0, domain_name="Source (Dermoscopy)", masks=src_all_m[src_train_idx])
    src_test_ds = SkinLesionDataset(src_all_x[src_test_idx], src_all_y[src_test_idx], domain_id=0, domain_name="Source (Dermoscopy)", masks=src_all_m[src_test_idx])

    # Unseen Target Domain (used for unsupervised adaptation and zero-shot cross-domain evaluation)
    n_tgt_total = tgt_all_x.size(0)
    perm_tgt = torch.randperm(n_tgt_total)
    split_tgt = int(0.5 * n_tgt_total)
    tgt_adapt_idx = perm_tgt[:split_tgt]  # Unlabeled during DANN training
    tgt_eval_idx = perm_tgt[split_tgt:]   # Strictly held-out for cross-domain evaluation

    tgt_adapt_ds = SkinLesionDataset(tgt_all_x[tgt_adapt_idx], tgt_all_y[tgt_adapt_idx], domain_id=1, domain_name="Target (Shifted Domain)")
    tgt_eval_ds = SkinLesionDataset(tgt_all_x[tgt_eval_idx], tgt_all_y[tgt_eval_idx], domain_id=1, domain_name="Target (Shifted Domain)", masks=tgt_all_m[tgt_eval_idx])

    # Distinct Segmentation Dataset (kept clearly separated per instructions)
    seg_imgs, seg_masks = generate_synthetic_dermoscopy(class_id=4, num_samples=80, img_size=img_size, random_seed=999)
    seg_dataset = SkinLesionDataset(seg_imgs, torch.full((80,), 4, dtype=torch.long), domain_id=0, domain_name="Segmentation Benchmark", masks=seg_masks)

    # Save benchmark sample images for the live Streamlit gallery if requested
    if save_sample_gallery and sample_dir:
        os.makedirs(os.path.join(sample_dir, "source"), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "target"), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "masks"), exist_ok=True)

        for i in range(min(14, len(src_test_ds))):
            s_img = src_test_ds.images[i]
            s_lbl = src_test_ds.labels[i].item()
            s_code = CLASS_CODES[s_lbl]
            save_image(s_img, os.path.join(sample_dir, "source", f"src_sample_{i}_{s_code}.png"))

        for i in range(min(14, len(tgt_eval_ds))):
            t_img = tgt_eval_ds.images[i]
            t_lbl = tgt_eval_ds.labels[i].item()
            t_code = CLASS_CODES[t_lbl]
            save_image(t_img, os.path.join(sample_dir, "target", f"tgt_sample_{i}_{t_code}.png"))

        for i in range(min(10, len(seg_dataset))):
            save_image(seg_dataset.images[i], os.path.join(sample_dir, "masks", f"seg_img_{i}.png"))
            save_image(seg_dataset.masks[i], os.path.join(sample_dir, "masks", f"seg_gt_mask_{i}.png"))

    return {
        "src_train": src_train_ds,
        "src_test": src_test_ds,
        "tgt_adapt": tgt_adapt_ds,
        "tgt_eval": tgt_eval_ds,
        "segmentation_suite": seg_dataset
    }
