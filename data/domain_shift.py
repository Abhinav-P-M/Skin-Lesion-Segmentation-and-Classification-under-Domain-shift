"""
Domain Shift Simulator and Dataset Constructor for Cross-Dataset Skin Lesion Analysis.

Source Domain:
  - Standardized Contact Dermoscopy (ISIC / HAM10000 style)
  - Calibrated polarized lighting, high-contrast, uniform illumination field.

Target Domain:
  - Unseen Shifted Domain (Smartphone / Clinical Photography / Non-polarized style)
  - Severe color temperature shifts (warm halogen / yellow tint or cold blue tint)
  - Optical vignetting, uneven illumination, camera sensor noise, and resolution drift.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Dict


class DomainShiftTransform:
    """
    Applies realistic domain shifts representing clinical acquisition differences.
    """
    def __init__(self, domain_type: str = "source", severity: float = 1.0):
        self.domain_type = domain_type.lower()
        self.severity = max(0.1, min(severity, 3.0))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (3, H, W) normalized to [0, 1].
        Returns:
            Domain-shifted image tensor of shape (3, H, W).
        """
        if self.domain_type == "source":
            # Source: Clean, standardized dermoscopy
            return img

        out = img.clone()
        _, H, W = out.shape

        # 1. Color Temperature / White-balance Shift (common across camera brands)
        # Shift towards warm yellow/tungsten or clinical fluorescent green-blue
        r_scale = 1.0 + 0.25 * self.severity
        b_scale = max(0.4, 1.0 - 0.35 * self.severity)
        out[0] = out[0] * r_scale
        out[2] = out[2] * b_scale

        # 2. Vignetting (optical intensity falloff from smartphone / non-telecentric lens)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        dist_from_center = torch.sqrt(x**2 + y**2)
        vignette_mask = 1.0 - 0.35 * (dist_from_center / np.sqrt(2.0)) * self.severity
        vignette_mask = torch.clamp(vignette_mask, 0.45, 1.0)
        out = out * vignette_mask.unsqueeze(0)

        # 3. Dynamic Range & Contrast Shift (different sensor dynamic range)
        out = TF.adjust_contrast(out, 0.8)
        out = TF.adjust_brightness(out, 0.95)

        # 4. Sensor ISO Noise (smartphone / low-cost clinical imaging)
        noise = torch.randn_like(out) * (0.04 * self.severity)
        out = torch.clamp(out + noise, 0.0, 1.0)

        return torch.clamp(out, 0.0, 1.0)


def generate_synthetic_dermoscopy(
    class_id: int,
    num_samples: int,
    img_size: int = 64,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates realistic dermoscopic patterns with matching ground-truth segmentation masks.
    Returns:
        images: (num_samples, 3, H, W) in [0, 1]
        masks:  (num_samples, 1, H, W) in {0, 1}
    """
    np.random.seed(random_seed + class_id * 100)
    torch.manual_seed(random_seed + class_id * 100)

    images = []
    masks = []

    y_grid, x_grid = np.mgrid[-1:1:complex(0, img_size), -1:1:complex(0, img_size)]

    for i in range(num_samples):
        # Lesion centroid with slight random jitter
        cx = np.random.uniform(-0.15, 0.15)
        cy = np.random.uniform(-0.15, 0.15)

        # Lesion radius & eccentricity based on malignancy category
        # Malignant categories (mel, bcc, akiec) have irregular/asymmetric borders
        is_malignant = class_id in [0, 1, 4]  # akiec, bcc, mel
        rx = np.random.uniform(0.35, 0.55)
        ry = rx * np.random.uniform(0.65, 1.0)
        angle = np.random.uniform(0, np.pi)

        # Rotated coordinates
        x_rot = (x_grid - cx) * np.cos(angle) + (y_grid - cy) * np.sin(angle)
        y_rot = -(x_grid - cx) * np.sin(angle) + (y_grid - cy) * np.cos(angle)

        # Border perturbation (asymmetry / irregularity)
        theta = np.arctan2(y_rot, x_rot)
        border_distortion = 1.0
        if is_malignant:
            # Multi-frequency perturbation for irregular border
            border_distortion += 0.18 * np.sin(3 * theta) + 0.12 * np.cos(5 * theta)
        else:
            border_distortion += 0.05 * np.sin(2 * theta)

        dist = np.sqrt((x_rot / rx)**2 + (y_rot / ry)**2) / border_distortion
        mask_np = (dist <= 1.0).astype(np.float32)

        # Base skin tone (Fitzpatrick II-IV dermoscopic background)
        skin_r = np.random.uniform(0.80, 0.90)
        skin_g = np.random.uniform(0.65, 0.75)
        skin_b = np.random.uniform(0.55, 0.65)

        img_np = np.zeros((3, img_size, img_size), dtype=np.float32)
        img_np[0, :, :] = skin_r + np.random.randn(img_size, img_size) * 0.02
        img_np[1, :, :] = skin_g + np.random.randn(img_size, img_size) * 0.02
        img_np[2, :, :] = skin_b + np.random.randn(img_size, img_size) * 0.02

        # Pigment colors depending on diagnosis
        if class_id == 4:  # Melanoma (dark brown/black, variegated colors)
            lesion_r = np.random.uniform(0.20, 0.35)
            lesion_g = np.random.uniform(0.12, 0.22)
            lesion_b = np.random.uniform(0.10, 0.20)
        elif class_id == 1:  # BCC (translucent pink/red, arborizing vessels)
            lesion_r = np.random.uniform(0.65, 0.78)
            lesion_g = np.random.uniform(0.30, 0.42)
            lesion_b = np.random.uniform(0.32, 0.44)
        elif class_id == 5:  # Nevus (uniform tan/brown, regular pigment network)
            lesion_r = np.random.uniform(0.40, 0.52)
            lesion_g = np.random.uniform(0.28, 0.38)
            lesion_b = np.random.uniform(0.20, 0.30)
        elif class_id == 6:  # Vascular (red to purple lacunae)
            lesion_r = np.random.uniform(0.68, 0.85)
            lesion_g = np.random.uniform(0.10, 0.22)
            lesion_b = np.random.uniform(0.15, 0.28)
        else:  # akiec, bkl, df (keratotic scales, yellow-brown)
            lesion_r = np.random.uniform(0.50, 0.65)
            lesion_g = np.random.uniform(0.40, 0.52)
            lesion_b = np.random.uniform(0.25, 0.35)

        # Blend lesion into skin using smooth boundary
        smooth_mask = np.clip(1.0 - (dist - 0.8) / 0.3, 0.0, 1.0) * mask_np

        img_np[0] = img_np[0] * (1.0 - smooth_mask) + lesion_r * smooth_mask
        img_np[1] = img_np[1] * (1.0 - smooth_mask) + lesion_g * smooth_mask
        img_np[2] = img_np[2] * (1.0 - smooth_mask) + lesion_b * smooth_mask

        img_t = torch.from_numpy(np.clip(img_np, 0.0, 1.0))
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)

        images.append(img_t)
        masks.append(mask_t)

    return torch.stack(images, dim=0), torch.stack(masks, dim=0)
