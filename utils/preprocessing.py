"""
Clinically Sound Preprocessing and Data Augmentations for Skin Lesion Images.
Preserves lesion border morphology and diagnostic pigment networks.
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


# Standard dermoscopy normalization
DERM_MEAN = [0.763, 0.545, 0.570]
DERM_STD = [0.140, 0.152, 0.169]


def get_inference_transforms(img_size: int = 64) -> T.Compose:
    """Standard evaluation and test-time inference transform."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])


def get_training_transforms(img_size: int = 64) -> T.Compose:
    """
    Mild, medically realistic augmentations:
    - Random horizontal and vertical flips (dermoscopy orientation is arbitrary)
    - Small rotation (up to 15 degrees)
    - Mild color jitter (avoids unrealistic distortion of diagnostic colors)
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor()
    ])


def preprocess_pil_image(image: Image.Image, img_size: int = 64) -> torch.Tensor:
    """Preprocesses a PIL Image into a (1, 3, H, W) PyTorch Tensor."""
    rgb = image.convert("RGB")
    transform = get_inference_transforms(img_size=img_size)
    tensor = transform(rgb).unsqueeze(0)  # (1, 3, H, W)
    return tensor
