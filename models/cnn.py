"""
Baseline Conventional CNN Classifier for Skin Lesion Diagnosis.
Maps dermoscopic images to 7 HAM10000 disease categories without domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNFeatureExtractor(nn.Module):
    """
    Hierarchical convolutional feature extractor.
    Last conv layer 'layer4' serves as target for Grad-CAM explainability.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.layer1 = ConvBlock(in_channels, 32)   # (B, 32, H/2, W/2)
        self.layer2 = ConvBlock(32, 64)            # (B, 64, H/4, W/4)
        self.layer3 = ConvBlock(64, 128)           # (B, 128, H/8, W/8)
        self.layer4 = nn.Sequential(               # (B, 256, H/16, W/16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        feat = self.fc_proj(x)
        return feat


class BaselineCNN(nn.Module):
    """
    Conventional CNN Classifier without domain adaptation.
    Pipeline: Input -> Preprocessing -> CNN Feature Extractor -> Fully Connected Layer -> Disease Class.
    """
    def __init__(self, num_classes: int = 7, feature_dim: int = 256):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities across disease categories"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
