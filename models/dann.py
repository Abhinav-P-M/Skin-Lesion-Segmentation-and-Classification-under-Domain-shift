"""
Domain-Adversarial Neural Network (DANN) for Cross-Dataset Skin Lesion Classification.
Paper: Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016).

Architecture:
                    Input Image
                         │
                         ▼
                 Feature Extractor G_f
                         │
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
         Label Classifier G_y    Gradient Reversal Layer (GRL)
                 │                │
                 ▼                ▼
           Disease Class      Domain Classifier G_d
           (HAM10000 7)           │
                                  ▼
                            Source / Target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import CNNFeatureExtractor
from models.gradient_reversal import GradientReversalLayer


class DomainClassifier(nn.Module):
    """
    Discriminator predicting whether features originate from Source (0) or Target (1) domain.
    """
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Binary domain output: 0 = Source, 1 = Target
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network.
    Jointly optimizes task-specific classification on source domain
    and domain invariance via adversarial gradient reversal on source + target data.
    """
    def __init__(self, num_classes: int = 7, feature_dim: int = 256):
        super().__init__()
        # Shared Feature Extractor
        self.feature_extractor = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        
        # Label Classifier (Task Head)
        self.class_classifier = nn.Linear(feature_dim, num_classes)
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambd=1.0)
        
        # Domain Classifier (Adversarial Head)
        self.domain_classifier = DomainClassifier(feature_dim=feature_dim)

    def forward(self, x: torch.Tensor, lambd: float = 1.0):
        features = self.feature_extractor(x)
        class_logits = self.class_classifier(features)

        # Apply Gradient Reversal
        self.grl.set_lambda(lambd)
        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)

        return class_logits, domain_logits, features

    def predict_class_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates disease category probabilities (used during test-time inference)"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            logits = self.class_classifier(features)
            return F.softmax(logits, dim=-1)

    def predict_domain_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates domain discrimination probabilities (Source vs Target)"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            logits = self.domain_classifier(features)
            return F.softmax(logits, dim=-1)
