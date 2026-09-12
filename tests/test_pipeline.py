"""
Unit and Integration Tests for Skin Lesion DANN Domain Shift Pipeline.
Uses Python's standard library unittest (no external test runners required).
"""

import unittest
import torch
import numpy as np

from models.unet import UNet
from models.cnn import BaselineCNN
from models.dann import DANN
from models.gradient_reversal import GradientReversalLayer, calc_lambda
from data.domain_shift import DomainShiftTransform, generate_synthetic_dermoscopy
from evaluation.segmentation_metrics import compute_dice, compute_iou
from evaluation.classification_metrics import compute_classification_metrics
from utils.gradcam import GradCAM


class TestPipeline(unittest.TestCase):

    def test_unet_forward(self):
        x = torch.rand(2, 3, 64, 64)
        unet = UNet(in_channels=3, out_channels=1, base_channels=16)
        out = unet(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))
        self.assertTrue(out.min() >= 0.0)
        self.assertTrue(out.max() <= 1.0)
        binary_mask = unet.predict_mask(x, threshold=0.5)
        self.assertEqual(binary_mask.shape, (2, 1, 64, 64))

    def test_cnn_forward(self):
        x = torch.rand(3, 3, 64, 64)
        cnn = BaselineCNN(num_classes=7, feature_dim=128)
        logits = cnn(x)
        self.assertEqual(logits.shape, (3, 7))
        probs = cnn.predict_proba(x)
        self.assertEqual(probs.shape, (3, 7))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-4))

    def test_dann_and_grl(self):
        x = torch.rand(4, 3, 64, 64, requires_grad=True)
        dann = DANN(num_classes=7, feature_dim=128)
        class_logits, domain_logits, feat = dann(x, lambd=0.75)
        self.assertEqual(class_logits.shape, (4, 7))
        self.assertEqual(domain_logits.shape, (4, 2))

        # Check gradient reversal
        loss = domain_logits.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

        # Dynamic lambda scheduling test
        l_start = calc_lambda(0, 100)
        l_end = calc_lambda(100, 100)
        self.assertAlmostEqual(l_start, 0.0, places=2)
        self.assertGreater(l_end, 0.9)

    def test_domain_transforms(self):
        img = torch.rand(3, 64, 64)
        t_src = DomainShiftTransform(domain_type="source")
        t_tgt = DomainShiftTransform(domain_type="target", severity=1.5)

        out_src = t_src(img)
        out_tgt = t_tgt(img)

        self.assertEqual(out_src.shape, (3, 64, 64))
        self.assertEqual(out_tgt.shape, (3, 64, 64))
        self.assertTrue(out_tgt.min() >= 0.0)
        self.assertTrue(out_tgt.max() <= 1.0)

    def test_segmentation_metrics(self):
        m1 = torch.zeros(1, 64, 64)
        m2 = torch.zeros(1, 64, 64)
        m1[:, 10:30, 10:30] = 1.0
        m2[:, 10:30, 10:30] = 1.0
        dice = compute_dice(m1, m2)
        iou = compute_iou(m1, m2)
        self.assertAlmostEqual(dice, 1.0, places=3)
        self.assertAlmostEqual(iou, 1.0, places=3)

    def test_classification_metrics(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_probs = np.eye(7)
        res = compute_classification_metrics(y_true, y_probs, num_classes=7)
        self.assertAlmostEqual(res["accuracy"], 1.0, places=3)
        self.assertAlmostEqual(res["macro_auroc"], 1.0, places=3)

    def test_gradcam(self):
        cnn = BaselineCNN(num_classes=7, feature_dim=128)
        gradcam = GradCAM(cnn, cnn.feature_extractor.layer4, is_dann=False)
        x = torch.rand(1, 3, 64, 64)
        heatmap, overlay = gradcam.generate_heatmap(x, target_class=4)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertEqual(overlay.shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
