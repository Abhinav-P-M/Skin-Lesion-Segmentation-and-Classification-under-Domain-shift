"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Model Explainability.
Paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017).
Allows verifying whether CNN/DANN attends to the actual lesion vs background/lighting artifacts.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from typing import Tuple, Optional


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, is_dann: bool = False):
        self.model = model
        self.target_layer = target_layer
        self.is_dann = is_dann

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates Grad-CAM heatmap and blended overlay.
        Args:
            input_tensor: (1, 3, H, W)
            target_class: optional class index. If None, uses top predicted class.
        Returns:
            heatmap: (H, W) float array in [0, 1]
            overlay: (H, W, 3) uint8 image with jet colormap blended
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        if self.is_dann:
            class_logits, _, _ = self.model(input_tensor)
        else:
            class_logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(class_logits, dim=-1).item()

        # Target score for backprop
        score = class_logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        grads = self.gradients[0]  # (C, H_feat, W_feat)
        acts = self.activations[0]  # (C, H_feat, W_feat)

        weights = torch.mean(grads, dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=0)  # (H_feat, W_feat)
        cam = F.relu(cam)  # Only features that positively contribute

        # Normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-6:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        # Upsample heatmap to input image size
        _, _, H, W = input_tensor.shape
        heatmap_resized = F.interpolate(
            torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        # Convert original input to uint8 RGB
        img_np = input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        # Apply colormap (jet / turbo)
        import matplotlib.pyplot as plt
        cmap = plt.colormaps["jet"]
        colored_cam = cmap(heatmap_resized)[:, :, :3]  # drop alpha, shape (H, W, 3)
        colored_cam = (colored_cam * 255.0).astype(np.uint8)

        # Alpha blend with original image
        alpha = 0.5
        overlay = (img_np * (1.0 - alpha) + colored_cam * alpha).astype(np.uint8)

        return heatmap_resized, overlay
