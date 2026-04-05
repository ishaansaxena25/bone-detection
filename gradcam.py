"""
Grad-CAM implementation for BoneCancerNet.

Improvements over naive implementation:
- LANCZOS4 interpolation for smooth 7×7 → 224×224 upscaling
- Gaussian blur to remove blocky artefacts
- Percentile thresholding — only the top activations are highlighted
- JET colormap for maximum contrast on greyscale X-ray images
- Weighted blend that preserves original image detail
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage
    -----
    cam_gen = GradCAM(model)
    cam     = cam_gen.generate(img_tensor)   # (1, 3, 224, 224), no torch.no_grad()
    overlay = create_heatmap_overlay(original_pil, cam)
    """

    def __init__(self, model, target_layer=None):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Default: last residual block group of ResNet-50 backbone
        if target_layer is None:
            target_layer = model.backbone.layer4

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ─────────────────────────────────────────────────────────
    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    # ── Main method ────────────────────────────────────────────────────────────
    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Returns
        -------
        cam : (224, 224) float32 ndarray, values in [0, 1]
        """
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Grad-CAM: global-average-pool the gradients to get channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1)          # (1, H, W)
        cam     = F.relu(cam).squeeze().cpu().numpy()              # (H, W)  — e.g. 7×7

        # ── Post-processing ───────────────────────────────────────────────────

        # 1. Smooth upscaling: LANCZOS4 avoids the blocky 7×7 grid artefact
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # 2. Gaussian blur for organic-looking activation regions
        cam = cv2.GaussianBlur(cam, (11, 11), sigmaX=4, sigmaY=4)

        # 3. Percentile thresholding — suppress the bottom 40% of activations
        #    so only genuinely high-activation regions are highlighted
        thresh = float(np.percentile(cam, 40))
        cam    = np.maximum(cam - thresh, 0)

        # 4. Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)

        self._remove_hooks()
        return cam

    def _remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def create_heatmap_overlay(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.50,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap onto the original PIL image.

    Strategy
    --------
    - Convert X-ray to RGB numpy (it may be greyscale)
    - Apply JET colormap to the CAM
    - Alpha-blend: where activation is LOW the original image dominates,
      where activation is HIGH the heatmap dominates (adaptive blend)

    Parameters
    ----------
    pil_image : original PIL image (any size)
    cam       : (224, 224) ndarray from GradCAM.generate(), values in [0, 1]
    alpha     : maximum heatmap opacity (at highest activation)
    colormap  : OpenCV colormap constant

    Returns
    -------
    PIL.Image blended result
    """
    # Original image as float32 RGB
    img = np.array(pil_image.resize((224, 224))).astype(np.float32)
    if img.ndim == 2:                          # greyscale → RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:                    # RGBA → RGB
        img = img[:, :, :3]

    # Heatmap coloured with chosen colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Adaptive alpha: blend weight scales with local activation strength
    # → dim areas keep original detail; bright spots show heatmap clearly
    weight = cam[:, :, np.newaxis] * alpha     # (224, 224, 1), range [0, alpha]
    overlay = (1.0 - weight) * img + weight * heatmap
    overlay  = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
