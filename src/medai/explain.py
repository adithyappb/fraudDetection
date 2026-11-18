"""Grad-CAM explanations (MONAI) + quantitative localisation against radiologist boxes."""
from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps
from monai.visualize import GradCAM

from .models import CAM_LAYER, to_input


class Explainer:
    def __init__(self, model: torch.nn.Module, arch: str):
        self.model = model
        self.cam = GradCAM(nn_module=model, target_layers=CAM_LAYER[arch])

    def __call__(self, img01: np.ndarray, class_idx: int) -> np.ndarray:
        """img01 [H, W] in [0,1] -> CAM [H, W] in [0,1] (upsampled to the input size)."""
        self.model.eval()
        x = to_input(img01).requires_grad_(True)
        with torch.enable_grad():
            cam = self.cam(x=x, class_idx=class_idx)[0, 0].detach().numpy()
        cam = cam - cam.min()
        return cam / cam.max() if cam.max() > 0 else cam


def localisation(cam: np.ndarray, box: tuple, threshold: float = 0.5) -> dict:
    """box = (x, y, w, h) in CAM pixel coordinates.
    pointing game: is the CAM maximum inside the box?  IoU / IoBB: thresholded CAM vs box."""
    h, w = cam.shape
    x0, y0 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x1, y1 = int(min(w, np.ceil(box[0] + box[2]))), int(min(h, np.ceil(box[1] + box[3])))
    bmask = np.zeros_like(cam, bool)
    bmask[y0:y1, x0:x1] = True
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    hot = cam >= threshold
    inter = (hot & bmask).sum()
    return {"hit": bool(bmask[peak]), "iou": float(inter / max((hot | bmask).sum(), 1)),
            "iobb": float(inter / max(hot.sum(), 1))}


def overlay(img01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """-> uint8 RGB with the jet-coloured CAM blended where it is strong."""
    base = np.repeat(img01[..., None], 3, -1)
    heat = colormaps["jet"](cam)[..., :3]
    a = alpha * np.clip((cam - 0.2) / 0.8, 0, 1)[..., None]
    return (np.clip(base * (1 - a) + heat * a, 0, 1) * 255).astype(np.uint8)
