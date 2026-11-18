"""Grad-CAM explanations + quantitative localisation against radiologist boxes.

Grad-CAM (Selvaraju et al., 2017): weight each feature map of a convolutional layer by the spatially
averaged gradient of the class logit, sum, ReLU, upsample. The layer is a stride-16 block (10 x 10 map
at 160 px input): the last block (5 x 5) is too coarse for small findings such as nodules - measured
pointing-game accuracy on the radiologist boxes is 0.31 vs 0.14 for the last block (chance 0.07).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps

from .models import to_input

CAM_LAYER = {"efficientnet_b0": "features.4", "densenet121": "features.denseblock3"}   # stride-16 blocks


class Explainer:
    def __init__(self, model: torch.nn.Module, arch: str):
        self.model = model
        layer = dict(model.named_modules())[CAM_LAYER[arch]]
        self._store = {}
        layer.register_forward_hook(lambda m, i, o: self._store.__setitem__("act", o))
        layer.register_full_backward_hook(lambda m, gi, go: self._store.__setitem__("grad", go[0]))

    def __call__(self, img01: np.ndarray, class_idx: int) -> np.ndarray:
        """img01 [H, W] in [0,1] -> CAM [H, W] in [0,1] (upsampled to the input size)."""
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            logit = self.model(to_input(img01).requires_grad_(True))[0, class_idx]
            logit.backward()
        act, grad = self._store["act"].detach(), self._store["grad"].detach()
        cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=img01.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
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
