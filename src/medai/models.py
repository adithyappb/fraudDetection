"""Model builders + the input transform shared by training, evaluation and the service."""
from __future__ import annotations

import logging

import numpy as np
import torch
from monai.networks.nets import UNet
from torchvision import models as tvm

log = logging.getLogger(__name__)
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226            # grayscale equivalent of ImageNet stats
CAM_LAYER = {"efficientnet_b0": "features.8", "densenet121": "features.norm5"}


def build_classifier(arch: str, n_classes: int, pretrained: str = "none") -> tuple[torch.nn.Module, str]:
    """Transfer learning: ImageNet backbone, new multi-label head. Returns (model, weights actually used)."""
    weights_used = "none"
    if arch == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        try:
            m = tvm.efficientnet_b0(weights=w)
            weights_used = "imagenet" if w else "none"
        except Exception as e:  # offline: no access to download.pytorch.org
            log.warning("ImageNet weights unavailable (%s) -> training from scratch", type(e).__name__)
            m = tvm.efficientnet_b0(weights=None)
        m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(m.classifier[1].in_features, n_classes))
    elif arch == "densenet121":                          # CheXNet architecture
        w = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        try:
            m = tvm.densenet121(weights=w)
            weights_used = "imagenet" if w else "none"
        except Exception as e:
            log.warning("ImageNet weights unavailable (%s) -> training from scratch", type(e).__name__)
            m = tvm.densenet121(weights=None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, n_classes)
    else:
        raise ValueError(arch)
    return m, weights_used


def to_input(img01: np.ndarray | torch.Tensor) -> torch.Tensor:
    """[H, W] or [B, H, W] float in [0,1] -> normalised [B, 3, H, W] tensor."""
    t = torch.as_tensor(img01, dtype=torch.float32)
    if t.ndim == 2:
        t = t[None]
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t[:, None].expand(-1, 3, -1, -1).contiguous()


def build_unet(channels) -> torch.nn.Module:
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=tuple(channels),
                strides=tuple([2] * (len(channels) - 1)), num_res_units=2, norm="batch")
