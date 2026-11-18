"""Model inference used by the service: chest X-ray classification + Grad-CAM, 3D spleen segmentation."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted, Compose, Invertd
from PIL import Image

from .config import MODELS, params
from .data.dicom_io import ct_volume, cxr_pixels
from .explain import Explainer
from .models import build_unet, to_input
from .train_seg import base_transforms


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


class Predictor:
    def __init__(self):
        p = params()
        from .evaluate import load_classifier
        self.cls, ck = load_classifier()
        self.findings, self.thresholds, self.size = ck["findings"], ck["thresholds"], ck["image_size"]
        self.explainer = Explainer(self.cls, ck["arch"])
        sk = torch.load(MODELS / "spleen_unet3d.pt", weights_only=False)
        self.seg_cfg = p["segmentation"]
        self.unet = build_unet(self.seg_cfg["channels"])
        self.unet.load_state_dict(sk["state_dict"])
        self.unet.eval()
        self.version = {"classifier": f"{ck['arch']}@{_sha(MODELS / 'cxr_classifier.pt')}",
                        "segmenter": f"unet3d@{_sha(MODELS / 'spleen_unet3d.pt')}"}

    # ------------------------------------------------------------------ chest X-ray
    def analyze_cxr(self, ds) -> dict:
        full = cxr_pixels(ds)
        small = np.asarray(Image.fromarray((full * 255).astype(np.uint8)).resize((self.size, self.size), Image.BOX),
                           np.float32) / 255
        with torch.no_grad():
            prob = torch.sigmoid(self.cls(to_input(small)))[0].numpy()
        cams = np.stack([self.explainer(small, k) for k in range(len(self.findings))]).astype(np.float16)
        findings = []
        for k, f in enumerate(self.findings):
            peak = np.unravel_index(int(np.argmax(cams[k])), cams[k].shape)
            findings.append({"finding": f, "probability": round(float(prob[k]), 4),
                             "threshold": round(self.thresholds[f], 4), "positive": bool(prob[k] >= self.thresholds[f]),
                             "cam_peak": [round(peak[1] / self.size, 3), round(peak[0] / self.size, 3)]})
        findings.sort(key=lambda d: -d["probability"])
        positives = [d["finding"] for d in findings if d["positive"]]
        return {"kind": "cxr", "findings": findings, "positives": positives,
                "summary": ", ".join(positives) if positives else "No target finding above threshold",
                "image": (full * 255).astype(np.uint8), "cams": cams}

    # ------------------------------------------------------------------ CT spleen
    def analyze_ct(self, slices) -> dict:
        vol, spacing = ct_volume(slices)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ct.nii.gz"
            nib.save(nib.Nifti1Image(vol.astype(np.float32), np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])), f)
            pre = Compose(base_transforms(self.seg_cfg, keys=("image",)))
            d = pre({"image": str(f)})
            with torch.no_grad():
                logits = sliding_window_inference(d["image"][None], self.seg_cfg["roi"], 4, self.unet, overlap=0.25)[0]
            d["pred"] = torch.softmax(logits, 0)
            post = Compose([Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False),
                            AsDiscreted(keys="pred", argmax=False)])
            d = post(d)
            prob = d["pred"][1].numpy()                                     # back on the original voxel grid (x, y, z)
        mask = prob > 0.5
        voxel_ml = spacing[0] * spacing[1] * spacing[2] / 1000
        zs = np.flatnonzero(mask.any((0, 1)))
        conf = float(prob[mask].mean()) if mask.any() else 0.0
        win = np.clip((vol + 160) / 400, 0, 1)                              # W400 / L40 soft-tissue window
        return {"kind": "ct", "spleen_volume_ml": round(float(mask.sum() * voxel_ml), 1),
                "spleen_slices": [int(zs.min()), int(zs.max())] if len(zs) else None,
                "mean_confidence": round(conf, 4), "slices": int(vol.shape[2]),
                "summary": f"Spleen segmented: {mask.sum() * voxel_ml:.0f} mL" if mask.any() else "No spleen found",
                "volume": (win * 255).astype(np.uint8), "mask": mask.astype(np.uint8), "prob": prob.astype(np.float16)}
