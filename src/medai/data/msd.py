"""Medical Segmentation Decathlon Task09_Spleen (portal-venous CT, spleen label).

Real data: data/raw/Task09_Spleen/{imagesTr,labelsTr}/*.nii.gz + dataset.json
Phantom fallback: abdominal CT phantoms in Hounsfield units with the same layout, orientation
and spacing conventions (liver, spleen, kidneys, aorta, spine, fat, bowel gas, noise).
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

SHAPE = (160, 160, 56)            # x, y, z voxels
SPACING = (1.6, 1.6, 4.0)         # mm, typical of MSD spleen after acquisition


def _blob(shape, c, r, rng, rough=0.18):
    """Ellipsoid deformed by smooth noise -> organic organ shape."""
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape[::-1]], indexing="ij")
    xx, yy, zz = xx.transpose(2, 1, 0), yy.transpose(2, 1, 0), zz.transpose(2, 1, 0)
    d = ((xx - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((zz - c[2]) / r[2]) ** 2
    noise = ndi.gaussian_filter(rng.normal(size=shape), 6) * 8
    return d < 1 + rough * noise


def render_ct(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X, Y, Z = SHAPE
    hu = np.full(SHAPE, -1000.0, np.float32)
    body = _blob(SHAPE, (X / 2, Y / 2 + 4, Z / 2), (X * 0.44, Y * 0.33, Z * 4), rng, 0.05)
    hu[body] = -90                                                     # subcutaneous fat
    inner = _blob(SHAPE, (X / 2, Y / 2 + 4, Z / 2), (X * 0.39, Y * 0.28, Z * 4), rng, 0.05)
    hu[inner] = 35                                                      # muscle / soft tissue
    sc = rng.uniform(0.85, 1.15)
    liver = _blob(SHAPE, (X * 0.33, Y * 0.48, Z * 0.62), (X * 0.17 * sc, Y * 0.16, Z * 0.42), rng) & inner
    hu[liver] = rng.normal(105, 8)
    for side in (0.3, 0.7):                                             # kidneys
        k = _blob(SHAPE, (X * side, Y * 0.66, Z * 0.35), (X * 0.05, Y * 0.06, Z * 0.2), rng, 0.08) & inner
        hu[k] = 150
    aorta = _blob(SHAPE, (X * 0.48, Y * 0.62, Z / 2), (X * 0.03, Y * 0.03, Z * 3), rng, 0.02) & inner
    hu[aorta] = 170
    spine = _blob(SHAPE, (X * 0.5, Y * 0.74, Z / 2), (X * 0.06, Y * 0.06, Z * 3), rng, 0.03) & body
    hu[spine] = 420
    gas = (ndi.gaussian_filter(rng.normal(size=SHAPE), 3) > 0.12) & _blob(
        SHAPE, (X * 0.52, Y * 0.4, Z * 0.45), (X * 0.15, Y * 0.12, Z * 0.35), rng) & inner & ~liver
    hu[gas] = -850
    # spleen: left upper quadrant (image right in radiological convention), variable size
    ss = rng.uniform(0.7, 1.35)
    spleen = _blob(SHAPE, (X * rng.uniform(0.66, 0.74), Y * rng.uniform(0.5, 0.58), Z * rng.uniform(0.55, 0.7)),
                   (X * 0.07 * ss, Y * 0.11 * ss, Z * 0.18 * ss), rng, 0.22) & inner & ~liver
    hu[spleen] = rng.normal(118, 8)                                     # enhances slightly more than liver
    hu = ndi.gaussian_filter(hu, (0.8, 0.8, 0.5)) + rng.normal(0, 14, SHAPE)
    return hu.astype(np.int16), spleen.astype(np.uint8)


def make_phantoms(out_dir: Path, n: int, seed: int) -> None:
    (out_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    affine = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])       # RAS like MSD NIfTI
    training = []
    for i in range(n):
        img, lab = render_ct(np.random.default_rng(seed * 1000 + i))
        name = f"spleen_{i + 1:03d}.nii.gz"
        nib.save(nib.Nifti1Image(img, affine), out_dir / "imagesTr" / name)
        nib.save(nib.Nifti1Image(lab, affine), out_dir / "labelsTr" / name)
        training.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})
    json.dump({"name": "Spleen (phantom)", "modality": {"0": "CT"}, "labels": {"0": "background", "1": "spleen"},
               "numTraining": n, "training": training, "phantom": True}, open(out_dir / "dataset.json", "w"), indent=2)


def list_pairs(root: Path) -> list[dict]:
    meta = json.load(open(root / "dataset.json"))
    return [{"image": str(root / t["image"]), "label": str(root / t["label"])} for t in meta["training"]]
