"""NIH ChestX-ray14 (classification + bounding boxes).

Real data: data/raw/nih-cxr14/{images/*.png, Data_Entry_2017*.csv, BBox_List_2017.csv}
(see README "Get the data").

Phantom fallback: when the 42 GB of images are not available, `make_phantoms` renders a
radiograph-like image for each of N *real* NIH metadata rows (real labels and co-occurrence,
real patient IDs, age, sex and view position). The files and CSVs use exactly the NIH layout,
so every downstream step is identical. Rendering uses:
  * 500+ real expert lung masks (v7labs, shipped in torchxrayvision) for lung shape,
  * the real radiologist bounding boxes (BBox_List_2017) to place findings where NIH annotated them,
  * known radiographic physics/confounders: AP portable films magnify the heart (~15%) and are
    noisier/lower contrast - the same confounder that biases models trained on real NIH data.
"""
from __future__ import annotations

import io
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from ..config import params, path

ASSETS = Path(__file__).parent / "assets"
S = 320                       # phantom resolution (real NIH PNGs are 1024 x 1024)
NIH_SIZE = 1024


# ----------------------------------------------------------------------------- metadata
def read_metadata(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = raw_dir or path(params()["data"]["cxr_raw_dir"])
    entries = sorted(raw_dir.glob("Data_Entry_2017*.csv"))
    meta = pd.read_csv(entries[0] if entries else ASSETS / "Data_Entry_2017_v2020.csv.gz")
    meta = meta.rename(columns={"Image Index": "image", "Finding Labels": "labels", "Patient ID": "patient",
                                "Patient Age": "age", "Patient Gender": "sex", "View Position": "view"})
    bb_file = raw_dir / "BBox_List_2017.csv"
    bb = pd.read_csv(bb_file if bb_file.exists() else ASSETS / "BBox_List_2017.csv.gz")
    bb = bb.iloc[:, :6]
    bb.columns = ["image", "finding", "x", "y", "w", "h"]
    bb["finding"] = bb.finding.replace({"Infiltrate": "Infiltration"})
    return meta[["image", "labels", "patient", "age", "sex", "view"]], bb


def label_matrix(meta: pd.DataFrame, findings: list[str]) -> np.ndarray:
    return np.stack([meta.labels.str.contains(f, regex=False).to_numpy() for f in findings], 1).astype(np.float32)


# ----------------------------------------------------------------------------- lung masks
def load_lung_masks() -> list[np.ndarray]:
    """Real expert lung masks -> cropped to the lung field, keep clean two-lung masks."""
    masks = []
    with zipfile.ZipFile(ASSETS / "semantic_masks_v7labs_lungs.zip") as z:
        for name in sorted(z.namelist()):
            if name.endswith("/"):
                continue
            m = np.array(Image.open(io.BytesIO(z.read(name))).convert("L")) > 127
            if not 0.15 < m.mean() < 0.6:
                continue
            lab, n = ndi.label(m)
            if n < 2:
                continue
            sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
            if sizes[1] < 0.4 * sizes[0]:                       # need two comparable lungs
                continue
            keep = np.isin(lab, np.argsort(np.bincount(lab.ravel()))[::-1][1:3])
            ys, xs = np.nonzero(keep)
            crop = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = crop.shape
            if 0.6 < h / w < 1.3:
                masks.append(crop)
    return masks


# ----------------------------------------------------------------------------- rendering
def _ellipse(yy, xx, cy, cx, ry, rx, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    y, x = yy - cy, xx - cx
    return ((x * c + y * s) / rx) ** 2 + ((-x * s + y * c) / ry) ** 2 <= 1


def _soft(mask, sigma):
    return ndi.gaussian_filter(mask.astype(np.float32), sigma)


def _sample_in(mask, rng, rows=None):
    ys, xs = np.nonzero(mask)
    if rows is not None:
        sel = (ys >= rows[0]) & (ys <= rows[1])
        if sel.any():
            ys, xs = ys[sel], xs[sel]
    i = rng.integers(len(ys))
    return int(ys[i]), int(xs[i])


def render_cxr(rng: np.random.Generator, findings: set[str], view: str, sex: str, lung_mask: np.ndarray,
               boxes: dict[str, tuple] | None = None) -> tuple[np.ndarray, dict[str, tuple]]:
    """Returns (uint8 S x S image, {finding: (x, y, w, h) in NIH 1024 coordinates})."""
    boxes = boxes or {}
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ap = view == "AP"
    truth = {}

    # torso + soft tissue
    torso = _ellipse(yy, xx, S * 0.62, S * 0.5, S * 0.62, S * 0.47)
    img = np.where(torso, 0.40 + 0.12 * (yy / S), 0.04).astype(np.float32)
    img += 0.05 * _soft(_ellipse(yy, xx, S * 0.12, S * 0.5, S * 0.15, S * 0.12), 6)  # neck

    # lungs from a real mask
    lw, lh = int(S * rng.uniform(0.66, 0.76)), int(S * rng.uniform(0.56, 0.66))
    lm = np.array(Image.fromarray(lung_mask.astype(np.uint8) * 255).resize((lw, lh), Image.BILINEAR)) > 127
    lungs = np.zeros((S, S), bool)
    top, left = int(S * rng.uniform(0.12, 0.17)) + (8 if ap else 0), (S - lw) // 2 + int(rng.integers(-6, 7))
    lungs[top:top + lh, left:left + lw] = lm
    lung_soft = _soft(lungs, 2.0)
    img -= 0.30 * lung_soft
    lung_bottom = top + lh

    # vascular markings, denser near the hila
    hil_y, hil_x = top + lh * 0.45, S * 0.5
    dist = np.hypot((yy - hil_y) / S, (xx - hil_x) / S)
    marks = np.abs(ndi.gaussian_filter(rng.normal(size=(S, S)), 1.6))
    marks = (marks > np.percentile(marks, 82)) * np.clip(1.1 - 2.2 * dist, 0.15, 1)
    markings = _soft(marks, 0.8) * lung_soft

    # heart (cardiothoracic ratio), mediastinum, spine
    ctr = rng.uniform(0.40, 0.48)
    if "Cardiomegaly" in findings:
        ctr = rng.uniform(0.56, 0.66) if rng.random() > 0.15 else rng.uniform(0.51, 0.54)
    magnification = 1.15 if ap else 1.0                                 # AP projection magnifies the heart
    hw = ctr * lw * magnification / 2
    hcy, hcx = top + lh * rng.uniform(0.66, 0.74), S * 0.5 + hw * 0.22
    heart = _ellipse(yy, xx, hcy, hcx, hw * 0.72, hw, angle=rng.uniform(-0.35, -0.1)) & (yy < lung_bottom + 4)
    img += 0.22 * _soft(heart, 4)
    img += 0.15 * _soft((np.abs(xx - S * 0.5) < S * 0.055) & (yy > S * 0.06) & (yy < hcy), 4)
    img += 0.04 * _soft(np.abs(xx - S * 0.5) < S * 0.03, 2) * (0.7 + 0.3 * np.sin(yy / 5.0))
    if "Cardiomegaly" in findings:
        truth["Cardiomegaly"] = (hcx - hw, hcy - hw * 0.68, 2 * hw, 1.36 * hw)

    # ribs + clavicles
    ribs = np.zeros((S, S), np.float32)
    dx = np.abs(xx - S * 0.5)
    for k in range(10):                                   # posterior ribs: rise from the spine, sweep down laterally
        y0 = top - 6 + k * lh / 9.0
        curve = y0 - 0.25 * dx + 0.0042 * dx ** 2
        ribs += ((np.abs(yy - curve) < 2.2) & (dx > S * 0.04) & (dx < lw * 0.53)).astype(np.float32)
    for side in (-1, 1):                                  # clavicles
        clav = top + 2 + 0.22 * (S * 0.5 - xx) * side * -1
        ribs += ((np.abs(yy - clav) < 3) & (np.abs(xx - (S * 0.5 + side * S * 0.17)) < S * 0.13)).astype(np.float32)
    img += 0.06 * _soft(ribs > 0, 1.0) * torso * (yy < lung_bottom + 6)
    if sex == "F":                                                       # breast shadows over lower zones
        for side in (-1, 1):
            img += 0.06 * _soft(_ellipse(yy, xx, top + lh * 0.85, S * 0.5 + side * lw * 0.33, S * 0.12, S * 0.15), 8)

    def box_or(f, default_mask, rows=None, size=(0.10, 0.10)):
        if f in boxes:
            x, y, w, h = (v * S / NIH_SIZE for v in boxes[f])
            return y + h / 2, x + w / 2, max(h, 6), max(w, 6)
        cy, cx = _sample_in(default_mask, rng, rows)
        return cy, cx, S * size[0], S * size[1]

    strength = lambda: 1.0 if rng.random() > 0.15 else 0.45   # noqa: E731  - some findings are subtle
    path_layer = np.zeros((S, S), np.float32)

    if "Effusion" in findings:
        cy, cx, h, w = box_or("Effusion", lungs, (lung_bottom - lh * 0.25, lung_bottom))
        side_mask = lungs & ((xx < S * 0.5) if cx < S * 0.5 else (xx >= S * 0.5))
        level = lung_bottom - rng.uniform(0.12, 0.30) * lh
        meniscus = level - 0.25 * np.abs(xx - S * 0.5)                     # rises laterally
        fluid = side_mask & (yy > meniscus)
        path_layer += 0.32 * strength() * _soft(fluid, 2.5)
        ys, xs = np.nonzero(fluid)
        if len(ys):
            truth["Effusion"] = (xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1)
    if "Atelectasis" in findings:
        cy, cx, h, w = box_or("Atelectasis", lungs, (top + lh * 0.55, lung_bottom - 5))
        band = _ellipse(yy, xx, cy, cx, max(h, w) * 0.18, max(h, w) * 0.6, angle=rng.uniform(-0.5, 0.5)) & lungs
        path_layer += 0.22 * strength() * _soft(band, 1.5)
        truth["Atelectasis"] = (cx - w / 2, cy - h / 2, w, h)
    if "Infiltration" in findings:
        cy, cx, h, w = box_or("Infiltration", lungs, size=(0.22, 0.18))
        region = _ellipse(yy, xx, cy, cx, h / 2, w / 2)
        patch = ndi.gaussian_filter(rng.normal(size=(S, S)), 4) > 0.03
        path_layer += 0.16 * strength() * _soft(region & patch & lungs, 3)
        truth["Infiltration"] = (cx - w / 2, cy - h / 2, w, h)
    if "Pneumonia" in findings:
        cy, cx, h, w = box_or("Pneumonia", lungs, size=(0.24, 0.2))
        region = _ellipse(yy, xx, cy, cx, h / 2, w / 2)
        patch = ndi.gaussian_filter(rng.normal(size=(S, S)), 3) > -0.15
        cons = _soft(region & patch & lungs, 2)
        bronch = (np.abs(ndi.gaussian_filter(rng.normal(size=(S, S)), 2)) < 0.02) & region
        path_layer += 0.28 * strength() * cons - 0.08 * bronch * cons
        truth["Pneumonia"] = (cx - w / 2, cy - h / 2, w, h)
    if "Mass" in findings:
        cy, cx, h, w = box_or("Mass", lungs)
        r = np.clip(min(h, w) / 2, S * 0.04, S * 0.09)
        ang = np.arctan2(yy - cy, xx - cx)
        lob = 1 + 0.12 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.06 * np.sin(5 * ang)
        path_layer += 0.30 * strength() * _soft(np.hypot(yy - cy, xx - cx) < r * lob, 1.8)
        truth["Mass"] = (cx - r, cy - r, 2 * r, 2 * r)
    if "Nodule" in findings:
        cy, cx, h, w = box_or("Nodule", lungs)
        r = np.clip(min(h, w) / 2, S * 0.012, S * 0.028)
        path_layer += 0.34 * strength() * _soft(np.hypot(yy - cy, xx - cx) < r, 0.7)
        truth["Nodule"] = (cx - r, cy - r, 2 * r, 2 * r)
    if "Pneumothorax" in findings:
        cy, cx, h, w = box_or("Pneumothorax", lungs, (top, top + lh * 0.5))
        side = (xx < S * 0.5) if cx < S * 0.5 else (xx >= S * 0.5)
        collapsed = ndi.binary_erosion(lungs & side, iterations=int(rng.uniform(7, 14)))
        rim = lungs & side & ~collapsed & (yy < top + lh * 0.7)
        markings[rim] = 0                                                   # no lung markings beyond the pleural line
        path_layer -= 0.06 * strength() * _soft(rim, 1)
        edge = collapsed & ~ndi.binary_erosion(collapsed) & (yy < top + lh * 0.7)
        path_layer += 0.10 * _soft(edge, 0.6)
        ys, xs = np.nonzero(rim)
        if len(ys):
            truth["Pneumothorax"] = (xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1)

    img += 0.07 * markings + path_layer
    # acquisition: AP portable films are noisier, lower contrast, often rotated
    if ap:
        img = 0.5 + (img - 0.5) * 0.85
        img = ndi.rotate(img, rng.uniform(-4, 4), reshape=False, order=1, mode="nearest")
    img = ndi.gaussian_filter(img, 0.7) + rng.normal(0, 0.03 if ap else 0.018, (S, S))
    img = np.clip(img, 0, 1) ** rng.uniform(0.85, 1.15)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    truth = {k: tuple(float(v * NIH_SIZE / S) for v in b) for k, b in truth.items()}
    return out, truth


def _render_row(args):
    row, seed, mask, boxes, findings, out = args
    rng = np.random.default_rng(seed)
    labels = {f for f in findings if f in row["labels"]}
    img, truth = render_cxr(rng, labels, row["view"], row["sex"], mask, boxes)
    Image.fromarray(img).save(out / "images" / row["image"], optimize=False)
    return [(row["image"], f, *b) for f, b in truth.items()]


def make_phantoms(out_dir: Path, n: int, seed: int) -> None:
    findings = params()["classification"]["findings"]
    meta, bb = read_metadata(ASSETS.parent / "_none_")
    rng = np.random.default_rng(seed)
    bb_imgs = set(bb.image)
    base = meta[meta.image.isin(bb_imgs)]
    box_patients = set(base.patient)
    others = meta[~meta.patient.isin(box_patients)]     # box patients are test-only: don't spend images on them
    # positive-enriched sample (documented): rows with any target finding are 3x as likely to be drawn,
    # so rare findings (Pneumonia 1.3%, Cardiomegaly 2.5% in NIH) have enough training positives.
    weight = np.where(others.labels.apply(lambda s: any(f in s for f in findings)), 3.0, 1.0)
    rest = others.sample(n - len(base), weights=weight, random_state=seed)
    sel = pd.concat([base, rest]).sort_values("image").reset_index(drop=True)
    masks = load_lung_masks()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, row in enumerate(sel.to_dict("records")):
        boxes = {r.finding: (r.x, r.y, r.w, r.h) for r in bb[bb.image == row["image"]].itertuples()} \
            if row["image"] in bb_imgs else None
        jobs.append((row, seed * 100003 + i, masks[rng.integers(len(masks))], boxes, findings, out_dir))
    with Pool(2) as pool:
        truth = [t for res in pool.imap(_render_row, jobs, chunksize=32) for t in res]
    full = pd.read_csv(ASSETS / "Data_Entry_2017_v2020.csv.gz")
    full[full["Image Index"].isin(set(sel.image))].to_csv(out_dir / "Data_Entry_2017_v2020.csv", index=False)
    bbo = pd.read_csv(ASSETS / "BBox_List_2017.csv.gz")
    bbo[bbo["Image Index"].isin(set(sel.image))].to_csv(out_dir / "BBox_List_2017.csv", index=False)
    pd.DataFrame(truth, columns=["image", "finding", "x", "y", "w", "h"]).to_csv(out_dir / "phantom_regions.csv", index=False)
    (out_dir / "PHANTOM.txt").write_text(
        "Phantom images rendered from real NIH ChestX-ray14 metadata rows (labels, patients, age, sex, view)\n"
        "and real bounding boxes. Replace this folder with the real NIH data for real results.\n")
