"""Act as an imaging modality: convert held-out test cases to DICOM and send them to Orthanc
over the DICOM network protocol (C-STORE, pynetdicom) - exactly how a scanner or PACS would.

  python scripts/send_studies.py --cxr 12 --ct 2 [--host localhost --port 4242]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from pynetdicom import AE, StoragePresentationContexts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medai.config import PROCESSED, params, path  # noqa: E402
from medai.data import dicom_io  # noqa: E402


def cxr_cases(n: int, seed: int):
    idx = pd.read_csv(PROCESSED / "cxr_index.csv")
    findings = params()["classification"]["findings"]
    te = idx[idx.split == "test"]
    pos = te[te[findings].sum(1) > 0].sample(n - n // 4, random_state=seed)      # mostly positives + some normals
    neg = te[te[findings].sum(1) == 0].sample(n // 4, random_state=seed)
    raw = path(params()["data"]["cxr_raw_dir"]) / "images"
    for r in pd.concat([pos, neg]).sample(frac=1, random_state=seed).itertuples():
        img = np.asarray(Image.open(raw / r.image).convert("L"))
        patient = {"id": f"NIH-{r.patient:05d}", "sex": r.sex, "age": r.age, "accession": r.image.split(".")[0]}
        yield [dicom_io.cxr_dataset(img, patient, r.view)], r.labels


def ct_cases(n: int):
    for i, case in enumerate(json.load(open(PROCESSED / "ct_split.json"))["test"][:n]):
        im = nib.load(case["image"])
        hu = np.asarray(im.dataobj).astype(np.float32)
        spacing = tuple(float(abs(v)) for v in im.header.get_zooms()[:3])
        patient = {"id": f"MSD-SPLEEN-{i + 1:03d}", "sex": "O", "age": None, "accession": Path(case["image"]).name[:-7]}
        yield dicom_io.ct_series(hu, spacing, patient), "spleen CT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr", type=int, default=12)
    ap.add_argument("--ct", type=int, default=2)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=4242)
    ap.add_argument("--aet", default="ORTHANC")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    ae = AE(ae_title="MEDAI-MODALITY")
    ae.requested_contexts = StoragePresentationContexts
    assoc = ae.associate(a.host, a.port, ae_title=a.aet)
    if not assoc.is_established:
        sys.exit(f"Could not associate with {a.aet}@{a.host}:{a.port} - is Orthanc running?")
    sent = 0
    for gen in (cxr_cases(a.cxr, a.seed) if a.cxr else [], ct_cases(a.ct) if a.ct else []):
        for datasets, truth in gen:
            for ds in datasets:
                status = assoc.send_c_store(ds)
                if status.Status != 0x0000:
                    print("C-STORE failed", hex(status.Status))
            sent += 1
            print(f"C-STORE {datasets[0].Modality:2} {datasets[0].PatientID:<18} {len(datasets):3d} instance(s)  "
                  f"[ground truth: {truth}]")
    assoc.release()
    print(f"sent {sent} studies to {a.aet}@{a.host}:{a.port}")


if __name__ == "__main__":
    main()
