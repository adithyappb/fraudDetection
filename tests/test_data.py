import numpy as np
import pandas as pd

from medai.config import PROCESSED
from medai.data import dicom_io, msd, nih


def test_nih_metadata_and_boxes_are_real(tmp_path):
    meta, bb = nih.read_metadata(tmp_path)                           # bundled official NIH metadata
    assert len(meta) == 112120 and meta.patient.nunique() == 30805      # official NIH ChestX-ray14 counts
    assert len(bb) == 984 and set(bb.finding) == {"Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                                                   "Mass", "Nodule", "Pneumonia", "Pneumothorax"}


def test_phantom_renders_findings_where_boxes_say():
    masks = nih.load_lung_masks()[:3]
    rng = np.random.default_rng(0)
    box = (600.0, 300.0, 120.0, 120.0)
    img, truth = nih.render_cxr(rng, {"Mass"}, "PA", "M", masks[0], {"Mass": box})
    assert img.shape == (nih.S, nih.S) and img.dtype == np.uint8
    x, y, w, h = truth["Mass"]
    assert abs((x + w / 2) - (box[0] + box[2] / 2)) < 40 and abs((y + h / 2) - (box[1] + box[3] / 2)) < 40


def test_split_is_by_patient_and_boxes_are_in_test():
    idx = pd.read_csv(PROCESSED / "cxr_index.csv")
    per_patient = idx.groupby("patient").split.nunique()
    assert (per_patient == 1).all(), "a patient appears in more than one split (leakage)"
    boxes = pd.read_csv(PROCESSED / "cxr_boxes.csv")
    assert set(idx.set_index("image").loc[boxes.image.unique(), "split"]) == {"test"}


def test_cxr_dicom_roundtrip():
    img = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)
    ds = dicom_io.read(dicom_io.to_bytes(dicom_io.cxr_dataset(img, {"id": "P1", "sex": "F", "age": 57}, "AP")))
    assert ds.Modality == "DX" and ds.ViewPosition == "AP" and ds.PatientAge == "057Y" and ds.BitsStored == 12
    back = dicom_io.cxr_pixels(ds)
    # serving input must match the training input (PNG / 255): no train/serve intensity skew
    assert np.abs(back - img / 255).max() < 0.005


def test_ct_dicom_roundtrip_preserves_hounsfield_and_geometry():
    hu, _ = msd.render_ct(np.random.default_rng(2))
    hu = hu[::4, ::4, :10]
    series = [dicom_io.read(dicom_io.to_bytes(d)) for d in dicom_io.ct_series(hu, (1.6, 1.6, 4.0), {"id": "P2"})]
    vol, spacing = dicom_io.ct_volume(series[::-1])                    # order on the wire must not matter
    assert np.array_equal(vol, np.clip(hu, -1024, 3071).astype(np.float32))
    assert spacing == (1.6, 1.6, 4.0)
    assert len({d.SeriesInstanceUID for d in series}) == 1
