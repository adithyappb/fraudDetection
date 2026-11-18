"""DICOM read/write (pydicom). See docs/DICOM.md for the format primer.

Writes standards-conformant objects:
  * chest radiographs  -> Digital X-Ray Image Storage - For Presentation (DX), 12-bit MONOCHROME2
  * CT volumes          -> CT Image Storage, one instance per axial slice, HU via RescaleSlope/Intercept
  * AI overlays         -> Secondary Capture (RGB), new series in the same study, flagged as derived
"""
from __future__ import annotations

import datetime as dt
from io import BytesIO

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

DX_STORAGE = "1.2.840.10008.5.1.4.1.1.1.1"
CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
SC_STORAGE = "1.2.840.10008.5.1.4.1.1.7"


def _base(sop_class: str, patient: dict, study_uid: str, series_uid: str, modality: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    now = dt.datetime.now()
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.Modality = modality
    ds.PatientID = patient["id"]
    ds.PatientName = patient.get("name", f"ANON^{patient['id']}")
    ds.PatientSex = patient.get("sex", "O")
    if patient.get("age") is not None:
        ds.PatientAge = f"{int(patient['age']):03d}Y"
    ds.StudyDate = ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.StudyTime = ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.AccessionNumber = patient.get("accession", "")
    ds.StudyID = "1"
    ds.Manufacturer = "medimaging-ai (education only)"
    ds.PatientIdentityRemoved = "YES"
    return ds


def cxr_dataset(img_u8: np.ndarray, patient: dict, view: str, study_uid: str | None = None,
                pixel_spacing_mm: float = 0.143 * 1024 / 320) -> FileDataset:
    study_uid = study_uid or generate_uid()
    ds = _base(DX_STORAGE, patient, study_uid, generate_uid(), "DX")
    ds.StudyDescription = f"CHEST {view}"
    ds.SeriesDescription = f"CHEST {view}"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = view
    ds.PresentationIntentType = "FOR PRESENTATION"
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    ds.SeriesNumber, ds.InstanceNumber = 1, 1
    px = (img_u8.astype(np.uint16) * 16).astype(np.uint16)              # 8-bit source -> 12-bit stored
    ds.Rows, ds.Columns = px.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelSpacing = [pixel_spacing_mm, pixel_spacing_mm]
    ds.WindowCenter, ds.WindowWidth = 2048, 4096
    ds.PixelData = px.tobytes()
    return ds


def ct_series(hu: np.ndarray, spacing: tuple, patient: dict, study_uid: str | None = None,
              description: str = "ABDOMEN CT PORTAL VENOUS") -> list[FileDataset]:
    """hu: (x, y, z) volume in Hounsfield units."""
    study_uid = study_uid or generate_uid()
    series_uid, frame_uid = generate_uid(), generate_uid()
    out = []
    for k in range(hu.shape[2]):
        ds = _base(CT_STORAGE, patient, study_uid, series_uid, "CT")
        ds.StudyDescription, ds.SeriesDescription = description, description
        ds.BodyPartExamined = "ABDOMEN"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.SeriesNumber, ds.InstanceNumber = 2, k + 1
        ds.FrameOfReferenceUID = frame_uid
        ds.ImagePositionPatient = [0.0, 0.0, float(k * spacing[2])]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceThickness, ds.SpacingBetweenSlices = spacing[2], spacing[2]
        ds.PixelSpacing = [spacing[1], spacing[0]]                      # row spacing (y), column spacing (x)
        sl = (np.clip(hu[:, :, k].T, -1024, 3071) + 1024).astype(np.uint16)
        ds.Rows, ds.Columns = sl.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = -1024, 1, "HU"
        ds.WindowCenter, ds.WindowWidth = 50, 400
        ds.PixelData = sl.tobytes()
        out.append(ds)
    return out


def secondary_capture(rgb: np.ndarray, ref: Dataset, description: str, series_number: int = 900,
                      instance: int = 1, series_uid: str | None = None) -> FileDataset:
    patient = {"id": ref.PatientID, "name": str(ref.PatientName), "sex": ref.get("PatientSex", "O")}
    ds = _base(SC_STORAGE, patient, ref.StudyInstanceUID, series_uid or generate_uid(), "OT")
    ds.SeriesDescription = description
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.ConversionType = "WSD"
    ds.DerivationDescription = "AI overlay - FOR EDUCATION ONLY, NOT FOR CLINICAL USE"
    ds.SeriesNumber, ds.InstanceNumber = series_number, instance
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = np.ascontiguousarray(rgb.astype(np.uint8)).tobytes()
    return ds


def to_bytes(ds: FileDataset) -> bytes:
    buf = BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def read(b: bytes) -> Dataset:
    return pydicom.dcmread(BytesIO(b))


def cxr_pixels(ds: Dataset) -> np.ndarray:
    """-> float32 image in [0, 1], MONOCHROME2 (bright = dense)."""
    a = ds.pixel_array.astype(np.float32)
    a = a * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return 1 - a if ds.PhotometricInterpretation == "MONOCHROME1" else a


def ct_volume(slices: list[Dataset]) -> tuple[np.ndarray, tuple]:
    """Sort by z position, apply rescale -> (x, y, z) HU volume and voxel spacing (x, y, z)."""
    slices = sorted(slices, key=lambda d: float(d.ImagePositionPatient[2]))
    vol = np.stack([s.pixel_array.T.astype(np.float32) * float(s.RescaleSlope) + float(s.RescaleIntercept)
                    for s in slices], axis=2)
    dz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])) if len(slices) > 1 \
        else float(slices[0].SliceThickness)
    return vol, (float(slices[0].PixelSpacing[1]), float(slices[0].PixelSpacing[0]), dz)
