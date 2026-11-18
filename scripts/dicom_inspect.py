"""Print the key DICOM tags of a file (milestone 1: learn the DICOM format).

  python scripts/dicom_inspect.py path/to/file.dcm
"""
import sys

import pydicom

KEYS = ["SOPClassUID", "Modality", "PatientID", "PatientSex", "PatientAge", "StudyInstanceUID", "SeriesInstanceUID",
        "SOPInstanceUID", "StudyDescription", "SeriesDescription", "ViewPosition", "BodyPartExamined", "Rows",
        "Columns", "PixelSpacing", "SliceThickness", "ImagePositionPatient", "ImageOrientationPatient",
        "PhotometricInterpretation", "BitsAllocated", "BitsStored", "RescaleSlope", "RescaleIntercept",
        "WindowCenter", "WindowWidth", "ImageType", "DerivationDescription"]

ds = pydicom.dcmread(sys.argv[1])
print(f"Transfer syntax: {ds.file_meta.TransferSyntaxUID.name}")
for k in KEYS:
    if k in ds:
        v = ds.data_element(k)
        print(f"({v.tag.group:04X},{v.tag.element:04X}) {v.VR}  {k:<28} {v.value}")
a = ds.pixel_array
print(f"pixel array: {a.shape} {a.dtype} min={a.min()} max={a.max()}")
