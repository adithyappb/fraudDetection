# DICOM primer: what this project reads and writes

DICOM (Digital Imaging and Communications in Medicine) is two things: a **file format** (a header of tagged
attributes plus pixel data) and a **network protocol** (how scanners, PACS and viewers exchange images).
This project uses both.

## The information model: Patient → Study → Series → Instance

| Level | Identified by | In this project |
|---|---|---|
| Patient | `PatientID (0010,0020)` | `NIH-00013118` (NIH patient), `MSD-SPLEEN-001` |
| Study (one exam/visit) | `StudyInstanceUID (0020,000D)` | one chest X-ray, or one abdominal CT |
| Series (one acquisition/reconstruction) | `SeriesInstanceUID (0020,000E)` | the radiograph; the CT slices; **our AI overlay series** |
| Instance (one image/object) | `SOPInstanceUID (0008,0018)` | one radiograph, one CT slice |

`SOPClassUID (0008,0016)` says what kind of object an instance is. We use three:

| SOP Class | UID | Used for |
|---|---|---|
| Digital X-Ray Image Storage – For Presentation | 1.2.840.10008.5.1.4.1.1.1.1 | chest radiographs (`Modality = DX`) |
| CT Image Storage | 1.2.840.10008.5.1.4.1.1.2 | one instance per axial CT slice |
| Secondary Capture Image Storage | 1.2.840.10008.5.1.4.1.1.7 | the AI overlay (Grad-CAM / mask), `ImageType = DERIVED\SECONDARY` |

## Tags that matter for AI

| Tag | Why it matters |
|---|---|
| `PhotometricInterpretation` | `MONOCHROME2` = bright is dense. `MONOCHROME1` is inverted and must be flipped before the model (`dicom_io.cxr_pixels`) |
| `BitsAllocated` / `BitsStored` | Radiographs are typically 12–16 bit, never feed raw 16-bit values to an 8-bit-trained model; we rescale by percentiles |
| `RescaleSlope` / `RescaleIntercept` | CT stores unsigned integers; **HU = value × slope + intercept** (we write intercept −1024) |
| `PixelSpacing`, `SliceThickness` | physical size in mm; the 3D U-Net resamples to 1.5 × 1.5 × 2.0 mm, and spleen volume in mL uses them |
| `ImagePositionPatient`, `ImageOrientationPatient` | slice order is the **z of ImagePositionPatient**, *not* `InstanceNumber` or file order (`dicom_io.ct_volume` sorts on it; a test sends slices reversed) |
| `ViewPosition` | PA vs AP. AP portable films magnify the heart, which is a known confounder (see MODEL_CARD.md) |
| `WindowCenter` / `WindowWidth` | display defaults (CT soft tissue: L 40–50 / W 400) |
| `PatientIdentityRemoved` | we only use de-identified data and set this to `YES` |

## Network: C-STORE and Orthanc

* `scripts/send_studies.py` behaves like a **modality**: it opens a DICOM association (AE title `MEDAI-MODALITY`) to
  Orthanc on port 4242 and sends each instance with **C-STORE**.
* Orthanc indexes them and exposes them over REST (`/studies`, `/series`, `/instances/{id}/file`) and a change feed
  (`/changes`). Once a study receives no new instance for `StableAge` seconds, Orthanc emits `StableStudy`. That is
  the worker's trigger, so a CT is processed only once all its slices have arrived.
* The worker writes its result back as a new **Secondary Capture series in the same study**, so any DICOM viewer
  connected to the PACS shows the AI overlay next to the original images, clearly labelled as derived and for
  education only.

Inspect any file: `python scripts/dicom_inspect.py file.dcm`.
