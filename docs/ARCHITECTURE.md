# Architecture and data documentation

![architecture](figures/architecture.png)

## Components

| Component | Code | Responsibility |
|---|---|---|
| Data acquisition | `acquire.py`, `data/nih.py`, `data/msd.py` | real NIH / MSD data, or phantoms in the identical layout; versioned with `dvc add data/raw` |
| Preprocess | `preprocess.py` | resize to 160 px (area interpolation), uint8 cache, **patient-level** split, boxes rescaled |
| Classifier | `train_cls.py`, `models.py` | ImageNet backbone + new 8-way multi-label head, BCE with √(neg/pos) class weights, AdamW, one-cycle LR, augmentation (affine, contrast, noise; **no horizontal flip**, which would move the heart) |
| Calibration | `calibrate.py` | per-finding thresholds at 80% specificity on **validation** |
| Segmenter | `train_seg.py` | MONAI 3D U-Net, Dice-CE, 64³ positive/negative patch sampling, sliding-window validation |
| Evaluation | `evaluate.py`, `explain.py` | AUROC/AP/sens/spec, Grad-CAM vs radiologist boxes, subgroup audit, Dice/HD95, figures |
| DICOM I/O | `data/dicom_io.py` | DX, CT series and Secondary Capture writers; pixel/HU readers |
| Worker | `service/worker.py` | Orthanc `/changes` → `StableStudy` → inference → results + audit → overlay series back to Orthanc |
| Results + audit | `service/db.py` | SQLite tables `studies`, `audit` (hash chain), `worker_state` (change-feed cursor) |
| API | `service/api.py` | REST for the viewer; serves the built React app |
| Viewer | `web/` | worklist, scan + overlay, findings, decision, audit log, model card |

## Data flow for one study

1. A modality (`send_studies.py`) sends DICOM over **C-STORE** to Orthanc on port 4242.
2. Orthanc stores the instances. After 3 s with no new instance (`StableAge`), it emits **`StableStudy`** in `/changes`.
3. The worker (cursor kept in `worker_state`) downloads every instance and routes by `Modality`:
   * **DX/CR:** pixels / (2^BitsStored − 1), flipped if MONOCHROME1, then resized to 160 px. This is the same mapping as training (a test asserts < 0.005 difference). Outputs: 8 probabilities and 8 Grad-CAM maps (stride-16 layer, 10 × 10, upsampled).
   * **CT:** slices sorted by `ImagePositionPatient` z, HU = value × RescaleSlope + RescaleIntercept. The volume goes through the training transforms (window, crop, RAS orientation, 1.5 × 1.5 × 2 mm spacing), then sliding-window 3D U-Net. MONAI `Invertd` maps the probabilities back to the original voxel grid, giving the mask, volume in mL and confidence.
4. Arrays are saved under `data/service/results/{study}`. The JSON result goes into `studies`, and audit entries `study_received` and `ai_analyzed` record the model versions.
5. The overlay is uploaded to Orthanc as a **Secondary Capture** series in the same study. Orthanc then re-emits `StableStudy`, which the worker ignores because the study is already analysed.
6. The viewer polls `/api/studies`. The radiologist reviews and signs Approve or Reject, recorded as an audit entry.

## Data

### NIH ChestX-ray14 (classification)

| File | Content |
|---|---|
| `images/*.png` | 1024 × 1024, 8-bit (phantoms: 320 × 320) |
| `Data_Entry_2017*.csv` | Image Index, Finding Labels (`\|`-separated), Follow-up #, Patient ID, Age, Gender, View Position, original size, pixel spacing |
| `BBox_List_2017.csv` | 984 radiologist boxes (x, y, w, h in 1024-px space) for the 8 modelled findings |

**This build** (`metrics/data_summary.json`):
- **Images:** 7,000 from 5,235 patients, split by patient: train 4,585 · val 733 · test 1,682.
- **Test boxes:** all 880 box images, with their patients, are in test.
- **Prevalence:** Infiltration 27.9%, Effusion 19.2%, Atelectasis 18.4%, Nodule 10.5%, Mass 9.2%, Pneumothorax 8.0%, Cardiomegaly 5.4%, Pneumonia 3.6%; 31% have none of the 8.
- **Sampling:** rows with a target finding were drawn 3× as often, because real NIH has Pneumonia at 1.3% and Cardiomegaly at 2.5%.
- **Demographics:** PA 61% / AP 39%, male 57%, median age 50.

### MSD Task09 Spleen (segmentation)

| File | Content |
|---|---|
| `imagesTr/*.nii.gz` | CT volumes in HU |
| `labelsTr/*.nii.gz` | 0 = background, 1 = spleen |
| `dataset.json` | training list |

**This build:** 40 phantom volumes (160 × 160 × 56, 1.6 × 1.6 × 4 mm), split 30 / 4 / 6.

### DVC

- `data/raw.dvc` pins the exact raw dataset (md5 of 6,000+ files).
- `dvc.yaml` declares the stages with their deps, params and outputs.
- `dvc.lock` pins the result of each run.

`dvc repro` re-runs only what changed. `dvc metrics diff` compares experiments, and `dvc push`/`pull` share data through a remote (`dvc remote add -d store <s3|gs|ssh|path>`).

## Deployment

`docker-compose.yml` runs three services: `orthanc` (official image with `orthanc/orthanc.json`), `worker`, and `api` (serves the React build). One multi-stage image (Node build → Python 3.11 slim, CPU PyTorch) is shared by worker and API, with a shared volume for results.

**For any non-demo deployment:**
- enable Orthanc authentication/TLS
- put the API behind SSO
- move SQLite to Postgres
