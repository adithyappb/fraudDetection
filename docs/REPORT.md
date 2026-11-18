# Project report: AI-Assisted Medical Imaging and Diagnosis System

> **FOR EDUCATION AND RESEARCH ONLY: NOT FOR CLINICAL USE.**

## 1. Problem

Radiology volumes grow faster than the number of radiologists. AI can pre-read studies, but it is only useful if four things hold:

* It plugs into the hospital's imaging infrastructure: **DICOM** and a **PACS** (here Orthanc), not ad-hoc uploads.
* It shows **why** it flagged something (Grad-CAM), so a radiologist can check it quickly.
* A **human stays in charge**: nothing is final until a named radiologist approves or rejects it, and every decision is **auditable**.
* Its **limits and biases are measured and stated**.

This project builds that loop end to end for two tasks:
- multi-label chest X-ray classification (8 findings)
- 3D spleen segmentation on abdominal CT

## 2. Design

Full detail is in [ARCHITECTURE.md](ARCHITECTURE.md).

* **Datasets.**
  - NIH ChestX-ray14 is chosen for classification: open, 112k images, and **984 radiologist boxes**, so Grad-CAM can be *scored*.
  - MSD Task09 Spleen is chosen for 3D segmentation: MONAI's reference task.
* **Fallback data.** Neither image set was downloadable in the build environment. Phantoms reproduce both datasets' exact file layouts:
  - **Radiographs:** each is rendered from a *real* NIH metadata row (real labels, patients, age, sex, view position), with findings placed at the *real* radiologist box locations and lungs shaped by 500+ *real* expert lung masks.
  - **CT:** abdominal phantoms in Hounsfield units.

  Swapping in the real data is one command.
* **Reproducibility.** The raw data is versioned with `dvc add`. A 5-stage `dvc.yaml` pipeline is driven by `params.yaml`, and metrics are tracked with `dvc metrics`.
* **Leakage control.**
  - Splits are **by patient**; NIH has up to 184 images per patient.
  - Thresholds are set on validation only, and the test split is used once.
  - Box patients are test-only.
* **Classifier.**
  - EfficientNet-B0 at 160 px, chosen over DenseNet-121 because it trains 2× faster on CPU; DenseNet is one config line away.
  - New multi-label head, √-balanced BCE.
  - Anatomically valid augmentation: no left-right flips.
* **Grad-CAM.** Implemented with hooks on a stride-16 block. It is evaluated with the pointing game and IoU against the radiologist boxes, not just displayed.
* **Segmentation.** MONAI 3D U-Net with the standard spleen recipe (HU window, RAS orientation, isotropic resampling, positive/negative patch sampling, sliding-window inference, `Invertd` back to scanner space).
* **Service.**
  - **Worker:** follows Orthanc's change feed and processes `StableStudy`. It stores results and writes the overlay back to Orthanc as a DICOM **Secondary Capture** series.
  - **FastAPI:** serves the React viewer.
  - **Audit log:** hash-chained (SHA-256 of previous hash + entry), so tampering is detectable.

## 3. Results

Held-out, patient-disjoint test split, **phantom data**. These figures validate the pipeline; they are **not** clinical performance claims.

**Chest X-ray classifier (1,682 test images):**
- **Macro-AUROC: 0.784.**
- **Strong:** cardiomegaly 0.987, effusion 0.975, mass 0.845, pneumonia 0.805.
- **Weak:** nodule 0.768, atelectasis 0.688, infiltration 0.610, pneumothorax 0.592.

The weak findings are the subtle ones (faint patchy opacity, thin pleural line) and were trained from scratch: ImageNet weights could not be downloaded.

![ROC](figures/roc_per_finding.png)

**Grad-CAM vs 984 radiologist boxes:**
- **Pointing game: 0.31**, against 0.07 by chance, so the heat map peak falls inside the box 4.4× more often than random.
- **By finding:** cardiomegaly 0.93; mass 0.33; pneumonia 0.29; nodules 0.05, too small for a 10 × 10 map.

![Grad-CAM](figures/gradcam_examples.png)

**Spleen 3D U-Net (6 test CT volumes):** **Dice 0.954** (0.939–0.965 per volume), **HD95 1.9 mm**.

**Bias audit:**
- macro-AUROC female 0.797 vs male 0.772; across age bands 0.773–0.793; PA 0.782 = AP 0.782.
- Cardiomegaly false-positive rate is higher on **AP** films (34.9% vs 31.8%). That is the expected magnification confounder. Details are in [MODEL_CARD.md](MODEL_CARD.md).

**System:**
- 14 studies were sent over DICOM C-STORE (12 CXR + 2 CT, 124 instances) and processed in about 25 s.
- Inference per study: CXR ~0.6 s with 8 Grad-CAMs; CT ~0.9 s.
- Live service probabilities match offline evaluation within 0.03.
- **11 / 11 tests pass**, including end to end against a live Orthanc 1.12.2.

## 4. Challenges

1. **No access to real images.** The build environment could only reach package registries. Solution: phantoms driven by the real NIH metadata, radiologist boxes and lung masks bundled in the `torchxrayvision` package, in byte-identical layouts. The loaders and the download script for the real data were tested on the same code path.
2. **Grad-CAM first looked fine but scored ~0 on the boxes.** Two problems:
   - **Wiring:** the library CAM peaked in the image corner every time.
   - **Coarseness:** the last block is a 5 × 5 grid at 160 px, too coarse for small findings.

   A layer comparison against the boxes (5×5: 0.14 → 10×10: 0.31) chose the stride-16 block, with a small hook-based implementation. **Lesson:** an explanation method must be *measured*, not just visualised.
3. **Training/serving skew.** The first service version normalised DICOM pixels by percentiles while training used value/255, which inflated probabilities. Now both use one mapping, and a test pins it to < 0.005.
4. **Alert fatigue.** Youden-optimal thresholds flagged 5–7 findings per film. Thresholds are now a separate DVC stage (`calibrate`) at 80% specificity on validation. On test that gave 67–72% specificity: a real calibration shift, reported rather than hidden.
5. **Rare classes and leakage together.** Putting all box patients in test starved training of positives. The fix was a positive-enriched sample and keeping box patients test-only without spending their extra images.
6. **Orthanc feedback loop.** Uploading the overlay re-triggers `StableStudy`. The worker is idempotent: it skips studies it has already analysed, and a test asserts this.
7. **CPU-only budget.** 2 cores, so: 160 px inputs, EfficientNet-B0, a cached uint8 dataset, and a 64³-patch 3D U-Net. The full pipeline takes about 35 minutes.

## 5. Next steps

* **Real data first.** `make download && make data && make train`, with ImageNet transfer learning, at 224–320 px, on all 14 NIH findings, then re-run the bias audit.
* **Better explanations.** Grad-CAM++ / HiResCAM at higher resolution, plus box-supervised attention to improve nodule localisation.
* **Calibration.** Temperature scaling and per-site threshold recalibration, with monitored drift in the live specificity.
* **Feedback loop.** Use the radiologists' "disagree" labels from the audit log as curated hard negatives for the next DVC data version.
* **Standards.** Emit DICOM **SR / Segmentation (SEG)** objects in place of Secondary Capture, and add DICOMweb (QIDO/WADO) plus an OHIF viewer integration.
* **Security.** Orthanc authentication and TLS, SSO on the API, Postgres with WORM storage for the audit log, and PHI de-identification checks on ingest.
