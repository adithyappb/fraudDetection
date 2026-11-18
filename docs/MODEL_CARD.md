# Model card: limits and bias

> **FOR EDUCATION AND RESEARCH ONLY: NOT FOR CLINICAL USE.** Not a medical device, not validated, not cleared or approved by any regulator. Never use these outputs for diagnosis, triage or treatment.

## Models

| | Chest X-ray classifier | CT spleen segmenter |
|---|---|---|
| Architecture | EfficientNet-B0 (torchvision); DenseNet-121 selectable | MONAI 3D U-Net, channels 16-32-64-128, residual units |
| Input | frontal CXR, 160 × 160, 1 channel replicated to 3 | portal-venous abdominal CT, resampled to 1.5 × 1.5 × 2.0 mm, HU window −57…164 |
| Output | 8 independent probabilities + Grad-CAM per finding | spleen mask, volume (mL), mean voxel confidence |
| Training data | NIH ChestX-ray14 schema; *this build:* 4,585 phantom radiographs from real NIH metadata | MSD Task09 Spleen schema; *this build:* 30 phantom CT volumes |
| Initialisation | ImageNet (transfer learning) when downloadable; **this build: random** (no internet) | random |
| Decision thresholds | per finding, 80% specificity on the **validation** split | 0.5 on softmax |

## Intended use

Teaching how a medical-imaging AI pipeline is built: DICOM handling, training, explainability, human review and audit.

**Out of scope:** any real patient, any clinical decision, paediatric images, lateral views, non-portal-venous CT.

## Performance (held-out test split, patient-disjoint)

| Finding | AUROC | Sensitivity | Specificity | Grad-CAM pointing (radiologist boxes) |
|---|---|---|---|---|
| Cardiomegaly | 0.987 | 1.00 | 0.67 | 0.93 |
| Effusion | 0.975 | 0.99 | 0.68 | 0.19 |
| Mass | 0.845 | 0.83 | 0.72 | 0.33 |
| Pneumonia | 0.805 | 0.75 | 0.68 | 0.29 |
| Nodule | 0.768 | 0.76 | 0.69 | 0.05 |
| Atelectasis | 0.688 | 0.58 | 0.69 | 0.18 |
| Infiltration | 0.610 | 0.41 | 0.71 | 0.23 |
| Pneumothorax | 0.592 | 0.17 | 0.89 | 0.11 |
| **Macro** | **0.784** | | | **0.31** (chance ≈ 0.07) |

Spleen segmentation: **Dice 0.954**, HD95 1.9 mm on 6 test volumes.

These numbers come from **phantom** images and do not predict performance on real radiographs. Published models on real NIH ChestX-ray14 reach a macro-AUROC of roughly 0.80–0.84 over 14 findings.

## Bias audit (test split)

| Subgroup | n | macro-AUROC |
|---|---|---|
| Female | 742 | 0.797 |
| Male | 940 | 0.772 |
| Age < 40 | 519 | 0.792 |
| Age 40–60 | 781 | 0.773 |
| Age > 60 | 382 | 0.793 |
| View PA | 967 | 0.782 |
| View AP (portable) | 715 | 0.782 |

**Cardiomegaly false-positive rate: AP 34.9% vs PA 31.8%.** AP (portable, bedside) films magnify the heart by about 15%, so a model can read "enlarged heart" from the projection rather than the disease. On real data this effect is much larger, because AP films also come from sicker inpatients: a textbook shortcut.

The 2.5-point AUROC gap between female and male patients is within the noise for these group sizes. It must still be re-measured on real data.

## Known limitations

1. **Label noise:** NIH labels were mined from radiology reports by NLP (estimated ≥10% wrong for some findings); the model inherits it.
2. **Population shift:** one US hospital (NIH Clinical Center), adult inpatients/outpatients. Performance on other hospitals, scanners, countries or children is unknown.
3. **Calibration shift:** thresholds set at 80% specificity on validation gave 67–72% specificity on test. Operating points must be re-calibrated per site.
4. **Coverage:** only 8 of the many possible chest findings, so an alert-free study is **not** a normal study.
5. **Weak findings:** pneumothorax and infiltration are close to chance at this training budget; their alerts are unreliable.
6. **Grad-CAM shows *where*, not *why*.** Maps are coarse (10 × 10 grid) and can highlight correlated regions (devices, markers, the heart for effusion). Small nodules are poorly localised (pointing 0.05).
7. **Segmentation scope:** trained on portal-venous CT only. Splenomegaly, splenectomy, trauma and arterial-phase scans are untested.
8. **Phantom training data (this build):** the shipped weights learned synthetic textures. Retrain on real data before drawing any conclusion (`make download && make data && make train`).

## Human oversight built in

- The worklist never auto-closes a study. Every result needs a named radiologist to **Approve** or **Reject**, optionally listing the AI findings they disagree with.
- Every AI result and every decision is written to a **hash-chained audit log**. Editing or deleting a past entry breaks the chain, and the viewer shows it.
- The disclaimer is displayed on every screen, in the API description, in every AI response, and inside the DICOM overlay (`DerivationDescription`).
