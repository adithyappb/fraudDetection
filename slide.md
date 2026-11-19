# AI-Assisted Medical Imaging and Diagnosis System: 10-minute talk script

Speaker script for the 10-minute final presentation (problem → design → challenges → results → next steps). One section per slide; the timings add up to ~10 minutes. The same text is in each slide's speaker notes in the deck.

## Slide 1: Title  _(~30 s, ends at 0:30)_

Hi, I'm dp. My project is an AI-assisted medical imaging system: X-rays and CT scans arrive in a DICOM server, deep learning models analyse them, the results are shown with explanation heatmaps, and a radiologist approves or rejects each one. I'll cover the problem, the design, the challenges, the results and next steps. One thing up front: this is an educational system and is not for clinical use. That notice is built into every screen.

## Slide 2: The problem  _(~55 s, ends at 1:25)_

Radiology workloads keep growing, and AI can pre-read studies. But in practice it only helps if four things hold. It has to speak DICOM and plug into the hospital's image server, not need manual uploads. It has to explain itself so a radiologist can check it quickly. A human has to stay in charge, with every decision audited. And its limits and biases have to be measured and stated. I built that loop end to end for two tasks: classifying eight findings on chest X-rays, and segmenting the spleen in 3D on abdominal CT.

## Slide 3: Data and DICOM  _(~55 s, ends at 2:20)_

For data I chose NIH ChestX-ray14 for classification. It's openly downloadable, unlike MIMIC or CheXpert, and it ships 984 radiologist bounding boxes, which let me measure whether Grad-CAM points at the right place instead of just showing pretty pictures. For segmentation I used the Decathlon spleen task, MONAI's reference 3D problem. A big part of milestone one was learning DICOM: the patient, study, series, instance hierarchy; that CT stores integers you must rescale to Hounsfield units; and that slices must be ordered by their z position, never by file order. I wrote a test that sends slices in reverse to prove it.

## Slide 4: Architecture  _(~55 s, ends at 3:15)_

Here's the architecture. The top row is the offline DVC pipeline: data, preprocess, train the classifier and the 3D U-Net, then evaluate. The bottom row is the live system. A modality sends images to Orthanc over the DICOM network protocol. My worker watches Orthanc's change feed, and when a study is complete it downloads it and routes by modality: X-rays go to the classifier and Grad-CAM, CTs to the 3D U-Net. It stores the results, writes an audit entry and pushes the overlay back into Orthanc as a new DICOM series. The radiologist uses a React viewer served by FastAPI.

## Slide 5: Preprocessing and DVC  _(~50 s, ends at 4:05)_

Milestone two was the data pipeline. Images are resized to 160 pixels and normalised, and augmented with rotation, scaling, contrast and noise. I deliberately left out left-right flips, because flipping a chest X-ray puts the heart on the wrong side, which a model should never see. The split is by patient, not by image: some NIH patients have 184 images, so an image-level split would leak the same patient into test. Everything is versioned with DVC. The raw data is tracked by dvc add, and a six-stage pipeline means dvc repro only re-runs what changed.

## Slide 6: Models  _(~55 s, ends at 5:00)_

For classification I used transfer learning with EfficientNet-B0, which trains twice as fast as DenseNet-121 on CPU. DenseNet, the CheXNet architecture, is one config line away. It has an eight-finding multi-label head with a class-balanced loss. Decision thresholds are chosen on validation for 80 percent specificity, in their own pipeline stage, so they never touch test data. For segmentation I used MONAI's 3D U-Net with the standard spleen recipe: HU windowing, reorientation, resampling, patch sampling, Dice plus cross-entropy, sliding-window inference, and inverting the transforms so the mask lands back on the original CT grid.

## Slide 7: Grad-CAM  _(~55 s, ends at 5:55)_

Every prediction gets a Grad-CAM heatmap, one per finding. The key decision was to measure it, not just display it. On 984 radiologist boxes, the heatmap's peak falls inside the box 31 percent of the time, against seven percent by chance. That's four times better than random. It's excellent for cardiomegaly at 0.93, and poor for tiny nodules at 0.05, because the heatmap is only a ten-by-ten grid. That number goes straight into the model's limitations.

## Slide 8: Orthanc service and viewer  _(~55 s, ends at 6:50)_

Here's the live system. A scanner, simulated by my script, sends DICOM to Orthanc over the network. The worker waits for Orthanc's StableStudy event, so it only processes a CT once all its slices have arrived. It runs inference, and writes the overlay back into the same study as a standard DICOM series, so any PACS viewer shows it too. In my React viewer the radiologist sees the scan, the overlay with an opacity control, each finding's confidence against its threshold, and Approve and Reject buttons. They can mark findings they disagree with. Every event goes into an audit log where each entry hashes the previous one, so editing history is detectable.

## Slide 9: Results  _(~60 s, ends at 7:50)_

Results on the patient-disjoint test set. The classifier reaches a macro-AUROC of 0.784. It's very strong on cardiomegaly and effusion, at 0.987 and 0.975, and good on mass and pneumonia. It's weak on the subtle findings: atelectasis, infiltration and pneumothorax, which is close to chance at this training budget. I report that openly rather than hide it. The 3D U-Net segments the spleen with a Dice of 0.954 and a boundary error of 1.9 millimetres. In the live system an X-ray, including all eight heatmaps, takes about 0.6 seconds, and a CT about 0.9 seconds. All 11 tests pass, including the end-to-end one against a live Orthanc.

## Slide 10: Bias and limits  _(~50 s, ends at 8:40)_

Milestone seven asked for limits and bias. Performance is similar across sex, age bands and view position. But cardiomegaly false positives are higher on AP portable films, 34.9 versus 31.8 percent. AP films magnify the heart by about 15 percent, so a model can learn the projection instead of the disease. It's a well-known shortcut, and on real data it's worse because AP films come from sicker patients. The model card also lists: noisy NLP-mined labels, a single-site population, only eight findings so no alert doesn't mean normal, thresholds that need recalibration per site, and that Grad-CAM shows where the model looked, not why.

## Slide 11: Challenges  _(~65 s, ends at 9:45)_

Four challenges taught me the most. First, my Grad-CAM looked fine but scored almost zero against the boxes: it peaked in the image corner every time. I rewrote it with hooks and compared layers, and a finer layer took it from 0.14 to 0.31. Lesson: measure explanations. Second, training/serving skew: the service normalised DICOM pixels differently from training, which inflated probabilities. Now there's one mapping, pinned by a test. Third, alert fatigue: the first thresholds flagged five to seven findings per film, so I moved threshold calibration into its own stage at 80 percent specificity. Fourth, uploading my overlay to Orthanc re-triggered the worker, so I made it idempotent.

## Slide 12: Next steps  _(~30 s, ends at 10:15)_

Next, I'd run the full NIH dataset with ImageNet transfer learning at higher resolution on all fourteen findings, and use higher-resolution explanation methods to fix nodule localisation. I'd also feed the radiologists' disagreement labels back as the next DVC data version. On the standards side: emit proper DICOM structured reports and segmentation objects, and integrate the OHIF viewer. And for anything beyond a demo: authentication, TLS and write-once audit storage. Thank you, I'm happy to take questions.

---
Total ≈ 10 min 15 s. Keep ~30 s spare for the switch to Q&A.
