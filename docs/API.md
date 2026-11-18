# API reference

Base URL `http://localhost:8000`. Swagger UI at `/docs`. Every response concerning AI output carries the education-only disclaimer.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + disclaimer |
| GET | `/api/studies?status=analyzed` | worklist (status: received · analyzed · approved · rejected · error) |
| GET | `/api/studies/{id}` | full study: metadata, AI result, model version, decision, study audit trail |
| GET | `/api/studies/{id}/image.png?slice=k` | the scan as PNG (CT: axial slice `k`) |
| GET | `/api/studies/{id}/overlay.png?finding=Effusion` | transparent RGBA overlay: Grad-CAM for that finding (CXR) |
| GET | `/api/studies/{id}/overlay.png?slice=k` | transparent RGBA spleen mask for slice `k` (CT) |
| POST | `/api/studies/{id}/decision` | radiologist decision → audit log |
| GET | `/api/audit?limit=200` | audit entries + hash-chain verification |
| GET | `/api/model-card` | metrics, bias audit, intended use |

`{id}` is the Orthanc study ID.

## GET /api/studies/{id} (chest X-ray, abridged)

```json
{
  "study_id": "8a4f0e1c-...", "patient_id": "NIH-27278", "modality": "DX", "kind": "cxr",
  "description": "CHEST PA", "status": "analyzed",
  "ai": {
    "summary": "Effusion, Atelectasis",
    "positives": ["Effusion", "Atelectasis"],
    "findings": [
      {"finding": "Effusion",    "probability": 0.993, "threshold": 0.053, "positive": true,  "cam_peak": [0.29, 0.74]},
      {"finding": "Infiltration","probability": 0.375, "threshold": 0.453, "positive": false, "cam_peak": [0.62, 0.41]}
    ],
    "inference_seconds": 0.59,
    "disclaimer": "FOR EDUCATION AND RESEARCH ONLY - NOT FOR CLINICAL USE. Not a medical device."
  },
  "model_version": "{\"classifier\": \"efficientnet_b0@bc4f5f6b6292\", \"segmenter\": \"unet3d@3f57d7c0aff9\"}",
  "overlay_series": "1.2.826.0.1.3680043.8.498....",
  "audit": [{"action": "ai_analyzed", "actor": "ai-worker", "hash": "…"}, {"action": "study_received", "…": "…"}]
}
```

`cam_peak` is the Grad-CAM maximum as `[x, y]`, a fraction of the image width and height. A CT study's `ai` instead holds `spleen_volume_ml`, `spleen_slices` `[first, last]`, `mean_confidence` and `slices`.

## POST /api/studies/{id}/decision

```json
{"decision": "approved", "reviewer": "Dr. Rao", "comment": "agree with effusion", "disagree_findings": ["Atelectasis"]}
```
`decision` ∈ {approved, rejected}; `reviewer` is required. A decision can be changed later; every change is a new audit entry.

**Response:** `{"study_id": "...", "status": "approved", "audit_hash": "3f9c…"}`.

**Errors:**
- `404` unknown study
- `409` study not analysed yet
- `422` invalid body

## GET /api/audit

```json
{"verification": {"valid": true, "entries": 31, "head": "e1d2…"},
 "entries": [{"id": 31, "ts": 1795021012.4, "actor": "Dr. Rao", "action": "radiologist_approved",
              "study_id": "8a4f…", "details": {"comment": "agree with effusion", "disagree_findings": ["Atelectasis"],
              "ai_summary": "Effusion, Atelectasis", "previous_status": "analyzed"},
              "prev_hash": "…", "hash": "…"}]}
```

Actions:
- `study_received`
- `ai_analyzed` (includes model versions)
- `ai_failed`
- `radiologist_approved`
- `radiologist_rejected`

## DICOM interfaces

| Direction | Protocol | Details |
|---|---|---|
| Modality → Orthanc | DICOM C-STORE, port 4242 | any AE title; `scripts/send_studies.py` |
| Worker ← Orthanc | REST: `/changes`, `/studies/{id}`, `/series/{id}`, `/instances/{id}/file` | triggered by `StableStudy` |
| Worker → Orthanc | REST `POST /instances` | Secondary Capture series *"AI overlay (education only)"* in the same study |
