"""REST API for the React viewer (and anything else). Serves the built viewer at /.

  uvicorn medai.service.api:app --port 8000          (RUN_WORKER=1 also starts the Orthanc worker)
"""
from __future__ import annotations

import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps
from PIL import Image
from pydantic import BaseModel, Field

from ..config import DISCLAIMER, ROOT, params, path
from .db import ResultsDB

STATE: dict = {}


class Decision(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(..., min_length=1, max_length=80)
    comment: Optional[str] = Field(None, max_length=2000)
    disagree_findings: list[str] = Field(default_factory=list, description="AI findings the reviewer disagrees with")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = params()["service"]
    STATE["db"] = STATE.get("db") or ResultsDB(path(cfg["db_path"]))
    stop = threading.Event()
    if os.environ.get("RUN_WORKER") == "1":
        from .worker import Worker
        w = Worker(db=STATE["db"])
        threading.Thread(target=w.run_forever, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="AI-Assisted Medical Imaging API", version="1.0.0", lifespan=lifespan,
              description=f"**{DISCLAIMER}**  Chest X-ray findings with Grad-CAM, CT spleen segmentation, "
                          "radiologist review and a tamper-evident audit log.")


def db() -> ResultsDB:
    return STATE["db"]


def _study_or_404(sid):
    s = db().study(sid)
    if not s:
        raise HTTPException(404, "study not found")
    return s


def _png(arr: np.ndarray) -> Response:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def _dir(sid):
    return path(params()["service"]["results_dir"]) / sid


@app.get("/api/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/api/studies")
def studies(status: Optional[str] = None):
    rows = db().studies()
    out = []
    for r in rows:
        if status and r["status"] != status:
            continue
        ai = r["ai"] or {}
        top = (ai.get("findings") or [{}])[0]
        out.append({k: r[k] for k in ("study_id", "patient_id", "patient_sex", "patient_age", "modality", "kind",
                                      "description", "status", "received_at", "reviewer", "decision")}
                   | {"summary": ai.get("summary"), "top_finding": top.get("finding"),
                      "top_probability": top.get("probability"), "spleen_volume_ml": ai.get("spleen_volume_ml")})
    return out


@app.get("/api/studies/{sid}")
def study(sid: str):
    s = _study_or_404(sid)
    s["audit"] = db().audit_log(sid, 50)
    s["disclaimer"] = DISCLAIMER
    return s


@app.get("/api/studies/{sid}/image.png")
def image(sid: str, slice: int = Query(None, ge=0)):
    s, d = _study_or_404(sid), _dir(sid)
    if s["kind"] == "cxr":
        return _png(np.load(d / "image.npy"))
    vol = np.load(d / "volume.npy", mmap_mode="r")
    z = vol.shape[2] // 2 if slice is None else min(slice, vol.shape[2] - 1)
    return _png(np.ascontiguousarray(vol[:, :, z].T))                   # rows = y: anterior up, patient left on the right


@app.get("/api/studies/{sid}/overlay.png")
def overlay_png(sid: str, finding: Optional[str] = None, slice: int = Query(None, ge=0)):
    """Transparent RGBA overlay: Grad-CAM heat map (CXR) or segmentation mask (CT)."""
    s, d = _study_or_404(sid), _dir(sid)
    if s["kind"] == "cxr":
        ai = s["ai"]
        names = [f["finding"] for f in sorted(ai["findings"], key=lambda f: f["finding"])]
        order = params()["classification"]["findings"]
        name = finding or ai["findings"][0]["finding"]
        if name not in names:
            raise HTTPException(404, "unknown finding")
        cam = np.load(d / "cams.npy")[order.index(name)].astype(np.float32)
        h, w = np.load(d / "image.npy", mmap_mode="r").shape
        cam = np.asarray(Image.fromarray(cam).resize((w, h), Image.BILINEAR))
        rgba = (colormaps["jet"](cam) * 255).astype(np.uint8)
        rgba[..., 3] = (np.clip((cam - 0.25) / 0.75, 0, 1) * 170).astype(np.uint8)
        return _png(rgba)
    mask = np.load(d / "mask.npy", mmap_mode="r")
    z = mask.shape[2] // 2 if slice is None else min(slice, mask.shape[2] - 1)
    m = np.ascontiguousarray(mask[:, :, z].T).astype(bool)
    rgba = np.zeros(m.shape + (4,), np.uint8)
    rgba[m] = (255, 70, 70, 120)
    edge = m & ~np.pad(m, 1)[2:, 1:-1] | m & ~np.pad(m, 1)[:-2, 1:-1] | m & ~np.pad(m, 1)[1:-1, 2:] | m & ~np.pad(m, 1)[1:-1, :-2]
    rgba[edge] = (255, 40, 40, 255)
    return _png(rgba)


@app.post("/api/studies/{sid}/decision")
def decide(sid: str, d: Decision):
    s = _study_or_404(sid)
    if s["status"] not in ("analyzed", "approved", "rejected"):
        raise HTTPException(409, f"study is '{s['status']}', not ready for review")
    import time
    db().upsert_study(study_id=sid, status=d.decision, reviewer=d.reviewer, decision=d.decision,
                      decision_comment=d.comment, decided_at=time.time())
    entry = db().audit(d.reviewer, f"radiologist_{d.decision}", sid,
                       {"comment": d.comment, "disagree_findings": d.disagree_findings,
                        "ai_summary": (s["ai"] or {}).get("summary"), "previous_status": s["status"]})
    return {"study_id": sid, "status": d.decision, "audit_hash": entry["hash"]}


@app.get("/api/audit")
def audit(limit: int = Query(200, le=5000)):
    return {"verification": db().verify_audit(), "entries": db().audit_log(limit=limit)}


@app.get("/api/model-card")
def model_card():
    import json
    out = {"disclaimer": DISCLAIMER, "intended_use": "Education and research demonstration only."}
    for name in ("classification", "segmentation", "data_summary", "train_cls"):
        f = path(f"metrics/{name}.json")
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


_web = ROOT / "web" / "dist"
if _web.exists():
    app.mount("/assets", StaticFiles(directory=_web / "assets"), name="assets")

    @app.get("/{full:path}", include_in_schema=False)
    def spa(full: str):
        return FileResponse(_web / "index.html")
