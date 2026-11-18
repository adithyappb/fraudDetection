"""Watches Orthanc for new studies, runs inference, stores results, pushes an AI overlay back.

Loop: GET /changes since the last seen sequence number -> on `StableStudy` (Orthanc has received
no new instance for StableAge seconds) -> download every instance -> route by modality
(DX/CR -> classifier + Grad-CAM, CT -> 3D U-Net) -> save arrays + JSON, audit entry ->
upload a Secondary Capture overlay series into the same study (visible in any DICOM viewer).

  python -m medai.service.worker
"""
from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np

from ..config import DISCLAIMER, params, path
from ..data import dicom_io
from ..explain import overlay
from .db import ResultsDB
from .orthanc import Orthanc

log = logging.getLogger("medai.worker")
AI_SERIES = "AI overlay (education only)"


class Worker:
    def __init__(self, db: ResultsDB | None = None, orthanc: Orthanc | None = None, predictor=None):
        cfg = params()["service"]
        self.cfg = cfg
        self.db = db or ResultsDB(path(cfg["db_path"]))
        self.orthanc = orthanc or Orthanc(cfg["orthanc_url"])
        self._predictor = predictor
        self.results = path(cfg["results_dir"])

    @property
    def predictor(self):
        if self._predictor is None:
            from ..inference import Predictor
            self._predictor = Predictor()
        return self._predictor

    def poll_once(self) -> list[str]:
        since = int(self.db.get_state("orthanc_seq", 0))
        ch = self.orthanc.changes(since)
        done = []
        for c in ch["Changes"]:
            if c["ChangeType"] == "StableStudy":
                existing = self.db.study(c["ID"])
                if existing and existing["status"] not in ("received", "error"):
                    continue                        # our own overlay upload re-triggers StableStudy
                try:
                    self.process(c["ID"])
                    done.append(c["ID"])
                except Exception as e:
                    log.exception("study %s failed", c["ID"])
                    self.db.upsert_study(study_id=c["ID"], status="error", ai=json.dumps({"error": str(e)}))
                    self.db.audit("ai-worker", "ai_failed", c["ID"], {"error": str(e)})
        self.db.set_state("orthanc_seq", ch["Last"])
        return done

    def process(self, sid: str) -> None:
        t0 = time.time()
        st = self.orthanc.study(sid)
        tags, pt = st["MainDicomTags"], st["PatientMainDicomTags"]
        series = [self.orthanc.series(s) for s in st["Series"]]
        series = [s for s in series if s["MainDicomTags"].get("SeriesDescription") != AI_SERIES]
        modality = series[0]["MainDicomTags"].get("Modality", "")
        datasets = [dicom_io.read(self.orthanc.instance_file(i)) for i in series[0]["Instances"]]
        common = dict(study_id=sid, study_uid=tags.get("StudyInstanceUID"), patient_id=pt.get("PatientID"),
                      patient_sex=datasets[0].get("PatientSex", ""), patient_age=datasets[0].get("PatientAge", ""),
                      modality=modality, description=tags.get("StudyDescription", ""), received_at=time.time())
        self.db.upsert_study(**common, status="received")
        self.db.audit("ai-worker", "study_received", sid, {"modality": modality, "instances": len(datasets)})
        out = self.results / sid
        out.mkdir(parents=True, exist_ok=True)
        if modality in ("DX", "CR"):
            r = self.predictor.analyze_cxr(datasets[0])
            np.save(out / "image.npy", r.pop("image"))
            np.save(out / "cams.npy", r.pop("cams"))
            kind = "cxr"
        elif modality == "CT":
            r = self.predictor.analyze_ct(datasets)
            np.save(out / "volume.npy", r.pop("volume"))
            np.save(out / "mask.npy", r.pop("mask"))
            np.save(out / "prob.npy", r.pop("prob"))
            kind = "ct"
        else:
            self.db.upsert_study(study_id=sid, status="unsupported", kind="other")
            return
        r["inference_seconds"] = round(time.time() - t0, 2)
        r["disclaimer"] = DISCLAIMER
        version = json.dumps(self.predictor.version)
        overlay_series = self._push_overlay(kind, out, datasets[0], r) if self.cfg["upload_overlay_to_orthanc"] else None
        self.db.upsert_study(study_id=sid, kind=kind, status="analyzed", analyzed_at=time.time(), ai=json.dumps(r),
                             model_version=version, overlay_series=overlay_series)
        self.db.audit("ai-worker", "ai_analyzed", sid, {"summary": r["summary"], "model": self.predictor.version,
                                                        "seconds": r["inference_seconds"]})
        log.info("study %s (%s): %s in %.1fs", sid[:8], kind, r["summary"], r["inference_seconds"])

    def _push_overlay(self, kind, out, ref, result) -> str | None:
        from pydicom.uid import generate_uid
        series_uid = generate_uid()
        if kind == "cxr":
            img = np.load(out / "image.npy").astype(np.float32) / 255
            cams = np.load(out / "cams.npy").astype(np.float32)
            k = self.predictor.findings.index(result["findings"][0]["finding"])   # most likely finding
            from PIL import Image
            cam = np.asarray(Image.fromarray(cams[k]).resize(img.shape[::-1], Image.BILINEAR))
            rgbs = [overlay(img, cam)]
        else:
            vol, mask = np.load(out / "volume.npy"), np.load(out / "mask.npy")
            zs = np.flatnonzero(mask.any((0, 1)))
            rgbs = []
            for z in zs[:: max(1, len(zs) // 12)]:
                base = np.repeat(vol[:, :, z].T[..., None], 3, -1).astype(np.float32)
                m = mask[:, :, z].T.astype(bool)
                base[m] = base[m] * 0.55 + np.array([255, 60, 60]) * 0.45
                rgbs.append(base.astype(np.uint8))
        for i, rgb in enumerate(rgbs):
            ds = dicom_io.secondary_capture(rgb, ref, AI_SERIES, instance=i + 1, series_uid=series_uid)
            self.orthanc.upload(dicom_io.to_bytes(ds))
        return series_uid

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("watching %s", self.cfg["orthanc_url"])
        while not stop.is_set():
            try:
                self.poll_once()
            except Exception as e:
                log.warning("Orthanc unreachable (%s); retrying", type(e).__name__)
            stop.wait(self.cfg["poll_seconds"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Worker().run_forever()
