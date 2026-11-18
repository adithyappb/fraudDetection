"""Audit log integrity + end-to-end: DICOM -> real Orthanc -> worker -> API -> radiologist decision."""
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medai.config import PROCESSED, params, path
from medai.data import dicom_io
from medai.service.db import ResultsDB


def test_audit_chain_detects_tampering(tmp_path):
    db = ResultsDB(tmp_path / "a.db")
    for i in range(5):
        db.audit("dr.a", "radiologist_approved", f"s{i}", {"comment": f"ok {i}"})
    assert db.verify_audit() == {**db.verify_audit(), "valid": True, "entries": 5}
    db.conn.execute("UPDATE audit SET details='{\"comment\": \"edited\"}' WHERE id=3")
    v = db.verify_audit()
    assert not v["valid"] and v["broken_at_id"] == 3


@pytest.fixture(scope="module")
def stack(trained, orthanc_url, tmp_path_factory):
    from medai.service import api
    from medai.service.orthanc import Orthanc
    from medai.service.worker import Worker
    d = tmp_path_factory.mktemp("svc")
    params()["service"]["results_dir"] = str(d / "results")
    db = ResultsDB(d / "results.db")
    orth = Orthanc(orthanc_url)
    db.set_state("orthanc_seq", orth.last_change())               # only react to what this test sends
    worker = Worker(db=db, orthanc=orth)
    api.STATE["db"] = db
    with TestClient(api.app) as client:
        yield worker, orth, client


def _send_cxr(orth):
    idx = pd.read_csv(PROCESSED / "cxr_index.csv")
    row = idx[(idx.split == "test") & (idx.Effusion == 1)].iloc[0]
    img = np.asarray(Image.open(path(params()["data"]["cxr_raw_dir"]) / "images" / row.image).convert("L"))
    ds = dicom_io.cxr_dataset(img, {"id": f"TEST-{time.time_ns()}", "sex": row["sex"], "age": row["age"]}, row["view"])
    orth.upload(dicom_io.to_bytes(ds))
    return ds


def _wait(worker, n=1, timeout=40):
    done, t0 = [], time.time()
    while len(done) < n and time.time() - t0 < timeout:
        done += worker.poll_once()
        time.sleep(1)
    return done


def test_end_to_end_review_flow(stack):
    worker, orth, client = stack
    ds = _send_cxr(orth)
    done = _wait(worker)
    assert len(done) == 1, "worker did not pick up the stable study"
    sid = done[0]
    s = client.get(f"/api/studies/{sid}").json()
    assert s["status"] == "analyzed" and s["kind"] == "cxr" and len(s["ai"]["findings"]) == 8
    assert "NOT FOR CLINICAL USE" in s["disclaimer"]
    # overlay pushed back into the same study as a Secondary Capture series
    descs = [orth.series(x)["MainDicomTags"].get("SeriesDescription") for x in orth.study(sid)["Series"]]
    assert "AI overlay (education only)" in descs
    assert orth.study(sid)["MainDicomTags"]["StudyInstanceUID"] == ds.StudyInstanceUID
    # viewer assets
    assert client.get(f"/api/studies/{sid}/image.png").headers["content-type"] == "image/png"
    ov = client.get(f"/api/studies/{sid}/overlay.png", params={"finding": "Effusion"})
    assert ov.status_code == 200 and Image.open(__import__("io").BytesIO(ov.content)).mode == "RGBA"
    # radiologist decision -> audit
    assert client.post(f"/api/studies/{sid}/decision", json={"decision": "maybe", "reviewer": "x"}).status_code == 422
    r = client.post(f"/api/studies/{sid}/decision",
                    json={"decision": "approved", "reviewer": "dr.test", "comment": "agree", "disagree_findings": []})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    actions = [a["action"] for a in client.get(f"/api/studies/{sid}").json()["audit"]]
    assert actions[:3] == ["radiologist_approved", "ai_analyzed", "study_received"]
    assert client.get("/api/audit").json()["verification"]["valid"]
    # the overlay upload re-triggers StableStudy; the worker must not re-process the study
    assert _wait(worker, timeout=8) == []
