"""Latency + throughput benchmark: in-process scorer and over HTTP (if the API is running).

  python scripts/benchmark.py [--n 5000] [--http http://localhost:8000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.config import load_config, path  # noqa: E402
from fraudplat.serving.scorer import Scorer  # noqa: E402
from fraudplat.streaming.producer import load_replay, records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--http", default=None)
    a = ap.parse_args()
    recs = [r for r, _ in records(load_replay(a.n))]
    s = Scorer(warm_start=True)
    s.cases.conn.execute("PRAGMA synchronous=NORMAL")
    lat = []
    t0 = time.perf_counter()
    for r in recs:
        t = time.perf_counter()
        s.score_raw(r, record_case=False)
        lat.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t0
    out = {"in_process": {"n": len(recs), "p50_ms": np.percentile(lat, 50), "p95_ms": np.percentile(lat, 95),
                          "p99_ms": np.percentile(lat, 99), "throughput_tps_single_core": len(recs) / wall}}
    if a.http:
        import httpx
        c = httpx.Client(base_url=a.http)
        hl = []
        for r in recs[:1000]:
            t = time.perf_counter()
            c.post("/score", json=r).raise_for_status()
            hl.append((time.perf_counter() - t) * 1000)
        out["http_end_to_end"] = {"n": len(hl), "p50_ms": np.percentile(hl, 50), "p95_ms": np.percentile(hl, 95),
                                  "p99_ms": np.percentile(hl, 99)}
    out = json.loads(json.dumps(out, default=float))
    for v in out.values():
        for k in v:
            v[k] = round(v[k], 2)
    print(json.dumps(out, indent=2))
    (path(load_config()["serving"]["artifacts_dir"]) / "benchmark.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
