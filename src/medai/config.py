"""Loads params.yaml (also read by DVC). Paths are resolved against the project root."""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = "FOR EDUCATION AND RESEARCH ONLY - NOT FOR CLINICAL USE. Not a medical device."


@lru_cache(maxsize=1)
def params() -> dict:
    p = yaml.safe_load(open(os.environ.get("MEDAI_PARAMS", ROOT / "params.yaml")))
    if os.environ.get("ORTHANC_URL"):
        p["service"]["orthanc_url"] = os.environ["ORTHANC_URL"]
    return p


def path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"
