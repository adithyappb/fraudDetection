"""Configuration loading. One YAML file drives every component."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("FRAUD_CONFIG", ROOT / "configs" / "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Environment overrides used by docker-compose / Kubernetes
    if os.environ.get("KAFKA_BOOTSTRAP"):
        cfg["kafka"]["bootstrap"] = os.environ["KAFKA_BOOTSTRAP"]
    if os.environ.get("ARTIFACTS_DIR"):
        cfg["serving"]["artifacts_dir"] = os.environ["ARTIFACTS_DIR"]
        cfg["serving"]["db_path"] = str(Path(os.environ["ARTIFACTS_DIR"]) / "cases.db")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    return cfg


def path(rel: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
