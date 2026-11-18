import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("WARM_START", "1")


@pytest.fixture(scope="session")
def trained():
    """Make sure a trained model exists (runs the quick pipeline once if needed)."""
    from fraudplat.config import load_config, path
    art = path(load_config()["serving"]["artifacts_dir"])
    if not (art / "training_report.json").exists():
        cfg = load_config()
        cfg["data"]["synthetic_rows"] = 60_000
        cfg["xgboost"]["n_estimators"] = 200
        from fraudplat import pipeline
        pipeline.run("synthetic", variants=["class_weight"])
    return art


@pytest.fixture(scope="session")
def small_raw():
    from fraudplat.data import generate_synthetic
    return generate_synthetic(n_rows=20_000, seed=7)
