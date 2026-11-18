import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def trained():
    if not (ROOT / "models" / "cxr_classifier.pt").exists() or not (ROOT / "metrics" / "classification.json").exists():
        pytest.skip("run `make train` first")
    return ROOT


@pytest.fixture(scope="session")
def orthanc_url():
    import httpx
    from medai.config import params
    url = params()["service"]["orthanc_url"]
    try:
        httpx.get(url + "/system", timeout=2).raise_for_status()
    except Exception:
        pytest.skip(f"Orthanc not reachable at {url} (run `make orthanc`)")
    return url
