"""Make data/raw ready: keep real NIH / MSD data if present, otherwise render phantoms.

  python -m medai.acquire            # then: dvc add data/raw   (versions the raw data)
"""
from __future__ import annotations

import logging

from .config import params, path
from .data import msd, nih

log = logging.getLogger("medai.acquire")


def has_real_cxr(d) -> bool:
    return (d / "images").exists() and any((d / "images").glob("*.png")) and not (d / "PHANTOM.txt").exists()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = params()["data"]
    cxr, ct = path(p["cxr_raw_dir"]), path(p["ct_raw_dir"])
    src = p["source"]
    if src == "real" or (src == "auto" and has_real_cxr(cxr)):
        log.info("NIH ChestX-ray14: using real images in %s", cxr)
    elif not (cxr / "PHANTOM.txt").exists():
        log.info("NIH ChestX-ray14 images not found -> rendering %d phantoms from real NIH metadata", p["phantom_cxr_images"])
        nih.make_phantoms(cxr, p["phantom_cxr_images"], p["seed"])
    if (ct / "dataset.json").exists():
        log.info("MSD Task09_Spleen: found %s", ct)
    else:
        log.info("MSD Task09_Spleen not found -> rendering %d CT phantoms", p["phantom_ct_volumes"])
        msd.make_phantoms(ct, p["phantom_ct_volumes"], p["seed"])
    log.info("data/raw ready")


if __name__ == "__main__":
    main()
