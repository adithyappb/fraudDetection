"""Minimal Orthanc REST client (https://orthanc.uclouvain.be/book/users/rest.html)."""
from __future__ import annotations

import httpx


class Orthanc:
    def __init__(self, url: str, auth: tuple[str, str] | None = None, timeout: float = 30):
        self.c = httpx.Client(base_url=url.rstrip("/"), auth=auth, timeout=timeout)

    def _get(self, p, **kw):
        r = self.c.get(p, **kw)
        r.raise_for_status()
        return r

    def system(self) -> dict:
        return self._get("/system").json()

    def changes(self, since: int, limit: int = 100) -> dict:
        """Change log: NewInstance, NewSeries, StableSeries, StableStudy, ... (monotonic `Seq`)."""
        return self._get("/changes", params={"since": since, "limit": limit}).json()

    def last_change(self) -> int:
        return self._get("/changes", params={"last": True}).json()["Last"]

    def study(self, sid: str) -> dict:
        return self._get(f"/studies/{sid}").json()

    def series(self, sid: str) -> dict:
        return self._get(f"/series/{sid}").json()

    def instance_file(self, iid: str) -> bytes:
        return self._get(f"/instances/{iid}/file").content

    def upload(self, dicom_bytes: bytes) -> dict:
        r = self.c.post("/instances", content=dicom_bytes, headers={"Content-Type": "application/dicom"})
        r.raise_for_status()
        return r.json()

    def studies(self) -> list[str]:
        return self._get("/studies").json()
