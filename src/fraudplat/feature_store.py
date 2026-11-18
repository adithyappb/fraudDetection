"""Feature store access.

`FeatureStoreClient` wraps Feast (feature_repo/). If Feast is not installed (or
`feast.use_feast: false`), it falls back to `LocalPointInTimeStore`, which implements the
same two operations with identical semantics:

  * historical retrieval = point-in-time join: for each (entity, event time) take the latest
    snapshot with snapshot_ts <= event time (and within TTL)
  * online retrieval = latest snapshot per entity

Both read the same parquet files, so models trained with either backend are interchangeable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, path
from .features import ACCOUNT_PROFILE, DEVICE_PROFILE

log = logging.getLogger(__name__)

VIEWS = {
    "account_profile": ("uid", ACCOUNT_PROFILE),
    "device_profile": ("device_id", DEVICE_PROFILE),
    "account_graph_risk": ("uid", ["gnn_score"]),
    "device_graph_risk": ("device_id", ["gnn_dev_mean", "gnn_dev_max"]),
}
TTL_DAYS = 400


def offline_dir() -> Path:
    d = path(load_config()["data"]["processed_dir"]) / "feast"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_view(name: str, df: pd.DataFrame) -> Path:
    p = offline_dir() / f"{name}.parquet"
    df = df.copy()
    df["created"] = pd.Timestamp.now(tz="UTC")
    df.to_parquet(p, index=False)
    return p


class LocalPointInTimeStore:
    def __init__(self):
        self._online: dict[str, pd.DataFrame] = {}

    def _load(self, view):
        return pd.read_parquet(offline_dir() / f"{view}.parquet")

    def get_historical(self, entity_df: pd.DataFrame, view: str) -> pd.DataFrame:
        key, cols = VIEWS[view]
        snap = self._load(view).sort_values("event_timestamp")
        left = entity_df[[key, "event_timestamp"]].reset_index().sort_values("event_timestamp")
        merged = pd.merge_asof(left, snap[[key, "event_timestamp"] + cols].rename(
            columns={"event_timestamp": "snap_ts"}), left_on="event_timestamp", right_on="snap_ts",
            by=key, direction="backward", allow_exact_matches=True)
        stale = (merged.event_timestamp - merged.snap_ts) > pd.Timedelta(days=TTL_DAYS)
        merged.loc[stale, cols] = np.nan
        return merged.set_index("index").loc[entity_df.index, cols]

    def materialize(self, end: pd.Timestamp | None = None):
        """Load the latest snapshot per entity with snapshot_ts <= end into the online table and
        persist it (the serving process reads the persisted file)."""
        for view, (key, cols) in VIEWS.items():
            f = offline_dir() / f"{view}.parquet"
            if f.exists():
                d = pd.read_parquet(f)
                if end is not None and not view.endswith("graph_risk"):
                    d = d[d.event_timestamp <= end]
                d = d.sort_values("event_timestamp").drop_duplicates(key, keep="last")
                d[[key] + cols].to_parquet(offline_dir() / f"{view}.online.parquet", index=False)
                self._online[view] = d.set_index(key)[cols]

    def _load_online(self):
        for view, (key, cols) in VIEWS.items():
            f = offline_dir() / f"{view}.online.parquet"
            if f.exists():
                self._online[view] = pd.read_parquet(f).set_index(key)[cols]
            else:
                self.materialize()
                return

    def get_online(self, view: str, keys: list[str]) -> pd.DataFrame:
        key, cols = VIEWS[view]
        if not self._online:
            self._load_online()
        tbl = self._online.get(view)
        if tbl is None:
            return pd.DataFrame(np.nan, index=range(len(keys)), columns=cols)
        return tbl.reindex(keys).reset_index(drop=True)


def online_profile(store, uid: str, device_id: str) -> dict:
    """All serving-time Feast features for one transaction in a single round trip."""
    if isinstance(store, FeastStore):
        refs = [f"{v}:{c}" for v, (_, cols) in VIEWS.items() for c in cols]
        resp = store.fs.get_online_features(features=refs, entity_rows=[{"uid": uid, "device_id": device_id}]).to_dict()
        return {k.split("__")[-1]: (np.nan if v[0] is None else float(v[0])) for k, v in resp.items()
                if k not in ("uid", "device_id")}
    out = {}
    for v, (key, cols) in VIEWS.items():
        row = store.get_online(v, [uid if key == "uid" else device_id]).iloc[0]
        out.update({c: float(row[c]) for c in cols})
    return out


class FeastStore:
    """Thin adapter over feast.FeatureStore for the three views in feature_repo/."""

    def __init__(self, repo: Path):
        from feast import FeatureStore  # noqa: WPS433
        self.repo = repo
        self.fs = FeatureStore(repo_path=str(repo))

    def apply(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("feast_defs", self.repo / "definitions.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.fs.apply(mod.ALL_OBJECTS)

    def get_historical(self, entity_df: pd.DataFrame, view: str) -> pd.DataFrame:
        key, cols = VIEWS[view]
        ent = entity_df[[key, "event_timestamp"]].copy()
        ent["_row"] = np.arange(len(ent))
        job = self.fs.get_historical_features(entity_df=ent, features=[f"{view}:{c}" for c in cols])
        res = job.to_df().drop_duplicates("_row").set_index("_row").reindex(range(len(ent)))
        res.index = entity_df.index
        return res[cols].astype(float)

    def materialize(self, end: pd.Timestamp | None = None):
        start = pd.Timestamp("1970-01-02", tz="UTC").to_pydatetime()
        end = (end or pd.Timestamp.now(tz="UTC")).to_pydatetime()
        self.fs.materialize(start_date=start, end_date=end,
                            feature_views=["account_profile", "device_profile"])
        # graph risk scores are a single snapshot at end of data -> always load latest
        self.fs.materialize(start_date=start, end_date=pd.Timestamp.now(tz="UTC").to_pydatetime(),
                            feature_views=["account_graph_risk", "device_graph_risk"])

    def get_online(self, view: str, keys: list[str]) -> pd.DataFrame:
        key, cols = VIEWS[view]
        resp = self.fs.get_online_features(features=[f"{view}:{c}" for c in cols],
                                           entity_rows=[{key: k} for k in keys]).to_dict()
        return pd.DataFrame({c: resp[c] for c in cols}).astype(float)


def get_store(prefer_feast: bool | None = None):
    cfg = load_config()["feast"]
    want = cfg["use_feast"] if prefer_feast is None else prefer_feast
    if want in (True, "auto"):
        try:
            store = FeastStore(path(cfg["repo_path"]))
            log.info("Using Feast feature store")
            return store
        except Exception as e:  # ImportError or misconfiguration
            if want is True:
                raise
            log.warning("Feast unavailable (%s); using built-in point-in-time store", e)
    return LocalPointInTimeStore()
