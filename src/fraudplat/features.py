"""Feature engineering.

Two feature families, with a single definition each:

1. **Entity profile features** (long-horizon, batch) -> stored in **Feast**.
   Daily snapshots per account (uid) and per device. Training reads them with a
   point-in-time join (`get_historical_features`), serving reads the same views from the
   online store. Same code, same values -> no training/serving skew.

2. **Real-time features** (velocity, change flags) -> computed from strictly *prior*
   events. `build_realtime_features` is the vectorized batch version used for training;
   `OnlineFeatureEngine` is the incremental version used by the streaming service.
   `tests/test_features.py` asserts both produce identical values.

Derived features (amount vs. the user's average etc.) combine both families through one
function, `derive`, used in both training and serving.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import load_config
from .data import NUM_PASSTHROUGH

CATEGORICAL = ["product", "card4", "card6", "device_type", "p_email"]
WINDOWS = load_config()["features"]["velocity_windows_sec"]
WIN_NAMES = {3600: "1h", 86400: "24h", 604800: "7d"}

REALTIME_FEATURES = (
    [f"uid_cnt_{WIN_NAMES.get(w, w)}" for w in WINDOWS]
    + ["uid_amt_sum_24h", "uid_secs_since_last", "uid_prior_cnt",
       "dev_new_for_uid", "dev_changed", "email_changed",
       "dev_prior_uids", "card_prior_uids", "dev_cnt_24h",
       "card_addr_changed", "card_prior_addrs"]
)
ACCOUNT_PROFILE = ["acct_txn_count", "acct_amt_mean", "acct_amt_std", "acct_n_devices",
                   "acct_night_ratio", "acct_active_days"]
DEVICE_PROFILE = ["dev_n_accounts", "dev_txn_count", "dev_amt_mean"]
DERIVED = ["log_amount", "amt_cents", "hour", "dow", "amt_to_acct_mean", "amt_zscore",
           "amt_to_dev_mean"]
MODEL_FEATURES = (DERIVED + REALTIME_FEATURES + ACCOUNT_PROFILE + DEVICE_PROFILE
                  + NUM_PASSTHROUGH + CATEGORICAL)


# --------------------------------------------------------------------------------------
# 1. Real-time features: vectorized batch version
# --------------------------------------------------------------------------------------
def _window_counts(g_ts: np.ndarray, w: int) -> np.ndarray:
    """For sorted timestamps of one group: number of *previous* events with ts >= t - w."""
    idx = np.arange(len(g_ts))
    return idx - np.searchsorted(g_ts, g_ts - w, side="left")


def _grouped_window(df: pd.DataFrame, key: str, w: int, values: np.ndarray | None = None) -> np.ndarray:
    """Count (or sum of `values`) of prior events within window w, per group. df must be time-sorted."""
    out = np.zeros(len(df))
    order = np.lexsort((np.arange(len(df)), df[key].to_numpy()))   # group, then time order
    keys = df[key].to_numpy()[order]
    ts = df["ts"].to_numpy()[order]
    vals = None if values is None else values[order]
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:], strict=False):
        t = ts[s:e]
        lo = np.searchsorted(t, t - w, side="left")
        if vals is None:
            res = np.arange(e - s) - lo
        else:
            cs = np.r_[0.0, np.cumsum(vals[s:e])]
            res = cs[np.arange(e - s)] - cs[lo]
        out[order[s:e]] = res
    return out


def _prior_distinct(df: pd.DataFrame, group: str, member: str) -> np.ndarray:
    """Distinct `member` values seen for `group` strictly before each row (excluding this row's own member
    if first seen now)."""
    first = ~df.duplicated([group, member])
    cum = first.groupby(df[group]).cumsum()
    return (cum - first).to_numpy().astype(float)


def build_realtime_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: canonical transactions sorted by (ts, transaction_id)."""
    out = pd.DataFrame(index=df.index)
    for w in WINDOWS:
        out[f"uid_cnt_{WIN_NAMES.get(w, w)}"] = _grouped_window(df, "uid", w)
    out["uid_amt_sum_24h"] = _grouped_window(df, "uid", 86400, df["amount"].to_numpy())
    g = df.groupby("uid", sort=False)
    out["uid_secs_since_last"] = (df["ts"] - g["ts"].shift(1)).astype(float)
    out["uid_prior_cnt"] = g.cumcount().astype(float)
    known = df.duplicated(["uid", "device_id"]) | (df.device_id == "d_unknown")
    out["dev_new_for_uid"] = (~known).astype(float)
    prev_dev = g["device_id"].shift(1)
    out["dev_changed"] = (prev_dev.notna() & (prev_dev != df.device_id)).astype(float)
    prev_em = g["email_id"].shift(1)
    out["email_changed"] = (prev_em.notna() & (prev_em != df.email_id)).astype(float)
    pdu = _prior_distinct(df, "device_id", "uid")
    out["dev_prior_uids"] = np.where(df.device_id == "d_unknown", np.nan, pdu)
    out["card_prior_uids"] = _prior_distinct(df, "card_id", "uid")
    # location change: same card used from a different billing address than its previous txn
    prev_addr = df.groupby("card_id", sort=False)["addr_id"].shift(1)
    out["card_addr_changed"] = (prev_addr.notna() & (prev_addr != df.addr_id)).astype(float)
    out["card_prior_addrs"] = _prior_distinct(df, "card_id", "addr_id")
    dc = _grouped_window(df, "device_id", 86400)
    out["dev_cnt_24h"] = np.where(df.device_id == "d_unknown", np.nan, dc)
    return out[REALTIME_FEATURES]


# --------------------------------------------------------------------------------------
# 1b. Real-time features: incremental (online) version - identical semantics
# --------------------------------------------------------------------------------------
@dataclass
class OnlineFeatureEngine:
    """Per-entity streaming state. Memory is bounded by the 7-day window for velocity
    deques; distinct-member sets grow with entity cardinality (fine for one node; shard by
    uid via Kafka partition key to scale out)."""
    uid_ts: dict = field(default_factory=lambda: defaultdict(deque))
    uid_amt24: dict = field(default_factory=lambda: defaultdict(deque))
    uid_last: dict = field(default_factory=dict)          # uid -> (ts, device, email)
    uid_cnt: dict = field(default_factory=lambda: defaultdict(int))
    uid_devices: dict = field(default_factory=lambda: defaultdict(set))
    dev_uids: dict = field(default_factory=lambda: defaultdict(set))
    card_uids: dict = field(default_factory=lambda: defaultdict(set))
    dev_ts: dict = field(default_factory=lambda: defaultdict(deque))
    card_last_addr: dict = field(default_factory=dict)
    card_addrs: dict = field(default_factory=lambda: defaultdict(set))

    def compute(self, t: dict, update: bool = True) -> dict:
        ts, uid, dev, card, email = int(t["ts"]), t["uid"], t["device_id"], t["card_id"], t["email_id"]
        f: dict = {}
        q = self.uid_ts[uid]
        for w in WINDOWS:
            f[f"uid_cnt_{WIN_NAMES.get(w, w)}"] = float(sum(1 for x in q if x >= ts - w))
        f["uid_amt_sum_24h"] = float(sum(a for (x, a) in self.uid_amt24[uid] if x >= ts - 86400))
        last = self.uid_last.get(uid)
        f["uid_secs_since_last"] = float(ts - last[0]) if last else np.nan
        f["uid_prior_cnt"] = float(self.uid_cnt[uid])
        unknown = dev == "d_unknown"
        f["dev_new_for_uid"] = 0.0 if (unknown or dev in self.uid_devices[uid]) else 1.0
        f["dev_changed"] = float(bool(last) and last[1] != dev)
        f["email_changed"] = float(bool(last) and last[2] != email)
        # distinct accounts seen on this device / card before this event
        f["dev_prior_uids"] = np.nan if unknown else float(len(self.dev_uids[dev]))
        f["card_prior_uids"] = float(len(self.card_uids[card]))
        dq = self.dev_ts[dev]
        f["dev_cnt_24h"] = np.nan if unknown else float(sum(1 for x in dq if x >= ts - 86400))
        last_addr = self.card_last_addr.get(card)
        f["card_addr_changed"] = float(last_addr is not None and last_addr != t["addr_id"])
        f["card_prior_addrs"] = float(len(self.card_addrs[card]))
        if update:
            self.update(t)
        return f

    def update(self, t: dict) -> None:
        ts, uid, dev, card, email = int(t["ts"]), t["uid"], t["device_id"], t["card_id"], t["email_id"]
        maxw = max(WINDOWS)
        q = self.uid_ts[uid]
        q.append(ts)
        while q and q[0] < ts - maxw:
            q.popleft()
        a = self.uid_amt24[uid]
        a.append((ts, float(t["amount"])))
        while a and a[0][0] < ts - 86400:
            a.popleft()
        self.uid_last[uid] = (ts, dev, email)
        self.uid_cnt[uid] += 1
        self.uid_devices[uid].add(dev)
        self.dev_uids[dev].add(uid)
        self.card_uids[card].add(uid)
        self.card_last_addr[card] = t["addr_id"]
        self.card_addrs[card].add(t["addr_id"])
        d = self.dev_ts[dev]
        d.append(ts)
        while d and d[0] < ts - 86400:
            d.popleft()

    def warm_start(self, df: pd.DataFrame) -> "OnlineFeatureEngine":
        for rec in df[["ts", "uid", "device_id", "card_id", "email_id", "addr_id", "amount"]].to_dict("records"):
            self.update(rec)
        return self


# --------------------------------------------------------------------------------------
# 2. Entity profile features (Feast) - daily snapshots, cumulative through end of day
# --------------------------------------------------------------------------------------
def _snapshot_ts(ts: pd.Series) -> pd.Series:
    """A day's snapshot becomes visible at the end of that day (start of the next)."""
    return pd.to_datetime((ts // 86400 + 1) * 86400, unit="s", utc=True)


def build_account_profiles(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["uid", "ts", "amount", "device_id"]].copy()
    d["day"] = d.ts // 86400
    d["night"] = ((d.ts % 86400) // 3600 < 6).astype(float)
    d["amt2"] = d.amount ** 2
    d["new_dev"] = (~d.duplicated(["uid", "device_id"])).astype(float)
    daily = d.groupby(["uid", "day"], sort=True).agg(
        n=("amount", "size"), s=("amount", "sum"), s2=("amt2", "sum"),
        nd=("new_dev", "sum"), night=("night", "sum"), ts=("ts", "max")).reset_index()
    g = daily.groupby("uid", sort=False)
    n = g["n"].cumsum()
    mean = g["s"].cumsum() / n
    var = (g["s2"].cumsum() / n - mean ** 2).clip(lower=0)
    return pd.DataFrame({
        "uid": daily.uid,
        "event_timestamp": _snapshot_ts(daily.ts),
        "acct_txn_count": n.astype(float),
        "acct_amt_mean": mean,
        "acct_amt_std": np.sqrt(var),
        "acct_n_devices": g["nd"].cumsum().astype(float),
        "acct_night_ratio": g["night"].cumsum() / n,
        "acct_active_days": (g.cumcount() + 1).astype(float),
    })


def build_device_profiles(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[df.device_id != "d_unknown", ["device_id", "uid", "ts", "amount"]].copy()
    d["day"] = d.ts // 86400
    d["new_uid"] = (~d.duplicated(["device_id", "uid"])).astype(float)
    daily = d.groupby(["device_id", "day"], sort=True).agg(
        n=("amount", "size"), s=("amount", "sum"), nu=("new_uid", "sum"), ts=("ts", "max")).reset_index()
    g = daily.groupby("device_id", sort=False)
    n = g["n"].cumsum()
    return pd.DataFrame({
        "device_id": daily.device_id,
        "event_timestamp": _snapshot_ts(daily.ts),
        "dev_n_accounts": g["nu"].cumsum().astype(float),
        "dev_txn_count": n.astype(float),
        "dev_amt_mean": g["s"].cumsum() / n,
    })


# --------------------------------------------------------------------------------------
# 3. Derived features - shared by training and serving
# --------------------------------------------------------------------------------------
def derive(df: pd.DataFrame) -> pd.DataFrame:
    """df must contain amount, ts and the profile columns. Returns DERIVED columns."""
    out = pd.DataFrame(index=df.index)
    amt = df["amount"].astype(float)
    out["log_amount"] = np.log1p(amt)
    out["amt_cents"] = (amt - np.floor(amt)).round(3)
    out["hour"] = ((df["ts"] % 86400) // 3600).astype(float)
    out["dow"] = ((df["ts"] // 86400) % 7).astype(float)
    out["amt_to_acct_mean"] = amt / df["acct_amt_mean"]
    out["amt_zscore"] = (amt - df["acct_amt_mean"]) / (df["acct_amt_std"] + 1.0)
    out["amt_to_dev_mean"] = amt / df["dev_amt_mean"]
    return out


class CategoryEncoder:
    """Stable category -> int mapping learned on training data, saved with the model."""

    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}

    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        for c in CATEGORICAL:
            vc = df[c].value_counts()
            vc = vc[vc >= 20]                                  # rare -> "other" (-1)
            self.mapping[c] = {k: i for i, k in enumerate(vc.index.astype(str))}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in CATEGORICAL:
            out[c] = df[c].astype(str).map(self.mapping[c]).fillna(-1).astype(float)
        return out


def assemble_record(rec: dict, encoder: CategoryEncoder) -> np.ndarray:
    """Single-record, pure-Python twin of `derive` + `assemble` for low-latency serving.
    Parity with the pandas version is asserted in tests/test_features.py."""
    amt, ts = float(rec["amount"]), int(rec["ts"])
    am, sd, dm = rec["acct_amt_mean"], rec["acct_amt_std"], rec["dev_amt_mean"]
    r = dict(rec)
    r["log_amount"] = float(np.log1p(amt))
    r["amt_cents"] = round(amt - np.floor(amt), 3)
    r["hour"] = float((ts % 86400) // 3600)
    r["dow"] = float((ts // 86400) % 7)
    r["amt_to_acct_mean"] = amt / am if am == am else np.nan
    r["amt_zscore"] = (amt - am) / (sd + 1.0) if am == am else np.nan
    r["amt_to_dev_mean"] = amt / dm if dm == dm else np.nan
    for c in CATEGORICAL:
        r[c] = float(encoder.mapping[c].get(str(rec[c]), -1))
    return np.array([r[f] for f in MODEL_FEATURES], dtype=np.float32)


def assemble(df: pd.DataFrame, encoder: CategoryEncoder) -> pd.DataFrame:
    """Final model matrix in the exact column order the model expects."""
    full = pd.concat([df, derive(df)], axis=1)
    full = encoder.transform(full)
    return full[MODEL_FEATURES].astype("float32")
