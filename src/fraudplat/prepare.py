"""Step 1-2: load data, EDA summary, feature engineering, register features in Feast,
build the point-in-time-correct training table."""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from .config import load_config, path
from .data import load_raw, time_split, to_canonical
from .feature_store import get_store, write_view
from .features import (ACCOUNT_PROFILE, DEVICE_PROFILE, build_account_profiles,
                       build_device_profiles, build_realtime_features)

log = logging.getLogger(__name__)


def eda(df: pd.DataFrame, source: str) -> dict:
    split = time_split(df)
    rep = {
        "source": source,
        "rows": int(len(df)),
        "fraud_rate_pct": round(100 * df.is_fraud.mean(), 3),
        "imbalance_ratio": f"1:{(1 - df.is_fraud.mean()) / max(df.is_fraud.mean(), 1e-9):.0f}",
        "days": round((df.ts.max() - df.ts.min()) / 86400, 1),
        "accounts": int(df.uid.nunique()),
        "devices": int(df.device_id.nunique()),
        "cards": int(df.card_id.nunique()),
        "missing_device_pct": round(100 * (df.device_id == "d_unknown").mean(), 1),
        "split_rows": split.value_counts().to_dict(),
        "split_fraud_rate_pct": (100 * df.groupby(split).is_fraud.mean()).round(3).to_dict(),
        "amount_median_legit": float(df.loc[df.is_fraud == 0, "amount"].median()),
        "amount_median_fraud": float(df.loc[df.is_fraud == 1, "amount"].median()),
        "fraud_rate_by_product_pct": (100 * df.groupby("product").is_fraud.mean()).round(2).to_dict(),
        "night_fraud_rate_pct": round(100 * df.loc[(df.ts % 86400) // 3600 < 6, "is_fraud"].mean(), 2),
        "day_fraud_rate_pct": round(100 * df.loc[(df.ts % 86400) // 3600 >= 6, "is_fraud"].mean(), 2),
    }
    return rep


def run(source: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    t0 = time.time()
    raw, src = load_raw(source)
    df = to_canonical(raw)
    out_dir = path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # held-out test period, in raw IEEE-CIS format, for the Kafka replay
    split = time_split(df)
    test_start = int(df.ts[split == "test"].min())
    raw[raw.TransactionDT >= test_start].sort_values("TransactionDT").to_parquet(out_dir / "replay.parquet", index=False)
    # history before the replay window, used to warm-start the streaming feature state
    df.loc[df.ts < test_start, ["ts", "uid", "device_id", "card_id", "email_id", "addr_id", "amount"]].to_parquet(
        out_dir / "stream_history.parquet", index=False)
    del raw
    report = eda(df, src)
    (out_dir / "eda.json").write_text(json.dumps(report, indent=2, default=str))
    sp = report["split_fraud_rate_pct"]
    log.info("EDA: %s txns over %s days | fraud %.2f%% (%s) | %s accounts, %s devices, %s cards",
             f"{report['rows']:,}", report["days"], report["fraud_rate_pct"], report["imbalance_ratio"],
             f"{report['accounts']:,}", f"{report['devices']:,}", f"{report['cards']:,}")
    log.info("EDA: fraud rate by split train %.2f%% / valid %.2f%% / test %.2f%% | night %.1f%% vs day %.1f%% | "
             "median amount legit $%.0f vs fraud $%.0f", sp["train"], sp["valid"], sp["test"],
             report["night_fraud_rate_pct"], report["day_fraud_rate_pct"],
             report["amount_median_legit"], report["amount_median_fraud"])

    # --- real-time features (strictly prior events) ---
    rt = build_realtime_features(df)

    # --- entity profiles -> Feast offline store ---
    write_view("account_profile", build_account_profiles(df))
    write_view("device_profile", build_device_profiles(df))
    store = get_store()
    if hasattr(store, "apply"):
        store.apply()

    ent = pd.DataFrame({"uid": df.uid, "device_id": df.device_id,
                        "event_timestamp": pd.to_datetime(df.ts, unit="s", utc=True)})
    # Point-in-time join: a transaction only sees profile snapshots from *completed* days.
    acct = store.get_historical(ent, "account_profile")
    dev = store.get_historical(ent, "device_profile")

    table = pd.concat([df, rt, acct[ACCOUNT_PROFILE], dev[DEVICE_PROFILE]], axis=1)
    table["split"] = time_split(table)
    table.to_parquet(out_dir / "training_table.parquet", index=False)
    log.info("Training table: %s rows x %s cols in %.1fs", *table.shape, time.time() - t0)
    return table
