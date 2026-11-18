"""Training/serving parity and leakage tests - the most important tests in the repo:
if these fail, offline metrics do not describe production behaviour."""
import numpy as np
import pandas as pd
import pytest

from fraudplat.data import canonical_record, to_canonical
from fraudplat.features import (MODEL_FEATURES, REALTIME_FEATURES, CategoryEncoder, OnlineFeatureEngine,
                                assemble, assemble_record, build_account_profiles, build_realtime_features)


def test_canonical_record_matches_batch(small_raw):
    batch = to_canonical(small_raw).set_index("transaction_id")
    for raw in small_raw.sample(500, random_state=0).to_dict("records"):
        raw = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in raw.items()}
        rec = canonical_record(raw)
        b = batch.loc[rec["transaction_id"]]
        for k in ["uid", "card_id", "device_id", "email_id", "product", "card4", "card6", "p_email", "device_type"]:
            assert rec[k] == b[k], k
        assert rec["amount"] == pytest.approx(b["amount"])


def test_online_engine_matches_batch_features(small_raw):
    df = to_canonical(small_raw).head(8000)
    batch = build_realtime_features(df)
    eng = OnlineFeatureEngine()
    online = pd.DataFrame([eng.compute(r) for r in df.to_dict("records")], index=df.index)[REALTIME_FEATURES]
    np.testing.assert_allclose(online.to_numpy(float), batch.to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True)


def test_realtime_features_use_only_prior_events(small_raw):
    df = to_canonical(small_raw)
    f = build_realtime_features(df)
    first = ~df.uid.duplicated()
    assert (f.loc[first, "uid_prior_cnt"] == 0).all()
    assert (f.loc[first, "uid_cnt_1h"] == 0).all()
    assert f.loc[first, "uid_secs_since_last"].isna().all()
    # a transaction never counts itself
    assert (f.uid_cnt_7d <= f.uid_prior_cnt).all()


def test_profiles_are_point_in_time(small_raw):
    df = to_canonical(small_raw)
    prof = build_account_profiles(df)
    # each snapshot becomes visible only after the day it summarises has ended
    last_ts = df.groupby(["uid", df.ts // 86400]).ts.max().reset_index(drop=True)
    snap_secs = prof.event_timestamp.astype("int64") // 10**9
    assert (snap_secs.to_numpy() > last_ts.to_numpy()).all()


def test_assemble_record_matches_batch(small_raw):
    df = to_canonical(small_raw).head(3000)
    rt = build_realtime_features(df)
    rng = np.random.default_rng(0)
    prof = pd.DataFrame({c: rng.lognormal(3, 1, len(df)) for c in
                         ["acct_txn_count", "acct_amt_mean", "acct_amt_std", "acct_n_devices", "acct_night_ratio",
                          "acct_active_days", "dev_n_accounts", "dev_txn_count", "dev_amt_mean"]}, index=df.index)
    prof.iloc[::7] = np.nan                                   # missing profiles (new accounts)
    full = pd.concat([df, rt, prof], axis=1)
    enc = CategoryEncoder().fit(full)
    batch = assemble(full, enc).to_numpy()
    online = np.vstack([assemble_record(r, enc) for r in full.to_dict("records")])
    assert online.shape[1] == len(MODEL_FEATURES)
    np.testing.assert_allclose(online, batch, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_feast_matches_builtin_point_in_time_store(trained):
    pytest.importorskip("feast")
    from fraudplat.config import path
    from fraudplat.feature_store import FeastStore, LocalPointInTimeStore
    t = pd.read_parquet(path("data/processed/training_table.parquet")).sample(5000, random_state=0)
    ent = pd.DataFrame({"uid": t.uid, "device_id": t.device_id,
                        "event_timestamp": pd.to_datetime(t.ts, unit="s", utc=True)})
    fs, lo = FeastStore(path("feature_repo")), LocalPointInTimeStore()
    a, b = fs.get_historical(ent, "account_profile"), lo.get_historical(ent, "account_profile")
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-4, equal_nan=True)
