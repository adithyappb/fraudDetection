"""Data loading.

Primary dataset: IEEE-CIS Fraud Detection (Kaggle competition `ieee-fraud-detection`).
It is the best fit for this project among the suggested datasets because it is the only
one that has real *entity* columns (card, address, e-mail domain, device fingerprint)
needed to build an account/device/card graph for fraud-ring detection, plus a real
time axis (TransactionDT) for velocity features and chronological evaluation.

If the Kaggle files are not present, `generate_synthetic()` produces a dataset with the
*same column names and semantics*, including injected fraud rings and account takeovers,
so the whole platform runs end-to-end offline. Downstream code never knows which source
was used.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import load_config, path

log = logging.getLogger(__name__)

# Raw IEEE-CIS columns kept (a compact, high-signal subset; the 339 V-columns are mostly
# redundant and would blow up memory without adding much over our engineered features).
ID_COLS = ["TransactionID", "isFraud", "TransactionDT"]
BASE_COLS = ["TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5",
             "card6", "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain"]
NUM_PASSTHROUGH = ["C1", "C2", "C5", "C6", "C9", "C11", "C13", "C14",
                   "D1", "D2", "D3", "D4", "D10", "D11", "D15", "dist1"]
IDENTITY_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_31", "id_33"]


def _ieee_available() -> bool:
    raw = path(load_config()["data"]["raw_dir"])
    return (raw / "train_transaction.csv").exists()


def load_ieee() -> pd.DataFrame:
    raw = path(load_config()["data"]["raw_dir"])
    tx_cols = list(dict.fromkeys(ID_COLS + BASE_COLS + NUM_PASSTHROUGH))
    tx = pd.read_csv(raw / "train_transaction.csv", usecols=lambda c: c in tx_cols)
    idn_file = raw / "train_identity.csv"
    if idn_file.exists():
        idn = pd.read_csv(idn_file, usecols=lambda c: c in IDENTITY_COLS)
        tx = tx.merge(idn, on="TransactionID", how="left")
    for c in IDENTITY_COLS[1:]:
        if c not in tx:
            tx[c] = np.nan
    log.info("Loaded IEEE-CIS: %d rows, fraud rate %.3f%%", len(tx), 100 * tx.isFraud.mean())
    return tx


def generate_synthetic(n_rows: int = 300_000, seed: int = 42, days: int = 182) -> pd.DataFrame:
    """IEEE-CIS-schema synthetic transactions with realistic fraud patterns.

    Fraud mechanisms (all detectable only through behaviour / relationships, not a label leak):
      * Fraud rings: groups of mule accounts sharing a small pool of devices and e-mails,
        transacting at night, in bursts, with higher amounts.
      * Account takeover (ATO): a legitimate account suddenly transacts from a ring device,
        in a short burst, at 3-8x its normal amount.
      * Opportunistic fraud: isolated fraud with weak signal (keeps the problem honest).
    Legit noise: families sharing devices, travel (new device/address), big one-off purchases.
    """
    rng = np.random.default_rng(seed)
    T = days * 86400
    n_acc = max(200, n_rows // 25)

    browsers = np.array(["chrome 63.0", "mobile safari 11.0", "chrome 64.0 for android",
                         "edge 16.0", "firefox 57.0", "samsung browser 6.2", "ie 11.0 for desktop"])
    screens = np.array(["1920x1080", "1366x768", "2208x1242", "1334x750", "2436x1125",
                        "1440x900", "2560x1440", "1280x800"])
    dinfo = np.array(["Windows", "iOS Device", "MacOS", "SM-G930V", "SM-G955U", "Trident/7.0",
                      "Moto G (5) Plus", "Pixel 2", "LG-H870", "rv:57.0"])
    emails = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
                       "comcast.net", "icloud.com", "outlook.com", "att.net", "live.com"])
    n_dev_pool = n_acc * 2

    def new_devices(k):
        return (rng.integers(0, len(dinfo), k), rng.integers(0, len(browsers), k),
                rng.integers(0, len(screens), k), rng.integers(0, n_dev_pool * 10, k))

    # ---------------- legitimate accounts ----------------
    acc_card1 = rng.integers(1000, 18000, n_acc)
    acc_addr1 = rng.integers(100, 540, n_acc)
    acc_anchor = rng.integers(-400, days - 20, n_acc)          # account "opening" day
    acc_base_amt = rng.lognormal(3.8, 0.8, n_acc)
    acc_email = rng.choice(len(emails), n_acc, p=np.r_[0.4, 0.17, 0.1, 0.08, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03])
    acc_card4 = rng.choice(4, n_acc, p=[0.65, 0.3, 0.04, 0.01])
    acc_card6 = rng.choice(2, n_acc, p=[0.75, 0.25])
    acc_product = rng.choice(5, n_acc, p=[0.74, 0.08, 0.07, 0.06, 0.05])
    d_info, d_br, d_sc, d_salt = new_devices(n_acc)
    acc_dev = np.stack([d_info, d_br, d_sc, d_salt], 1)
    # families: ~6% of accounts share a device with another account
    fam = rng.choice(n_acc, int(0.06 * n_acc), replace=False)
    acc_dev[fam] = acc_dev[rng.choice(n_acc, len(fam))]

    rows = []

    def emit(acc_idx, ts, amt, dev, email_i, is_fraud, product, c1, dist, d10):
        rows.append(pd.DataFrame({
            "acc": acc_idx, "TransactionDT": ts, "TransactionAmt": np.round(amt, 2),
            "dev_info": dev[:, 0], "dev_br": dev[:, 1], "dev_sc": dev[:, 2], "dev_salt": dev[:, 3],
            "email_i": email_i, "isFraud": is_fraud, "prod_i": product,
            "C1": c1, "dist1": dist, "D10": d10,
        }))

    n_legit_target = int(n_rows * 0.965)
    activity = rng.gamma(1.2, 1.0, n_acc)
    counts = rng.multinomial(n_legit_target, activity / activity.sum())
    acc_idx = np.repeat(np.arange(n_acc), counts)
    start = np.clip(acc_anchor[acc_idx], 0, None) * 86400
    ts = (start + rng.random(len(acc_idx)) * (T - start)).astype(np.int64)
    hour_shift = (rng.normal(14, 3.5, len(acc_idx)) % 24 * 3600).astype(np.int64)
    ts = ts - ts % 86400 + hour_shift
    amt = acc_base_amt[acc_idx] * rng.lognormal(0, 0.45, len(acc_idx))
    big = rng.random(len(acc_idx)) < 0.01                   # legit big purchases
    amt[big] *= rng.uniform(3, 8, big.sum())
    dev = acc_dev[acc_idx].copy()
    travel = rng.random(len(acc_idx)) < 0.04                # legit new device / travel
    dev[travel] = np.stack(new_devices(travel.sum()), 1)
    emit(acc_idx, ts, amt, dev, acc_email[acc_idx], np.zeros(len(acc_idx), int),
         acc_product[acc_idx], rng.poisson(1.5, len(acc_idx)) + 1,
         np.where(travel, rng.exponential(300, len(acc_idx)), rng.exponential(15, len(acc_idx))),
         np.clip(rng.normal(120, 60, len(acc_idx)), 0, None))

    # ---------------- fraud rings ----------------
    n_rings = max(5, n_acc // 200)
    ring_pools = []
    next_acc = n_acc
    extra = {"card1": [], "addr1": [], "anchor": [], "email": [], "card4": [], "card6": []}
    for _ in range(n_rings):
        k_dev = rng.integers(2, 7)
        pool = np.stack(new_devices(k_dev), 1)
        ring_pools.append(pool)
        ring_emails = rng.choice(len(emails), rng.integers(1, 3))
        n_mules = rng.integers(4, 25)
        ring_start = rng.integers(0, days - 15)
        ring_card1 = rng.integers(1000, 18000, max(1, n_mules // 3))   # stolen card BIN reuse
        for _m in range(n_mules):
            a = next_acc
            next_acc += 1
            extra["card1"].append(rng.choice(ring_card1))
            extra["addr1"].append(rng.integers(100, 540))
            anchor = ring_start + rng.integers(0, 10)
            extra["anchor"].append(anchor)
            extra["email"].append(rng.choice(ring_emails))
            extra["card4"].append(rng.choice(4))
            extra["card6"].append(rng.choice(2, p=[0.4, 0.6]))
            k = rng.integers(3, 25)
            # bursty: a few sessions, each with several txns minutes apart
            n_sess = max(1, k // 4)
            sess_t = anchor * 86400 + rng.integers(0, 12 * 86400, n_sess)
            t = np.repeat(sess_t, int(np.ceil(k / n_sess)))[:k] + rng.integers(0, 3600, k)
            t = t - t % 86400 + (rng.normal(2.5, 2.5, k) % 24 * 3600).astype(np.int64)
            is_f = (rng.random(k) < 0.88).astype(int)       # mules sometimes "warm up" with legit txns
            # Real rings rotate devices: half the mules mostly use their own fresh device and touch
            # the shared pool only occasionally -> first-order device counts look benign; the ring
            # is only visible multi-hop (mule -card- mule -device- mule), which is what the GNN is for.
            if rng.random() < 0.5:
                own = np.stack(new_devices(1), 1)
                use_pool = rng.random(k) < 0.25
                devs = np.where(use_pool[:, None], pool[rng.integers(0, k_dev, k)], own[[0] * k])
            else:
                devs = pool[rng.integers(0, k_dev, k)]
            emit(np.full(k, a), t, rng.lognormal(4.9, 0.9, k), devs,
                 np.full(k, extra["email"][-1]), is_f, np.where(rng.random(k) < 0.6, 1, 0),
                 rng.poisson(6, k) + 1, rng.exponential(120, k), np.clip(rng.normal(3, 3, k), 0, None))

    # ---------------- account takeover (links legit accounts to ring devices) ----------------
    ato = rng.choice(n_acc, int(0.015 * n_acc), replace=False)
    for a in ato:
        pool = ring_pools[rng.integers(0, n_rings)]
        k = rng.integers(2, 9)
        t0 = rng.integers(max(0, acc_anchor[a]) * 86400 + 86400, T - 3600) if acc_anchor[a] < days - 2 else T - 7200
        t = t0 + np.sort(rng.integers(0, 6 * 3600, k))
        emit(np.full(k, a), t, acc_base_amt[a] * rng.uniform(3, 8, k), pool[rng.integers(0, len(pool), k)],
             np.full(k, acc_email[a]), np.ones(k, int), np.full(k, acc_product[a]),
             rng.poisson(2, k) + 1, rng.exponential(400, k), np.clip(rng.normal(120, 60, k), 0, None))

    # ---------------- opportunistic, weak-signal fraud ----------------
    k = int(0.004 * n_rows)
    a = rng.integers(0, n_acc, k)
    emit(a, (rng.random(k) * T).astype(np.int64), acc_base_amt[a] * rng.lognormal(0.4, 0.6, k),
         acc_dev[a], acc_email[a], np.ones(k, int), acc_product[a], rng.poisson(1.5, k) + 1,
         rng.exponential(30, k), np.clip(rng.normal(100, 60, k), 0, None))

    df = pd.concat(rows, ignore_index=True)
    df = df[(df.TransactionDT >= 0) & (df.TransactionDT < T)]

    card1 = np.r_[acc_card1, extra["card1"]]
    addr1 = np.r_[acc_addr1, extra["addr1"]]
    anchor = np.r_[acc_anchor, extra["anchor"]]
    card4 = np.r_[acc_card4, extra["card4"]]
    card6 = np.r_[acc_card6, extra["card6"]]
    a = df.acc.to_numpy()
    day = df.TransactionDT.to_numpy() // 86400
    out = pd.DataFrame({
        "TransactionID": 0,
        "isFraud": df.isFraud.to_numpy(),
        "TransactionDT": df.TransactionDT.to_numpy() + 86400,   # IEEE-CIS starts at 86400
        "TransactionAmt": df.TransactionAmt.to_numpy(),
        "ProductCD": np.array(["W", "C", "R", "H", "S"])[df.prod_i.to_numpy()],
        "card1": card1[a],
        "card2": (card1[a] * 7) % 500 + 100.0,
        "card3": 150.0,
        "card4": np.array(["visa", "mastercard", "american express", "discover"])[card4[a]],
        "card5": (card1[a] * 3) % 150 + 100.0,
        "card6": np.array(["debit", "credit"])[card6[a]],
        "addr1": addr1[a].astype(float),
        "addr2": 87.0,
        "dist1": np.round(df.dist1.to_numpy(), 0),
        "P_emaildomain": emails[df.email_i.to_numpy()],
        "R_emaildomain": np.where(rng.random(len(df)) < 0.25, emails[df.email_i.to_numpy()], None),
        "C1": df.C1.to_numpy().astype(float),
        "C13": (df.C1.to_numpy() * rng.uniform(1, 4, len(df))).round(),
        "C14": df.C1.to_numpy().astype(float),
        "D1": (day - anchor[a]).astype(float).clip(0),           # days since card first seen
        "D10": np.round(df.D10.to_numpy()),
        "D15": np.round(df.D10.to_numpy() * rng.uniform(0.8, 1.2, len(df))),
        "DeviceType": np.where(np.isin(df.dev_info.to_numpy(), [0, 2, 5, 9]), "desktop", "mobile"),
        "DeviceInfo": dinfo[df.dev_info.to_numpy()],
        "id_31": browsers[df.dev_br.to_numpy()],
        # the salt makes otherwise-identical device models distinct fingerprints
        "id_33": [f"{screens[s]}_{h:x}" for s, h in zip(df.dev_sc.to_numpy(), df.dev_salt.to_numpy(), strict=False)],
    })
    # opened-before-dataset accounts: D1 must be consistent with IEEE semantics (day - anchor)
    out = out.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    out["TransactionID"] = np.arange(2_987_000, 2_987_000 + len(out))
    log.info("Synthetic data: %d rows, %d accounts, fraud rate %.2f%%", len(out), next_acc, 100 * out.isFraud.mean())
    return out


def load_raw(source: str | None = None) -> tuple[pd.DataFrame, str]:
    cfg = load_config()["data"]
    source = source or cfg["source"]
    if source == "auto":
        source = "ieee" if _ieee_available() else "synthetic"
    if source == "ieee":
        return load_ieee(), "ieee"
    return generate_synthetic(cfg["synthetic_rows"], cfg["seed"]), "synthetic"


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Map IEEE-CIS columns to the platform's canonical schema + entity keys."""
    df = pd.DataFrame({
        "transaction_id": raw["TransactionID"].astype("int64"),
        "ts": raw["TransactionDT"].astype("int64"),
        "is_fraud": raw["isFraud"].astype("int8") if "isFraud" in raw else 0,
        "amount": raw["TransactionAmt"].astype("float64"),
        "product": raw["ProductCD"].astype(str),
        "card4": raw["card4"].fillna("na").astype(str),
        "card6": raw["card6"].fillna("na").astype(str),
        "p_email": raw["P_emaildomain"].fillna("na").astype(str),
        "r_email": raw["R_emaildomain"].fillna("na").astype(str),
        "device_type": raw["DeviceType"].fillna("na").astype(str),
    })
    for c in NUM_PASSTHROUGH:
        df[c] = raw[c].astype("float64") if c in raw else np.nan
    # ---- entity keys ----
    # Card: IEEE-CIS card1..card6 jointly identify a card product/issuer.
    df["card_id"] = ("c" + raw["card1"].astype(str) + "_" + raw["card2"].fillna(-1).astype(int).astype(str)
                     + "_" + raw["card5"].fillna(-1).astype(int).astype(str) + "_" + df.card6)
    # Account (uid): the well-known IEEE-CIS client key = card1 + addr1 + (day - D1),
    # i.e. card + billing address + the day the card relationship started.
    d1n = (raw["TransactionDT"] // 86400 - raw["D1"].fillna(0)).astype(int)
    df["uid"] = ("u" + raw["card1"].astype(str) + "_" + raw["addr1"].fillna(-1).astype(int).astype(str)
                 + "_" + d1n.astype(str))
    # Device fingerprint: model + browser + screen; missing identity -> no device node.
    has_dev = raw["DeviceInfo"].notna() | raw["id_31"].notna()
    dev = ("d" + raw["DeviceInfo"].fillna("na").astype(str) + "|" + raw["id_31"].fillna("na").astype(str)
           + "|" + raw["id_33"].fillna("na").astype(str))
    df["device_id"] = np.where(has_dev, dev, "d_unknown")
    df["addr_id"] = "a" + raw["addr1"].fillna(-1).astype(int).astype(str)
    df["email_id"] = "e" + df.p_email
    return df.sort_values(["ts", "transaction_id"], kind="stable").reset_index(drop=True)


def _s(v, default="na"):
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _i(v):
    return -1 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)


def _f(v):
    return np.nan if v is None else float(v)


def canonical_record(raw: dict) -> dict:
    """Single-record, pure-Python twin of `to_canonical` (≈50x faster for online scoring).
    tests/test_features.py asserts both produce identical output."""
    ts = int(raw["TransactionDT"])
    d1 = raw.get("D1")
    d1 = 0 if d1 is None or (isinstance(d1, float) and np.isnan(d1)) else d1
    card6 = _s(raw.get("card6"))
    p_email = _s(raw.get("P_emaildomain"))
    rec = {
        "transaction_id": int(raw["TransactionID"]), "ts": ts, "is_fraud": int(raw.get("isFraud") or 0),
        "amount": float(raw["TransactionAmt"]), "product": str(raw.get("ProductCD")),
        "card4": _s(raw.get("card4")), "card6": card6, "p_email": p_email,
        "r_email": _s(raw.get("R_emaildomain")), "device_type": _s(raw.get("DeviceType")),
    }
    for c in NUM_PASSTHROUGH:
        rec[c] = _f(raw.get(c))
    rec["card_id"] = f"c{raw['card1']}_{_i(raw.get('card2'))}_{_i(raw.get('card5'))}_{card6}"
    rec["uid"] = f"u{raw['card1']}_{_i(raw.get('addr1'))}_{int(ts // 86400 - d1)}"
    info, br = raw.get("DeviceInfo"), raw.get("id_31")
    has_dev = (info is not None and info == info) or (br is not None and br == br)
    rec["device_id"] = f"d{_s(info)}|{_s(br)}|{_s(raw.get('id_33'))}" if has_dev else "d_unknown"
    rec["addr_id"] = f"a{_i(raw.get('addr1'))}"
    rec["email_id"] = "e" + p_email
    return rec


def time_split(df: pd.DataFrame) -> pd.Series:
    cfg = load_config()["data"]
    q1, q2 = df.ts.quantile([cfg["train_frac"], cfg["train_frac"] + cfg["valid_frac"]])
    return pd.Series(np.where(df.ts <= q1, "train", np.where(df.ts <= q2, "valid", "test")), index=df.index)
