"""Analyst dashboard (Streamlit). Talks only to the API, so it scales independently.

  API_URL=http://localhost:8000 streamlit run src/fraudplat/dashboard/app.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Fraud Operations", page_icon="🛡️", layout="wide")


@st.cache_resource
def client():
    return httpx.Client(base_url=API, timeout=10)


def get(p, **params):
    r = client().get(p, params=params)
    r.raise_for_status()
    return r.json()


def patch(p, body):
    r = client().patch(p, json=body)
    r.raise_for_status()
    return r.json()


try:
    stats = get("/stats")
except Exception as e:
    st.error(f"API not reachable at {API}: {e}")
    st.stop()

st.title("Fraud Operations Console")
k = st.columns(6)
k[0].metric("Transactions scored", f"{stats['scored_total']:,}")
k[1].metric("Alerts raised", f"{sum(stats['cases_by_status'].values()):,}")
k[2].metric("Open cases", stats["cases_by_status"].get("open", 0))
k[3].metric("Confirmed fraud", stats["cases_by_label"].get("fraud", 0))
k[4].metric("Latency p95 (ms)", stats["latency_ms_p95"])
k[5].metric("Feedback labels", stats["feedback_labels"])

tab_q, tab_case, tab_model = st.tabs(["Alert queue", "Case review", "Model & drift"])

with tab_q:
    c1, c2 = st.columns([1, 3])
    status = c1.selectbox("Status", ["open", "investigating", "closed", "all"], index=0)
    min_score = c2.slider("Minimum fraud score", 0.0, 1.0, 0.0, 0.01)
    cases = get("/cases", status=None if status == "all" else status, limit=1000)
    if cases:
        df = pd.DataFrame(cases)
        df = df[df.fraud_score >= min_score]
        df["top_reason"] = df.reasons.apply(lambda r: r[0]["feature"] if r else "")
        df["created"] = pd.to_datetime(df.created_at, unit="s").dt.strftime("%H:%M:%S")
        st.dataframe(df[["transaction_id", "fraud_score", "xgb_score", "gnn_score", "amount", "uid", "top_reason",
                         "status", "label", "created"]].sort_values("fraud_score", ascending=False),
                     hide_index=True, width="stretch",
                     column_config={"fraud_score": st.column_config.ProgressColumn("fraud_score", min_value=0, max_value=1),
                                    "gnn_score": st.column_config.ProgressColumn("ring risk", min_value=0, max_value=1)})
        st.caption(f"{len(df)} cases. Select one in the *Case review* tab.")
    else:
        st.info("No cases yet. Start the producer to stream transactions.")

with tab_case:
    all_cases = get("/cases", limit=1000)
    if not all_cases:
        st.info("No cases yet.")
    else:
        # stable order (by score) so the selected case does not jump after an analyst action
        ids = [c["transaction_id"] for c in sorted(all_cases, key=lambda c: (-c["fraud_score"], c["transaction_id"]))]
        tid = st.selectbox("Case (transaction id)", ids)
        c = get(f"/cases/{tid}")
        a, b = st.columns([2, 1])
        with a:
            st.subheader(f"Transaction {tid} — score {c['fraud_score']:.3f}")
            st.write(f"**Account** `{c['uid']}`  \n**Device** `{c['device_id']}`  \n**Amount** ${c['amount']:,.2f}")
            st.write(f"XGBoost score **{c['xgb_score']:.3f}** · Ring (GNN) score **{c['gnn_score']:.3f}**")
            if c["reasons"]:
                st.markdown("**Why it was flagged**")
                st.dataframe(pd.DataFrame(c["reasons"]), hide_index=True, width="stretch")
            related = [x for x in all_cases if x["uid"] == c["uid"] or x["device_id"] == c["device_id"]]
            if len(related) > 1:
                st.markdown(f"**{len(related) - 1} related case(s)** on the same account or device")
                st.dataframe(pd.DataFrame(related)[["transaction_id", "uid", "device_id", "fraud_score", "status", "label"]],
                             hide_index=True, width="stretch")
        with b:
            st.markdown(f"Status: **{c['status']}** · Label: **{c['label'] or '—'}**")
            analyst = st.text_input("Analyst", value=os.environ.get("USER", "analyst"))
            note = st.text_area("Note", value=c.get("note") or "")
            if st.button("Start investigation", width="stretch"):
                patch(f"/cases/{tid}", {"status": "investigating", "analyst": analyst, "note": note})
                st.rerun()
            if st.button("Confirm fraud", type="primary", width="stretch"):
                patch(f"/cases/{tid}", {"status": "closed", "label": "fraud", "analyst": analyst, "note": note})
                st.rerun()
            if st.button("Mark legitimate", width="stretch"):
                patch(f"/cases/{tid}", {"status": "closed", "label": "legit", "analyst": analyst, "note": note})
                st.rerun()

with tab_model:
    info = get("/model")
    rep = info.get("training_report")
    if rep:
        t = rep["gnn_blend"]["test"]
        m = pd.DataFrame({k: {"PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Recall@P80": v["recall_at_p80"],
                              "Precision@top1%": v["precision_at_top1pct"]} for k, v in t.items()}).T
        st.subheader("Offline test-period performance")
        st.dataframe(m.style.format("{:.4f}"), width="stretch")
        st.caption(f"{rep['eda']['rows']:,} transactions · fraud rate {rep['eda']['fraud_rate_pct']}% · "
                   f"alert threshold {rep['gnn_blend']['threshold']:.4f}")
    d = info.get("drift_status")
    st.subheader("Drift monitor")
    if d:
        when = datetime.fromtimestamp(d["checked_at"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        live = f"{d['live_pr_auc']:.3f}" if d["live_pr_auc"] is not None else "n/a"
        st.write(f"Last check {when} · drift share **{d['data_drift']['drift_share']:.0%}** · "
                 f"live PR-AUC on feedback labels **{live}** (offline baseline {d['baseline_pr_auc']:.3f})")
        if d["retrain"]:
            st.warning("Retraining triggered: " + "; ".join(d["reasons"]))
        else:
            st.success("No retraining needed")
        if d["data_drift"]["drifted_features"]:
            st.write("Drifted features: " + ", ".join(d["data_drift"]["drifted_features"]))
    else:
        st.info("Run `python -m fraudplat.monitoring.drift` to produce a drift report.")
