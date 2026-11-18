"""Case management store (SQLite; swap for Postgres in production - plain SQL only)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  transaction_id INTEGER PRIMARY KEY, uid TEXT, device_id TEXT, amount REAL, ts INTEGER,
  fraud_score REAL, xgb_score REAL, gnn_score REAL, reasons TEXT,
  status TEXT DEFAULT 'open', label TEXT, analyst TEXT, note TEXT,
  created_at REAL, updated_at REAL);
CREATE INDEX IF NOT EXISTS ix_cases_status ON cases(status);
CREATE TABLE IF NOT EXISTS scores (
  transaction_id INTEGER PRIMARY KEY, ts INTEGER, fraud_score REAL, decision TEXT,
  features TEXT, scored_at REAL);
CREATE TABLE IF NOT EXISTS feedback (
  transaction_id INTEGER PRIMARY KEY, label INTEGER, source TEXT, created_at REAL);
"""
STATUSES = {"open", "investigating", "closed"}
LABELS = {"fraud", "legit"}


class CaseStore:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            cols = [c[0] for c in cur.description] if cur.description else []
            return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

    def log_score(self, res: dict, canon: dict, feats: dict):
        self._q("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)",
                (res["transaction_id"], int(canon["ts"]), res["fraud_score"], res["decision"],
                 json.dumps({k: (None if v != v else v) for k, v in feats.items()}), time.time()))

    def open_case(self, res: dict, canon: dict):
        now = time.time()
        self._q("INSERT OR IGNORE INTO cases (transaction_id, uid, device_id, amount, ts, fraud_score, xgb_score,"
                " gnn_score, reasons, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,'open',?,?)",
                (res["transaction_id"], res["uid"], res["device_id"], float(canon["amount"]), int(canon["ts"]),
                 res["fraud_score"], res["xgb_score"], res["gnn_score"], json.dumps(res["reasons"]), now, now))

    def list_cases(self, status: str | None = None, limit: int = 200):
        if status:
            rows = self._q("SELECT * FROM cases WHERE status=? ORDER BY fraud_score DESC LIMIT ?", (status, limit))
        else:
            rows = self._q("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["reasons"] = json.loads(r["reasons"] or "[]")
        return rows

    def get_case(self, tid: int):
        rows = self._q("SELECT * FROM cases WHERE transaction_id=?", (tid,))
        if rows:
            rows[0]["reasons"] = json.loads(rows[0]["reasons"] or "[]")
        return rows[0] if rows else None

    def update_case(self, tid: int, status=None, label=None, analyst=None, note=None):
        if status and status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        if label and label not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)}")
        sets, args = [], []
        for col, val in (("status", status), ("label", label), ("analyst", analyst), ("note", note)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        sets.append("updated_at=?")
        args += [time.time(), tid]
        self._q(f"UPDATE cases SET {', '.join(sets)} WHERE transaction_id=?", args)
        if label:
            self.add_feedback(tid, 1 if label == "fraud" else 0, "analyst")
        return self.get_case(tid)

    def add_feedback(self, tid: int, label: int, source: str = "analyst"):
        self._q("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)", (tid, int(label), source, time.time()))

    def recent_scores(self, limit: int = 5000):
        rows = self._q("SELECT s.transaction_id, s.ts, s.fraud_score, s.decision, s.features, f.label "
                       "FROM scores s LEFT JOIN feedback f USING(transaction_id) ORDER BY s.ts DESC LIMIT ?", (limit,))
        for r in rows:
            r["features"] = json.loads(r["features"])
        return rows

    def summary(self):
        by_status = {r["status"]: r["n"] for r in self._q("SELECT status, COUNT(*) n FROM cases GROUP BY status")}
        by_label = {(r["label"] or "unlabelled"): r["n"] for r in self._q("SELECT label, COUNT(*) n FROM cases GROUP BY label")}
        n_scores = self._q("SELECT COUNT(*) n FROM scores")[0]["n"]
        n_fb = self._q("SELECT COUNT(*) n FROM feedback")[0]["n"]
        return {"cases_by_status": by_status, "cases_by_label": by_label, "scored_total": n_scores, "feedback_labels": n_fb}
