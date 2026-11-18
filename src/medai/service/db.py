"""Results store + append-only, hash-chained audit log (SQLite; swap for Postgres in production).

Every audit entry stores sha256(previous_hash + canonical JSON of the entry). Editing or deleting
any past row breaks the chain, which `verify_audit()` detects - tamper-evident record of every
AI result and every radiologist decision.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY, study_uid TEXT, patient_id TEXT, patient_sex TEXT, patient_age TEXT,
  modality TEXT, description TEXT, kind TEXT, status TEXT, received_at REAL, analyzed_at REAL,
  ai TEXT, model_version TEXT, overlay_series TEXT, reviewer TEXT, decision TEXT, decision_comment TEXT,
  decided_at REAL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, action TEXT, study_id TEXT,
  details TEXT, prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS worker_state (k TEXT PRIMARY KEY, v TEXT);
"""


class ResultsDB:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def q(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            cols = [c[0] for c in cur.description] if cur.description else []
            return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

    # ---------------------------------------------------------------- studies
    def upsert_study(self, **f):
        cols = ",".join(f)
        self.q(f"INSERT INTO studies ({cols}) VALUES ({','.join('?' * len(f))}) "
               f"ON CONFLICT(study_id) DO UPDATE SET {','.join(f'{k}=excluded.{k}' for k in f if k != 'study_id')}",
               tuple(f.values()))

    def study(self, sid):
        r = self.q("SELECT * FROM studies WHERE study_id=?", (sid,))
        if r and r[0]["ai"]:
            r[0]["ai"] = json.loads(r[0]["ai"])
        return r[0] if r else None

    def studies(self):
        rows = self.q("SELECT * FROM studies ORDER BY received_at DESC")
        for r in rows:
            r["ai"] = json.loads(r["ai"]) if r["ai"] else None
        return rows

    # ---------------------------------------------------------------- worker cursor
    def get_state(self, k, default=None):
        r = self.q("SELECT v FROM worker_state WHERE k=?", (k,))
        return r[0]["v"] if r else default

    def set_state(self, k, v):
        self.q("INSERT OR REPLACE INTO worker_state VALUES (?,?)", (k, str(v)))

    # ---------------------------------------------------------------- audit
    @staticmethod
    def _digest(prev, entry):
        return hashlib.sha256((prev + json.dumps(entry, sort_keys=True)).encode()).hexdigest()

    def audit(self, actor: str, action: str, study_id: str | None, details: dict) -> dict:
        with self.lock:
            last = self.conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = last[0] if last else "GENESIS"
            entry = {"ts": round(time.time(), 3), "actor": actor, "action": action, "study_id": study_id,
                     "details": details}
            h = self._digest(prev, entry)
            self.conn.execute("INSERT INTO audit (ts, actor, action, study_id, details, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
                              (entry["ts"], actor, action, study_id, json.dumps(details, sort_keys=True), prev, h))
        return {**entry, "hash": h}

    def audit_log(self, study_id: str | None = None, limit: int = 500):
        sql, args = "SELECT * FROM audit", ()
        if study_id:
            sql, args = sql + " WHERE study_id=?", (study_id,)
        rows = self.q(sql + " ORDER BY id DESC LIMIT ?", args + (limit,))
        for r in rows:
            r["details"] = json.loads(r["details"])
        return rows

    def verify_audit(self) -> dict:
        prev, n = "GENESIS", 0
        for r in self.q("SELECT * FROM audit ORDER BY id"):
            entry = {"ts": r["ts"], "actor": r["actor"], "action": r["action"], "study_id": r["study_id"],
                     "details": json.loads(r["details"])}
            if r["prev_hash"] != prev or self._digest(prev, entry) != r["hash"]:
                return {"valid": False, "entries": n, "broken_at_id": r["id"]}
            prev, n = r["hash"], n + 1
        return {"valid": True, "entries": n, "head": prev}
