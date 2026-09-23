"""Append-only journal of Jev decisions.

Every call stores a SHA-256 of the canonical state, the model and question
schema versions, token cost, and a slot for the later outcome label. Raw
probabilities stay attached to the row. They are not treated as win rates.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.jev.config import QUESTION_SCHEMA_VERSION, STATE_BUILDER_VERSION, estimate_cost_usd


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def default_journal_path() -> Path:
    configured = os.getenv("JEV_JOURNAL_PATH", "").strip()
    if configured:
        return Path(configured)
    root = Path(os.getenv("JEV_DATA_DIR", "/tmp/quantumtrade-jev"))
    return root / "journal.db"


class DecisionJournal:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_journal_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jev_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT,
                    provider TEXT,
                    model_version TEXT,
                    question_schema_version TEXT,
                    state_builder_version TEXT,
                    state_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    raw_answers_json TEXT,
                    input_tokens INTEGER,
                    latency_ms REAL,
                    cost_usd REAL,
                    label INTEGER,
                    label_horizon TEXT,
                    labeled_at TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jev_decisions_symbol ON jev_decisions(symbol, decision_id DESC)"
            )
            self._conn.commit()

    def record(
        self,
        *,
        symbol: str,
        state: Any,
        status: str,
        provider: str = "typesafe",
        model_version: str | None = None,
        raw_answers: dict | None = None,
        input_tokens: int | None = None,
        latency_ms: float | None = None,
        horizon: str = "intraday",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        state_hash = canonical_hash(state)
        cost = estimate_cost_usd(input_tokens)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO jev_decisions (
                    timestamp_utc, symbol, horizon, provider, model_version,
                    question_schema_version, state_builder_version, state_hash,
                    status, raw_answers_json, input_tokens, latency_ms, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    symbol,
                    horizon,
                    provider,
                    model_version,
                    QUESTION_SCHEMA_VERSION,
                    STATE_BUILDER_VERSION,
                    state_hash,
                    status,
                    json.dumps(raw_answers, default=str) if raw_answers is not None else None,
                    input_tokens,
                    latency_ms,
                    cost,
                ),
            )
            self._conn.commit()
            decision_id = int(cursor.lastrowid)
        return {
            "decision_id": decision_id,
            "state_hash": state_hash,
            "question_schema_version": QUESTION_SCHEMA_VERSION,
            "state_builder_version": STATE_BUILDER_VERSION,
            "cost_usd": cost,
            "input_tokens": input_tokens,
        }

    def label(self, decision_id: int, outcome: int, horizon: str = "t+1") -> bool:
        if outcome not in (-1, 0, 1):
            raise ValueError("outcome must be -1, 0, or 1")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE jev_decisions
                SET label = ?, label_horizon = ?, labeled_at = ?
                WHERE decision_id = ?
                """,
                (outcome, horizon, now, decision_id),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def labeled_choices(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT decision_id, symbol, raw_answers_json, label
                FROM jev_decisions
                WHERE label IS NOT NULL AND raw_answers_json IS NOT NULL
                ORDER BY decision_id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        parsed: list[dict[str, Any]] = []
        for row in rows:
            try:
                answers = json.loads(row["raw_answers_json"])
            except json.JSONDecodeError:
                continue
            parsed.append({
                "decision_id": row["decision_id"],
                "symbol": row["symbol"],
                "label": int(row["label"]),
                "answers": answers,
            })
        return parsed

    def recent(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = """
            SELECT decision_id, timestamp_utc, symbol, horizon, provider, model_version,
                   question_schema_version, state_builder_version, state_hash, status,
                   input_tokens, latency_ms, cost_usd, label
            FROM jev_decisions
        """
        params: list[Any] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY decision_id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def label_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM jev_decisions WHERE label IS NOT NULL").fetchone()
        return int(row["n"] if row else 0)


_journal: DecisionJournal | None = None
_journal_lock = threading.Lock()


def get_journal() -> DecisionJournal:
    global _journal
    with _journal_lock:
        path = default_journal_path()
        if _journal is None or _journal.path != path:
            _journal = DecisionJournal(path)
        return _journal


def reset_journal_cache() -> None:
    global _journal
    with _journal_lock:
        _journal = None
