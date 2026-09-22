"""SQLite dedup store for ingested posts.

Embeddings are optional and live elsewhere. This table only remembers ids
so a later pull can stop when it reaches a post it already stored.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    configured = os.getenv("JEV_TWEET_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    root = Path(os.getenv("JEV_DATA_DIR", "/tmp/quantumtrade-jev"))
    return root / "tweets.db"


class TweetStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jev_tweets (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    text TEXT,
                    created_at TEXT,
                    timestamp_epoch INTEGER,
                    likes INTEGER,
                    retweets INTEGER,
                    replies INTEGER,
                    author_username TEXT,
                    payload_json TEXT,
                    inserted_at TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jev_tweets_symbol ON jev_tweets(symbol, timestamp_epoch DESC)"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def known_ids(self, symbol: str, limit: int = 5000) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jev_tweets WHERE symbol = ? ORDER BY timestamp_epoch DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return {str(row["id"]) for row in rows}

    def save(self, symbol: str, posts: list[dict[str, Any]]) -> int:
        if not posts:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        saved = 0
        with self._lock:
            for post in posts:
                post_id = str(post.get("id") or "").strip()
                if not post_id:
                    continue
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO jev_tweets (
                        id, symbol, text, created_at, timestamp_epoch, likes, retweets,
                        replies, author_username, payload_json, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post_id,
                        symbol,
                        str(post.get("text") or "")[:1000],
                        str(post.get("created_at") or ""),
                        int(post.get("timestamp_epoch") or 0),
                        int(post.get("likes") or 0),
                        int(post.get("retweets") or 0),
                        int(post.get("replies") or 0),
                        str(post.get("author_username") or ""),
                        json.dumps(post, default=str)[:4000],
                        now,
                    ),
                )
                saved += self._conn.execute("SELECT changes()").fetchone()[0]
            self._conn.commit()
        return saved

    def recent(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, text, created_at, timestamp_epoch, likes, retweets, replies, author_username
                FROM jev_tweets WHERE symbol = ?
                ORDER BY timestamp_epoch DESC LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        return [dict(row) for row in rows]
