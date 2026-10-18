"""
Supabase Dual-Write Projector (WP4)
===================================
Asynchronously projects decision records and JEV advisory opinions into Supabase (`public.jev_signals`).
Guarantees:
- Fully decoupled from the trading execution engine (outbox/background driven).
- Supabase network outages or rate limits never block ingestion, decisions, or order routing.
- Idempotent upserts via deterministic UUIDs and merge-duplicates resolution.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

import requests

from backend.database.models import DecisionOpinion, DecisionRecord

logger = logging.getLogger(__name__)

DEFAULT_SUPABASE_URL = "https://ailcdqbhxvukqxbujpvs.supabase.co"
DEFAULT_SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpbGNkcWJoeHZ1a3F4YnVqcHZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQyNjc0NDQsImV4cCI6MjA2OTg0MzQ0NH0."
    "Ue-UlwhQ_1CC9HrHPwrJn7TII8pjlIVD9_8oenBIrHM"
)


class SupabaseProjector:
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ):
        self.url = (supabase_url or os.getenv("SUPABASE_URL") or DEFAULT_SUPABASE_URL).rstrip("/")
        self.key = (supabase_key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY") or DEFAULT_SUPABASE_KEY).strip()
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.url and self.key)

    def project_decision(
        self,
        record: DecisionRecord,
        opinion: Optional[DecisionOpinion] = None,
    ) -> bool:
        """
        Idempotently project a single DecisionRecord to Supabase public.jev_signals.
        Returns True on success, False on failure without raising exceptions.
        """
        if not self.is_configured():
            logger.debug("Supabase projector not configured; skipping projection")
            return False

        # Generate deterministic UUID based on scan_id and symbol
        namespace = uuid.NAMESPACE_DNS
        row_id = str(uuid.uuid5(namespace, f"{record.scan_id}_{record.symbol}"))

        if record.created_at and hasattr(record.created_at, "isoformat"):
            created_str = record.created_at.isoformat()
        elif record.created_at:
            created_str = str(record.created_at)
        else:
            from datetime import datetime, timezone
            created_str = datetime.now(timezone.utc).isoformat()

        indicators = (record.details or {}).get("indicators") or {}
        raw_answers = (opinion.raw_payload or {}) if opinion else {}

        decision_label = record.with_jev_action
        blocked_reason = None
        if record.jev_vetoed:
            decision_label = "NO_TRADE_VETOED"
            blocked_reason = record.jev_reason or "Vetoed by JEV risk model"
        elif record.with_jev_action == "HOLD":
            if record.divergence == "DOWNGRADED":
                decision_label = "NO_TRADE_CONFLICT"
                blocked_reason = record.with_jev_reason or "Contradicted by JEV opinion"
            else:
                decision_label = "NO_TRADE_NEUTRAL"
                blocked_reason = record.baseline_reason

        payload = {
            "id": row_id,
            "created_at": created_str,
            "symbol": record.symbol,
            "timeframe": record.timeframe or "M5",
            "decision": decision_label,
            "allowed_side": record.with_jev_action if record.with_jev_action in ("BUY", "SELL") else None,
            "blocked_reason": blocked_reason,
            "confidence": float(record.with_jev_confidence or 0.0),
            "signal_strength": float(record.baseline_confidence or 0.0),
            "probabilities": {
                "baseline": record.baseline_confidence,
                "with_jev": record.with_jev_confidence,
                "divergence": record.divergence,
            },
            "funding_annualized": float(indicators.get("funding_rate", 0.0) or 0.0) * 100.0,
            "oi_change_pct_1h": float(indicators.get("oi_change_pct_1h", 0.0) or 0.0),
            "raw_answers": raw_answers,
            "outcome": "PENDING",
            "timeframe_scanned": record.timeframe or "M5",
        }

        endpoint = f"{self.url}/rest/v1/jev_signals"
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code in (200, 201, 204):
                logger.info(f"[SUPABASE PROJECTOR] Successfully projected signal for {record.symbol} (id={row_id})")
                return True
            else:
                logger.warning(
                    f"[SUPABASE PROJECTOR] Failed to project signal {record.symbol}: HTTP {resp.status_code} - {resp.text}"
                )
                return False
        except Exception as exc:
            logger.warning(f"[SUPABASE PROJECTOR] Network error projecting signal {record.symbol} to Supabase: {exc}")
            return False


# Global default projector instance
default_supabase_projector = SupabaseProjector()
