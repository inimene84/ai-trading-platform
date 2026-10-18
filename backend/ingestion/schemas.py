"""
Strict Pydantic schemas for the Canonical Market Ingestion API (v1).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Instrument(BaseModel):
    """Normalized financial instrument reference."""
    symbol: str = Field(..., min_length=1, max_length=32, description="Trading ticker symbol, e.g. BTCUSDT, EURUSD")
    venue: str = Field(default="binance_futures", min_length=1, max_length=32, description="Exchange or broker venue, e.g. binance_futures, ctrader")
    market_type: str = Field(default="crypto_perp", min_length=1, max_length=20, description="Market type, e.g. crypto_perp, spot, forex, metal")

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol cannot be blank")
        return s

    @field_validator("venue")
    @classmethod
    def normalize_venue(cls, v: str) -> str:
        s = v.strip().lower()
        if not s:
            raise ValueError("venue cannot be blank")
        return s


class MarketObservation(BaseModel):
    """One single-instrument observation at a given timeframe."""
    instrument: Instrument
    timeframe: str = Field(default="M5", min_length=1, max_length=10, description="Observation timeframe, e.g. M1, M5, M15, H1, H4, D1")
    observed_at: datetime = Field(..., description="Timestamp of observation (must be timezone-aware)")
    price: Optional[float] = Field(default=None, ge=0.0, description="Last mark/close price")
    signal_direction: Optional[str] = Field(default=None, max_length=10, description="Signal direction: BUY, SELL, HOLD, NEUTRAL")
    confidence: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Model or indicator confidence (-1.0 to 1.0)")
    signal_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized signal strength (0.0 to 1.0)")
    indicators: dict[str, Any] = Field(default_factory=dict, description="Dictionary of indicator readings (e.g. RSI, MACD, ATR)")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Arbitrary observation metadata")

    @field_validator("observed_at")
    @classmethod
    def validate_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware (e.g. UTC)")
        return v

    @field_validator("timeframe")
    @classmethod
    def normalize_tf(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("signal_direction")
    @classmethod
    def normalize_sig(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.strip().upper()


class MarketScan(BaseModel):
    """Canonical multi-observation scan envelope submitted by scanners (n8n, microservices)."""
    scan_id: str = Field(..., min_length=1, max_length=64, description="Unique scan identifier preserved across retries (UUID or scan ID)")
    source: str = Field(..., min_length=1, max_length=50, description="Producer name, e.g. n8n_market_scanner, n8n_forex_scanner")
    scan_timestamp: datetime = Field(..., description="Timestamp of the scan batch (must be timezone-aware)")
    observations: List[MarketObservation] = Field(..., min_length=1, max_length=50, description="1 to 50 market observations")
    trace_id: Optional[str] = Field(default=None, max_length=64, description="Distributed correlation / trace ID")

    @field_validator("scan_timestamp")
    @classmethod
    def validate_scan_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("scan_timestamp must be timezone-aware (e.g. UTC)")
        return v

    @model_validator(mode="after")
    def check_no_duplicate_observations(self) -> MarketScan:
        seen = set()
        for obs in self.observations:
            key = (obs.instrument.symbol, obs.instrument.venue, obs.timeframe)
            if key in seen:
                raise ValueError(
                    f"Duplicate observation in scan envelope for instrument: {obs.instrument.symbol} on {obs.instrument.venue} ({obs.timeframe})"
                )
            seen.add(key)
        return self


class ScanReceipt(BaseModel):
    """Acknowledgement receipt returned upon scan ingestion."""
    ingestion_id: int
    scan_id: str
    source: str
    received_at: datetime
    observation_count: int
    duplicate: bool
    payload_sha256: str
    event_id: Optional[str] = None
    trace_id: Optional[str] = None


def canonical_payload_digest(scan: MarketScan) -> tuple[dict[str, Any], bytes, str]:
    """
    Produce a deterministic JSON serialization and SHA-256 hex digest of a MarketScan.
    Ensures identical payloads yield identical hashes across producers and retries.
    """
    data = scan.model_dump(mode="json")
    # Sort observations deterministically by symbol, venue, timeframe
    if "observations" in data and isinstance(data["observations"], list):
        data["observations"] = sorted(
            data["observations"],
            key=lambda x: (
                x.get("instrument", {}).get("symbol", ""),
                x.get("instrument", {}).get("venue", ""),
                x.get("timeframe", ""),
            ),
        )

    # Canonical json string with sorted keys and no unnecessary whitespace
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return data, encoded, digest
