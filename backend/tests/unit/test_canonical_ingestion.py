"""
Unit tests for WP1 Canonical Ingestion API (/api/ingestion/v1/market-scans):
- Strict Pydantic schemas (Instrument, MarketObservation, MarketScan)
- Canonical JSON serialization and SHA-256 hash determinism
- Authentication enforcement (X-Ingestion-Key, Bearer token)
- Idempotency on (source, scan_id): 201 Created -> 200 OK duplicate=True
- Conflict detection on payload modification: 409 Conflict
- Freshness verification on first insert vs duplicate
- Transactional scan + outbox atomic write
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.database.connection import Base, engine, get_db
from backend.database.models import IngestionMarketScan, IngestionOutbox
from backend.ingestion.schemas import (
    Instrument,
    MarketObservation,
    MarketScan,
    canonical_payload_digest,
)
from backend.main import app

INGESTION_URL = "/api/ingestion/v1/market-scans"
TEST_KEY = "test-n8n-ingestion-secret"


@pytest.fixture(autouse=True)
def setup_ingestion_test(monkeypatch, db_session):
    monkeypatch.setenv("INGESTION_N8N_KEY", TEST_KEY)
    yield



def _sample_scan_payload(scan_id: str = None, symbol: str = "BTCUSDT", price: float = 65000.0, age_sec: float = 10.0) -> dict:
    now = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    return {
        "scan_id": scan_id or f"scan_{uuid.uuid4().hex[:12]}",
        "source": "n8n_market_scanner",
        "scan_timestamp": now.isoformat(),
        "observations": [
            {
                "instrument": {
                    "symbol": symbol,
                    "venue": "binance_futures",
                    "market_type": "crypto_perp",
                },
                "timeframe": "M5",
                "observed_at": now.isoformat(),
                "price": price,
                "signal_direction": "BUY",
                "confidence": 0.82,
                "signal_strength": 0.75,
                "indicators": {"rsi": 62.4, "macd": 12.5, "atr": 240.0},
            }
        ],
        "trace_id": "trace-test-123",
    }


def test_schema_validation_rejects_duplicate_observations():
    now = datetime.now(timezone.utc)
    obs = {
        "instrument": {"symbol": "BTCUSDT", "venue": "binance_futures", "market_type": "crypto_perp"},
        "timeframe": "M5",
        "observed_at": now,
        "price": 60000.0,
    }
    # Duplicate same instrument + timeframe in one scan envelope
    with pytest.raises(ValueError, match="Duplicate observation in scan envelope"):
        MarketScan(
            scan_id="scan-dup-test",
            source="n8n_test",
            scan_timestamp=now,
            observations=[MarketObservation(**obs), MarketObservation(**obs)],
        )


def test_schema_requires_timezone_aware():
    naive_dt = datetime(2026, 10, 3, 6, 0, 0)
    with pytest.raises(ValueError, match="must be timezone-aware"):
        MarketObservation(
            instrument={"symbol": "BTCUSDT", "venue": "binance_futures", "market_type": "crypto_perp"},
            timeframe="M5",
            observed_at=naive_dt,
        )


def test_canonical_payload_digest_determinism():
    now = datetime.now(timezone.utc)
    obs1 = MarketObservation(
        instrument=Instrument(symbol="ETHUSDT", venue="binance_futures", market_type="crypto_perp"),
        timeframe="M5",
        observed_at=now,
        price=3200.0,
    )
    obs2 = MarketObservation(
        instrument=Instrument(symbol="BTCUSDT", venue="binance_futures", market_type="crypto_perp"),
        timeframe="M5",
        observed_at=now,
        price=65000.0,
    )

    scan_a = MarketScan(
        scan_id="scan-hash-test",
        source="scanner",
        scan_timestamp=now,
        observations=[obs1, obs2],
    )
    scan_b = MarketScan(
        scan_id="scan-hash-test",
        source="scanner",
        scan_timestamp=now,
        observations=[obs2, obs1],  # reversed order
    )

    _, _, hash_a = canonical_payload_digest(scan_a)
    _, _, hash_b = canonical_payload_digest(scan_b)
    assert hash_a == hash_b, "Observations must be canonically sorted before hashing"


def test_ingestion_authentication():
    client = TestClient(app)
    payload = _sample_scan_payload()

    # 1. Missing key -> 401
    resp = client.post(INGESTION_URL, json=payload)
    assert resp.status_code == 401

    # 2. Invalid key -> 403
    resp = client.post(
        INGESTION_URL,
        json=payload,
        headers={"X-Ingestion-Key": "wrong-key"},
    )
    assert resp.status_code == 403

    # 3. Valid key via X-Ingestion-Key -> 201
    resp = client.post(
        INGESTION_URL,
        json=payload,
        headers={"X-Ingestion-Key": TEST_KEY},
    )
    assert resp.status_code == 201


def test_ingestion_idempotency_and_outbox(db_session):
    client = TestClient(app)
    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    payload = _sample_scan_payload(scan_id=scan_id, symbol="SOLUSDT", price=145.5)
    headers = {"X-Ingestion-Key": TEST_KEY}

    # First submission -> 201 Created
    resp1 = client.post(INGESTION_URL, json=payload, headers=headers)
    assert resp1.status_code == 201
    data1 = resp1.json()
    assert data1["duplicate"] is False
    assert data1["scan_id"] == scan_id
    assert data1["observation_count"] == 1
    assert data1["event_id"] is not None

    # Check database records
    scan_row = (
        db_session.query(IngestionMarketScan)
        .filter(IngestionMarketScan.scan_id == scan_id)
        .first()
    )
    assert scan_row is not None
    assert scan_row.payload_sha256 == data1["payload_sha256"]

    outbox_row = (
        db_session.query(IngestionOutbox)
        .filter(IngestionOutbox.aggregate_id == str(scan_row.id))
        .first()
    )
    assert outbox_row is not None
    assert outbox_row.event_type == "market.scan.received"
    assert outbox_row.status == "pending"

    # Resubmission (identical retry) -> 200 OK with duplicate=True
    resp2 = client.post(INGESTION_URL, json=payload, headers=headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["duplicate"] is True
    assert data2["ingestion_id"] == data1["ingestion_id"]
    assert data2["payload_sha256"] == data1["payload_sha256"]


def test_ingestion_conflict_on_modified_payload(db_session):
    client = TestClient(app)
    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    payload = _sample_scan_payload(scan_id=scan_id, price=100.0)
    headers = {"X-Ingestion-Key": TEST_KEY}

    # Initial submission
    resp1 = client.post(INGESTION_URL, json=payload, headers=headers)
    assert resp1.status_code == 201

    # Modify payload under same scan_id (e.g. price changed)
    payload_modified = dict(payload)
    payload_modified["observations"] = [
        dict(payload["observations"][0], price=200.0)
    ]

    resp2 = client.post(INGESTION_URL, json=payload_modified, headers=headers)
    assert resp2.status_code == 409
    assert "different payload digest" in resp2.json()["detail"]


def test_stale_scan_rejected_on_first_insert_but_duplicate_allowed(db_session):
    client = TestClient(app)
    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    headers = {"X-Ingestion-Key": TEST_KEY}

    # 1. Fresh initial submission (10s old)
    fresh_payload = _sample_scan_payload(scan_id=scan_id, age_sec=10.0)
    resp1 = client.post(INGESTION_URL, json=fresh_payload, headers=headers)
    assert resp1.status_code == 201

    # 2. Resubmission of identical scan is permitted even if considered older
    resp2 = client.post(INGESTION_URL, json=fresh_payload, headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["duplicate"] is True

    # 3. New scan that is 20 minutes old (> 900s) is rejected on first insert
    stale_scan_id = f"stale_{uuid.uuid4().hex[:8]}"
    stale_payload = _sample_scan_payload(scan_id=stale_scan_id, age_sec=1200.0)
    resp3 = client.post(INGESTION_URL, json=stale_payload, headers=headers)
    assert resp3.status_code == 422
    assert "too stale" in resp3.json()["detail"]
