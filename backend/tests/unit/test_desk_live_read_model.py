"""
Unit tests for WP5: Desk UI Live Read Model API
- GET /api/desk/status
- GET /api/desk/positions
- GET /api/desk/tape
- GET /api/desk/decisions
- GET /api/desk/opinions
- GET /api/desk/outbox-health
"""

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.database.models import (
    DecisionOpinion,
    DecisionRecord,
    IngestionOutbox,
    Trade,
)
from backend.main import app


def test_desk_status_endpoint(db_session):
    client = TestClient(app)
    resp = client.get("/api/desk/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "effective_mode" in data
    assert "is_trading_allowed" in data
    assert "kill_switch_engaged" in data
    assert "open_positions_count" in data
    assert data["is_read_only"] is True


def test_desk_positions_endpoint(db_session):
    client = TestClient(app)

    # Insert an open trade
    t = Trade(
        symbol="BTCUSDT",
        direction="BUY",
        quantity=0.5,
        entry_price=64000.0,
        exit_price=64500.0,
        stop_loss=63000.0,
        take_profit=67000.0,
        status="open",
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(t)
    db_session.commit()

    resp = client.get("/api/desk/positions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    pos = [p for p in data if p["symbol"] == "BTCUSDT"][0]
    assert pos["side"] == "BUY"
    assert pos["quantity"] == 0.5
    assert pos["entry_price"] == 64000.0


def test_desk_tape_endpoint(db_session):
    client = TestClient(app)

    # Insert a closed trade
    t = Trade(
        symbol="ETHUSDT",
        direction="SELL",
        quantity=2.0,
        entry_price=3200.0,
        exit_price=3100.0,
        pnl=200.0,
        status="closed",
        timestamp=datetime.now(timezone.utc),
        closed_at=datetime.now(timezone.utc),
    )
    db_session.add(t)
    db_session.commit()

    resp = client.get("/api/desk/tape?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    fill = [f for f in data if f["symbol"] == "ETHUSDT"][0]
    assert fill["side"] == "SELL"
    assert fill["realized_pnl"] == 200.0


def test_desk_decisions_and_opinions(db_session):
    client = TestClient(app)

    dec_id = f"dec_desk_{uuid.uuid4().hex[:8]}"
    op_id = f"op_desk_{uuid.uuid4().hex[:8]}"
    scan_id = "scan-desk-999"

    record = DecisionRecord(
        decision_id=dec_id,
        scan_id=scan_id,
        event_id="evt-desk-999",
        symbol="SOLUSDT",
        observed_price=145.0,
        baseline_action="BUY",
        baseline_confidence=0.8,
        with_jev_action="BUY",
        with_jev_confidence=0.9,
        divergence="AGREED",
        jev_signal="BUY",
        jev_confidence=0.9,
        execution_mode="shadow",
        executed=False,
    )
    opinion = DecisionOpinion(
        opinion_id=op_id,
        scan_id=scan_id,
        symbol="SOLUSDT",
        signal="BUY",
        confidence=0.9,
        reasoning="Desk test opinion",
    )
    db_session.add(record)
    db_session.add(opinion)
    db_session.commit()

    # Check decisions
    d_resp = client.get("/api/desk/decisions?limit=10")
    assert d_resp.status_code == 200
    d_list = [d for d in d_resp.json() if d["decision_id"] == dec_id]
    assert len(d_list) == 1
    assert d_list[0]["symbol"] == "SOLUSDT"
    assert d_list[0]["divergence"] == "AGREED"

    # Check opinions
    o_resp = client.get("/api/desk/opinions?symbol=SOLUSDT")
    assert o_resp.status_code == 200
    o_list = o_resp.json()
    assert len(o_list) == 1
    assert o_list[0]["opinion_id"] == op_id


def test_desk_outbox_health(db_session):
    client = TestClient(app)

    # Insert an outbox row
    outbox_row = IngestionOutbox(
        event_id=f"evt_health_{uuid.uuid4().hex[:6]}",
        event_type="market.scan.received",
        aggregate_type="market_scan",
        aggregate_id="scan-health",
        payload={},
        status="pending",
    )
    db_session.add(outbox_row)
    db_session.commit()

    resp = client.get("/api/desk/outbox-health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pending_count"] >= 1
    assert "healthy" in data
    assert "oldest_pending_age_sec" in data
