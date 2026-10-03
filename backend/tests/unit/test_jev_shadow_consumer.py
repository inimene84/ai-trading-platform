"""
Unit tests for WP2: JEV Shadow Consumer & Decision Trace
- Outbox event claiming and processing lifecycle
- Dual decision trace creation (baseline vs counterfactual with-JEV)
- Divergence classification (AGREED, UPGRADED, DOWNGRADED, FLIPPED)
- JEV advisory opinion storage
- Shadow mode verification (zero broker calls, executed=False)
- Fail-closed behavior on vetoes and conflicting signals
"""

import uuid
from datetime import datetime, timezone

import pytest

from backend.database.models import (
    DecisionOpinion,
    DecisionRecord,
    IngestionMarketScan,
    IngestionOutbox,
)
from backend.services.outbox_worker import (
    claim_pending_events,
    classify_divergence,
    process_outbox_event,
    run_outbox_worker_iteration,
)
from backend.services.trading_mode import TradingMode


def test_divergence_classification():
    assert classify_divergence("BUY", "BUY") == "AGREED"
    assert classify_divergence("SELL", "SELL") == "AGREED"
    assert classify_divergence("HOLD", "HOLD") == "AGREED"
    assert classify_divergence("HOLD", "BUY") == "UPGRADED"
    assert classify_divergence("HOLD", "SELL") == "UPGRADED"
    assert classify_divergence("BUY", "HOLD") == "DOWNGRADED"
    assert classify_divergence("SELL", "HOLD") == "DOWNGRADED"
    assert classify_divergence("BUY", "SELL") == "FLIPPED"
    assert classify_divergence("SELL", "BUY") == "FLIPPED"


@pytest.mark.asyncio
async def test_outbox_worker_dual_decision_trace(db_session, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")

    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    trace_id = "trace-shadow-test-001"

    now = datetime.now(timezone.utc)
    scan_payload = {
        "scan_id": scan_id,
        "source": "n8n_market_scanner",
        "scan_timestamp": now.isoformat(),
        "trace_id": trace_id,
        "observations": [
            {
                "instrument": {"symbol": "BTCUSDT", "venue": "binance_futures", "market_type": "crypto_perp"},
                "timeframe": "M5",
                "observed_at": now.isoformat(),
                "price": 64500.0,
                "signal_direction": "BUY",
                "confidence": 0.75,
                "indicators": {"rsi": 58.2, "macd": 24.1},
            }
        ],
    }

    # Insert test outbox record
    outbox_row = IngestionOutbox(
        event_id=event_id,
        event_type="market.scan.received",
        aggregate_type="market_scan",
        aggregate_id=scan_id,
        payload=scan_payload,
        status="pending",
    )
    db_session.add(outbox_row)
    db_session.commit()

    # Mock JEV agreeing with the signal
    jev_mock = {
        "status": "ok",
        "signal": "BUY",
        "confidence": 0.85,
        "vetoed": False,
        "conflict": False,
        "reason": "Strong orderbook liquidity and momentum confirmation",
    }

    # Process outbox event
    claimed = claim_pending_events(db_session, batch_size=10)
    assert len(claimed) == 1
    assert claimed[0].status == "processing"
    assert claimed[0].attempt_count == 1

    res = await process_outbox_event(db_session, claimed[0], jev_override_eval=jev_mock)
    assert res["status"] == "completed"
    assert res["decisions_count"] == 1

    # Verify event state in DB
    refreshed_event = db_session.query(IngestionOutbox).filter(IngestionOutbox.event_id == event_id).first()
    assert refreshed_event.status == "completed"
    assert refreshed_event.processed_at is not None

    # Verify DecisionRecord
    record = db_session.query(DecisionRecord).filter(DecisionRecord.scan_id == scan_id).first()
    assert record is not None
    assert record.symbol == "BTCUSDT"
    assert record.trace_id == trace_id
    assert record.event_id == event_id
    assert record.baseline_action == "BUY"
    assert record.baseline_confidence == 0.75
    assert record.with_jev_action == "BUY"
    assert record.with_jev_confidence > 0.75  # Boosted by agreeing JEV
    assert record.divergence == "AGREED"
    assert record.jev_signal == "BUY"
    assert record.jev_confidence == 0.85
    assert record.jev_vetoed is False
    assert record.execution_mode == "shadow"
    assert record.executed is False  # Zero broker orders in shadow mode
    assert record.order_intent_id is None

    # Verify DecisionOpinion
    opinion = db_session.query(DecisionOpinion).filter(DecisionOpinion.scan_id == scan_id).first()
    assert opinion is not None
    assert opinion.symbol == "BTCUSDT"
    assert opinion.signal == "BUY"
    assert opinion.confidence == 0.85
    assert opinion.source == "jev_analyst"


@pytest.mark.asyncio
async def test_outbox_worker_veto_downgrades_to_hold(db_session, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")

    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    event_id = f"evt_{uuid.uuid4().hex[:8]}"

    now = datetime.now(timezone.utc)
    scan_payload = {
        "scan_id": scan_id,
        "source": "n8n_market_scanner",
        "scan_timestamp": now.isoformat(),
        "observations": [
            {
                "instrument": {"symbol": "ETHUSDT", "venue": "binance_futures", "market_type": "crypto_perp"},
                "timeframe": "M5",
                "observed_at": now.isoformat(),
                "price": 3100.0,
                "signal_direction": "BUY",
                "confidence": 0.80,
                "indicators": {},
            }
        ],
    }

    outbox_row = IngestionOutbox(
        event_id=event_id,
        event_type="market.scan.received",
        aggregate_type="market_scan",
        aggregate_id=scan_id,
        payload=scan_payload,
        status="pending",
    )
    db_session.add(outbox_row)
    db_session.commit()

    # JEV issues an uncertainty veto
    jev_veto_mock = {
        "status": "ok",
        "signal": "BUY",
        "confidence": 0.20,
        "vetoed": True,
        "conflict": False,
        "reason": "Uncertainty spike / macroeconomic event veto",
    }

    claimed = claim_pending_events(db_session, batch_size=5)
    await process_outbox_event(db_session, claimed[0], jev_override_eval=jev_veto_mock)

    record = db_session.query(DecisionRecord).filter(DecisionRecord.scan_id == scan_id).first()
    assert record is not None
    assert record.baseline_action == "BUY"
    assert record.with_jev_action == "HOLD"  # Vetoed down to HOLD
    assert record.divergence == "DOWNGRADED"
    assert record.jev_vetoed is True
    assert record.executed is False


@pytest.mark.asyncio
async def test_outbox_worker_conflicting_signals_fail_closed(db_session, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")

    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    event_id = f"evt_{uuid.uuid4().hex[:8]}"

    now = datetime.now(timezone.utc)
    scan_payload = {
        "scan_id": scan_id,
        "source": "n8n_market_scanner",
        "scan_timestamp": now.isoformat(),
        "observations": [
            {
                "instrument": {"symbol": "SOLUSDT", "venue": "binance_futures", "market_type": "crypto_perp"},
                "timeframe": "M5",
                "observed_at": now.isoformat(),
                "price": 142.0,
                "signal_direction": "BUY",
                "confidence": 0.70,
                "indicators": {},
            }
        ],
    }

    outbox_row = IngestionOutbox(
        event_id=event_id,
        event_type="market.scan.received",
        aggregate_type="market_scan",
        aggregate_id=scan_id,
        payload=scan_payload,
        status="pending",
    )
    db_session.add(outbox_row)
    db_session.commit()

    # JEV directly opposes the baseline signal
    jev_oppose_mock = {
        "status": "ok",
        "signal": "SELL",
        "confidence": 0.75,
        "vetoed": False,
        "conflict": False,
        "reason": "Top divergence on order flow",
    }

    claimed = claim_pending_events(db_session, batch_size=5)
    await process_outbox_event(db_session, claimed[0], jev_override_eval=jev_oppose_mock)

    record = db_session.query(DecisionRecord).filter(DecisionRecord.scan_id == scan_id).first()
    assert record is not None
    assert record.baseline_action == "BUY"
    assert record.with_jev_action == "HOLD"  # Conflict fails closed to HOLD
    assert record.divergence == "DOWNGRADED"
    assert record.executed is False


@pytest.mark.asyncio
async def test_run_outbox_worker_iteration_batch(db_session, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")

    now = datetime.now(timezone.utc)
    for i in range(3):
        scan_id = f"scan_iter_{i}_{uuid.uuid4().hex[:6]}"
        event_id = f"evt_iter_{i}_{uuid.uuid4().hex[:6]}"
        outbox_row = IngestionOutbox(
            event_id=event_id,
            event_type="market.scan.received",
            aggregate_type="market_scan",
            aggregate_id=scan_id,
            payload={
                "scan_id": scan_id,
                "source": "n8n_test",
                "scan_timestamp": now.isoformat(),
                "observations": [
                    {
                        "instrument": {"symbol": f"SYM{i}USDT", "venue": "binance_futures", "market_type": "crypto_perp"},
                        "timeframe": "M5",
                        "observed_at": now.isoformat(),
                        "price": 100.0 * (i + 1),
                        "signal_direction": "HOLD",
                        "confidence": 0.5,
                        "indicators": {},
                    }
                ],
            },
            status="pending",
        )
        db_session.add(outbox_row)
    db_session.commit()

    processed = await run_outbox_worker_iteration(
        db_session,
        batch_size=5,
        jev_override_eval={"status": "ok", "signal": "HOLD", "confidence": 0.5, "vetoed": False, "conflict": False},
    )
    assert processed == 3

    # Verify all 3 decision records created
    records = db_session.query(DecisionRecord).all()
    assert len(records) == 3
