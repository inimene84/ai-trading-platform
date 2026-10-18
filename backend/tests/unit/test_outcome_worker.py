"""
Unit tests for WP3: Outcome Closure & Labeling Worker
- Forward return and Triple-Barrier calculation
- Long and short profit/loss symmetry
- Idempotency across re-runs (same decision_id, horizon, and policy_version)
- Batch mature decision labeling
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from backend.database.models import DecisionOutcome, DecisionRecord
from backend.services.outcome_worker import (
    compute_triple_barrier_label,
    label_decision_record,
    run_outcome_worker_iteration,
)


def test_triple_barrier_long_calculations():
    # Long BUY from 100 to 102 (+2%) -> WIN
    label, ret, mfe, mae = compute_triple_barrier_label(
        direction="BUY",
        reference_price=100.0,
        horizon_price=102.0,
        high_price=102.5,
        low_price=99.8,
    )
    assert label == "WIN"
    assert ret == 2.0
    assert mfe == 2.5
    assert mae == -0.2

    # Long BUY from 100 to 98 (-2%) -> LOSS
    label, ret, mfe, mae = compute_triple_barrier_label(
        direction="BUY",
        reference_price=100.0,
        horizon_price=98.0,
        high_price=100.2,
        low_price=97.5,
    )
    assert label == "LOSS"
    assert ret == -2.0


def test_triple_barrier_short_calculations():
    # Short SELL from 100 to 97 (Price dropped 3% -> 3% profit for short) -> WIN
    label, ret, mfe, mae = compute_triple_barrier_label(
        direction="SELL",
        reference_price=100.0,
        horizon_price=97.0,
        high_price=100.5,
        low_price=96.5,
    )
    assert label == "WIN"
    assert ret == 3.0
    assert mfe == 3.5
    assert mae == -0.5

    # Short SELL from 100 to 103 (Price rose 3% -> 3% loss for short) -> LOSS
    label, ret, mfe, mae = compute_triple_barrier_label(
        direction="SELL",
        reference_price=100.0,
        horizon_price=103.0,
        high_price=103.5,
        low_price=99.5,
    )
    assert label == "LOSS"
    assert ret == -3.0


def test_label_decision_record_idempotent(db_session):
    dec_id = f"dec_test_{uuid.uuid4().hex[:8]}"
    record = DecisionRecord(
        decision_id=dec_id,
        scan_id="scan-123",
        event_id="evt-123",
        symbol="BTCUSDT",
        observed_price=60000.0,
        baseline_action="BUY",
        with_jev_action="BUY",
        divergence="AGREED",
        execution_mode="shadow",
        executed=False,
    )
    db_session.add(record)
    db_session.commit()

    # First label creation
    outcome1 = label_decision_record(
        db=db_session,
        record=record,
        horizon="1H",
        horizon_price=61200.0,  # +2%
    )
    assert outcome1 is not None
    assert outcome1.label == "WIN"
    assert outcome1.forward_return_pct == 2.0

    # Second label creation with identical params -> returns existing row without duplication
    outcome2 = label_decision_record(
        db=db_session,
        record=record,
        horizon="1H",
        horizon_price=61200.0,
    )
    assert outcome2.outcome_id == outcome1.outcome_id

    outcomes = db_session.query(DecisionOutcome).filter(DecisionOutcome.decision_id == dec_id).all()
    assert len(outcomes) == 1


def test_run_outcome_worker_iteration(db_session):
    past_time = datetime.now(timezone.utc) - timedelta(hours=2)
    dec_id = f"dec_mature_{uuid.uuid4().hex[:8]}"
    record = DecisionRecord(
        decision_id=dec_id,
        scan_id="scan-old",
        event_id="evt-old",
        symbol="ETHUSDT",
        observed_price=3000.0,
        baseline_action="BUY",
        with_jev_action="BUY",
        divergence="AGREED",
        created_at=past_time,
        execution_mode="shadow",
        executed=False,
    )
    db_session.add(record)
    db_session.commit()

    def mock_price_provider(sym, dt):
        return 3150.0  # +5% gain

    processed = run_outcome_worker_iteration(
        db=db_session,
        horizon="1H",
        batch_size=10,
        price_provider=mock_price_provider,
    )
    assert processed == 1

    outcome = db_session.query(DecisionOutcome).filter(DecisionOutcome.decision_id == dec_id).first()
    assert outcome is not None
    assert outcome.label == "WIN"
    assert outcome.forward_return_pct == 5.0
