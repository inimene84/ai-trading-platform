"""
Unit tests for WP0 Safety Baseline & Control Plane:
- Unified TradingMode (SHADOW, PAPER, LIVE, BACKTEST)
- Persistent Kill Switch engagement / release
- Execution blocking under SHADOW mode & Kill Switch in UnifiedTrading
- Control plane API endpoints
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.database.connection import Base, engine, get_db
from backend.database.models import ControlPlaneState
from backend.main import app
from backend.services.sentry_state import (
    TradingStatus,
    engage_kill_switch,
    get_trading_status,
    is_kill_switch_engaged,
    is_trading_allowed,
    read_state,
    release_kill_switch,
    write_state,
)
from backend.services.trading_mode import (
    TradingMode,
    get_trading_mode,
    is_live_mode,
    is_paper_mode,
    is_shadow_mode,
    live_exchange_orders_allowed,
)
from backend.services.unified_trading import OrderSide, OrderType, UnifiedOrder, UnifiedTrading


@pytest.fixture(autouse=True)
def setup_sentry_state(db_session):
    # Ensure sentry status is reset to ACTIVE for test isolation
    write_state(TradingStatus.ACTIVE)
    yield
    write_state(TradingStatus.ACTIVE)



def test_trading_mode_shadow(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")
    assert get_trading_mode() == TradingMode.SHADOW
    assert is_shadow_mode() is True
    assert is_paper_mode() is False
    assert is_live_mode() is False
    assert live_exchange_orders_allowed() is False


def test_trading_mode_paper(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    assert get_trading_mode() == TradingMode.PAPER
    assert is_paper_mode() is True
    assert is_shadow_mode() is False
    assert is_live_mode() is False
    assert live_exchange_orders_allowed() is False


def test_trading_mode_live(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    assert get_trading_mode() == TradingMode.LIVE
    assert is_live_mode() is True
    assert live_exchange_orders_allowed() is True


def test_kill_switch_lifecycle(db_session):
    # Initial state should be active
    assert is_trading_allowed(db=db_session) is True
    assert is_kill_switch_engaged(db=db_session) is False

    # Engage kill switch
    engage_res = engage_kill_switch(reason="Test anomaly detected", actor="tester", db=db_session)
    assert engage_res["status"] == TradingStatus.HALTED_MANUAL.value
    assert get_trading_status() == TradingStatus.HALTED_MANUAL
    assert is_trading_allowed(db=db_session) is False
    assert is_kill_switch_engaged(db=db_session) is True

    # Check DB record
    record = (
        db_session.query(ControlPlaneState)
        .order_by(ControlPlaneState.id.desc())
        .first()
    )
    assert record is not None
    assert record.kill_switch_engaged is True
    assert record.reason == "Test anomaly detected"
    assert record.actor == "tester"

    # Release kill switch
    release_res = release_kill_switch(actor="tester", reason="Test cleared", db=db_session)
    assert release_res["status"] == TradingStatus.ACTIVE.value
    assert get_trading_status() == TradingStatus.ACTIVE
    assert is_trading_allowed(db=db_session) is True
    assert is_kill_switch_engaged(db=db_session) is False

    # Check DB record updated
    db_session.refresh(record)
    assert record.kill_switch_engaged is False
    assert record.released_at is not None


def test_unified_trading_blocked_under_shadow_mode(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")
    ut = UnifiedTrading()
    sess = ut.init_session("binance_futures", mode="paper")

    order = UnifiedOrder(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        price=60000.0,
    )
    resp = ut.place_order(order, session_id=sess.broker + "_paper")
    assert resp.success is False
    assert "SHADOW" in resp.message
    assert resp.mode == "shadow"


def test_unified_trading_blocked_when_kill_switch_engaged(monkeypatch, db_session):
    monkeypatch.setenv("TRADING_MODE", "paper")
    engage_kill_switch(reason="Emergency market stop", actor="risk_sentinel", db=db_session)

    ut = UnifiedTrading()
    sess = ut.init_session("binance_futures", mode="paper")

    order = UnifiedOrder(
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        price=2500.0,
    )
    resp = ut.place_order(order, session_id=sess.broker + "_paper")
    assert resp.success is False
    assert "Trading blocked" in resp.message
    assert "Emergency market stop" in resp.message


def test_control_plane_endpoints(monkeypatch, db_session):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-test-key")

    client = TestClient(app)

    # 1. GET status
    resp = client.get("/api/control-plane/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["effective_mode"] == "paper"
    assert data["kill_switch_engaged"] is False
    assert data["is_trading_allowed"] is True

    # 2. POST engage without admin token -> 401
    resp = client.post(
        "/api/control-plane/kill-switch/engage",
        json={"reason": "Audit test", "actor": "operator"},
    )
    assert resp.status_code == 401

    # 3. POST engage with admin token
    headers = {"Authorization": "Bearer secret-test-key"}
    resp = client.post(
        "/api/control-plane/kill-switch/engage",
        json={"reason": "Audit test", "actor": "operator"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["kill_switch_engaged"] is True
    assert data["is_trading_allowed"] is False
    assert data["kill_switch_reason"] == "Audit test"

    # 4. POST release with admin token
    resp = client.post(
        "/api/control-plane/kill-switch/release",
        json={"reason": "Audit completed", "actor": "operator"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["kill_switch_engaged"] is False
    assert data["is_trading_allowed"] is True
