from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.services.ctrader_weekend_protect import (
    apply_weekend_protect,
    evaluate_swap_hold,
    evaluate_weekend_protect,
    is_friday_pre_close,
    swap_hold_mode,
    weekend_protect_mode,
)


FRIDAY_20_30 = datetime(2026, 9, 11, 20, 30, tzinfo=timezone.utc)
WEDNESDAY = datetime(2026, 9, 16, 14, 0, tzinfo=timezone.utc)


def test_default_modes_are_shadow(monkeypatch):
    monkeypatch.delenv("CTRADER_WEEKEND_PROTECT_MODE", raising=False)
    monkeypatch.delenv("CTRADER_SWAP_HOLD_MODE", raising=False)
    assert weekend_protect_mode() == "shadow"
    assert swap_hold_mode() == "shadow"


def test_friday_pre_close_window(monkeypatch):
    monkeypatch.setenv("CTRADER_WEEKEND_PROTECT_MINUTES", "60")
    assert is_friday_pre_close(FRIDAY_20_30) is True
    assert is_friday_pre_close(WEDNESDAY) is False
    assert is_friday_pre_close(datetime(2026, 9, 11, 21, 1, tzinfo=timezone.utc)) is False


def test_shadow_plans_but_does_not_call_broker():
    pos = {
        "symbol": "EURUSD",
        "side": "BUY",
        "position_id": "p1",
        "entry_price": 1.1000,
        "current_price": 1.1010,
        "quantity": 0.01,
    }
    broker = MagicMock()
    plan = apply_weekend_protect([pos], broker, now=FRIDAY_20_30)
    assert plan["mode"] == "shadow"
    assert plan["active_window"] is True
    assert plan["actions"]
    assert plan["actions"][0]["intended"] == "tighten"
    broker.close_position.assert_not_called()
    broker.amend_position_sltp.assert_not_called()


def test_flatten_mode_closes(monkeypatch):
    monkeypatch.setenv("CTRADER_WEEKEND_PROTECT_MODE", "flatten")
    pos = {
        "symbol": "GBPUSD",
        "side": "SELL",
        "position_id": "p2",
        "entry_price": 1.2700,
        "current_price": 1.2690,
        "quantity": 0.02,
    }
    broker = MagicMock()
    broker.close_position.return_value = {"status": "sent"}
    plan = apply_weekend_protect([pos], broker, now=FRIDAY_20_30)
    assert plan["actions"][0]["intended"] == "flatten"
    broker.close_position.assert_called_once()


def test_tighten_moves_winner_to_break_even(monkeypatch):
    monkeypatch.setenv("CTRADER_WEEKEND_PROTECT_MODE", "tighten")
    pos = {
        "symbol": "EURUSD",
        "side": "BUY",
        "position_id": "p3",
        "entry_price": 1.1000,
        "current_price": 1.1050,
        "quantity": 0.01,
    }
    broker = MagicMock()
    broker.amend_position_sltp.return_value = {"status": "sent"}
    plan = apply_weekend_protect([pos], broker, now=FRIDAY_20_30)
    assert plan["actions"][0]["detail"] == "tighten_to_break_even"
    broker.amend_position_sltp.assert_called_once_with("p3", stop_loss=1.1000)


def test_crypto_positions_are_ignored_on_friday():
    pos = {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "position_id": "p4",
        "entry_price": 70000,
        "current_price": 69900,
        "quantity": 0.01,
    }
    plan = evaluate_weekend_protect([pos], now=FRIDAY_20_30, mode="flatten")
    assert plan["actions"] == []


def test_off_mode_is_noop(monkeypatch):
    monkeypatch.setenv("CTRADER_WEEKEND_PROTECT_MODE", "off")
    plan = evaluate_weekend_protect(
        [{"symbol": "EURUSD", "side": "BUY", "position_id": "x", "entry_price": 1.1, "current_price": 1.1}],
        now=FRIDAY_20_30,
    )
    assert plan["active_window"] is False
    assert plan["actions"] == []


def test_swap_hold_shadow_does_not_flatten():
    opened = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)  # Monday
    now = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)  # Wednesday > 24h
    pos = {
        "symbol": "EURUSD",
        "side": "BUY",
        "position_id": "p5",
        "entry_price": 1.1000,
        "current_price": 1.1001,
        "take_profit": 1.1002,
        "quantity": 0.10,
        "opened_at": opened,
    }
    decision = evaluate_swap_hold(pos, now=now, expected_remaining_usd=0.01)
    assert decision["hold_hours"] > 24
    assert decision["would_flatten"] is True
    assert decision["action"] == "hold"
    assert "shadow" in decision["reason"]


def test_swap_hold_enforce_flattens_negative_ev(monkeypatch):
    monkeypatch.setenv("CTRADER_SWAP_HOLD_MODE", "enforce")
    opened = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)
    now = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)
    pos = {
        "symbol": "EURUSD",
        "side": "BUY",
        "opened_at": opened,
        "quantity": 0.10,
        "entry_price": 1.1,
        "current_price": 1.1,
    }
    decision = evaluate_swap_hold(pos, now=now, expected_remaining_usd=0.0)
    assert decision["action"] == "flatten"
