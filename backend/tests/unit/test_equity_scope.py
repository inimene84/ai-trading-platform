"""Equity / risk book scope — split demo cTrader vs live Binance."""

from backend.services.equity_scope import (
    SPLIT_BOOK_WARNING,
    compose_balance_payload,
    describe_equity_books,
    drawdown_suppressed_by_sandbox,
    effective_risk_scope,
    is_split_book,
    risk_broker_name,
)
from backend.services.risk_guard import is_drawdown_suppressed_in_testing


def test_split_book_when_ctrader_demo_and_binance_live(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    monkeypatch.setenv("EQUITY_RISK_SCOPE", "union")

    assert is_split_book() is True
    assert effective_risk_scope() == "broker"
    assert risk_broker_name() == "binance_futures"
    meta = describe_equity_books()
    assert meta["split_book"] is True
    assert meta["ctrader_env"] == "sandbox"
    assert meta["binance_env"] == "live"
    assert meta["split_book_warning"] == SPLIT_BOOK_WARNING


def test_compose_does_not_union_demo_and_live_equity(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "demo")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")

    payload = compose_balance_payload(
        [
            {"broker": "ctrader", "equity": 150.0, "balance": 150.0, "available": 150.0},
            {"broker": "binance_futures", "equity": 918.0, "balance": 900.0, "available": 800.0},
        ]
    )
    assert payload["split_book"] is True
    assert payload["equity"] == 918.0
    assert payload["display_union_equity"] is None
    assert payload["risk_broker"] == "binance_futures"


def test_compose_fails_closed_when_live_binance_book_missing(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")

    payload = compose_balance_payload(
        [{"broker": "ctrader", "equity": 150.0, "balance": 150.0, "available": 150.0}]
    )
    assert payload["error"] == "risk_book_missing"
    assert payload["equity"] == 0.0


def test_sandbox_does_not_suppress_drawdown_when_binance_is_live_cash(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    monkeypatch.setenv("DISABLE_DRAWDOWN_IN_TESTING", "false")
    monkeypatch.setenv("TESTING_MODE", "false")
    monkeypatch.setenv("ENFORCE_SANDBOX_DRAWDOWN", "false")

    assert drawdown_suppressed_by_sandbox() is False
    assert is_drawdown_suppressed_in_testing() is False


def test_sandbox_still_suppresses_drawdown_without_live_cash(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    monkeypatch.setenv("DISABLE_DRAWDOWN_IN_TESTING", "false")
    monkeypatch.setenv("TESTING_MODE", "false")

    assert drawdown_suppressed_by_sandbox() is True
    assert is_drawdown_suppressed_in_testing() is True


def test_unset_binance_testnet_is_live_cash_with_ctrader_sandbox(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    monkeypatch.delenv("BINANCE_TESTNET", raising=False)

    from backend.services.equity_scope import binance_env

    assert binance_env() == "live"
    assert is_split_book() is True
    assert risk_broker_name() == "binance_futures"
    assert drawdown_suppressed_by_sandbox() is False


def test_binance_testnet_one_is_live_like_adapter(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    from backend.services.equity_scope import binance_env

    assert binance_env() == "live"


def test_ctrader_env_live_without_confirm_is_not_live_cash(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "live")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.delenv("CTRADER_LIVE_CONFIRM", raising=False)
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")

    from backend.services.equity_scope import ctrader_money_mode

    assert ctrader_money_mode() == "demo"
    assert is_split_book() is True
    assert risk_broker_name() == "binance_futures"


def test_union_scope_sums_same_mode_books(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "live")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("CTRADER_LIVE_CONFIRM", "I_UNDERSTAND")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    monkeypatch.setenv("EQUITY_RISK_SCOPE", "union")

    payload = compose_balance_payload(
        [
            {"broker": "ctrader", "equity": 200.0, "balance": 200.0, "available": 180.0},
            {"broker": "binance_futures", "equity": 900.0, "balance": 900.0, "available": 800.0},
        ]
    )
    assert payload["split_book"] is False
    assert payload["equity"] == 1100.0
    assert payload["display_union_equity"] == 1100.0
    assert payload.get("error") is None
