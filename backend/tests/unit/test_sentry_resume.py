import asyncio

import pytest

from backend.services.sentry_state import TradingStatus, halt_trading, read_state
from backend.services import sentry_resume


@pytest.fixture
def sentry_state_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "sentry"
    monkeypatch.setenv("SENTRY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("SENTRY_RESUME_REQUIRE_RECONCILE", "false")
    return state_dir


def test_safe_resume_after_sentry_halt(sentry_state_dir, monkeypatch):
    async def _noop_telegram(*_args, **_kwargs):
        return True

    async def _noop_reconcile():
        return {"db_closed": 0, "exchange_only_symbols": [], "error": None}

    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setattr(sentry_resume, "send_telegram_message", _noop_telegram)
    monkeypatch.setattr(sentry_resume, "reconcile_positions", _noop_reconcile)

    halt_trading(reason="crash", halted_by="test")
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="watchdog"))
    assert result["ok"] is True
    assert read_state()["status"] == TradingStatus.ACTIVE.value


def test_live_auto_resume_defaults_off(sentry_state_dir, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("SENTRY_AUTO_RESUME_ENABLED", raising=False)
    monkeypatch.delenv("SENTRY_AUTO_RESUME_LIVE_CONFIRM", raising=False)
    halt_trading(reason="crash", halted_by="sentry")
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="sentry_watchdog"))
    assert result["ok"] is False
    assert result["reason"] == "live_default_off"


def test_live_auto_resume_requires_confirm_even_when_enabled(sentry_state_dir, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("SENTRY_AUTO_RESUME_ENABLED", "true")
    monkeypatch.delenv("SENTRY_AUTO_RESUME_LIVE_CONFIRM", raising=False)
    halt_trading(reason="crash", halted_by="sentry")
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="sentry_watchdog"))
    assert result["ok"] is False
    assert result["reason"] == "live_confirm_required"


def test_live_auto_resume_with_explicit_confirm(sentry_state_dir, monkeypatch):
    async def _noop_telegram(*_args, **_kwargs):
        return True

    async def _noop_reconcile():
        return {"db_closed": 0, "exchange_only_symbols": [], "error": None}

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("SENTRY_AUTO_RESUME_ENABLED", "true")
    monkeypatch.setenv("SENTRY_AUTO_RESUME_LIVE_CONFIRM", "I_UNDERSTAND")
    monkeypatch.setattr(sentry_resume, "send_telegram_message", _noop_telegram)
    monkeypatch.setattr(sentry_resume, "reconcile_positions", _noop_reconcile)

    halt_trading(reason="crash", halted_by="sentry")
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="sentry_watchdog"))
    assert result["ok"] is True
    assert read_state()["status"] == TradingStatus.ACTIVE.value


def test_operator_resume_still_works_in_live_without_auto_confirm(sentry_state_dir, monkeypatch):
    async def _noop_telegram(*_args, **_kwargs):
        return True

    async def _noop_reconcile():
        return {"db_closed": 0, "exchange_only_symbols": [], "error": None}

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("SENTRY_AUTO_RESUME_LIVE_CONFIRM", raising=False)
    monkeypatch.setattr(sentry_resume, "send_telegram_message", _noop_telegram)
    monkeypatch.setattr(sentry_resume, "reconcile_positions", _noop_reconcile)

    halt_trading(reason="crash", halted_by="sentry")
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="operator", allow_manual=True))
    assert result["ok"] is True


def test_auto_resume_skips_manual_halt(sentry_state_dir):
    halt_trading(reason="maintenance", halted_by="operator", manual=True)
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="watchdog", allow_manual=False))
    assert result["ok"] is False
    assert result["reason"] == "manual_halt_requires_operator_resume"


def test_operator_resume_clears_manual_halt(sentry_state_dir, monkeypatch):
    async def _noop_telegram(*_args, **_kwargs):
        return True

    async def _noop_reconcile():
        return {"db_closed": 0, "exchange_only_symbols": [], "error": None}

    monkeypatch.setattr(sentry_resume, "send_telegram_message", _noop_telegram)
    monkeypatch.setattr(sentry_resume, "reconcile_positions", _noop_reconcile)

    halt_trading(reason="maintenance", halted_by="operator", manual=True)
    result = asyncio.run(sentry_resume.safe_resume(resumed_by="operator", allow_manual=True))
    assert result["ok"] is True
    assert read_state()["status"] == TradingStatus.ACTIVE.value


def test_split_book_reconcile_includes_binance_and_fails_on_live_error(sentry_state_dir, monkeypatch):
    class _FakeBroker:
        def get_positions(self, raise_on_error=False):
            return []

    async def _fake_one(_db, _broker, name):
        if name == "binance_futures":
            return {
                "broker": name,
                "db_closed": 0,
                "exchange_only_symbols": ["BTCUSDC"],
                "error": "wallet_timeout",
            }
        return {
            "broker": name,
            "db_closed": 1,
            "exchange_only_symbols": [],
            "error": None,
        }

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setattr(
        sentry_resume,
        "_brokers_to_reconcile",
        lambda: [(_FakeBroker(), "ctrader"), (_FakeBroker(), "binance_futures")],
    )
    monkeypatch.setattr(sentry_resume, "_reconcile_one_broker", _fake_one)

    result = asyncio.run(sentry_resume.reconcile_positions())
    assert result["error"] == "wallet_timeout"
    assert "BTCUSDC" in result["exchange_only_symbols"]
    assert result["db_closed"] == 1
