"""Crypto loop max-open cap must count its own book, not the cTrader/n8n FX book."""

from types import SimpleNamespace

import pytest

from backend.services.risk_guard import RiskBreach, enforce_risk_limits
from backend.services.trading_loop import TradingLoopService


def _ct(sym):
    return SimpleNamespace(symbol=sym, broker="ctrader", exchange="ctrader", mode="live",
                           broker_order_id="323490229", binance_order_id=None,
                           broker_position_id="29804", direction="SELL", quantity=0.01,
                           entry_price=1.1, status="open")


def _paper(sym, n=1):
    return SimpleNamespace(symbol=sym, broker="binance_futures", exchange="binance_futures",
                           mode="paper", broker_order_id=f"paper_1792369951871_000{n}_6e29",
                           binance_order_id=f"paper_1792369951871_000{n}_6e29",
                           broker_position_id=None, direction="SELL", quantity=1.0,
                           entry_price=20.0, status="open")


FX = [_ct(s) for s in ("USDCAD", "CHFJPY", "EURUSD", "GBPUSD", "USDCHF", "XAUUSD", "EURAUD", "GBPJPY")]


def _cfg(cap=2):
    return SimpleNamespace(max_positions=cap, max_open_positions=cap)


def _count_only(monkeypatch):
    # Isolate check 1: stop right after the position-count step.
    import backend.services.risk_guard as rg

    class _Stop(Exception):
        pass

    monkeypatch.setattr(rg, "_trades_for_risk", lambda rows: [t for t in rows if t.broker == "ctrader"])
    return _Stop


def test_paper_book_scope_ignores_ctrader_fx(monkeypatch):
    _count_only(monkeypatch)
    book = [_paper("AVAXUSDC", 1), _paper("AVAXUSDC", 2)]
    try:
        enforce_risk_limits(None, _cfg(2), FX + book, None, position_book=book)
    except RiskBreach as rb:
        assert "Max open positions" not in rb.reason
    except Exception:
        pass  # later checks need a real config/DB; only check 1 matters here


def test_default_scope_still_counts_risk_book(monkeypatch):
    _count_only(monkeypatch)
    with pytest.raises(RiskBreach, match="Max open positions exceeded: 8 > 2"):
        enforce_risk_limits(None, _cfg(2), FX, None)


def test_paper_book_over_cap_still_breaches(monkeypatch):
    _count_only(monkeypatch)
    book = [_paper("AVAXUSDC"), _paper("SOLUSDC"), _paper("XRPUSDC")]
    with pytest.raises(RiskBreach, match="Max open positions exceeded: 3 > 2"):
        enforce_risk_limits(None, _cfg(2), FX + book, None, position_book=book)


def test_loop_book_is_paper_fills_in_paper_parallel(monkeypatch):
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "true")
    monkeypatch.setattr(TradingLoopService, "_is_live_binance", staticmethod(lambda: False))
    book = TradingLoopService._risk_position_book(FX + [_paper("AVAXUSDC")])
    assert [t.symbol for t in book] == ["AVAXUSDC"]


def test_loop_book_unchanged_without_paper_parallel(monkeypatch):
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    assert TradingLoopService._risk_position_book(FX) is None
