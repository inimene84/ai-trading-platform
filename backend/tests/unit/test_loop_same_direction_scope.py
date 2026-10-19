"""Same-direction cap (MAX_SAME_DIRECTION_POSITIONS) counts the loop's own book."""

from types import SimpleNamespace

from backend.services.trading_loop import TradingLoopService


def _ct(sym, d):
    return SimpleNamespace(symbol=sym, direction=d, broker="ctrader", exchange="ctrader",
                           mode="live", broker_order_id="323490229", binance_order_id=None,
                           broker_position_id="298", quantity=0.01, entry_price=1.1)


def _paper(sym, d, n=1):
    oid = f"paper_1792369951871_000{n}_6e29"
    return SimpleNamespace(symbol=sym, direction=d, broker="binance_futures",
                           exchange="binance_futures", mode="paper", broker_order_id=oid,
                           binance_order_id=oid, broker_position_id=None, quantity=1.0,
                           entry_price=20.0)


FX = [_ct("USDCHF", "BUY"), _ct("XAUUSD", "BUY"), _ct("EURAUD", "BUY"),
      _ct("USDCAD", "SELL"), _ct("EURUSD", "SELL")]


def _paper_parallel(monkeypatch, on=True):
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "true" if on else "false")
    monkeypatch.setattr(TradingLoopService, "_is_live_binance", staticmethod(lambda: False))


def test_paper_parallel_ignores_ctrader_buys(monkeypatch):
    _paper_parallel(monkeypatch)
    rows = FX + [_paper("AVAXUSDC", "SELL")]
    syms, label = TradingLoopService._same_direction_symbols(rows, FX, "BUY")
    assert syms == set()
    assert label == "loop paper book"


def test_paper_parallel_counts_own_paper_legs(monkeypatch):
    _paper_parallel(monkeypatch)
    own = [_paper("ETHUSDC", "BUY", 1), _paper("SOLUSDC", "BUY", 2), _paper("XRPUSDC", "BUY", 3),
           _paper("SOLUSDC", "BUY", 4)]
    syms, _ = TradingLoopService._same_direction_symbols(FX + own, FX, "BUY")
    assert syms == {"ETHUSDC", "SOLUSDC", "XRPUSDC"}  # distinct; 3 >= cap 3 still blocks


def test_without_paper_parallel_uses_risk_book_unchanged(monkeypatch):
    _paper_parallel(monkeypatch, on=False)
    syms, label = TradingLoopService._same_direction_symbols(FX, FX, "BUY")
    assert syms == {"USDCHF", "XAUUSD", "EURAUD"}
    assert label == "risk book"
