"""Execution-time MAX_POSITIONS check counts the loop's own book in paper-parallel."""

from backend.services.trading_loop import TradingLoopService


def _paper_parallel(monkeypatch, on=True):
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "true" if on else "false")
    monkeypatch.setattr(TradingLoopService, "_is_live_binance", staticmethod(lambda: False))


def test_paper_parallel_uses_own_book(monkeypatch):
    _paper_parallel(monkeypatch)
    # 8 cTrader FX symbols + 1 paper crypto symbol: only the 1 counts.
    assert TradingLoopService._execution_total_open(9, 1) == 1


def test_paper_parallel_own_book_at_cap_still_blocks(monkeypatch):
    _paper_parallel(monkeypatch)
    max_positions = 2
    assert TradingLoopService._execution_total_open(10, 2) >= max_positions


def test_without_paper_parallel_counts_all_books(monkeypatch):
    _paper_parallel(monkeypatch, on=False)
    assert TradingLoopService._execution_total_open(9, 1) == 9


def test_live_binance_counts_all_books(monkeypatch):
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "true")
    monkeypatch.setattr(TradingLoopService, "_is_live_binance", staticmethod(lambda: True))
    assert TradingLoopService._execution_total_open(9, 1) == 9
