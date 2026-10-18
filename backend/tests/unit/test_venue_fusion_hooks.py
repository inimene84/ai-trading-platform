"""Candidate-engine hooks for venue router, cutoff, and funding stay shadow-safe."""

from unittest.mock import patch

from backend.services.funding_filter import evaluate_funding_filter
from backend.services.signal_candidate_engine import SignalCandidateEngine
from backend.services.venue_router import venue_for_symbol


def test_entry_cutoff_shadow_does_not_skip(monkeypatch):
    monkeypatch.setenv("CTRADER_ENTRY_CUTOFF_MODE", "shadow")
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.evaluate_entry_cutoff",
        return_value={"mode": "shadow", "would_block": True, "blocked": False, "reason": "shadow"},
    ):
        assert engine._ctrader_entry_cutoff("EURUSD") is False


def test_entry_cutoff_enforce_skips(monkeypatch):
    monkeypatch.setenv("CTRADER_ENTRY_CUTOFF_MODE", "enforce")
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.evaluate_entry_cutoff",
        return_value={"mode": "enforce", "would_block": True, "blocked": True, "reason": "cutoff"},
    ):
        assert engine._ctrader_entry_cutoff("EURUSD") is True


def test_funding_enforce_vetoes_but_shadow_allows(monkeypatch):
    monkeypatch.setenv("FUNDING_FILTER_MODE", "enforce")
    decision = evaluate_funding_filter(direction="BUY", funding_rate=0.01, threshold=0.0005)
    assert decision["action"] == "veto"
    monkeypatch.setenv("FUNDING_FILTER_MODE", "shadow")
    decision = evaluate_funding_filter(direction="BUY", funding_rate=0.01, threshold=0.0005)
    assert decision["action"] == "allow"


def test_venue_router_used_for_scan_symbols():
    assert venue_for_symbol("BTCUSDC") == "binance_futures"
    assert venue_for_symbol("EURUSD") == "ctrader"
    assert venue_for_symbol("XAUUSD") == "ctrader"
