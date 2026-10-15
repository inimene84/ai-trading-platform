"""Stuck identical Jesse ML uncertainty vetoes fail-open after N hits."""

from backend.services.ml_uncertainty_guard import StuckUncertaintyTracker


def test_identical_margin_is_stuck_after_threshold(monkeypatch):
    monkeypatch.setenv("JESSE_ML_STUCK_VETO_HITS", "5")
    tracker = StuckUncertaintyTracker()
    for _ in range(4):
        assert tracker.observe(
            "BNBUSDC",
            margin=0.064,
            latest_close=1123.4,
            reason="uncertainty margin 0.064 < 0.08",
        ) is False
    assert tracker.observe(
        "BNBUSDC",
        margin=0.064,
        latest_close=1123.4,
        reason="uncertainty margin 0.064 < 0.08",
    ) is True
    # Next identical hit vetoes again — a frozen model must not trade every cycle.
    assert tracker.observe(
        "BNBUSDC",
        margin=0.064,
        latest_close=1123.4,
        reason="uncertainty margin 0.064 < 0.08",
    ) is False


def test_changing_close_resets_the_counter(monkeypatch):
    monkeypatch.setenv("JESSE_ML_STUCK_VETO_HITS", "3")
    tracker = StuckUncertaintyTracker()
    assert tracker.observe("BNBUSDC", margin=0.064, latest_close=100.0, reason="x") is False
    assert tracker.observe("BNBUSDC", margin=0.064, latest_close=100.0, reason="x") is False
    assert tracker.observe("BNBUSDC", margin=0.064, latest_close=101.0, reason="x") is False
    assert tracker.hits("BNBUSDC") == 1
