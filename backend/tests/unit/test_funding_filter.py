"""Funding filter: default shadow never vetoes; enforce can veto/size-down."""

from backend.services.funding_filter import (
    apply_size_mult,
    evaluate_funding_filter,
    funding_filter_mode,
    funding_opposes_side,
)


def test_default_mode_is_shadow(monkeypatch):
    monkeypatch.delenv("FUNDING_FILTER_MODE", raising=False)
    assert funding_filter_mode() == "shadow"


def test_long_pays_positive_funding():
    assert funding_opposes_side("BUY", 0.001) is True
    assert funding_opposes_side("SELL", 0.001) is False
    assert funding_opposes_side("SELL", -0.001) is True
    assert funding_opposes_side("BUY", -0.001) is False


def test_shadow_strong_opposing_funding_does_not_veto():
    decision = evaluate_funding_filter(
        direction="BUY",
        funding_rate=0.002,
        threshold=0.0005,
        mode="shadow",
    )
    assert decision["action"] == "allow"
    assert decision["would_action"] == "veto"
    assert decision["opposes"] is True
    assert "shadow" in decision["reason"]


def test_enforce_vetoes_strong_opposing_funding():
    decision = evaluate_funding_filter(
        direction="SELL",
        funding_rate=-0.001,
        threshold=0.0005,
        mode="enforce",
    )
    assert decision["action"] == "veto"
    assert decision["would_action"] == "veto"


def test_enforce_sizes_down_mild_opposing_funding():
    decision = evaluate_funding_filter(
        direction="BUY",
        funding_rate=0.0002,
        threshold=0.0005,
        size_mult=0.5,
        mode="enforce",
    )
    assert decision["action"] == "size_down"
    assert decision["size_mult"] == 0.5


def test_aligned_funding_always_allows():
    decision = evaluate_funding_filter(
        direction="BUY",
        funding_rate=-0.002,
        threshold=0.0001,
        mode="enforce",
    )
    assert decision["action"] == "allow"
    assert decision["opposes"] is False


def test_missing_funding_is_fail_open():
    decision = evaluate_funding_filter(direction="BUY", funding_rate=None, mode="enforce")
    assert decision["action"] == "allow"
    assert "fail_open" in decision["reason"]


def test_off_mode_is_noop():
    decision = evaluate_funding_filter(
        direction="BUY", funding_rate=0.01, mode="off",
    )
    assert decision["action"] == "allow"
    assert decision["would_action"] == "allow"


def test_apply_size_mult_scales_qty_and_lots():
    sized = apply_size_mult({"quantity": 2.0, "lots": 0.10, "risk_usd": 20.0}, 0.5)
    assert sized["quantity"] == 1.0
    assert sized["lots"] == 0.05
    assert sized["risk_usd"] == 10.0
    assert sized["funding_size_mult"] == 0.5
