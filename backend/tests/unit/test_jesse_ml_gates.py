"""Unit tests for Jesse ML promotion gates and empirical Fractional Kelly."""

from backend.services.jesse_ml_gates import (
    PBO_GATE,
    STRATEGY_PAYOFF_RATIO,
    STRATEGY_PT_ATR,
    STRATEGY_SL_ATR,
    annotate_ml_prediction,
    calculate_fractional_kelly,
    clip_kelly_for_thin_book,
    empirical_payoff_ratio,
    evaluate_promotion,
    geometry_matches_live,
    payoff_ratio_from_geometry,
)


def test_geometry_matches_live_strategy():
    assert geometry_matches_live(STRATEGY_PT_ATR, STRATEGY_SL_ATR) is True
    assert geometry_matches_live(4.0, 2.0) is False
    assert geometry_matches_live(5.5, 1.75) is True


def test_payoff_ratio_from_aligned_geometry():
    b = payoff_ratio_from_geometry(5.5, 1.75)
    assert abs(b - STRATEGY_PAYOFF_RATIO) < 1e-9
    assert 3.1 < b < 3.2


def test_evaluate_promotion_rejects_wrong_barrier_geometry():
    decision = evaluate_promotion(
        {"deflated_sharpe_ratio": 0.99, "prob_backtest_overfitting": 0.10},
        pt_mult=4.0,
        sl_mult=2.0,
    )
    assert decision.ok is False
    assert "untaken trade" in decision.reason


def test_evaluate_promotion_rejects_failing_pbo():
    decision = evaluate_promotion(
        {"deflated_sharpe_ratio": 0.99, "prob_backtest_overfitting": 0.344},
        pt_mult=5.5,
        sl_mult=1.75,
        allow_overfit=False,
    )
    assert decision.ok is False
    assert "PBO" in decision.reason
    assert PBO_GATE == 0.30


def test_evaluate_promotion_rejects_dsr_one_from_too_few_trials_if_below_gate():
    decision = evaluate_promotion(
        {"deflated_sharpe_ratio": 0.40, "prob_backtest_overfitting": 0.10},
        pt_mult=5.5,
        sl_mult=1.75,
    )
    assert decision.ok is False
    assert "DSR" in decision.reason


def test_evaluate_promotion_rejects_weak_but_not_fully_collapsed_recall():
    decision = evaluate_promotion(
        {
            "deflated_sharpe_ratio": 0.99,
            "prob_backtest_overfitting": 0.10,
            "bullish_recall": 0.08,
            "bearish_recall": 0.70,
        },
        pt_mult=5.5,
        sl_mult=1.75,
    )
    assert decision.ok is False
    assert "bullish recall" in decision.reason

    decision = evaluate_promotion(
        {
            "deflated_sharpe_ratio": 0.99,
            "prob_backtest_overfitting": 0.02,
            "bullish_recall": 0.0,
            "bearish_recall": 0.999,
        },
        pt_mult=5.5,
        sl_mult=1.75,
    )
    assert decision.ok is False
    assert "collapsed classifier" in decision.reason
    decision = evaluate_promotion(
        {"deflated_sharpe_ratio": 0.97, "prob_backtest_overfitting": 0.22},
        pt_mult=5.5,
        sl_mult=1.75,
    )
    assert decision.ok is True
    assert "PASS" in decision.reason


def test_evaluate_promotion_missing_metrics_fail_closed():
    decision = evaluate_promotion({}, pt_mult=5.5, sl_mult=1.75, allow_overfit=False)
    assert decision.ok is False
    assert "missing DSR or PBO" in decision.reason


def test_empirical_payoff_uses_geometry_until_30_closed_trades():
    fallback = empirical_payoff_ratio(avg_win=80.0, avg_loss_abs=20.0, closed_count=12)
    assert abs(fallback - STRATEGY_PAYOFF_RATIO) < 1e-9
    realized = empirical_payoff_ratio(avg_win=80.0, avg_loss_abs=20.0, closed_count=40)
    assert abs(realized - 4.0) < 1e-9


def test_fractional_kelly_uses_geometry_payoff_not_hardcoded_two():
    at_two = calculate_fractional_kelly(0.55, payoff_ratio=2.0)
    at_geometry = calculate_fractional_kelly(0.55, payoff_ratio=STRATEGY_PAYOFF_RATIO)
    assert at_geometry["payoff_ratio"] > 3.0
    assert at_geometry["fractional_kelly"] >= at_two["fractional_kelly"]
    assert 0.20 <= at_geometry["size_multiplier"] <= 2.0


def test_clip_kelly_for_thin_book():
    clipped, was_clipped = clip_kelly_for_thin_book(1.8, closed_count=5)
    assert was_clipped is True
    assert clipped == 1.0
    unclipped, was_clipped = clip_kelly_for_thin_book(1.8, closed_count=40)
    assert was_clipped is False
    assert unclipped == 1.8


def test_annotate_ml_prediction_fail_closes_overfit_model():
    payload = {
        "status": "success",
        "signal": "SELL",
        "metrics": {
            "deflated_sharpe_ratio": 1.0,
            "prob_backtest_overfitting": 0.636,
            "pt_mult": 4.0,
            "sl_mult": 2.0,
        },
    }
    out = annotate_ml_prediction(payload)
    assert out["status"] == "error"
    assert out["promotion_ok"] is False
    assert out["signal"] == "NEUTRAL"
    assert "promotion gate" in out["error"]
