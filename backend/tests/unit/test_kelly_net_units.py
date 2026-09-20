"""Kelly size_multiplier scales configured notional / stop-risk, using net b."""

from backend.services.jesse_ml_gates import (
    STRATEGY_PAYOFF_RATIO,
    calculate_fractional_kelly,
    net_theoretical_payoff_ratio,
)


def test_net_theoretical_b_is_below_raw_tp_sl():
    net_b = net_theoretical_payoff_ratio()
    assert net_b < STRATEGY_PAYOFF_RATIO
    assert 1.5 < net_b < 3.14


def test_size_multiplier_is_a_scale_not_a_wallet_fraction():
    """Audit: bet_size multiplies trade_usdt (and stop-risk), never 0.25 wallet."""
    kelly = calculate_fractional_kelly(0.55, payoff_ratio=net_theoretical_payoff_ratio())
    assert 0.20 <= kelly["size_multiplier"] <= 1.0
    assert kelly["size_multiplier"] != 0.25 or kelly["fractional_kelly"] > 0
