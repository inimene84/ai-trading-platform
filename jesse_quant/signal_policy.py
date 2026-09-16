#!/usr/bin/env python3
"""
Shared decision rule that turns calibrated class probabilities into a trade signal.

Used by train_ml.py (so holdout metrics reflect the model *as served*) and by
predict_server.py (live inference). Class ids: 0 = neutral, 1 = bullish (a long with the
deployed SL/TP geometry hits take-profit first), 2 = bearish (a short does).

The rule is geometry-aware: with payoff ratio b (TP distance / SL distance) a directional
bet only has positive expectancy when p > 1 / (1 + b). We therefore require a minimum
expected value in R-multiples rather than a fixed 0.45 probability threshold, which was
only meaningful for the old 2:1 geometry.
"""

from typing import Any, Dict

import numpy as np

from barrier_config import breakeven_win_probability, theoretical_payoff_ratio

DEFAULT_MIN_EV_R: float = 0.10     # minimum conservative expectancy, in units of the stop distance
DEFAULT_DOMINANCE: float = 1.2     # winning side must beat the opposing side by this factor


def expected_value_r(p_win: float, payoff_ratio: float) -> float:
    """Conservative expectancy in R: all non-take-profit outcomes are treated as a full stop."""
    return float(p_win * payoff_ratio - (1.0 - p_win))


def decide_signal(
    p_neutral: float,
    p_bullish: float,
    p_bearish: float,
    payoff_ratio: float = theoretical_payoff_ratio(),
    min_ev_r: float = DEFAULT_MIN_EV_R,
    dominance: float = DEFAULT_DOMINANCE,
    min_probability: float = 0.0,
) -> Dict[str, Any]:
    """
    Returns a dict with keys: signal ("BUY"/"SELL"/"NEUTRAL"), confidence (probability of
    the chosen side), expected_value_r, breakeven_probability and payoff_ratio.
    """
    ev_long = expected_value_r(p_bullish, payoff_ratio)
    ev_short = expected_value_r(p_bearish, payoff_ratio)
    p_be = breakeven_win_probability(payoff_ratio)

    signal = "NEUTRAL"
    conf = p_neutral
    ev = 0.0
    if (
        ev_long >= min_ev_r
        and p_bullish >= min_probability
        and p_bullish > p_bearish * dominance
        and ev_long >= ev_short
    ):
        signal, conf, ev = "BUY", p_bullish, ev_long
    elif (
        ev_short >= min_ev_r
        and p_bearish >= min_probability
        and p_bearish > p_bullish * dominance
    ):
        signal, conf, ev = "SELL", p_bearish, ev_short

    return {
        "signal": signal,
        "confidence": float(conf),
        "expected_value_r": round(float(ev), 4),
        "ev_long_r": round(float(ev_long), 4),
        "ev_short_r": round(float(ev_short), 4),
        "breakeven_probability": round(float(p_be), 4),
        "payoff_ratio": round(float(payoff_ratio), 4),
        "min_ev_r": float(min_ev_r),
    }


def decide_signals_vectorized(
    probs: np.ndarray,
    payoff_ratio: float = theoretical_payoff_ratio(),
    min_ev_r: float = DEFAULT_MIN_EV_R,
    dominance: float = DEFAULT_DOMINANCE,
    min_probability: float = 0.0,
) -> np.ndarray:
    """
    Vectorised twin of decide_signal for an (n, k) probability matrix with columns ordered
    [neutral, bullish, bearish] (bearish optional). Returns class ids 0 / 1 (BUY) / 2 (SELL).
    """
    probs = np.asarray(probs, dtype=float)
    p_bull = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(probs))
    p_bear = probs[:, 2] if probs.shape[1] > 2 else np.zeros(len(probs))
    ev_long = p_bull * payoff_ratio - (1.0 - p_bull)
    ev_short = p_bear * payoff_ratio - (1.0 - p_bear)
    buy = (ev_long >= min_ev_r) & (p_bull >= min_probability) & (p_bull > p_bear * dominance) & (ev_long >= ev_short)
    sell = (~buy) & (ev_short >= min_ev_r) & (p_bear >= min_probability) & (p_bear > p_bull * dominance)
    out = np.zeros(len(probs), dtype=int)
    out[buy] = 1
    out[sell] = 2
    return out
