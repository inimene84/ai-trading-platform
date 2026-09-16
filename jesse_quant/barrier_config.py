#!/usr/bin/env python3
"""
Single source of truth for the trade geometry and cost model shared by:
  - strategies/QuantumAIStrategy and strategies/QuantumMLStrategy (Jesse backtests)
  - triple_barrier.py / train_ml.py (triple-barrier label geometry)
  - predict_server.py (Kelly payoff ratio fallback, gate reporting)

The defaults mirror the *deployed* live risk configuration of ai-trading-platform-v3
(SL_ATR_MULT / TP_ATR_MULT / TRAIL_ACTIVATION_ATR / TRAIL_ATR_MULT in its .env, also
exposed via GET /api/jesse/status). Every value can be overridden through environment
variables so research runs can deviate without editing code, but the *default* is the
production geometry so that models are trained on the trade the strategy actually takes.
"""

import os
from typing import Any, Dict


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Deployed trade geometry (ATR multiples) -----------------------------------------
SL_ATR_MULT: float = _env_float("SL_ATR_MULT", 1.75)
TP_ATR_MULT: float = _env_float("TP_ATR_MULT", 5.5)
TRAIL_ACTIVATION_ATR: float = _env_float("TRAIL_ACTIVATION_ATR", 2.2)
TRAIL_ATR_MULT: float = _env_float("TRAIL_ATR_MULT", 1.6)
ATR_PERIOD: int = _env_int("ATR_PERIOD", 14)

# Vertical barrier for triple-barrier labelling (bars of the training timeframe).
# 48 x 1h = 2 days gives a 5.5 ATR target a realistic chance to resolve.
MAX_HOLDING_BARS: int = _env_int("BARRIER_MAX_HOLDING_BARS", 48)

# --- Cost model ----------------------------------------------------------------------
# Binance USDT-M perpetual taker fee (per side).
FEE_RATE: float = _env_float("BACKTEST_FEE_RATE", 0.0006)
# Base execution slippage (fraction of price), scaled by ATR regime in the strategies.
BASE_SLIPPAGE: float = _env_float("BASE_SLIPPAGE", 0.0003)
MAX_SLIPPAGE: float = _env_float("MAX_SLIPPAGE", 0.0015)
# Perpetual funding: default 0.01% per 8h interval (Binance baseline rate), paid by longs
# when positive. Sign handling: a flat positive rate is a conservative assumption for
# long-biased strategies; shorts *receive* funding under a positive rate.
FUNDING_RATE_8H: float = _env_float("FUNDING_RATE_8H", 0.0001)
FUNDING_INTERVAL_HOURS: int = _env_int("FUNDING_INTERVAL_HOURS", 8)
# Zero-cost sanity toggle (audit "edge exists only without costs" test).
ZERO_COST: bool = _env_bool("JESSE_ZERO_COST", False)

# --- Statistical gates -----------------------------------------------------------------
DSR_MIN: float = _env_float("ML_GATE_DSR_MIN", 0.95)
PBO_MAX: float = _env_float("ML_GATE_PBO_MAX", 0.30)


def theoretical_payoff_ratio(tp_mult: float = TP_ATR_MULT, sl_mult: float = SL_ATR_MULT) -> float:
    """Payoff ratio b implied by the barrier geometry (TP distance / SL distance)."""
    if sl_mult <= 0:
        return 1.0
    return float(tp_mult / sl_mult)


def breakeven_win_probability(payoff_ratio: float) -> float:
    """Win probability at which expected value of a b:1 bet is zero: p = 1 / (1 + b)."""
    return float(1.0 / (1.0 + max(payoff_ratio, 1e-9)))


def geometry_dict(
    sl_mult: float = SL_ATR_MULT,
    tp_mult: float = TP_ATR_MULT,
    max_holding_bars: int = MAX_HOLDING_BARS,
    atr_period: int = ATR_PERIOD,
) -> Dict[str, Any]:
    """Serialisable description of the geometry, stored in model artifacts."""
    return {
        "sl_atr_mult": float(sl_mult),
        "tp_atr_mult": float(tp_mult),
        "trail_activation_atr": float(TRAIL_ACTIVATION_ATR),
        "trail_atr_mult": float(TRAIL_ATR_MULT),
        "atr_period": int(atr_period),
        "max_holding_bars": int(max_holding_bars),
        "theoretical_payoff_ratio": round(theoretical_payoff_ratio(tp_mult, sl_mult), 4),
        "breakeven_win_probability": round(breakeven_win_probability(theoretical_payoff_ratio(tp_mult, sl_mult)), 4),
        "source": "barrier_config.py (defaults mirror live SL_ATR_MULT/TP_ATR_MULT)",
    }


def cost_model_dict() -> Dict[str, Any]:
    return {
        "fee_rate_per_side": FEE_RATE,
        "base_slippage": BASE_SLIPPAGE,
        "max_slippage": MAX_SLIPPAGE,
        "funding_rate_8h": FUNDING_RATE_8H,
        "funding_interval_hours": FUNDING_INTERVAL_HOURS,
        "zero_cost": ZERO_COST,
    }


def round_trip_cost_pct(holding_bars: int, bar_hours: float = 1.0, zero_cost: bool = ZERO_COST) -> float:
    """
    Approximate all-in cost (percent of notional) of a round trip held for `holding_bars`:
    two taker fees, two slippage legs and the expected number of funding intervals.
    """
    if zero_cost:
        return 0.0
    n_funding = int((holding_bars * bar_hours) // FUNDING_INTERVAL_HOURS)
    frac = 2.0 * FEE_RATE + 2.0 * BASE_SLIPPAGE + n_funding * FUNDING_RATE_8H
    return float(frac * 100.0)


if __name__ == "__main__":
    import json

    print(json.dumps({"geometry": geometry_dict(), "cost_model": cost_model_dict()}, indent=2))
