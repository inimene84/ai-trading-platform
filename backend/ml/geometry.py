"""Canonical live/training triple-barrier geometry (QTP Promotion Contract v1.0.0).

Training labels MUST use these locked ATR multipliers. A mismatch versus live
RiskConfig is GEOMETRY_LIVE_LOCK and is a hard REJECT.
"""

from __future__ import annotations

from typing import Any, Mapping

SPEC_VERSION = "1.0.0"
STRATEGY_ID = "qtp-lgbm-meta-v1"

# Canonical live house geometry — must match RiskConfig / compose / .env.example.
HOUSE_SL_ATR_MULT = 1.75
HOUSE_PT_ATR_MULT = 5.5
HOUSE_ATR_PERIOD = 14
HOUSE_VERTICAL_TIMEOUT_BARS = 48
HOUSE_BAR_TIMEFRAME = "1h"
HOUSE_BAR_PRIMARY_TYPE = "time"

DEFAULT_TAKER_FEE_RATE = 0.0004
DEFAULT_SLIPPAGE_RATE = 0.0002
DEFAULT_FUNDING_RATE_PER_8H = 0.0001

DEFAULT_GATES: dict[str, Any] = {
    "dsr_min": 0.95,
    "pbo_max": 0.30,
    "min_trades": 80,
    "min_net_sharpe_after_costs": 0.0,
    "min_costed_edge_bps": 0.0,
    "min_path_sharpe_p10": -0.50,
    "min_conformal_coverage": 0.70,
    "holdout_required_for_promote": True,
    "path_left_tail_hard": False,
    "conformal_coverage_hard": False,
    "minbtl_hard": False,
}

DEFAULT_LIVE_SIGNAL: dict[str, float] = {
    "p_win_min": 0.55,
    "width_max": 0.35,
}


def training_barrier_multipliers() -> tuple[float, float]:
    """Return (sl_atr_mult, pt_atr_mult) for Triple-Barrier labeling."""
    return HOUSE_SL_ATR_MULT, HOUSE_PT_ATR_MULT


def live_locked_fields(source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract the locked geometry fields the engine compares to live config."""
    src = dict(source or {})
    barrier = src.get("triple_barrier") if isinstance(src.get("triple_barrier"), Mapping) else src
    bar = src.get("bar") if isinstance(src.get("bar"), Mapping) else src
    costs = src.get("costs") if isinstance(src.get("costs"), Mapping) else {}
    gates = src.get("gates") if isinstance(src.get("gates"), Mapping) else {}
    return {
        "sl_atr_mult": float(barrier.get("sl_atr_mult", HOUSE_SL_ATR_MULT)),
        "pt_atr_mult": float(barrier.get("pt_atr_mult", HOUSE_PT_ATR_MULT)),
        "atr_period": int(barrier.get("atr_period", HOUSE_ATR_PERIOD)),
        "vertical_timeout_bars": int(barrier.get("vertical_timeout_bars", HOUSE_VERTICAL_TIMEOUT_BARS)),
        "timeframe": str(bar.get("timeframe", HOUSE_BAR_TIMEFRAME)),
        "primary_type": str(bar.get("primary_type", HOUSE_BAR_PRIMARY_TYPE)),
        "costs": dict(costs),
        "gates": dict(gates) if gates else dict(DEFAULT_GATES),
    }


def house_geometry(*, strategy_id: str = STRATEGY_ID, include_hashes_block: bool = False) -> dict[str, Any]:
    """Frozen geometry.json payload (hashes block optional / filled by the writer)."""
    payload: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "strategy_id": strategy_id,
        "triple_barrier": {
            "sl_atr_mult": HOUSE_SL_ATR_MULT,
            "pt_atr_mult": HOUSE_PT_ATR_MULT,
            "atr_period": HOUSE_ATR_PERIOD,
            "vertical_timeout_bars": HOUSE_VERTICAL_TIMEOUT_BARS,
        },
        "bar": {
            "timeframe": HOUSE_BAR_TIMEFRAME,
            "primary_type": HOUSE_BAR_PRIMARY_TYPE,
        },
        "costs": {
            "taker_fee_rate": DEFAULT_TAKER_FEE_RATE,
            "slippage_rate": DEFAULT_SLIPPAGE_RATE,
            "funding_rate_per_8h": DEFAULT_FUNDING_RATE_PER_8H,
        },
        "gates": dict(DEFAULT_GATES),
        "live": dict(DEFAULT_LIVE_SIGNAL),
        "validation": {
            "inner": "PurgedKFold",
            "outer": "CombinatorialPurgedCV",
            "pbo_n_splits": 16,
            "library": "purgedcv",
            "library_pin": "eslazarev/purged-cross-validation",
            "purge_horizon": HOUSE_VERTICAL_TIMEOUT_BARS,
            "embargo_bars": max(HOUSE_VERTICAL_TIMEOUT_BARS, 200),
        },
    }
    if include_hashes_block:
        payload["hashes"] = {}
    return payload


def locked_fields_match(left: Mapping[str, Any], right: Mapping[str, Any], *, abs_tol: float = 1e-9) -> bool:
    """True when locked ATR/bar/cost fields agree."""
    l_lock = live_locked_fields(left)
    r_lock = live_locked_fields(right)
    float_keys = ("sl_atr_mult", "pt_atr_mult")
    int_keys = ("atr_period", "vertical_timeout_bars")
    str_keys = ("timeframe", "primary_type")
    for key in float_keys:
        if abs(float(l_lock[key]) - float(r_lock[key])) > abs_tol:
            return False
    for key in int_keys:
        if int(l_lock[key]) != int(r_lock[key]):
            return False
    for key in str_keys:
        if str(l_lock[key]) != str(r_lock[key]):
            return False
    l_costs = l_lock.get("costs") or {}
    r_costs = r_lock.get("costs") or {}
    for key in ("taker_fee_rate", "slippage_rate"):
        if key not in l_costs or key not in r_costs:
            continue
        if abs(float(l_costs[key]) - float(r_costs[key])) > abs_tol:
            return False
    return True


def matches_house_lock(source: Mapping[str, Any], *, abs_tol: float = 1e-9) -> bool:
    """True when source locked fields equal the canonical 1.75 / 5.5 house lock."""
    return locked_fields_match(source, house_geometry(), abs_tol=abs_tol)
