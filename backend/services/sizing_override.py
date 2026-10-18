"""Dynamic Position-Sizing Override Module.

Computes a bounded, scale-down-only multiplier ``override_multiplier in [0.0, 1.0]``
that sits AFTER Fractional Kelly and equity-risk sizing and BEFORE exchange
hard notional caps in :meth:`DecisionEngine._create_entry_decision`.

Design principles:
  1. **Scale-down only (monotone conservative)**: ``multiplier <= 1.0`` always.
     Never inflates position sizes above Kelly or configured baseline risk.
  2. **Decomposable & GPU-tunable**: Every factor (regime volatility, regime
     confidence, circuit-breaker state, rolling drawdown, consecutive losing
     streak, and bear-debate strength) is parameterized via ``SizingOverrideConfig``
     or environment variables so historical grid-search / GPU sweeps can tune
     weights without modifying code.
  3. **Shadow-first rollout**: ``SIZING_OVERRIDE_MODE=off|shadow|enforce``
     (default ``shadow``). In ``shadow`` mode, ``effective_multiplier`` is ``1.0``
     while ``proposed_multiplier`` and ``components`` are logged and attached to
     ``DecisionEngine.last_evaluation`` for Grafana / Supabase inspection.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SizingOverrideConfig:
    mode: str = field(default_factory=lambda: os.getenv("SIZING_OVERRIDE_MODE", "shadow").strip().lower())
    min_multiplier: float = field(default_factory=lambda: float(os.getenv("SIZING_OVERRIDE_MIN_MULT", "0.20")))
    volatile_regime_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_VOLATILE_MULT", "0.50")))
    breakout_regime_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_BREAKOUT_MULT", "0.75")))
    ranging_regime_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_RANGING_MULT", "0.60")))
    low_regime_conf_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIZING_LOW_REGIME_CONF", "0.50"))
    )
    low_regime_conf_mult: float = field(
        default_factory=lambda: float(os.getenv("SIZING_LOW_REGIME_CONF_MULT", "0.75"))
    )
    dd_tier1_pct: float = field(default_factory=lambda: float(os.getenv("SIZING_DD_TIER1_PCT", "5.0")))
    dd_tier1_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_DD_TIER1_MULT", "0.75")))
    dd_tier2_pct: float = field(default_factory=lambda: float(os.getenv("SIZING_DD_TIER2_PCT", "10.0")))
    dd_tier2_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_DD_TIER2_MULT", "0.50")))
    streak_tier1: int = field(default_factory=lambda: int(os.getenv("SIZING_STREAK_TIER1", "3")))
    streak_tier1_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_STREAK_TIER1_MULT", "0.75")))
    streak_tier2: int = field(default_factory=lambda: int(os.getenv("SIZING_STREAK_TIER2", "5")))
    streak_tier2_mult: float = field(default_factory=lambda: float(os.getenv("SIZING_STREAK_TIER2_MULT", "0.50")))
    bear_slope: float = field(default_factory=lambda: float(os.getenv("SIZING_BEAR_SLOPE", "0.50")))


@dataclass
class SizingOverrideResult:
    mode: str
    proposed_multiplier: float
    effective_multiplier: float
    original_notional: float
    adjusted_notional: float
    components: Dict[str, float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "proposed_multiplier": round(self.proposed_multiplier, 4),
            "effective_multiplier": round(self.effective_multiplier, 4),
            "original_notional": round(self.original_notional, 2),
            "adjusted_notional": round(self.adjusted_notional, 2),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "reason": self.reason,
        }


def compute_sizing_override(
    notional: float,
    *,
    regime: str = "TRENDING",
    regime_confidence: float = 1.0,
    cb_state: str = "CLOSED",
    cb_mode: Optional[str] = None,
    drawdown_pct: float = 0.0,
    losing_streak: int = 0,
    bear_strength: float = 0.0,
    config: Optional[SizingOverrideConfig] = None,
) -> SizingOverrideResult:
    """Compute a bounded scale-down multiplier and adjusted USDT notional."""
    cfg = config or SizingOverrideConfig()
    mode = (cfg.mode or "shadow").strip().lower()
    base_notional = max(0.0, float(notional))

    if mode == "off":
        return SizingOverrideResult(
            mode="off",
            proposed_multiplier=1.0,
            effective_multiplier=1.0,
            original_notional=base_notional,
            adjusted_notional=base_notional,
            components={},
            reason="sizing_override_off",
        )

    reg_upper = (regime or "TRENDING").upper()
    if reg_upper == "VOLATILE":
        regime_mult = cfg.volatile_regime_mult
    elif reg_upper == "BREAKOUT":
        regime_mult = cfg.breakout_regime_mult
    elif reg_upper == "RANGING":
        regime_mult = cfg.ranging_regime_mult
    else:
        regime_mult = 1.0

    conf_mult = cfg.low_regime_conf_mult if float(regime_confidence) < cfg.low_regime_conf_threshold else 1.0

    # OPEN -> size 0.0 is gated on REGIME_CB_MODE=enforce. A shadow/off breaker
    # must never resize a live order, even if SIZING_OVERRIDE_MODE=enforce.
    resolved_cb_mode = (cb_mode if cb_mode is not None else os.getenv("REGIME_CB_MODE", "shadow")).strip().lower()
    cb_upper = (cb_state or "CLOSED").upper()
    if resolved_cb_mode == "enforce" and cb_upper == "OPEN":
        cb_mult = 0.0
    elif resolved_cb_mode == "enforce" and cb_upper == "HALF_OPEN":
        cb_mult = 0.50
    else:
        cb_mult = 1.0

    dd = max(0.0, float(drawdown_pct or 0.0))
    if dd >= cfg.dd_tier2_pct:
        dd_mult = cfg.dd_tier2_mult
    elif dd >= cfg.dd_tier1_pct:
        dd_mult = cfg.dd_tier1_mult
    else:
        dd_mult = 1.0

    streak = max(0, int(losing_streak or 0))
    if streak >= cfg.streak_tier2:
        streak_mult = cfg.streak_tier2_mult
    elif streak >= cfg.streak_tier1:
        streak_mult = cfg.streak_tier1_mult
    else:
        streak_mult = 1.0

    b_str = max(0.0, min(1.0, float(bear_strength or 0.0)))
    bear_mult = max(cfg.min_multiplier, 1.0 - cfg.bear_slope * b_str) if b_str > 0 else 1.0

    components = {
        "regime": regime_mult,
        "regime_conf": conf_mult,
        "circuit_breaker": cb_mult,
        "drawdown": dd_mult,
        "losing_streak": streak_mult,
        "bear_debate": bear_mult,
    }

    raw_mult = 1.0
    for val in components.values():
        raw_mult *= float(val)

    if cb_mult == 0.0:
        proposed = 0.0
    else:
        proposed = max(cfg.min_multiplier, min(1.0, raw_mult))

    effective = proposed if mode == "enforce" else 1.0
    adjusted = base_notional * effective

    active_reasons = [f"{k}={v:.2f}" for k, v in components.items() if v < 0.999]
    reason = ", ".join(active_reasons) if active_reasons else "full_size"

    return SizingOverrideResult(
        mode=mode,
        proposed_multiplier=proposed,
        effective_multiplier=effective,
        original_notional=base_notional,
        adjusted_notional=adjusted,
        components=components,
        reason=reason,
    )


def query_recent_streak_and_drawdown(
    account_equity: float = 0.0,
    lookback: int = 20,
) -> tuple[int, float]:
    """Query recent closed trades in the active partition for (losing_streak, drawdown_pct).

    Fail-safe: returns ``(0, 0.0)`` on any database error so sizing never crashes.
    """
    losing_streak = 0
    drawdown_pct = 0.0
    try:
        from sqlalchemy import or_
        from backend.database.connection import SessionLocal
        from backend.database.models import Trade
        from backend.services.equity_scope import risk_broker_name
        from backend.services.pnl_accounting import RECONCILIATION_STRATEGIES
        from backend.services.trading_mode import get_trading_mode

        active_broker = risk_broker_name().lower()
        mode_val = get_trading_mode().value if hasattr(get_trading_mode(), "value") else str(get_trading_mode())

        with SessionLocal() as db:
            q = (
                db.query(Trade)
                .filter(Trade.status.in_(["closed", "exit"]), Trade.pnl.isnot(None))
                .filter(
                    or_(
                        Trade.strategy.is_(None),
                        ~Trade.strategy.in_(sorted(RECONCILIATION_STRATEGIES)),
                    )
                )
                .filter(Trade.mode == mode_val.lower())
            )
            if active_broker == "ctrader":
                q = q.filter(Trade.broker.in_(["ctrader", "ctrader:paper", "ic", "icmarkets"]))
            elif active_broker in {"binance", "binance_futures"}:
                q = q.filter(Trade.broker.in_(["binance", "binance_futures", "binanceusdm"]))

            recent = q.order_by(Trade.closed_at.desc()).limit(lookback).all()
            pnls = [float(r.pnl or 0.0) for r in recent]

        for p in pnls:
            if p < 0:
                losing_streak += 1
            elif p > 0:
                break

        if pnls and account_equity > 0:
            # Compute peak-to-current drawdown across the lookback window in chronological order
            cum = 0.0
            peak = 0.0
            for p in reversed(pnls):
                cum += p
                if cum > peak:
                    peak = cum
            dd_abs = max(0.0, peak - cum)
            drawdown_pct = (dd_abs / float(account_equity)) * 100.0
    except Exception as e:
        logger.debug("sizing_override streak/drawdown query skipped: %s", e)
    return losing_streak, round(drawdown_pct, 3)
