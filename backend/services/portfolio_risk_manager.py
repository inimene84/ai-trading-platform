"""Deterministic PortfolioRiskManager — live/paper pre-order gateway.

Workstream 3 of the Antigravity cleanup. Five hard controls, no LLM, no
allocator FSM, no money-mode flip. Call :func:`evaluate_entry` before any
new order leaves the backend. Flatten / reduce-only exits are never blocked
by these gates.

This module does not set TRADING_MODE, PAPER_TRADING, or DRY_RUN_ALL.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from backend.services.multi_asset_bars import classify_symbol

logger = logging.getLogger(__name__)

# Inverse-vol: 20 calendar days of 1h bars, 12% target, 12.5% single-name cap.
INV_VOL_LOOKBACK_BARS = 480
INV_VOL_MIN_RETURNS = 20
HOURS_PER_YEAR = 24 * 365
_TF_BARS_PER_HOUR = {
    "1m": 60.0, "m1": 60.0,
    "5m": 12.0, "m5": 12.0,
    "15m": 4.0, "m15": 4.0,
    "30m": 2.0, "m30": 2.0,
    "1h": 1.0, "h1": 1.0,
    "4h": 0.25, "h4": 0.25,
    "12h": 1.0 / 12.0, "h12": 1.0 / 12.0,
    "1d": 1.0 / 24.0, "d1": 1.0 / 24.0,
}
TARGET_PORTFOLIO_VOL = 0.12
MAX_SYMBOL_WEIGHT = 0.125

# Correlated crypto cluster (ρ > 0.75 treated as one book).
MAX_CONCURRENT_CRYPTO_POSITIONS = 3
MAX_CRYPTO_CLUSTER_GROSS_WEIGHT = 0.45

# Two-stage rolling drawdown breaker (fraction of peak, not percent points).
DRAWDOWN_SOFT_PCT = 0.08
DRAWDOWN_HARD_PCT = 0.12
DRAWDOWN_SOFT_SIZE_MULT = 0.50

# Long perp funding hard-block (8h rate).
FUNDING_LONG_BLOCK = 0.0003

# Micro-size forward gate for Recovery 48/192 + meta-labeler (ARTIFACT_4).
FORWARD_GATE_NOTIONAL_USD = 150.0

SHADOW_ONLY = "SHADOW_ONLY"
SIZE_HALVED = "SIZE_HALVED"
NORMAL = "NORMAL"

# Strategies that may log in shadow but must not submit live/paper entries.
_SHADOW_STRATEGY_MARKERS = (
    "recovery_48",
    "recovery48",
    "recovery-48",
    "recovery 48",
    "recovery_192",
    "recovery192",
    "recovery-192",
    "recovery 192",
    "meta_labeler",
    "meta-labeler",
    "metalabeler",
    "jesse_ml",
    "quantumml",
    "quantum_ml",
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def target_portfolio_vol() -> float:
    return max(1e-6, _env_float("PRM_TARGET_PORTFOLIO_VOL", TARGET_PORTFOLIO_VOL))


def max_symbol_weight() -> float:
    return min(1.0, max(0.0, _env_float("PRM_MAX_SYMBOL_WEIGHT", MAX_SYMBOL_WEIGHT)))


def max_concurrent_crypto() -> int:
    return max(1, _env_int("PRM_MAX_CONCURRENT_CRYPTO", MAX_CONCURRENT_CRYPTO_POSITIONS))


def max_crypto_cluster_gross() -> float:
    return min(1.0, max(0.0, _env_float("PRM_MAX_CRYPTO_CLUSTER_GROSS", MAX_CRYPTO_CLUSTER_GROSS_WEIGHT)))


def drawdown_soft_pct() -> float:
    return max(0.0, _env_float("PRM_DD_SOFT_PCT", DRAWDOWN_SOFT_PCT))


def drawdown_hard_pct() -> float:
    return max(0.0, _env_float("PRM_DD_HARD_PCT", DRAWDOWN_HARD_PCT))


def funding_long_block() -> float:
    return max(0.0, _env_float("PRM_FUNDING_LONG_BLOCK", FUNDING_LONG_BLOCK))


def forward_gate_notional_usd() -> float:
    return max(0.0, _env_float("PRM_FORWARD_GATE_NOTIONAL_USD", FORWARD_GATE_NOTIONAL_USD))


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    notional: float


@dataclass
class PortfolioRiskDecision:
    allow: bool
    mode: str
    sized_notional: float
    size_mult: float
    prefer_post_only: bool
    reasons: list[str] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow": self.allow,
            "mode": self.mode,
            "sized_notional": round(float(self.sized_notional), 6),
            "size_mult": round(float(self.size_mult), 4),
            "prefer_post_only": self.prefer_post_only,
            "reasons": list(self.reasons),
            "gates": dict(self.gates),
        }


def is_crypto_symbol(symbol: str) -> bool:
    return classify_symbol(symbol or "") == "crypto"


def normalize_side(side: str) -> str:
    raw = (side or "").strip().upper()
    if raw in {"BUY", "LONG"}:
        return "BUY"
    if raw in {"SELL", "SHORT"}:
        return "SELL"
    if raw in {"CLOSE_LONG", "CLOSE_SHORT", "CLOSE", "FLATTEN"}:
        return raw
    return raw


def is_long_side(side: str) -> bool:
    return normalize_side(side) == "BUY"


def closes_from_bars(bars: Optional[Sequence[Any]]) -> list[float]:
    out: list[float] = []
    for bar in bars or []:
        if isinstance(bar, Mapping):
            raw = bar.get("close", bar.get("c"))
        else:
            raw = bar
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.append(value)
    return out


def periods_per_year_for_timeframe(timeframe: str) -> float:
    tf = (timeframe or "1h").strip().lower()
    return HOURS_PER_YEAR * _TF_BARS_PER_HOUR.get(tf, 1.0)


def lookback_bars_for_timeframe(timeframe: str) -> int:
    """~20 calendar days of bars at the given timeframe."""
    tf = (timeframe or "1h").strip().lower()
    per_hour = _TF_BARS_PER_HOUR.get(tf, 1.0)
    return max(INV_VOL_MIN_RETURNS + 1, int(20 * 24 * per_hour))


def annualized_realized_vol(
    closes: Sequence[float],
    *,
    lookback: int = INV_VOL_LOOKBACK_BARS,
    periods_per_year: float = HOURS_PER_YEAR,
    min_returns: int = INV_VOL_MIN_RETURNS,
    timeframe: str = "",
) -> Optional[float]:
    """20-day realized annualized vol from log returns.

    Default lookback/annualization is 1h (480 bars, 24*365). When
    ``timeframe`` is set, both scale so 15m/5m bars are not treated as 1h.
    """
    if timeframe:
        lookback = lookback_bars_for_timeframe(timeframe)
        periods_per_year = periods_per_year_for_timeframe(timeframe)
    window = [float(c) for c in closes[-lookback:] if c and float(c) > 0]
    if len(window) < min_returns + 1:
        return None
    rets: list[float] = []
    for prev, cur in zip(window, window[1:]):
        if prev > 0 and cur > 0:
            rets.append(math.log(cur / prev))
    if len(rets) < min_returns:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0:
        return None
    return math.sqrt(var) * math.sqrt(float(periods_per_year))


def inverse_vol_weight(sigma: Optional[float], *, target_vol: Optional[float] = None, cap: Optional[float] = None) -> float:
    """w = min(cap, target_vol / σ). Missing/zero σ → cap only (do not size up)."""
    resolved_cap = max_symbol_weight() if cap is None else min(1.0, max(0.0, float(cap)))
    resolved_target = target_portfolio_vol() if target_vol is None else max(1e-6, float(target_vol))
    if sigma is None or sigma <= 0:
        return resolved_cap
    return min(resolved_cap, resolved_target / float(sigma))


def inverse_vol_notional(equity: float, sigma: Optional[float]) -> float:
    if equity <= 0:
        return 0.0
    return float(equity) * inverse_vol_weight(sigma)


def rolling_drawdown_pct(current_equity: float, peak_equity: float) -> float:
    """Positive fraction: 0.08 means an 8% drop from peak."""
    peak = float(peak_equity or 0.0)
    current = float(current_equity or 0.0)
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - current) / peak)


def peak_equity_30d(equities: Sequence[float], current: float) -> float:
    vals = [float(v) for v in equities if v and float(v) > 0]
    current_f = float(current or 0.0)
    if current_f > 0:
        vals.append(current_f)
    return max(vals) if vals else 0.0


def drawdown_state(drawdown_pct: float) -> tuple[str, float, bool]:
    """Return (mode, size_mult, allow_new_entries)."""
    dd = max(0.0, float(drawdown_pct or 0.0))
    hard = drawdown_hard_pct()
    soft = drawdown_soft_pct()
    if hard > 0 and dd >= hard:
        return SHADOW_ONLY, 0.0, False
    if soft > 0 and dd >= soft:
        return SIZE_HALVED, DRAWDOWN_SOFT_SIZE_MULT, True
    return NORMAL, 1.0, True


def funding_blocks_long(side: str, funding_rate: Optional[float]) -> bool:
    """True when a long perp would pay more than +0.03% / 8h."""
    if not is_long_side(side):
        return False
    if funding_rate is None:
        return False
    try:
        rate = float(funding_rate)
    except (TypeError, ValueError):
        return False
    return rate > funding_long_block()


def is_shadow_strategy(strategy: str) -> bool:
    raw = (strategy or "").strip().lower().replace(" ", "_")
    compact = raw.replace("-", "_")
    for marker in _SHADOW_STRATEGY_MARKERS:
        key = marker.replace(" ", "_").replace("-", "_")
        if key in compact:
            return True
    return False


def prefer_post_only(*, timeframe: str = "", urgent: bool = False, reduce_only: bool = False) -> bool:
    """Non-urgent 4h/12h (and slower) entries prefer maker/post-only."""
    if reduce_only or urgent:
        return False
    tf = (timeframe or "").strip().lower()
    return tf in {"4h", "h4", "12h", "h12", "1d", "d1", "1w", "w1"}


def _position_symbol(item: Any) -> str:
    if isinstance(item, OpenPosition):
        return str(item.symbol or "")
    if isinstance(item, Mapping):
        return str(item.get("symbol") or "")
    return str(getattr(item, "symbol", "") or "")


def _position_notional(item: Any) -> float:
    if isinstance(item, OpenPosition):
        return abs(float(item.notional or 0.0))
    if isinstance(item, Mapping):
        if item.get("notional") is not None:
            try:
                return abs(float(item.get("notional") or 0.0))
            except (TypeError, ValueError):
                return 0.0
        try:
            qty = float(item.get("quantity") or item.get("positionAmt") or item.get("qty") or 0.0)
            px = float(item.get("entry_price") or item.get("price") or item.get("mark_price") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return abs(qty * px)
    try:
        qty = float(getattr(item, "quantity", 0.0) or 0.0)
        px = float(getattr(item, "entry_price", 0.0) or getattr(item, "price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return abs(qty * px)


def positions_from_objects(items: Optional[Iterable[Any]]) -> list[OpenPosition]:
    out: list[OpenPosition] = []
    for item in items or []:
        symbol = _position_symbol(item)
        notional = _position_notional(item)
        if symbol and notional > 0:
            out.append(OpenPosition(symbol=symbol, notional=notional))
    return out


def crypto_cluster_state(
    open_positions: Sequence[Any],
    *,
    new_symbol: str = "",
    new_notional: float = 0.0,
    equity: float = 0.0,
) -> dict[str, Any]:
    """Concurrent crypto count + gross weight including an optional new entry."""
    existing: dict[str, float] = {}
    for pos in positions_from_objects(open_positions):
        if not is_crypto_symbol(pos.symbol):
            continue
        key = str(pos.symbol).upper()
        existing[key] = existing.get(key, 0.0) + pos.notional

    new_is_crypto = bool(new_symbol) and is_crypto_symbol(new_symbol)
    new_key = str(new_symbol or "").upper()
    adding_new_name = new_is_crypto and new_key not in existing

    count = len(existing) + (1 if adding_new_name else 0)
    gross = sum(existing.values()) + (float(new_notional) if new_is_crypto else 0.0)
    weight = (gross / float(equity)) if equity > 0 else 0.0
    max_count = max_concurrent_crypto()
    max_weight = max_crypto_cluster_gross()
    count_ok = count <= max_count
    weight_ok = (equity <= 0) or (weight <= max_weight + 1e-12)
    return {
        "count": count,
        "gross_notional": gross,
        "gross_weight": weight,
        "max_count": max_count,
        "max_weight": max_weight,
        "count_ok": count_ok,
        "weight_ok": weight_ok,
        "ok": count_ok and weight_ok,
        "new_is_crypto": new_is_crypto,
        "adding_new_name": adding_new_name,
    }


def forward_gate_status() -> dict[str, Any]:
    """ARTIFACT_4-style confirmation of the enforced constants."""
    return {
        "active": True,
        "mode_note": "Does not flip TRADING_MODE / PAPER_TRADING / DRY_RUN_ALL",
        "inverse_vol": {
            "lookback_1h_bars": INV_VOL_LOOKBACK_BARS,
            "target_portfolio_vol": target_portfolio_vol(),
            "max_symbol_weight": max_symbol_weight(),
            "active": True,
        },
        "crypto_cluster": {
            "max_concurrent_positions": max_concurrent_crypto(),
            "max_gross_weight": max_crypto_cluster_gross(),
            "active": True,
        },
        "drawdown_breaker": {
            "soft_pct": drawdown_soft_pct(),
            "hard_pct": drawdown_hard_pct(),
            "soft_size_mult": DRAWDOWN_SOFT_SIZE_MULT,
            "hard_mode": SHADOW_ONLY,
            "active": True,
        },
        "funding_gate": {
            "long_block_rate": funding_long_block(),
            "active": True,
        },
        "forward_gate_notional_cap_usd": forward_gate_notional_usd(),
        "shadow_strategies": list(_SHADOW_STRATEGY_MARKERS),
        "post_only_for_4h_12h": True,
    }


def evaluate_entry(
    *,
    symbol: str,
    side: str,
    equity: float = 0.0,
    proposed_notional: float = 0.0,
    bars: Optional[Sequence[Any]] = None,
    hourly_closes: Optional[Sequence[float]] = None,
    open_positions: Optional[Sequence[Any]] = None,
    peak_equity: Optional[float] = None,
    rolling_drawdown_pct: Optional[float] = None,
    funding_rate: Optional[float] = None,
    strategy: str = "",
    timeframe: str = "",
    urgent: bool = False,
    reduce_only: bool = False,
) -> PortfolioRiskDecision:
    """Evaluate one new entry against the five deterministic gates.

    Flatten / reduce-only is always allowed. Missing vol/funding/peak data
    does not invent a halt: inv-vol falls back to the 12.5% cap, funding
    fail-opens, drawdown is 0 when no peak is given.
    """
    reasons: list[str] = []
    gates: dict[str, Any] = {}
    size_mult = 1.0
    mode = NORMAL
    allow = True
    side_n = normalize_side(side)
    proposed = max(0.0, float(proposed_notional or 0.0))
    eq = max(0.0, float(equity or 0.0))

    if reduce_only or side_n in {"CLOSE_LONG", "CLOSE_SHORT", "CLOSE", "FLATTEN"}:
        return PortfolioRiskDecision(
            allow=True,
            mode=NORMAL,
            sized_notional=proposed,
            size_mult=1.0,
            prefer_post_only=False,
            reasons=["reduce_only_bypass"],
            gates={"reduce_only": True},
        )

    closes = list(hourly_closes) if hourly_closes is not None else closes_from_bars(bars)
    sigma = annualized_realized_vol(closes, timeframe=timeframe) if closes else None
    inv_cap = inverse_vol_notional(eq, sigma) if eq > 0 else 0.0
    symbol_cap = eq * max_symbol_weight() if eq > 0 else 0.0
    fwd_cap = forward_gate_notional_usd()

    sized = proposed if proposed > 0 else inv_cap
    if eq > 0:
        if inv_cap > 0:
            sized = min(sized, inv_cap) if proposed > 0 else inv_cap
        elif symbol_cap > 0:
            sized = min(sized, symbol_cap) if proposed > 0 else symbol_cap
    if fwd_cap > 0 and sized > fwd_cap:
        sized = fwd_cap
        reasons.append(f"forward_gate_notional_cap ${fwd_cap:.2f}")

    gates["inverse_vol"] = {
        "sigma_ann": None if sigma is None else round(float(sigma), 6),
        "lookback_closes": len(closes),
        "inv_vol_notional": round(inv_cap, 6),
        "symbol_cap_notional": round(symbol_cap, 6),
        "active": True,
    }

    if rolling_drawdown_pct is not None:
        dd = max(0.0, float(rolling_drawdown_pct))
    else:
        peak = float(peak_equity) if peak_equity is not None else eq
        dd = _drawdown_fraction(eq, peak)

    dd_mode, dd_mult, _dd_allow = drawdown_state(dd)
    gates["drawdown"] = {
        "pct": round(dd, 6),
        "mode": dd_mode,
        "size_mult": dd_mult,
        "soft_pct": drawdown_soft_pct(),
        "hard_pct": drawdown_hard_pct(),
        "active": True,
    }
    if dd_mode == SHADOW_ONLY:
        mode = SHADOW_ONLY
        size_mult = 0.0
        allow = False
        reasons.append(f"drawdown {dd:.2%} >= {drawdown_hard_pct():.0%} hard kill (SHADOW_ONLY)")
    elif dd_mode == SIZE_HALVED:
        mode = SIZE_HALVED
        size_mult = dd_mult
        sized *= dd_mult
        reasons.append(f"drawdown {dd:.2%} >= {drawdown_soft_pct():.0%}: size x{dd_mult:.2f}")

    cluster = crypto_cluster_state(
        open_positions or [],
        new_symbol=symbol,
        new_notional=sized,
        equity=eq,
    )
    gates["crypto_cluster"] = cluster
    if cluster["new_is_crypto"] and not cluster["count_ok"]:
        allow = False
        reasons.append(
            f"crypto cluster count {cluster['count']} > {cluster['max_count']}"
        )
    if cluster["new_is_crypto"] and eq > 0 and not cluster["weight_ok"]:
        allow = False
        reasons.append(
            f"crypto cluster gross {cluster['gross_weight']:.2%} > {cluster['max_weight']:.0%}"
        )

    fund_block = funding_blocks_long(side_n, funding_rate)
    gates["funding"] = {
        "rate": None if funding_rate is None else float(funding_rate),
        "threshold": funding_long_block(),
        "blocks_long": fund_block,
        "active": True,
    }
    if fund_block:
        allow = False
        reasons.append(
            f"funding {float(funding_rate):+.6f} > +{funding_long_block():.4%} blocks long perp"
        )

    shadow = is_shadow_strategy(strategy)
    gates["shadow_strategy"] = {
        "strategy": strategy or "",
        "shadow_only": shadow,
        "forward_gate_notional_usd": fwd_cap,
        "active": True,
    }
    if shadow:
        mode = SHADOW_ONLY
        allow = False
        reasons.append(f"strategy {strategy!r} is SHADOW_ONLY (Recovery 48/192 / meta-labeler)")

    post_only = prefer_post_only(timeframe=timeframe, urgent=urgent, reduce_only=False)
    gates["post_only"] = {"prefer": post_only, "timeframe": timeframe, "active": True}

    if sized < 0:
        sized = 0.0

    return PortfolioRiskDecision(
        allow=allow,
        mode=mode,
        sized_notional=float(sized),
        size_mult=float(size_mult),
        prefer_post_only=post_only,
        reasons=reasons or ["ok"],
        gates=gates,
    )


def _drawdown_fraction(current_equity: float, peak_equity: float) -> float:
    return rolling_drawdown_pct(current_equity, peak_equity)
