"""Venue cost model for Binance USDM and cTrader / IC Markets.

Covers spread, maker/taker fees, Binance funding, and overnight swap
(including the triple-swap convention: Wednesday for FX/metals, Friday
for crypto). Used by ``backend/backtesting`` and
``backend/backtesting_ctrader`` so later IC BTC-short experiments can
share one set of assumptions.

Application in the engines is gated by ``BACKTEST_VENUE_COSTS``
(default false) so existing backtest numbers do not change on merge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from backend.services.binance_fees import DEFAULT_MAKER_RATE, DEFAULT_TAKER_RATE
from backend.services.multi_asset_bars import classify_symbol

Venue = Literal["binance", "ctrader"]
AssetClass = str


def venue_costs_enabled() -> bool:
    return (os.getenv("BACKTEST_VENUE_COSTS", "false") or "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class VenueCostConfig:
    venue: Venue
    maker_fee: float
    taker_fee: float
    spread_bps: float
    funding_rate_8h: float
    swap_long_pips: float
    swap_short_pips: float
    pip_size: float


def default_cost_config(venue: Venue, symbol: str = "") -> VenueCostConfig:
    asset = classify_symbol(symbol) if symbol else ("crypto" if venue == "binance" else "forex")
    if venue == "binance":
        return VenueCostConfig(
            venue="binance",
            maker_fee=_f("BACKTEST_BINANCE_MAKER_FEE", DEFAULT_MAKER_RATE),
            taker_fee=_f("BACKTEST_BINANCE_TAKER_FEE", DEFAULT_TAKER_RATE),
            spread_bps=_f("BACKTEST_BINANCE_SPREAD_BPS", 1.0),
            funding_rate_8h=_f("BACKTEST_BINANCE_FUNDING_8H", 0.0001),
            swap_long_pips=0.0,
            swap_short_pips=0.0,
            pip_size=0.1 if "BTC" in (symbol or "").upper() else 0.01,
        )
    pip = 0.01 if (symbol or "").upper().endswith("JPY") else 0.0001
    if asset == "metal":
        pip = 0.01
    return VenueCostConfig(
        venue="ctrader",
        maker_fee=_f("BACKTEST_CTRADER_COMMISSION", 0.0),
        taker_fee=_f("BACKTEST_CTRADER_COMMISSION", 0.0),
        spread_bps=_f("BACKTEST_CTRADER_SPREAD_BPS", 1.2),
        funding_rate_8h=0.0,
        swap_long_pips=_f("BACKTEST_CTRADER_SWAP_LONG_PIPS", 0.4),
        swap_short_pips=_f("BACKTEST_CTRADER_SWAP_SHORT_PIPS", 0.6),
        pip_size=pip,
    )


def triple_swap_weekday(asset_class: str) -> int:
    """Weekday (Mon=0) that posts 3x swap.

    Crypto books settle Friday; FX / metals settle Wednesday.
    """
    if (asset_class or "").lower() == "crypto":
        return 4  # Friday
    return 2  # Wednesday


def swap_multiplier(when: datetime, asset_class: str) -> int:
    """Nights of swap charged for a position held over ``when``'s rollover."""
    current = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    if current.weekday() == triple_swap_weekday(asset_class):
        return 3
    # Weekend: FX is closed Sat/Sun so no daily swap; crypto still funds.
    if (asset_class or "").lower() != "crypto" and current.weekday() >= 5:
        return 0
    return 1


def spread_cost(notional: float, spread_bps: float) -> float:
    return abs(notional) * (float(spread_bps) / 10_000.0)


def fee_cost(notional: float, rate: float) -> float:
    return abs(notional) * abs(float(rate))


def funding_cost(notional: float, rate_8h: float, hours: float, *, side: str) -> float:
    """Funding paid (+) / received (−). Longs pay a positive rate."""
    intervals = max(0.0, float(hours)) / 8.0
    signed = 1.0 if (side or "").upper() in {"BUY", "LONG"} else -1.0
    return signed * abs(notional) * float(rate_8h) * intervals


def swap_cost_pips(
    *,
    side: str,
    nights: list[datetime],
    asset_class: str,
    long_pips: float,
    short_pips: float,
) -> float:
    """Sum of swap pips across rollover nights (triple-aware)."""
    per_night = float(long_pips) if (side or "").upper() in {"BUY", "LONG"} else float(short_pips)
    total = 0.0
    for night in nights:
        total += per_night * swap_multiplier(night, asset_class)
    return total


def swap_cost_usd(
    *,
    lots: float,
    pip_value_per_lot: float,
    side: str,
    nights: list[datetime],
    asset_class: str,
    long_pips: float,
    short_pips: float,
) -> float:
    pips = swap_cost_pips(
        side=side,
        nights=nights,
        asset_class=asset_class,
        long_pips=long_pips,
        short_pips=short_pips,
    )
    return abs(float(lots)) * float(pip_value_per_lot) * pips


def estimate_round_trip(
    *,
    venue: Venue,
    symbol: str,
    notional: float,
    side: str,
    hours_held: float = 0.0,
    nights: Optional[list[datetime]] = None,
    maker_entry: bool = False,
    maker_exit: bool = False,
    lots: float = 0.0,
    pip_value_per_lot: float = 10.0,
    config: Optional[VenueCostConfig] = None,
) -> dict[str, float]:
    """One-trade cost breakdown (spread + fees + funding/swap)."""
    cfg = config or default_cost_config(venue, symbol)
    asset = classify_symbol(symbol)
    entry_fee = fee_cost(notional, cfg.maker_fee if maker_entry else cfg.taker_fee)
    exit_fee = fee_cost(notional, cfg.maker_fee if maker_exit else cfg.taker_fee)
    spread = spread_cost(notional, cfg.spread_bps)
    funding = 0.0
    swap = 0.0
    if venue == "binance":
        funding = funding_cost(notional, cfg.funding_rate_8h, hours_held, side=side)
    else:
        swap = swap_cost_usd(
            lots=lots or 1.0,
            pip_value_per_lot=pip_value_per_lot,
            side=side,
            nights=nights or [],
            asset_class=asset,
            long_pips=cfg.swap_long_pips,
            short_pips=cfg.swap_short_pips,
        )
    total = entry_fee + exit_fee + spread + funding + swap
    return {
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "spread": spread,
        "funding": funding,
        "swap": swap,
        "total": total,
    }
