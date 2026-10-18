"""Asset-class → venue table for dual-broker QuantumTrade.

Crypto stays on Binance USDC-M futures (24/7). Forex and metals stay on
cTrader (IC Markets) during sessions. The mapping is env-driven so a later
IC crypto experiment can flip ``VENUE_CRYPTO`` without a code change.

Defaults match today's hard-coded scan routing. The existing cTrader
crypto-order refusal in ``ctrader_service.place_order`` remains the backstop.

``CTRADER_INVERT_SIDE`` is unchanged here. Re-measure that edge before
turning it off — see ``docs/ops/VENUE_FUSION.md``.
"""

from __future__ import annotations

import os
from typing import Literal

from backend.services.multi_asset_bars import classify_symbol

VenueName = Literal["binance_futures", "ctrader"]

_VENUE_ALIASES: dict[str, VenueName] = {
    "binance": "binance_futures",
    "binance_futures": "binance_futures",
    "binanceusdm": "binance_futures",
    "binance_usdm": "binance_futures",
    "ctrader": "ctrader",
    "ic": "ctrader",
    "icmarkets": "ctrader",
    "ic_markets": "ctrader",
}

_ASSET_ENV: dict[str, tuple[str, str]] = {
    "crypto": ("VENUE_CRYPTO", "binance"),
    "forex": ("VENUE_FOREX", "ctrader"),
    "metal": ("VENUE_METALS", "ctrader"),
    "oil": ("VENUE_OIL", "ctrader"),
    "index": ("VENUE_INDEX", "ctrader"),
    "stock": ("VENUE_STOCK", "ctrader"),
}


def _normalize_venue(raw: str, default: str) -> VenueName:
    key = (raw or default or "").strip().lower()
    if key in _VENUE_ALIASES:
        return _VENUE_ALIASES[key]
    fallback = (default or "ctrader").strip().lower()
    return _VENUE_ALIASES.get(fallback, "ctrader")


def venue_for_asset_class(asset_class: str) -> VenueName:
    """Resolve a classified asset class to a broker venue."""
    env_name, default = _ASSET_ENV.get((asset_class or "").strip().lower(), ("VENUE_OTHER", "ctrader"))
    return _normalize_venue(os.getenv(env_name, default), default)


def venue_for_symbol(symbol: str) -> VenueName:
    """Resolve a trading symbol to a broker venue via asset class."""
    return venue_for_asset_class(classify_symbol(symbol))


def describe_venue_table() -> dict[str, str]:
    """Env snapshot for /status and tests — no I/O."""
    return {
        asset: venue_for_asset_class(asset)
        for asset in ("crypto", "forex", "metal", "oil", "index", "stock")
    }
