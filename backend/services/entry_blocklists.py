"""Config blocklists for new entries (Oct 19 "cut the losers", approved by Toomas).

Nothing here closes or touches an open position. Each list only refuses NEW
entries; open legs keep their broker SL/TP and normal management.

Env overrides (comma-separated; set to ``none`` to clear a list):

* ``CTRADER_DISABLED_STRATEGIES`` — default ``MOMENTUM_TREND_PULSE``
* ``CTRADER_SYMBOL_BLOCKLIST``    — default ``XAUUSD,XAGUSD``
* ``BINANCE_BASE_BLOCKLIST``      — default ``CELR,AGT,ICNT,ONE,AKE,BULLA,EPIC``
  (base assets; matches every USDT/USDC/BUSD quote)
* ``BINANCE_ORPHAN_ADOPTION``     — default ``false``: exchange legs opened
  outside the loop are logged, never adopted into loop management.
"""

from __future__ import annotations

import os
from typing import Optional

from backend.services.symbol_aliases import crypto_perp_base

DEFAULT_CTRADER_DISABLED_STRATEGIES = "MOMENTUM_TREND_PULSE"
DEFAULT_CTRADER_SYMBOL_BLOCKLIST = "XAUUSD,XAGUSD"
DEFAULT_BINANCE_BASE_BLOCKLIST = "CELR,AGT,ICNT,ONE,AKE,BULLA,EPIC"

_METAL_ALIASES = {"GOLD": "XAUUSD", "SILVER": "XAGUSD"}


def _csv_env(name: str, default: str) -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    raw = raw.strip()
    if raw.lower() in {"", "none", "off", "false"}:
        return frozenset()
    return frozenset(p.strip().upper() for p in raw.split(",") if p.strip())


def ctrader_disabled_strategies() -> frozenset[str]:
    return _csv_env("CTRADER_DISABLED_STRATEGIES", DEFAULT_CTRADER_DISABLED_STRATEGIES)


def ctrader_symbol_blocklist() -> frozenset[str]:
    return _csv_env("CTRADER_SYMBOL_BLOCKLIST", DEFAULT_CTRADER_SYMBOL_BLOCKLIST)


def binance_base_blocklist() -> frozenset[str]:
    return _csv_env("BINANCE_BASE_BLOCKLIST", DEFAULT_BINANCE_BASE_BLOCKLIST)


def binance_orphan_adoption_enabled() -> bool:
    return (os.getenv("BINANCE_ORPHAN_ADOPTION", "false") or "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _norm_ctrader_symbol(symbol: Optional[str]) -> str:
    s = str(symbol or "").upper().replace("/", "").replace("-", "").strip().rstrip(".")
    return _METAL_ALIASES.get(s, s)


def is_binance_blocked_symbol(symbol: Optional[str]) -> bool:
    base = crypto_perp_base(symbol)
    return bool(base and base in binance_base_blocklist())


def _is_ctrader(broker: Optional[str]) -> bool:
    b = str(broker or "").strip().lower()
    return b.startswith("ctrader") or b in {"ic", "icmarkets"}


def entry_block_reason(
    broker: Optional[str],
    symbol: Optional[str],
    strategy: Optional[str] = None,
) -> Optional[str]:
    """Why a NEW entry is refused, or None. Never applies to exits."""
    if _is_ctrader(broker):
        sym = _norm_ctrader_symbol(symbol)
        if sym in ctrader_symbol_blocklist():
            return f"BLOCKLIST — {sym} is blocked for new cTrader entries (CTRADER_SYMBOL_BLOCKLIST)"
        strat = str(strategy or "").strip().upper()
        if strat and strat in ctrader_disabled_strategies():
            return f"BLOCKLIST — strategy {strat} is disabled on cTrader (CTRADER_DISABLED_STRATEGIES)"
        return None
    if is_binance_blocked_symbol(symbol):
        return (
            f"BLOCKLIST — {str(symbol or '').upper()} is a blocked small-cap "
            f"(BINANCE_BASE_BLOCKLIST)"
        )
    return None
