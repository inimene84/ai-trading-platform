"""QuantumAIStrategy equity universe lock.

Task 1.2: QuantumAIStrategy is allowed only on US tech equities that showed
a positive edge (NVDA, MSFT, AAPL). Crypto and FX must never enter.

This module has no Jesse import so backend unit tests can cover the rule
without the Jesse runtime.
"""

from __future__ import annotations

ALLOWED_EQUITY_BASES = frozenset({"NVDA", "MSFT", "AAPL"})

_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD", "US")


def equity_base(symbol: str) -> str:
    """Extract the ticker base from Jesse / yfinance / Binance-style names."""
    raw = (symbol or "").strip().upper().replace("/", "-").replace("_", "-")
    if not raw:
        return ""
    if "-" in raw:
        raw = raw.split("-", 1)[0]
    for suffix in _QUOTE_SUFFIXES:
        if raw.endswith(suffix) and len(raw) > len(suffix):
            return raw[: -len(suffix)]
    return raw


def is_allowed_equity_symbol(symbol: str) -> bool:
    return equity_base(symbol) in ALLOWED_EQUITY_BASES
