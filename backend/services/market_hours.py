"""Venue session hours for entry filters.

Crypto is 24/7. cTrader FX / metals / oil follow the existing QuantumTrade
window (Sun 21:00 UTC – Fri 21:00 UTC). Used to skip weekend IC order
attempts without weakening weekday fail-closed entry filters.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.services.multi_asset_bars import classify_symbol

_CRYPTO_YFINANCE = {"BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"}


def is_crypto_symbol(symbol: str) -> bool:
    """Match the trading-loop crypto set so weekend skip does not close perps."""
    if classify_symbol(symbol) == "crypto":
        return True
    upper = symbol.upper()
    if upper.endswith(("USDT", "USDC", "BUSD")):
        return True
    if upper in _CRYPTO_YFINANCE or (upper.endswith("-USD") and "=" not in upper):
        return True
    return False


def is_venue_open(symbol: str, now: datetime | None = None) -> bool:
    """True when new entries may be sent for ``symbol``.

    Crypto (USDT/USDC perps): always open.
    FX, metals, oil, index: closed Fri 21:00 UTC through Sun 21:00 UTC.
    """
    if is_crypto_symbol(symbol):
        return True

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    weekday = current.weekday()  # Mon=0 ... Sun=6
    if weekday == 4 and current.hour >= 21:  # Friday after 21:00 UTC
        return False
    if weekday == 5:  # Saturday
        return False
    if weekday == 6 and current.hour < 21:  # Sunday before 21:00 UTC
        return False
    return True
