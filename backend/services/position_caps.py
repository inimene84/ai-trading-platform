"""Single source of truth for the concurrent-position cap.

Live week: the account is supposed to hold at most 2 Binance USDC legs.
Three knobs existed (``MAX_POSITIONS``, ``MAX_OPEN_POSITIONS``,
``MAX_BINANCE_POSITIONS``) and the entry paths did not agree:

* Decision engine used ``max_binance_positions`` only (default 10).
* Risk guard used ``min(max_positions, max_open_positions)``.
* Signal candidate engine used ``MAX_BINANCE_POSITIONS`` (default 10) and
  ignored ``MAX_POSITIONS``.
* The loop counted distinct *DB* symbols, so 6 live shorts could exist
  while the DB still thought the book was under the cap.

Every Binance entry path should call ``effective_binance_position_cap``
and compare it to the live exchange book when that snapshot is available.
Defaults are unchanged; the function only takes the stricter of the knobs
the operator already set.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def effective_binance_position_cap(config: Any) -> int:
    """Stricter of max_positions / max_open_positions / max_binance_positions."""
    caps: list[int] = []
    for attr in ("max_positions", "max_open_positions", "max_binance_positions"):
        number = _positive_int(getattr(config, attr, None))
        if number is not None:
            caps.append(number)
    if not caps:
        return 10
    return min(caps)


def effective_ctrader_position_cap(config: Any) -> int:
    caps: list[int] = []
    for attr in ("max_ctrader_positions", "max_positions", "max_open_positions"):
        number = _positive_int(getattr(config, attr, None))
        if number is not None:
            caps.append(number)
    if not caps:
        return 10
    return min(caps)


def live_binance_position_count(positions: Optional[Iterable[Any]]) -> int:
    """Distinct live Binance symbols with non-zero size."""
    symbols: set[str] = set()
    for raw in positions or []:
        if not isinstance(raw, dict):
            continue
        qty = raw.get("quantity")
        if qty is None:
            qty = raw.get("positionAmt")
        try:
            size = abs(float(qty or 0))
        except (TypeError, ValueError):
            size = 0.0
        if size <= 0:
            continue
        symbol = str(raw.get("symbol") or "").upper()
        if symbol:
            symbols.add(symbol)
    return len(symbols)


def binance_book_at_cap(
    config: Any,
    *,
    db_open_count: int,
    live_positions: Optional[Iterable[Any]] = None,
) -> tuple[bool, int, int]:
    """True when either the DB count or the live book is at/over the cap.

    Returns ``(at_cap, observed_count, cap)``. Observed count is the max of
    DB distinct-symbol count and live exchange count so a stale DB cannot
    hide extra exchange legs.
    """
    cap = effective_binance_position_cap(config)
    live_count = live_binance_position_count(live_positions)
    observed = max(int(db_open_count or 0), live_count)
    return observed >= cap, observed, cap
