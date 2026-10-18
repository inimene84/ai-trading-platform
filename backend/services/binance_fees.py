"""Binance USDM commission-rate cache and fill-role logging.

Reads the live maker/taker rates from ``GET /fapi/v1/commissionRate`` and
caches them. Fill logs use the exchange ``maker`` flag plus the cached
rate instead of the historical 0.02% / 0.05% assumption.

Observational only — does not change order placement. Cache miss / API
error falls back to the last known rate, then to conservative defaults.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Conservative public-tier defaults (taker 5 bps, maker 2 bps) used only
# when the exchange has never answered for this process.
DEFAULT_MAKER_RATE = 0.0002
DEFAULT_TAKER_RATE = 0.0005

_CACHE: dict[str, dict[str, Any]] = {}


def commission_cache_ttl_sec() -> float:
    try:
        return max(30.0, float(os.getenv("BINANCE_COMMISSION_RATE_TTL_SEC", "3600") or 3600))
    except (TypeError, ValueError):
        return 3600.0


def _cache_key(symbol: str) -> str:
    return (symbol or "").upper()


def get_cached_commission_rates(symbol: str) -> Optional[dict[str, float]]:
    """Return cached rates if fresh, else None."""
    entry = _CACHE.get(_cache_key(symbol))
    if not entry:
        return None
    if time.time() - float(entry.get("fetched_at") or 0) > commission_cache_ttl_sec():
        return None
    return {
        "maker_rate": float(entry["maker_rate"]),
        "taker_rate": float(entry["taker_rate"]),
    }


def store_commission_rates(symbol: str, maker_rate: float, taker_rate: float) -> dict[str, float]:
    payload = {
        "maker_rate": float(maker_rate),
        "taker_rate": float(taker_rate),
        "fetched_at": time.time(),
    }
    _CACHE[_cache_key(symbol)] = payload
    return {"maker_rate": payload["maker_rate"], "taker_rate": payload["taker_rate"]}


def clear_commission_cache() -> None:
    _CACHE.clear()


def parse_commission_rate_payload(payload: dict[str, Any]) -> tuple[float, float]:
    """Extract maker/taker rates from a /fapi/v1/commissionRate body."""
    maker = payload.get("makerCommissionRate", payload.get("makerCommission"))
    taker = payload.get("takerCommissionRate", payload.get("takerCommission"))
    return float(maker if maker is not None else DEFAULT_MAKER_RATE), float(
        taker if taker is not None else DEFAULT_TAKER_RATE
    )


def fetch_commission_rates(client: Any, symbol: str) -> dict[str, float]:
    """Fetch + cache rates. Uses cache when fresh. Never raises to callers."""
    cached = get_cached_commission_rates(symbol)
    if cached is not None:
        return cached
    try:
        raw = client.futures_commission_rate(symbol=symbol)
        maker, taker = parse_commission_rate_payload(raw or {})
        stored = store_commission_rates(symbol, maker, taker)
        logger.info(
            "[FEE] %s commissionRate maker=%.6f taker=%.6f (cached %.0fs)",
            symbol, stored["maker_rate"], stored["taker_rate"], commission_cache_ttl_sec(),
        )
        return stored
    except Exception as exc:
        stale = _CACHE.get(_cache_key(symbol))
        if stale:
            logger.warning(
                "[FEE] %s commissionRate fetch failed (%s) — using stale cache",
                symbol, exc,
            )
            return {
                "maker_rate": float(stale["maker_rate"]),
                "taker_rate": float(stale["taker_rate"]),
            }
        logger.warning(
            "[FEE] %s commissionRate fetch failed (%s) — using defaults maker=%.6f taker=%.6f",
            symbol, exc, DEFAULT_MAKER_RATE, DEFAULT_TAKER_RATE,
        )
        return {"maker_rate": DEFAULT_MAKER_RATE, "taker_rate": DEFAULT_TAKER_RATE}


def classify_fill_role(fill: dict[str, Any]) -> str:
    """Binance account-trade ``maker`` flag → maker|taker."""
    flag = fill.get("maker")
    if flag is True or str(flag).lower() in {"true", "1", "yes"}:
        return "maker"
    return "taker"


def summarize_fills(
    fills: list[dict[str, Any]],
    *,
    rates: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Aggregate commission and maker/taker notionals from account trades."""
    maker_notional = 0.0
    taker_notional = 0.0
    commission = 0.0
    maker_fills = 0
    taker_fills = 0
    for fill in fills or []:
        qty = abs(float(fill.get("qty") or fill.get("quantity") or 0.0))
        px = abs(float(fill.get("price") or 0.0))
        notional = qty * px
        commission += abs(float(fill.get("commission") or 0.0))
        role = classify_fill_role(fill)
        if role == "maker":
            maker_notional += notional
            maker_fills += 1
        else:
            taker_notional += notional
            taker_fills += 1
    total_notional = maker_notional + taker_notional
    role = "mixed"
    if maker_fills and not taker_fills:
        role = "maker"
    elif taker_fills and not maker_fills:
        role = "taker"
    elif not maker_fills and not taker_fills:
        role = "unknown"
    expected = None
    if rates and total_notional > 0:
        expected = maker_notional * float(rates.get("maker_rate") or 0.0) + taker_notional * float(
            rates.get("taker_rate") or 0.0
        )
    return {
        "role": role,
        "maker_fills": maker_fills,
        "taker_fills": taker_fills,
        "maker_notional": maker_notional,
        "taker_notional": taker_notional,
        "commission": commission,
        "expected_commission": expected,
        "maker_rate": None if not rates else float(rates.get("maker_rate") or 0.0),
        "taker_rate": None if not rates else float(rates.get("taker_rate") or 0.0),
    }


def log_fill_fees(symbol: str, order_id: Any, summary: dict[str, Any]) -> None:
    logger.info(
        "[FEE] %s order=%s role=%s maker_fills=%s taker_fills=%s "
        "commission=%.6f expected=%.6f maker_rate=%s taker_rate=%s",
        symbol,
        order_id,
        summary.get("role"),
        summary.get("maker_fills"),
        summary.get("taker_fills"),
        float(summary.get("commission") or 0.0),
        float(summary.get("expected_commission") or 0.0),
        summary.get("maker_rate"),
        summary.get("taker_rate"),
    )
