"""Explicit equity / risk book scope for dual-broker QuantumTrade.

Live audits (2026-09-12 and 2026-09-15) showed status and kill logic mixing
cTrader ``CTRADER_ENV=sandbox|demo`` equity with Binance live cash. Those
books are independent. This module never flips ``CTRADER_ENV`` to live.

Scopes
------
``broker`` (default)
    Kill / drawdown / sizing use one broker's equity.
``union``
    Opt-in via ``EQUITY_RISK_SCOPE=union``, and only when every contributing
    book shares the same money mode (all live cash, or all demo). Demo+live
    is never unioned — that is forced back to broker-scoped and labeled
    ``split_book``.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

from backend.services.trading_mode import (
    TradingMode,
    binance_paper_parallel_enabled,
    get_trading_mode,
    live_binance_orders_allowed,
    live_exchange_orders_allowed,
)

MoneyMode = Literal["live_cash", "demo", "paper", "testnet", "unknown"]

_BROKER_ALIASES: dict[str, set[str]] = {
    "ctrader": {"ctrader", "ic", "icmarkets", "ctrader:paper"},
    "binance_futures": {"binance_futures", "binance", "binanceusdm"},
}

SPLIT_BOOK_WARNING = (
    "ACTIVE_BROKER is cTrader demo/sandbox while Binance is live cash. "
    "Kill and drawdown use the Binance live book only. "
    "cTrader will not be auto-switched to live — that needs an explicit "
    "Toomas money confirmation (CTRADER_ENV=live + CTRADER_LIVE_CONFIRM)."
)


def _env_lower(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip().lower()


def _truthy(name: str, default: str = "false") -> bool:
    return _env_lower(name, default) in {"1", "true", "yes", "on"}


def _binance_testnet_enabled() -> bool:
    """Match BinanceFuturesService: only the literal ``true`` is testnet.

    Unset, empty, ``false``, and ``1`` are live. ``_truthy()`` would treat
    ``BINANCE_TESTNET=1`` as testnet while the adapter sent live orders.
    """
    return (os.getenv("BINANCE_TESTNET", "false") or "").strip().lower() == "true"


def _ctrader_paper_mode() -> bool:
    """Match CTraderService: paper/demo host defaults ON."""
    raw = os.getenv("CTRADER_PAPER_MODE", os.getenv("CTRADE_PAPER_MODE", "true"))
    return (raw or "true").strip().lower() == "true"


def _ctrader_live_confirmed() -> bool:
    confirm = os.getenv("CTRADER_LIVE_CONFIRM", os.getenv("CTRADE_LIVE_CONFIRM", "")).strip()
    return confirm == "I_UNDERSTAND"


def active_broker_name() -> str:
    return _env_lower("ACTIVE_BROKER", "ctrader") or "ctrader"


def ctrader_env() -> str:
    raw = _env_lower("CTRADER_ENV", "demo")
    if raw in {"sandbox", "demo", "live"}:
        return raw
    return "demo"


def binance_env() -> str:
    if binance_paper_parallel_enabled():
        return "paper"
    # Must match BinanceFuturesService: unset/empty/false → live, not testnet.
    # Only the literal ``true`` is testnet (``1`` / ``yes`` still send live).
    if _binance_testnet_enabled():
        return "testnet"
    return "live"


def ctrader_money_mode() -> MoneyMode:
    if get_trading_mode() != TradingMode.LIVE:
        return "paper" if get_trading_mode() == TradingMode.PAPER else "unknown"
    if _ctrader_paper_mode():
        return "demo"
    env = ctrader_env()
    if env == "live" and _ctrader_live_confirmed() and live_exchange_orders_allowed():
        return "live_cash"
    return "demo"


def binance_money_mode() -> MoneyMode:
    if get_trading_mode() != TradingMode.LIVE or binance_paper_parallel_enabled():
        if get_trading_mode() == TradingMode.PAPER or binance_paper_parallel_enabled():
            return "paper"
    if not live_binance_orders_allowed():
        return "testnet" if binance_env() == "testnet" else "paper"
    if binance_env() == "testnet":
        return "testnet"
    return "live_cash"


def is_split_book() -> bool:
    """True when cTrader is not live cash while Binance is live cash."""
    return binance_money_mode() == "live_cash" and ctrader_money_mode() != "live_cash"


def has_live_cash_book() -> bool:
    return binance_money_mode() == "live_cash" or ctrader_money_mode() == "live_cash"


def configured_risk_scope() -> str:
    raw = _env_lower("EQUITY_RISK_SCOPE", "broker")
    if raw in {"union", "combined", "all"}:
        return "union"
    return "broker"


def effective_risk_scope() -> str:
    """Union is refused when books mix money modes (demo + live)."""
    if is_split_book():
        return "broker"
    if configured_risk_scope() == "union":
        return "union"
    return "broker"


def live_cash_broker() -> Optional[str]:
    """Broker whose equity must drive kill/drawdown for live money."""
    bn = binance_money_mode()
    ct = ctrader_money_mode()
    if bn == "live_cash" and ct != "live_cash":
        return "binance_futures"
    if ct == "live_cash" and bn != "live_cash":
        return "ctrader"
    if bn == "live_cash" and ct == "live_cash":
        active = active_broker_name()
        if active.startswith("ctrader"):
            return "ctrader"
        return "binance_futures"
    return None


def risk_broker_name() -> str:
    live = live_cash_broker()
    if live:
        return live
    return active_broker_name()


def drawdown_suppressed_by_sandbox() -> bool:
    """Sandbox/demo may suppress drawdown only when no live-cash book exists.

    ``CTRADER_ENV=sandbox`` must not disable drawdown for a live Binance account.
    """
    if has_live_cash_book():
        return False
    if _truthy("ENFORCE_SANDBOX_DRAWDOWN", "false"):
        return False
    if ctrader_env() == "sandbox":
        return True
    if _ctrader_paper_mode():
        return True
    return False


def describe_equity_books() -> dict[str, Any]:
    """Env classification for /status — no broker I/O."""
    split = is_split_book()
    return {
        "active_broker": active_broker_name(),
        "ctrader_env": ctrader_env(),
        "binance_env": binance_env(),
        "ctrader_money_mode": ctrader_money_mode(),
        "binance_money_mode": binance_money_mode(),
        "split_book": split,
        "configured_equity_scope": configured_risk_scope(),
        "equity_scope": effective_risk_scope(),
        "risk_broker": risk_broker_name(),
        "split_book_warning": SPLIT_BOOK_WARNING if split else None,
    }


def _broker_matches(book_broker: str, wanted_name: str) -> bool:
    raw = (book_broker or "").lower()
    aliases = _BROKER_ALIASES.get(wanted_name, {wanted_name})
    if raw in aliases:
        return True
    return any(alias in raw for alias in aliases if alias)


def _pick_book(books: list[dict[str, Any]], wanted: str) -> Optional[dict[str, Any]]:
    for book in books:
        if _broker_matches(str(book.get("broker") or ""), wanted):
            return book
    return None


def compose_balance_payload(books: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a kill/status payload from fetched broker books.

    Never reports demo+live union as ``equity``. Missing live-cash books fail
    closed (``error`` set, equity 0) rather than substituting the demo book.
    """
    meta = describe_equity_books()
    tagged = [book for book in books if book]
    risk_name = meta["risk_broker"]
    risk_book = _pick_book(tagged, risk_name)

    payload: dict[str, Any] = {
        **meta,
        "books": tagged,
        "broker": risk_name,
        "balance": 0.0,
        "available": 0.0,
        "equity": 0.0,
        "margin_used": 0.0,
        "display_union_equity": None,
    }

    same_mode = not meta["split_book"]
    if same_mode and effective_risk_scope() == "union":
        if any(b.get("error") for b in tagged):
            payload["error"] = "union_book_unverified"
            return payload
        healthy = [b for b in tagged if not b.get("error")]
        if healthy:
            payload["display_union_equity"] = sum(
                float(b.get("equity") or b.get("balance") or 0.0) for b in healthy
            )
            payload["equity"] = payload["display_union_equity"]
            payload["balance"] = sum(float(b.get("balance") or 0.0) for b in healthy)
            payload["available"] = sum(float(b.get("available") or 0.0) for b in healthy)
            payload["margin_used"] = sum(float(b.get("margin_used") or 0.0) for b in healthy)
            payload["broker"] = "union"
            return payload

    if risk_book is None:
        payload["error"] = "risk_book_missing"
        return payload
    if risk_book.get("error"):
        payload["error"] = str(risk_book.get("error") or "unverified")
        return payload

    payload["balance"] = float(risk_book.get("balance") or 0.0)
    payload["available"] = float(risk_book.get("available") or 0.0)
    payload["equity"] = float(risk_book.get("equity") or risk_book.get("balance") or 0.0)
    payload["margin_used"] = float(risk_book.get("margin_used") or 0.0)
    payload["broker"] = str(risk_book.get("broker") or risk_name)
    return payload
