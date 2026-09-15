"""
Safe auto-resume after sentry halt: reconcile positions, restore ACTIVE, notify.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import structlog

from backend.services.sentry_state import (
    TradingStatus,
    get_trading_status,
    read_state,
    resume_trading,
)
from backend.services.trading_mode import TradingMode, get_trading_mode
from backend.utils.telegram import send_telegram_message

logger = structlog.get_logger(__name__)

LIVE_AUTO_RESUME_CONFIRM = "I_UNDERSTAND"


def _env_truthy(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def auto_resume_policy() -> dict[str, Any]:
    """Whether sentry auto-resume may fire.

    Paper / backtest: default ON (existing watchdog). Override with
    SENTRY_AUTO_RESUME_ENABLED=false.

    Live cash: default OFF. Requires both SENTRY_AUTO_RESUME_ENABLED=true and
    SENTRY_AUTO_RESUME_LIVE_CONFIRM=I_UNDERSTAND. A stray
    SENTRY_AUTO_RESUME_ENABLED=true on live money is not enough.
    """
    mode = get_trading_mode()
    flag = _env_truthy("SENTRY_AUTO_RESUME_ENABLED")
    confirm = os.getenv("SENTRY_AUTO_RESUME_LIVE_CONFIRM", "").strip()
    if mode == TradingMode.LIVE:
        enabled = bool(flag) and confirm == LIVE_AUTO_RESUME_CONFIRM
        if enabled:
            reason = "live_confirmed"
        elif flag is True:
            reason = "live_confirm_required"
        else:
            reason = "live_default_off"
        return {
            "enabled": enabled,
            "trading_mode": mode.value,
            "requires_live_confirm": True,
            "live_confirm_set": confirm == LIVE_AUTO_RESUME_CONFIRM,
            "reason": reason,
        }
    enabled = True if flag is None else flag
    return {
        "enabled": enabled,
        "trading_mode": mode.value,
        "requires_live_confirm": False,
        "live_confirm_set": confirm == LIVE_AUTO_RESUME_CONFIRM,
        "reason": "paper_default_on" if enabled else "explicitly_disabled",
    }


def auto_resume_enabled() -> bool:
    return bool(auto_resume_policy()["enabled"])


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _normalize_broker_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key.startswith("ctrader") or key in {"ic", "icmarkets"}:
        return "ctrader"
    if "binance" in key:
        return "binance_futures"
    return key


def _brokers_to_reconcile() -> list[tuple[Any, str]]:
    """Active broker plus the live-cash risk book when the books are split."""
    from backend.services.binance_futures_service import binance_futures_broker
    from backend.services.ctrader_service import ctrader_broker
    from backend.services.equity_scope import is_split_book, risk_broker_name
    from backend.services.trading_loop import get_active_broker, get_active_broker_name

    active_name = get_active_broker_name()
    seen: set[str] = set()
    ordered: list[tuple[Any, str]] = []

    def _add(broker: Any, name: str) -> None:
        key = _normalize_broker_name(name) or "unknown"
        if key in seen:
            return
        seen.add(key)
        ordered.append((broker, key))

    _add(get_active_broker(), active_name)
    risk_name = risk_broker_name()
    if is_split_book() or _normalize_broker_name(risk_name) != _normalize_broker_name(active_name):
        if _normalize_broker_name(risk_name) == "binance_futures":
            _add(binance_futures_broker, "binance_futures")
        elif _normalize_broker_name(risk_name) == "ctrader":
            _add(ctrader_broker, "ctrader")
    return ordered


async def _reconcile_one_broker(db: Any, broker: Any, broker_name: str) -> dict[str, Any]:
    from backend.database.models import Trade
    from backend.services.ledger import is_binance_paper_fill
    from backend.services.trading_loop_helpers import (
        BrokerPositionSyncService,
        is_ctrader_trade,
    )

    synced = await BrokerPositionSyncService.sync_positions(
        db, broker, {}, {}, broker_name=broker_name
    )
    broker_raw = await asyncio.get_event_loop().run_in_executor(
        None, lambda: broker.get_positions(raise_on_error=False)
    )
    live_symbols = {
        p["symbol"]
        for p in (broker_raw or [])
        if abs(float(p.get("quantity") or p.get("positionAmt") or 0)) > 0
    }
    rows = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
    if _normalize_broker_name(broker_name) == "binance_futures":
        db_symbols = {
            t.symbol
            for t in rows
            if not is_ctrader_trade(t) and not is_binance_paper_fill(t)
        }
    elif _normalize_broker_name(broker_name) == "ctrader":
        db_symbols = {
            t.symbol for t in rows if is_ctrader_trade(t)
        }
    else:
        db_symbols = {t.symbol for t in rows}
    return {
        "broker": broker_name,
        "db_closed": synced,
        "exchange_only_symbols": sorted(live_symbols - db_symbols),
        "error": None,
    }


async def reconcile_positions() -> dict[str, Any]:
    """Sync DB open trades with live broker(s); report exchange-only positions.

    Split-book (cTrader demo + Binance live) must also reconcile the live
    Binance book. Resume fails closed if the live-cash book errors.
    """
    from backend.database.connection import SessionLocal
    from backend.services.equity_scope import risk_broker_name

    db = SessionLocal()
    summary: dict[str, Any] = {
        "db_closed": 0,
        "exchange_only_symbols": [],
        "error": None,
        "books": [],
    }
    risk_name = risk_broker_name()
    try:
        orphans: set[str] = set()
        for broker, name in _brokers_to_reconcile():
            try:
                result = await _reconcile_one_broker(db, broker, name)
            except Exception as exc:
                result = {
                    "broker": name,
                    "db_closed": 0,
                    "exchange_only_symbols": [],
                    "error": str(exc),
                }
                logger.warning(
                    "Sentry reconciliation failed",
                    broker=name,
                    error=str(exc),
                )
            summary["books"].append(result)
            summary["db_closed"] += int(result.get("db_closed") or 0)
            orphans.update(result.get("exchange_only_symbols") or [])
            if result.get("error") and _normalize_broker_name(name) == _normalize_broker_name(risk_name):
                summary["error"] = str(result["error"])
        summary["exchange_only_symbols"] = sorted(orphans)
    except Exception as exc:
        summary["error"] = str(exc)
        logger.warning("Sentry reconciliation failed", error=str(exc))
    finally:
        db.close()
    return summary


async def safe_resume(
    *,
    resumed_by: str,
    reconcile: bool = True,
    allow_manual: bool = False,
) -> dict[str, Any]:
    """
    Resume trading after sentry halt.

    Auto-resume (allow_manual=False) only clears HALTED_BY_SENTRY.
    Operator /resume (allow_manual=True) clears any halt state.

    Live-cash auto-resume is refused unless auto_resume_enabled().
    """
    current = get_trading_status()
    state = read_state()

    if current == TradingStatus.ACTIVE:
        return {"ok": True, "already_active": True, "state": state}

    if current == TradingStatus.HALTED_MANUAL and not allow_manual:
        return {
            "ok": False,
            "skipped": True,
            "reason": "manual_halt_requires_operator_resume",
            "state": state,
        }

    if not allow_manual and not auto_resume_enabled():
        policy = auto_resume_policy()
        return {
            "ok": False,
            "skipped": True,
            "reason": policy["reason"],
            "sentry_auto_resume": policy,
            "state": state,
        }

    reconcile_result: dict[str, Any] | None = None
    if reconcile:
        reconcile_result = await reconcile_positions()
        require_ok = os.getenv("SENTRY_RESUME_REQUIRE_RECONCILE", "true").lower() == "true"
        if require_ok and reconcile_result.get("error"):
            return {
                "ok": False,
                "skipped": True,
                "reason": "reconciliation_failed",
                "reconcile": reconcile_result,
                "state": state,
            }

    # Re-place SL/TP that may have been stripped during watchdog halt / deploy restart.
    protection_result: dict[str, Any] | None = None
    try:
        from backend.database.connection import SessionLocal
        from backend.services.binance_futures_service import binance_futures_broker
        from backend.services.trading_loop_helpers import ExchangeProtectionManager

        db = SessionLocal()
        try:
            protection_result = ExchangeProtectionManager.restore_all_open_positions(
                db, binance_futures_broker,
            )
        finally:
            db.close()
    except Exception as exc:
        protection_result = {"error": str(exc)}
        logger.warning("Protection restore on resume failed", error=str(exc))

    new_state = resume_trading(resumed_by=resumed_by)
    halted_at = state.get("halted_at")
    halt_reason = state.get("reason") or "unknown"

    msg_lines = [
        "✅ <b>TRADING RESUMED</b>",
        f"Resumed by: {resumed_by}",
        f"Previous halt: {halt_reason}",
    ]
    if halted_at:
        msg_lines.append(f"Halted at: {halted_at}")
    if reconcile_result:
        msg_lines.append(f"DB positions closed: {reconcile_result.get('db_closed', 0)}")
        orphans = reconcile_result.get("exchange_only_symbols") or []
        if orphans:
            msg_lines.append(f"⚠️ Exchange-only positions: {', '.join(orphans)}")
        if reconcile_result.get("error"):
            msg_lines.append(f"Reconcile note: {reconcile_result['error']}")
    if protection_result:
        msg_lines.append(
            f"Protection restore: checked={protection_result.get('checked', 0)} "
            f"restored={protection_result.get('restored', 0)}"
        )
        if protection_result.get("error"):
            msg_lines.append(f"Protection note: {protection_result['error']}")

    try:
        await send_telegram_message("\n".join(msg_lines), parse_mode="HTML")
    except Exception as exc:
        logger.warning("Telegram alert failed during resume", error=str(exc))

    return {
        "ok": True,
        "state": new_state,
        "reconcile": reconcile_result,
        "protection": protection_result,
    }


def seconds_since_halt() -> float | None:
    state = read_state()
    halted_at = _parse_iso(state.get("halted_at"))
    if not halted_at:
        return None
    return (datetime.now(timezone.utc) - halted_at).total_seconds()
