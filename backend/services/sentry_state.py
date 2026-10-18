"""
Persistent trading halt flag shared between backend and sentry_watchdog.

Stored on the app-data volume so both containers see the same state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_STATE_DIR = "/app/data/sentry"
STATE_FILENAME = "trading_status.json"


class TradingStatus(str, Enum):
    ACTIVE = "ACTIVE"
    HALTED_BY_SENTRY = "HALTED_BY_SENTRY"
    HALTED_MANUAL = "HALTED_MANUAL"


def _state_dir() -> Path:
    return Path(os.getenv("SENTRY_STATE_DIR", DEFAULT_STATE_DIR))


def _state_path() -> Path:
    return _state_dir() / STATE_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "status": TradingStatus.ACTIVE.value,
        "reason": None,
        "halted_at": None,
        "halted_by": None,
        "resumed_at": None,
        "updated_at": _utc_now(),
    }


def _ensure_dir() -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)


def read_state() -> dict[str, Any]:
    """Return current sentry state, defaulting to ACTIVE if missing."""
    path = _state_path()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # File vanished between exists() and read (concurrent atomic replace)
        # — no halt was recorded, so default to ACTIVE rather than failing closed.
        return _default_state()
    except Exception as exc:
        logger.error("Failed to read sentry state; fail-closed to HALTED_BY_SENTRY", error=str(exc))
        return {
            **_default_state(),
            "status": TradingStatus.HALTED_BY_SENTRY.value,
            "reason": f"state_read_error: {exc}",
            "halted_at": _utc_now(),
            "halted_by": "sentry_state",
        }
    if not isinstance(data, dict):
        logger.error("Sentry state file is not a JSON object; fail-closed to HALTED_BY_SENTRY")
        return {
            **_default_state(),
            "status": TradingStatus.HALTED_BY_SENTRY.value,
            "reason": "state_read_error: not a JSON object",
            "halted_at": _utc_now(),
            "halted_by": "sentry_state",
        }
    status = data.get("status", TradingStatus.ACTIVE.value)
    if status not in {s.value for s in TradingStatus}:
        logger.warning("Unknown trading status in state file; treating as HALTED_BY_SENTRY", status=status)
        data["status"] = TradingStatus.HALTED_BY_SENTRY.value
    return data


def write_state(
    status: TradingStatus,
    *,
    reason: str | None = None,
    halted_by: str | None = None,
    resumed_by: str | None = None,
) -> dict[str, Any]:
    """Atomically persist trading status."""
    _ensure_dir()
    current = read_state()
    now = _utc_now()
    payload: dict[str, Any] = {
        "status": status.value,
        "reason": reason if status != TradingStatus.ACTIVE else None,
        "halted_at": now if status != TradingStatus.ACTIVE else current.get("halted_at"),
        "halted_by": halted_by if status != TradingStatus.ACTIVE else current.get("halted_by"),
        "resumed_at": now if status == TradingStatus.ACTIVE else None,
        "resumed_by": resumed_by if status == TradingStatus.ACTIVE else None,
        "updated_at": now,
    }
    path = _state_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("Sentry trading status updated", **payload)
    return payload


def get_trading_status() -> TradingStatus:
    raw = read_state().get("status", TradingStatus.ACTIVE.value)
    return TradingStatus(raw)


def is_kill_switch_engaged(db=None) -> bool:
    """Return True if kill switch is engaged via sentry file or DB record."""
    status = get_trading_status()
    if status in (TradingStatus.HALTED_MANUAL, TradingStatus.HALTED_BY_SENTRY):
        return True

    # Check DB if available
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import ControlPlaneState

        def _check(s):
            latest = (
                s.query(ControlPlaneState)
                .order_by(ControlPlaneState.id.desc())
                .first()
            )
            return bool(latest and latest.kill_switch_engaged)

        if db is not None:
            return _check(db)
        with SessionLocal() as session:
            return _check(session)
    except Exception as exc:
        logger.debug("Could not check ControlPlaneState table for kill switch", error=str(exc))
        return False


def is_trading_allowed(db=None) -> bool:
    if get_trading_status() != TradingStatus.ACTIVE:
        return False
    if is_kill_switch_engaged(db=db):
        return False
    return True


def halt_trading(*, reason: str, halted_by: str, manual: bool = False, db=None) -> dict[str, Any]:
    status = TradingStatus.HALTED_MANUAL if manual else TradingStatus.HALTED_BY_SENTRY
    res = write_state(status, reason=reason, halted_by=halted_by)

    # Persist in DB if manual kill switch or sentry halt
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import ControlPlaneState
        now_dt = datetime.now(timezone.utc)

        def _persist(s):
            record = ControlPlaneState(
                kill_switch_engaged=True,
                actor=halted_by,
                reason=reason,
                details={"status": status.value, "manual": manual},
                engaged_at=now_dt,
            )
            s.add(record)
            s.commit()

        if db is not None:
            _persist(db)
        else:
            with SessionLocal() as session:
                _persist(session)
    except Exception as exc:
        logger.warning("Could not persist ControlPlaneState on halt", error=str(exc))

    return res


def resume_trading(*, resumed_by: str, reason: str = "Trading resumed", db=None) -> dict[str, Any]:
    res = write_state(TradingStatus.ACTIVE, resumed_by=resumed_by)

    # Persist in DB
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import ControlPlaneState
        now_dt = datetime.now(timezone.utc)

        def _persist(s):
            latest = (
                s.query(ControlPlaneState)
                .filter(ControlPlaneState.kill_switch_engaged.is_(True))
                .order_by(ControlPlaneState.id.desc())
                .first()
            )
            if latest:
                latest.kill_switch_engaged = False
                latest.released_at = now_dt
                if latest.details:
                    details = dict(latest.details)
                    details["released_by"] = resumed_by
                    details["release_reason"] = reason
                    latest.details = details
            else:
                record = ControlPlaneState(
                    kill_switch_engaged=False,
                    actor=resumed_by,
                    reason=reason,
                    details={"released_by": resumed_by, "action": "resume"},
                    released_at=now_dt,
                )
                s.add(record)
            s.commit()

        if db is not None:
            _persist(db)
        else:
            with SessionLocal() as session:
                _persist(session)
    except Exception as exc:
        logger.warning("Could not persist ControlPlaneState on resume", error=str(exc))

    return res


def engage_kill_switch(*, reason: str, actor: str = "admin", details: dict[str, Any] | None = None, db=None) -> dict[str, Any]:
    return halt_trading(reason=reason, halted_by=actor, manual=True, db=db)


def release_kill_switch(*, actor: str = "admin", reason: str = "Kill switch released", db=None) -> dict[str, Any]:
    return resume_trading(resumed_by=actor, reason=reason, db=db)

