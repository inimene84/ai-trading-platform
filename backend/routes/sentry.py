"""
Sentry safety endpoints — halt/resume trading and expose watchdog state.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.services.sentry_emergency import emergency_halt
from backend.services.sentry_resume import auto_resume_policy, safe_resume
from backend.services.sentry_state import (
    TradingStatus,
    halt_trading,
    is_trading_allowed,
    read_state,
)
from backend.services.ops_status import trading_ops_snapshot

router = APIRouter(prefix="/sentry", tags=["sentry"])


class EmergencyHaltRequest(BaseModel):
    reason: str = Field(default="heartbeat_failure", max_length=500)
    source: str = Field(default="unknown", max_length=120)
    manual: bool = False


class ResumeRequest(BaseModel):
    note: str = Field(default="", max_length=500)
    reconcile: bool = True


class AutoResumeRequest(BaseModel):
    source: str = Field(default="sentry_watchdog", max_length=120)


def _sentry_tokens() -> list[str]:
    tokens = []
    for name in ("SENTRY_WATCHDOG_TOKEN", "ADMIN_API_KEY", "API_AUTH_TOKEN", "BACKEND_API_KEY"):
        if val := os.getenv(name, "").strip():
            tokens.append(val)
    return tokens


def _validate_sentry_token(request: Request) -> None:
    tokens = _sentry_tokens()
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentry token is not configured; refusing unauthenticated halt/resume.",
        )

    supplied = request.headers.get("x-sentry-token", "").strip()
    if not supplied:
        auth = request.headers.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        supplied = supplied or request.headers.get("x-api-key", "").strip()

    if not supplied:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing sentry/watchdog token.",
        )
    if not any(hmac.compare_digest(supplied, token) for token in tokens):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid sentry token.")


@router.get("/status")
async def sentry_status() -> dict[str, Any]:
    state = read_state()
    ops = trading_ops_snapshot()
    return {
        "trading_allowed": is_trading_allowed(),
        "status": state.get("status", TradingStatus.ACTIVE.value),
        "state": state,
        "equity_books": ops["equity_books"],
        "llm_router": ops["llm_router"],
        "sentry_auto_resume": ops["sentry_auto_resume"],
    }


@router.post("/emergency-halt")
async def sentry_emergency_halt(body: EmergencyHaltRequest, request: Request) -> dict[str, Any]:
    """Halt trading, cancel open orders, notify operators. Used by sentry_watchdog and cron."""
    _validate_sentry_token(request)
    result = await emergency_halt(reason=body.reason, source=body.source, manual=body.manual)
    return {"ok": True, **result}


@router.post("/halt")
async def sentry_halt(body: EmergencyHaltRequest, request: Request) -> dict[str, Any]:
    """Set halt flag without cancelling orders (manual operator action)."""
    _validate_sentry_token(request)
    state = halt_trading(reason=body.reason, halted_by=body.source, manual=True)
    return {"ok": True, "state": state}


@router.post("/resume")
async def sentry_resume(body: ResumeRequest, request: Request) -> dict[str, Any]:
    """Clear halt flag (including manual halts) and send Telegram confirmation."""
    _validate_sentry_token(request)
    result = await safe_resume(
        resumed_by=body.note or "operator",
        reconcile=body.reconcile,
        allow_manual=True,
    )
    return result


@router.post("/auto-resume")
async def sentry_auto_resume(body: AutoResumeRequest, request: Request) -> dict[str, Any]:
    """Resume after sentry halt once backend is stable. Skips HALTED_MANUAL.

    Live cash requires SENTRY_AUTO_RESUME_ENABLED=true and
    SENTRY_AUTO_RESUME_LIVE_CONFIRM=I_UNDERSTAND. Paper/backtest keep the
    existing default-on behavior.
    """
    _validate_sentry_token(request)
    policy = auto_resume_policy()
    if not policy["enabled"]:
        return {
            "ok": False,
            "skipped": True,
            "reason": policy["reason"],
            "sentry_auto_resume": policy,
        }
    result = await safe_resume(resumed_by=body.source, reconcile=True)
    return result
