"""
Control Plane & Safety API
==========================
Authoritative surface for system safety state, global execution mode,
persistent kill-switch engagement/release, and daily loss monitoring.

Endpoints:
  GET  /control-plane/status              - View effective mode, kill switch, daily PnL, sentry status
  POST /control-plane/kill-switch/engage  - Admin halt of all order execution
  POST /control-plane/kill-switch/release - Admin resumption of order execution
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import Trade
from backend.security import validate_admin_request
from backend.services.pnl_accounting import realized_windows
from backend.services.risk_config import RiskConfig
from backend.services.sentry_state import (
    engage_kill_switch,
    is_kill_switch_engaged,
    is_trading_allowed,
    read_state,
    release_kill_switch,
)
from backend.services.trading_mode import get_trading_mode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/control-plane", tags=["control-plane"])


class KillSwitchEngageRequest(BaseModel):
    reason: str = Field(..., min_length=3, description="Audit reason for engaging the kill switch")
    actor: str = Field(default="admin", description="Operator or subsystem identity")
    confirm: bool = Field(default=True, description="Explicit confirmation flag")


class KillSwitchReleaseRequest(BaseModel):
    reason: str = Field(default="Manual operator release", min_length=3, description="Audit reason for releasing the kill switch")
    actor: str = Field(default="admin", description="Operator identity")


class ControlPlaneStatusResponse(BaseModel):
    effective_mode: str
    is_trading_allowed: bool
    kill_switch_engaged: bool
    kill_switch_reason: Optional[str] = None
    kill_switch_actor: Optional[str] = None
    kill_switch_engaged_at: Optional[str] = None
    sentry_status: str
    daily_realized_pnl: float
    daily_loss_limit_usdt: float
    daily_loss_limit_breached: bool
    reconciliation_pnl: float
    data_freshness_status: str
    last_change_at: Optional[str] = None
    timestamp: str


def _get_control_plane_status_data(db: Session) -> dict[str, Any]:
    global_mode = get_trading_mode()
    sentry_raw = read_state()
    sentry_status_val = sentry_raw.get("status", "ACTIVE")
    kill_engaged = is_kill_switch_engaged(db=db)
    trading_ok = is_trading_allowed(db=db)

    # Compute realized daily PnL from trades
    daily_pnl = 0.0
    recon_pnl = 0.0
    try:
        trades = (
            db.query(Trade)
            .filter(Trade.status == "closed")
            .order_by(Trade.closed_at.desc())
            .limit(500)
            .all()
        )
        windows = realized_windows(trades)
        daily_pnl = float(windows.get("realized_today", 0.0))
        recon_pnl = float(windows.get("reconciliation_pnl", 0.0))
    except Exception as exc:
        logger.warning(f"Could not calculate realized windows for control plane: {exc}")

    # Risk limit check
    cfg = RiskConfig()
    daily_limit_usdt = float(getattr(cfg, "max_daily_loss_usdt", 500.0))
    loss_breached = daily_pnl < -abs(daily_limit_usdt)

    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "effective_mode": global_mode.value,
        "is_trading_allowed": trading_ok and not loss_breached,
        "kill_switch_engaged": kill_engaged,
        "kill_switch_reason": sentry_raw.get("reason"),
        "kill_switch_actor": sentry_raw.get("halted_by"),
        "kill_switch_engaged_at": sentry_raw.get("halted_at"),
        "sentry_status": sentry_status_val,
        "daily_realized_pnl": daily_pnl,
        "daily_loss_limit_usdt": daily_limit_usdt,
        "daily_loss_limit_breached": loss_breached,
        "reconciliation_pnl": recon_pnl,
        "data_freshness_status": "fresh",
        "last_change_at": sentry_raw.get("updated_at"),
        "timestamp": now_iso,
    }


@router.get("/status", response_model=ControlPlaneStatusResponse)
def get_status(db: Session = Depends(get_db)):
    """Return effective execution mode, persistent kill-switch state, and safety telemetry."""
    data = _get_control_plane_status_data(db)
    return ControlPlaneStatusResponse(**data)


@router.post("/kill-switch/engage", response_model=ControlPlaneStatusResponse)
def engage_switch(
    body: KillSwitchEngageRequest,
    db: Session = Depends(get_db),
    _auth: bool = Depends(validate_admin_request),
):
    """Admin endpoint to engage the global kill switch and halt order routing."""
    if not body.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kill switch engage requires confirm=True",
        )

    logger.warning(
        f"[CONTROL PLANE] Kill switch ENGAGED by actor='{body.actor}'. Reason: {body.reason}"
    )
    engage_kill_switch(reason=body.reason, actor=body.actor, db=db)
    data = _get_control_plane_status_data(db)
    return ControlPlaneStatusResponse(**data)


@router.post("/kill-switch/release", response_model=ControlPlaneStatusResponse)
def release_switch(
    body: KillSwitchReleaseRequest,
    db: Session = Depends(get_db),
    _auth: bool = Depends(validate_admin_request),
):
    """Admin endpoint to release the kill switch and resume normal order routing."""
    logger.info(
        f"[CONTROL PLANE] Kill switch RELEASED by actor='{body.actor}'. Reason: {body.reason}"
    )
    release_kill_switch(actor=body.actor, reason=body.reason, db=db)
    data = _get_control_plane_status_data(db)
    return ControlPlaneStatusResponse(**data)
