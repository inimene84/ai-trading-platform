#!/usr/bin/env python3
"""
Isolated sentry watchdog — polls backend health and halts trading on failure.

When backend recovers after a sentry halt, auto-resumes trading and Telegram alerts you.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import httpx

from emergency import (
    call_backend_auto_resume,
    call_backend_emergency_halt,
    cancel_binance_orders,
    fetch_sentry_status,
    read_local_halt_status,
    seconds_since_local_halt,
    send_telegram,
    write_halt_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sentry-watchdog] %(levelname)s %(message)s",
)
logger = logging.getLogger("sentry_watchdog")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _auto_resume_enabled() -> bool:
    """Live cash defaults OFF; paper defaults ON.

    Live requires SENTRY_AUTO_RESUME_ENABLED=true AND
    SENTRY_AUTO_RESUME_LIVE_CONFIRM=I_UNDERSTAND. The watchdog container
    cannot import backend.services, so this mirrors sentry_resume.auto_resume_policy.
    """
    raw_mode = os.getenv("TRADING_MODE", "").strip().lower()
    if raw_mode not in {"backtest", "paper", "live"}:
        paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        dry_run = os.getenv("DRY_RUN_ALL", "true").lower() == "true"
        raw_mode = "paper" if (paper_trading or dry_run) else "live"

    flag_raw = os.getenv("SENTRY_AUTO_RESUME_ENABLED")
    flag = None if flag_raw is None or flag_raw.strip() == "" else (
        flag_raw.strip().lower() in {"1", "true", "yes", "on"}
    )
    confirm = os.getenv("SENTRY_AUTO_RESUME_LIVE_CONFIRM", "").strip()
    if raw_mode == "live":
        return bool(flag) and confirm == "I_UNDERSTAND"
    if flag is None:
        return True
    return flag


async def _check_health(base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return False
            data = resp.json()
            return data.get("status") == "ok"
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return False


async def _trigger_emergency(base_url: str, token: str, reason: str) -> None:
    source = "sentry_watchdog"
    logger.critical("EMERGENCY HALT: %s", reason)

    ok = await call_backend_emergency_halt(base_url, token, reason, source)
    if ok:
        logger.info("Backend accepted emergency-halt")
        return

    logger.warning("Backend unreachable — executing local halt + cancel")
    state = write_halt_file(reason=reason, source=source)
    cancel = cancel_binance_orders()
    await send_telegram(
        "\n".join(
            [
                "🛑 <b>SENTRY (local)</b>",
                "Backend unreachable — halt file written.",
                f"Reason: {reason}",
                f"Status: {state.get('status')}",
                f"Orders: {cancel}",
                "",
                "Auto-resume: paper defaults on; live cash requires "
                "SENTRY_AUTO_RESUME_ENABLED=true and "
                "SENTRY_AUTO_RESUME_LIVE_CONFIRM=I_UNDERSTAND.",
            ]
        )
    )


async def _try_auto_resume(
    base_url: str,
    token: str,
    *,
    resume_after_sec: int,
    resume_health_checks: int,
    consecutive_healthy: int,
) -> None:
    sentry = await fetch_sentry_status(base_url)
    status = (sentry or {}).get("status")
    if status is None:
        status = read_local_halt_status()

    if status not in {"HALTED_BY_SENTRY", None}:
        if status == "ACTIVE":
            return
        logger.debug("Auto-resume skipped: status=%s", status)
        return

    if status != "HALTED_BY_SENTRY":
        return

    since = seconds_since_local_halt()
    if since is None and sentry:
        state = (sentry.get("state") or {})
        halted_at = state.get("halted_at")
        if halted_at:
            try:
                parsed = datetime.fromisoformat(halted_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                since = (datetime.now(timezone.utc) - parsed).total_seconds()
            except ValueError:
                since = None

    if since is not None and since < resume_after_sec:
        logger.debug(
            "Auto-resume waiting: %.0fs / %ss since halt",
            since,
            resume_after_sec,
        )
        return

    if consecutive_healthy < resume_health_checks:
        logger.debug(
            "Auto-resume waiting: %s/%s healthy checks",
            consecutive_healthy,
            resume_health_checks,
        )
        return

    logger.info("Attempting auto-resume after stable recovery")
    result = await call_backend_auto_resume(base_url, token, "sentry_watchdog")
    if result and result.get("ok"):
        logger.info("Auto-resume succeeded")
        return

    reason = (result or {}).get("reason") or (result or {}).get("error") or "unknown"
    logger.warning("Auto-resume not completed: %s", reason)


async def run() -> None:
    base_url = os.getenv("BACKEND_URL", "http://backend:8000")
    token = os.getenv("SENTRY_WATCHDOG_TOKEN", "").strip()
    poll_sec = _env_int("SENTRY_POLL_INTERVAL_SEC", 10)
    # 6 x 10s = 60s of failed heartbeats before the emergency path runs —
    # 3x fired on ordinary backend restarts and cancelled Binance orders.
    fail_threshold = _env_int("SENTRY_FAIL_THRESHOLD", 6)
    cooldown_sec = _env_int("SENTRY_EMERGENCY_COOLDOWN_SEC", 300)
    auto_resume = _auto_resume_enabled()
    resume_after_sec = _env_int("SENTRY_RESUME_AFTER_SEC", 120)
    resume_health_checks = _env_int("SENTRY_RESUME_HEALTH_CHECKS", 6)
    startup_grace_sec = _env_int("SENTRY_STARTUP_GRACE_SEC", 120)

    logger.info(
        "Starting sentry watchdog base_url=%s poll=%ss auto_resume=%s startup_grace=%ss",
        base_url,
        poll_sec,
        auto_resume,
        startup_grace_sec,
    )

    consecutive_failures = 0
    consecutive_healthy = 0
    last_emergency_at: datetime | None = None
    started_at = datetime.now(timezone.utc)
    seen_healthy = False

    while True:
        healthy = await _check_health(base_url)
        if healthy:
            seen_healthy = True
            consecutive_failures = 0
            consecutive_healthy += 1
            if _auto_resume_enabled():
                await _try_auto_resume(
                    base_url,
                    token,
                    resume_after_sec=resume_after_sec,
                    resume_health_checks=resume_health_checks,
                    consecutive_healthy=consecutive_healthy,
                )
                sentry = await fetch_sentry_status(base_url)
                if sentry and sentry.get("trading_allowed"):
                    consecutive_healthy = 0
        else:
            consecutive_healthy = 0
            in_grace = (datetime.now(timezone.utc) - started_at).total_seconds() < startup_grace_sec
            if not seen_healthy and in_grace:
                logger.info("Startup grace: backend not ready yet — not counting failures")
            else:
                consecutive_failures += 1
                logger.warning("Unhealthy backend (%s/%s)", consecutive_failures, fail_threshold)
                if consecutive_failures >= fail_threshold:
                    now = datetime.now(timezone.utc)
                    if last_emergency_at and (now - last_emergency_at).total_seconds() < cooldown_sec:
                        logger.info("Emergency cooldown active — skipping duplicate halt")
                    else:
                        await _trigger_emergency(
                            base_url,
                            token,
                            reason=f"heartbeat_failed_{consecutive_failures}x",
                        )
                        last_emergency_at = now
                        consecutive_failures = 0
                        consecutive_healthy = 0

        await asyncio.sleep(poll_sec)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
