"""Sentry watchdog grace must cover backend restarts, not only watchdog boot."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WATCHDOG = ROOT / "sentry_watchdog"
if str(WATCHDOG) not in sys.path:
    sys.path.insert(0, str(WATCHDOG))

from watchdog import in_failure_grace  # noqa: E402


def test_startup_grace_covers_first_minutes():
    started = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    now = started + timedelta(seconds=50)
    assert in_failure_grace(
        now=now,
        started_at=started,
        last_healthy_at=None,
        startup_grace_sec=120,
    ) is True


def test_restart_grace_covers_healthy_then_down():
    started = datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc)
    last_healthy = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    now = last_healthy + timedelta(seconds=40)
    assert in_failure_grace(
        now=now,
        started_at=started,
        last_healthy_at=last_healthy,
        startup_grace_sec=120,
    ) is True


def test_failures_count_after_grace_expires():
    started = datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc)
    last_healthy = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    now = last_healthy + timedelta(seconds=121)
    assert in_failure_grace(
        now=now,
        started_at=started,
        last_healthy_at=last_healthy,
        startup_grace_sec=120,
    ) is False


@pytest.mark.parametrize("grace", [0, -1])
def test_disabled_grace_never_holds(grace):
    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    assert in_failure_grace(
        now=now,
        started_at=now,
        last_healthy_at=now,
        startup_grace_sec=grace,
    ) is False
