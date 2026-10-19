"""Cron-expression scheduler for feed refresh / dashboard snapshots / Kronos batch.

Jobs are defined with standard 5-field cron expressions (croniter) and run
inside a single supervised asyncio loop with 1s granularity. A job exception
never kills the loop — it is recorded on the job's ``last_error`` instead.

Env config:
  FEED_SCHEDULER_ENABLED   master switch (default true)
  FEED_REFRESH_CRON        default ``*/5 * * * *``
  DASHBOARD_SNAPSHOT_CRON  default ``*/15 * * * *``
  KRONOS_BATCH_CRON        default ``*/30 * * * *``
  FEED_REFRESH_ENABLED / DASHBOARD_SNAPSHOT_ENABLED / KRONOS_BATCH_ENABLED
                           per-job overrides (default true)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dotenv import load_dotenv

from backend.services import kronos_service
from backend.services.data_hub import DataHub
from backend.services.influxdb_writer import influx
from backend.services.unified_feed import unified_feed

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from croniter import croniter
except ImportError:  # pragma: no cover - dependency declared in pyproject
    croniter = None  # type: ignore[assignment]
    logger.error("croniter not installed — feed scheduler disabled (pip install croniter)")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _env_flag(key: str, default: bool = True) -> bool:
    return os.getenv(key, "true" if default else "false").strip().lower() == "true"


@dataclass
class CronJob:
    name: str
    cron_expr: str
    handler: Callable[[], Awaitable[None]]
    enabled: bool
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_error: Optional[str] = None
    _next_dt: Optional[datetime] = field(default=None, repr=False, compare=False)


_LATEST_BATCH: Dict[str, Any] = {"as_of": None, "results": []}


async def _job_feed_refresh() -> None:
    """Warm quotes for the full default universe + 1h bars for crypto."""
    await unified_feed.get_quotes()
    universe = unified_feed.default_universe()
    await asyncio.gather(
        *(unified_feed.get_bars(sym, "1h", limit=100) for sym in universe["crypto"]),
        return_exceptions=True,
    )
    logger.info("FeedScheduler: feed_refresh completed")


async def _job_dashboard_snapshot() -> None:
    """Snapshot the /api/feed/overview payload to DataHub + InfluxDB."""
    universe = unified_feed.default_universe()
    crypto, equities, metals = await asyncio.gather(
        unified_feed.get_quotes(universe["crypto"]),
        unified_feed.get_quotes(universe["equities"]),
        unified_feed.get_quotes(universe["metals"]),
    )
    payload = {
        "as_of": _utc_now_iso(),
        "crypto": crypto,
        "equities": equities,
        "metals": metals,
    }
    DataHub().publish("feed:snapshot", payload, ttl_ms=20 * 60 * 1000)
    try:
        await influx.write_feed_snapshot(payload)
    except Exception as e:
        logger.warning("FeedScheduler: InfluxDB snapshot write failed: %s", e)
    logger.info("FeedScheduler: dashboard_snapshot completed")


_DEFAULT_FX_BATCH = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,NZDUSD,EURGBP,EURJPY,GBPJPY"


def _fx_batch_symbols() -> List[str]:
    """FX pairs for the Kronos batch, limited to what the active cTrader
    account actually lists (empty when cTrader is not connected)."""
    if not _env_flag("KRONOS_BATCH_FX_ENABLED", True):
        return []
    wanted = [s.strip().upper() for s in os.getenv("FEED_SYMBOLS_FX", _DEFAULT_FX_BATCH).split(",") if s.strip()]
    try:
        from backend.services.ctrader_service import ctrader_service
        if not ctrader_service.is_connected:
            return []
        ids = getattr(ctrader_service, "_symbol_ids", {}) or {}
        return [s for s in wanted if s in ids]
    except Exception:
        logger.debug("FeedScheduler: FX batch universe unavailable", exc_info=True)
        return []


def _fx_batch_pause_sec() -> float:
    try:
        return max(0.0, float(os.getenv("KRONOS_BATCH_FX_PAUSE_SEC", "1.0") or 1.0))
    except (TypeError, ValueError):
        return 1.0


async def _job_kronos_batch() -> None:
    """Run Kronos predictions for the default universe; cache + publish results.

    FX pairs run last, one at a time with a short pause, so the CPU-bound
    Kronos sidecar (2 cores) and Spotware's 5 req/s historical limit (the
    trendbar pacer already enforces CTRADER_BAR_REQ_PER_SEC) stay unstressed.
    """
    started = asyncio.get_event_loop().time()
    universe = unified_feed.default_universe()
    fx_symbols = _fx_batch_symbols()
    symbols = universe["crypto"] + universe["equities"] + universe["metals"] + fx_symbols
    fx_set = set(fx_symbols)
    results: List[Dict[str, Any]] = []
    for sym in symbols:
        if sym in fx_set:
            await asyncio.sleep(_fx_batch_pause_sec())
        try:
            bars_payload = await unified_feed.get_bars(sym, "1h", limit=400)
            pred = await kronos_service.predict(bars_payload.get("data"), sym, "1h")
            result = {
                "symbol": sym,
                "interval": "1h",
                "as_of": _utc_now_iso(),
                "signal": pred.get("signal"),
                "confidence": pred.get("confidence"),
                "predicted_close": pred.get("predicted_close"),
                "predicted_change_pct": pred.get("predicted_change_pct"),
                "cum_change_5_pct": pred.get("cum_change_5_pct"),
                "cum_change_10_pct": pred.get("cum_change_10_pct"),
                "reversal_risk": pred.get("reversal_risk"),
                "model_backend": pred.get("model_backend"),
                "error": pred.get("error"),
            }
        except Exception as e:
            logger.warning("FeedScheduler: kronos_batch failed for %s: %s", sym, e)
            result = {
                "symbol": sym, "interval": "1h", "as_of": _utc_now_iso(),
                "signal": "NEUTRAL", "confidence": 0.0, "predicted_close": None,
                "predicted_change_pct": 0.0, "cum_change_5_pct": 0.0,
                "cum_change_10_pct": 0.0, "reversal_risk": False,
                "model_backend": None, "error": str(e),
            }
        results.append(result)
        try:
            DataHub().publish(f"forecast:{sym}", result, ttl_ms=35 * 60 * 1000)
        except Exception:
            logger.debug("FeedScheduler: DataHub publish skipped for forecast:%s", sym, exc_info=True)

    _LATEST_BATCH["as_of"] = _utc_now_iso()
    _LATEST_BATCH["results"] = results
    logger.info(
        "FeedScheduler: kronos_batch completed (%s symbols, %s FX) in %.1fs",
        len(results), len(fx_symbols), asyncio.get_event_loop().time() - started,
    )


class FeedSchedulerService:
    """Async cron scheduler; jobs never raise out of the loop."""

    def __init__(self) -> None:
        self.enabled = _env_flag("FEED_SCHEDULER_ENABLED", True)
        self.jobs: List[CronJob] = [
            CronJob(
                name="feed_refresh",
                cron_expr=os.getenv("FEED_REFRESH_CRON", "*/5 * * * *"),
                handler=_job_feed_refresh,
                enabled=_env_flag("FEED_REFRESH_ENABLED", True),
            ),
            CronJob(
                name="dashboard_snapshot",
                cron_expr=os.getenv("DASHBOARD_SNAPSHOT_CRON", "*/15 * * * *"),
                handler=_job_dashboard_snapshot,
                enabled=_env_flag("DASHBOARD_SNAPSHOT_ENABLED", True),
            ),
            CronJob(
                name="kronos_batch",
                cron_expr=os.getenv("KRONOS_BATCH_CRON", "*/30 * * * *"),
                handler=_job_kronos_batch,
                enabled=_env_flag("KRONOS_BATCH_ENABLED", True),
            ),
        ]

    @staticmethod
    def _next_fire(cron_expr: str, base: datetime) -> Optional[datetime]:
        if croniter is None:
            return None
        try:
            return croniter(cron_expr, base).get_next(datetime)
        except Exception as e:
            logger.error("FeedScheduler: invalid cron expression '%s': %s", cron_expr, e)
            return None

    async def _run_job(self, job: CronJob) -> None:
        try:
            await job.handler()
            job.last_error = None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            job.last_error = str(e)
            logger.error("FeedScheduler: job '%s' failed: %s", job.name, e, exc_info=True)
        finally:
            job.last_run = _utc_now_iso()
            job._next_dt = self._next_fire(job.cron_expr, _utc_now())
            job.next_run = job._next_dt.isoformat() if job._next_dt else None

    async def start(self) -> None:
        """Supervised loop: fire due jobs concurrently, 1s-granularity sleep."""
        if croniter is None:
            logger.error("FeedScheduler: croniter unavailable — scheduler loop not started")
            return
        logger.info(
            "FeedScheduler: starting with jobs "
            + ", ".join(f"{j.name}({'on' if j.enabled else 'off'}:{j.cron_expr})" for j in self.jobs)
        )
        while True:
            now = _utc_now()
            for job in self.jobs:
                if job.enabled and job._next_dt is None:
                    job._next_dt = self._next_fire(job.cron_expr, now)
                    job.next_run = job._next_dt.isoformat() if job._next_dt else None

            due = [j for j in self.jobs if j.enabled and j._next_dt is not None and j._next_dt <= now]
            if due:
                await asyncio.gather(
                    *(self._run_job(j) for j in due), return_exceptions=True
                )
                continue
            await asyncio.sleep(1.0)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "jobs": [
                {
                    "name": j.name,
                    "cron_expr": j.cron_expr,
                    "enabled": j.enabled,
                    "last_run": j.last_run,
                    "next_run": j.next_run,
                    "last_error": j.last_error,
                }
                for j in self.jobs
            ],
        }

    async def run_job_now(self, name: str) -> Dict[str, Any]:
        """Manually trigger a job by name."""
        job = next((j for j in self.jobs if j.name == name), None)
        if job is None:
            return {"ok": False, "error": f"unknown job '{name}'",
                    "known_jobs": [j.name for j in self.jobs]}
        await self._run_job(job)
        return {
            "ok": job.last_error is None,
            "job": job.name,
            "last_run": job.last_run,
            "last_error": job.last_error,
        }

    def get_latest_batch(self) -> Dict[str, Any]:
        """Last kronos_batch results for /api/forecast/batch/latest."""
        return {"as_of": _LATEST_BATCH["as_of"], "results": list(_LATEST_BATCH["results"])}


feed_scheduler = FeedSchedulerService()
