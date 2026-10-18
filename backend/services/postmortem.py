"""Postmortem agent: runs on CLOSED trades (either inline or via background poller).

Writes a structured postmortem to Supabase (``public.postmortems``) and embeds it
into Qdrant ``finmem-memory`` deep tier (365d horizon, alpha=0.988) via
:class:`backend.services.finmem_service.FinMemService` so the pattern library
compounds over time and is retrievable by persona agents.

Design fixes over the prototype pack:
  - Normalizes real SQLAlchemy ``Trade`` attributes (``id``, ``pnl``, ``timestamp``,
    ``closed_at``, ``entry_price``, ``exit_price``, ``stop_loss``, ``take_profit``,
    ``broker``, ``mode``, ``broker_metadata``, ``notes``) as well as dict payloads.
  - Infers ``exit_reason`` from stop/target price proximity or ``notes`` / metadata
    when not explicitly provided, preventing every trade from misclassifying as
    ``fee_churn``.
  - Uses deterministic composite key ``{broker}:{mode}:{id}`` so upserts are
    idempotent across restarts and brokers.
  - Embeds into FINMEM through ``FinMemService.store_memory(..., layer="deep")``
    so stratified retrieval fields (``layer``, ``base_importance``, ``timestamp``,
    ``access_count``) are populated properly.
  - Includes a background poller (``run_postmortem_poller``) modeled on
    ``TradeMemory.run_recorder_loop`` so all ~9 trade close paths are covered
    without invasive hooks in the hot loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

CAUSES = (
    "fee_churn",
    "stop_placement",
    "regime_shift",
    "model_error",
    "news_shock",
    "liquidity",
    "timeout",
    "winner",
    "unknown",
)


def _supabase_credentials() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
        os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    return url, key


def _iso(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def normalize_trade(trade: Any) -> dict[str, Any]:
    """Convert either a SQLAlchemy ``Trade`` ORM row or a dict into a canonical dict."""
    if isinstance(trade, dict):
        raw = dict(trade)
    else:
        raw = {
            "id": getattr(trade, "id", None),
            "trade_id": getattr(trade, "trade_id", None),
            "symbol": getattr(trade, "symbol", ""),
            "direction": getattr(trade, "direction", ""),
            "broker": getattr(trade, "broker", None) or getattr(trade, "exchange", None) or "binance_futures",
            "mode": getattr(trade, "mode", None) or "paper",
            "opened_at": getattr(trade, "timestamp", None),
            "closed_at": getattr(trade, "closed_at", None),
            "entry_price": getattr(trade, "entry_price", None),
            "exit_price": getattr(trade, "exit_price", None) or getattr(trade, "filled_price", None),
            "stop_loss": getattr(trade, "stop_loss", None),
            "take_profit": getattr(trade, "take_profit", None),
            "quantity": getattr(trade, "quantity", 0.0),
            "pnl": getattr(trade, "pnl", None),
            "strategy": getattr(trade, "strategy", None),
            "notes": getattr(trade, "notes", None),
            "signal_id": getattr(trade, "signal_id", None),
            "broker_metadata": getattr(trade, "broker_metadata", None) or {},
        }

    broker = str(raw.get("broker") or raw.get("exchange") or "binance_futures").lower()
    mode = str(raw.get("mode") or "paper").lower()
    raw_id = raw.get("trade_id") or raw.get("id")
    if raw_id is not None and ":" not in str(raw_id):
        trade_id = f"{broker}:{mode}:{raw_id}"
    else:
        trade_id = str(raw_id or f"{broker}:{mode}:unknown")

    pnl_val = raw.get("realized_pnl")
    if pnl_val is None:
        pnl_val = raw.get("pnl")
    realized_pnl = float(pnl_val) if pnl_val is not None else 0.0

    meta = raw.get("broker_metadata") if isinstance(raw.get("broker_metadata"), dict) else {}
    fees_val = raw.get("total_fees")
    if fees_val is None:
        fees_val = meta.get("commission") or meta.get("total_fees") or meta.get("fee") or 0.0
    total_fees = float(fees_val or 0.0)

    entry_px = float(raw.get("entry_price") or 0.0)
    exit_px = float(raw.get("exit_price") or 0.0)
    sl_px = float(raw["stop_loss"]) if raw.get("stop_loss") is not None else None
    tp_px = float(raw["take_profit"]) if raw.get("take_profit") is not None else None

    exit_reason = str(raw.get("exit_reason") or meta.get("exit_reason") or "").strip().lower()
    notes_lower = str(raw.get("notes") or "").lower()
    if not exit_reason:
        if "stop_loss" in notes_lower or "sl hit" in notes_lower or "stop loss" in notes_lower:
            exit_reason = "stop_loss"
        elif "trailing" in notes_lower:
            exit_reason = "trailing_stop"
        elif "take_profit" in notes_lower or "tp hit" in notes_lower or "take profit" in notes_lower:
            exit_reason = "take_profit"
        elif "timeout" in notes_lower or "time_stop" in notes_lower:
            exit_reason = "timeout"
        elif sl_px and exit_px > 0 and entry_px > 0:
            tol = max(abs(entry_px - sl_px) * 0.15, entry_px * 0.001)
            if abs(exit_px - sl_px) <= tol and realized_pnl < 0:
                exit_reason = "stop_loss"
            elif tp_px and abs(exit_px - tp_px) <= max(abs(tp_px - entry_px) * 0.15, entry_px * 0.001) and realized_pnl > 0:
                exit_reason = "take_profit"

    gate_snapshot = raw.get("gate_snapshot") or raw.get("gates_at_entry") or meta.get("gates") or {}
    llm_conf = raw.get("llm_confidence")
    if llm_conf is None:
        llm_conf = meta.get("confidence") or gate_snapshot.get("confidence")

    return {
        "trade_id": trade_id,
        "symbol": str(raw.get("symbol") or "").upper(),
        "direction": str(raw.get("direction") or "").upper(),
        "broker": broker,
        "mode": mode,
        "opened_at": _iso(raw.get("opened_at") or raw.get("timestamp")),
        "closed_at": _iso(raw.get("closed_at")),
        "entry_price": entry_px if entry_px > 0 else None,
        "exit_price": exit_px if exit_px > 0 else None,
        "stop_loss": sl_px,
        "take_profit": tp_px,
        "realized_pnl": realized_pnl,
        "total_fees": total_fees,
        "exit_reason": exit_reason or "unknown",
        "gate_snapshot": gate_snapshot if isinstance(gate_snapshot, dict) else {},
        "llm_confidence": float(llm_conf) if llm_conf is not None else None,
        "regime_at_entry": raw.get("regime_at_entry") or gate_snapshot.get("regime"),
        "regime_at_exit": raw.get("regime_at_exit") or meta.get("regime_at_exit"),
    }


def classify_cause(trade: Any) -> str:
    """Deterministic first-pass cause classification."""
    norm = normalize_trade(trade)
    pnl = float(norm.get("realized_pnl") or 0.0)
    fees = float(norm.get("total_fees") or 0.0)
    exit_reason = str(norm.get("exit_reason") or "").lower()
    conf = float(norm.get("llm_confidence") or 0.0)
    reg_in = norm.get("regime_at_entry")
    reg_out = norm.get("regime_at_exit")

    # Small PnL eaten by fees or tiny scratch trade
    fee_floor = max(fees * 1.5, 0.50) if fees > 0 else 0.15
    if abs(pnl) <= fee_floor and exit_reason not in ("stop_loss", "take_profit"):
        return "fee_churn"

    if pnl > 0:
        return "winner"

    if reg_in and reg_out and str(reg_in).upper() != str(reg_out).upper():
        return "regime_shift"

    if exit_reason in ("stop_loss", "trailing_stop", "sl"):
        return "stop_placement"

    if exit_reason in ("timeout", "time_stop"):
        return "timeout"

    if pnl < 0 and conf >= 0.70:
        return "model_error"

    return "unknown"


async def build_postmortem(trade: Any) -> dict[str, Any]:
    """Build structured postmortem dict from a closed trade row or dict."""
    norm = normalize_trade(trade)
    cause = classify_cause(norm)
    pm: dict[str, Any] = {
        "trade_id": norm["trade_id"],
        "symbol": norm["symbol"],
        "direction": norm["direction"],
        "broker": norm["broker"],
        "mode": norm["mode"],
        "opened_at": norm["opened_at"],
        "closed_at": norm["closed_at"],
        "realized_pnl": norm["realized_pnl"],
        "total_fees": norm["total_fees"],
        "exit_reason": norm["exit_reason"],
        "cause": cause,
        "gates_at_entry": norm["gate_snapshot"],
        "llm_confidence": norm["llm_confidence"],
        "reflection": None,
        "reflection_model": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if os.getenv("POSTMORTEM_LLM_REFLECTION", "true").strip().lower() == "true":
        try:
            from backend.llm.router import call_llm_resilient_meta
            prompt = (
                "Write a concise postmortem (max 60 words) for this closed trade. "
                "State why it likely won or lost and one concrete rule for future setups.\n"
                + json.dumps(pm, default=str)[:2000]
            )
            txt, meta = await call_llm_resilient_meta("general", prompt, response_json=False)
            pm["reflection"] = (txt or "").strip()[:600]
            pm["reflection_model"] = meta.get("model")
        except Exception as e:
            logger.warning("postmortem reflection skipped for %s: %s", pm["trade_id"], e)
    return pm


async def _supabase_upsert(table: str, row: dict[str, Any]) -> bool:
    url, key = _supabase_credentials()
    if not (url and key):
        return False
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                f"{url}/rest/v1/{table}",
                params={"on_conflict": "trade_id"},
                json=row,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
            )
            if r.status_code in (200, 201, 204):
                return True
            logger.warning("postmortem supabase upsert HTTP %s: %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        logger.warning("postmortem supabase upsert failed: %s", e)
        return False


async def embed_to_finmem(pm: dict[str, Any]) -> bool:
    """Store postmortem into FINMEM deep tier via FinMemService (proper schema & vector)."""
    try:
        from backend.services.finmem_service import finmem_service

        pnl = float(pm.get("realized_pnl") or 0.0)
        reflection = pm.get("reflection") or f"Exit via {pm.get('exit_reason')} with PnL {pnl:+.2f}"
        content = (
            f"[POSTMORTEM {pm['trade_id']}] {pm['symbol']} {pm['direction']} "
            f"cause={pm['cause']} pnl={pnl:+.2f} exit={pm.get('exit_reason')}: {reflection}"
        )
        sents = (
            {"positive": 0.75, "neutral": 0.20, "negative": 0.05}
            if pnl > 0
            else {"positive": 0.05, "neutral": 0.20, "negative": 0.75}
        )
        mem_id = await finmem_service.memory_store.store_memory(
            symbol=pm["symbol"],
            layer="deep",
            content=content,
            source_type="postmortem",
            base_importance=85.0 if pnl < 0 else 70.0,
            sentiments=sents,
        )
        return bool(mem_id)
    except Exception as e:
        logger.warning("finmem embed failed for %s: %s", pm.get("trade_id"), e)
        return False


async def on_trade_closed(trade: Any) -> Optional[dict[str, Any]]:
    """Process a single closed trade: classify, reflect, upsert to Supabase, and embed to FINMEM."""
    try:
        pm = await build_postmortem(trade)
        ok_db = await _supabase_upsert("postmortems", pm)
        ok_vec = await embed_to_finmem(pm)
        logger.info(
            "postmortem %s cause=%s pnl=%s db=%s vec=%s",
            pm["trade_id"],
            pm["cause"],
            pm["realized_pnl"],
            ok_db,
            ok_vec,
        )
        return pm
    except Exception:
        logger.exception("postmortem failed for %s", getattr(trade, "id", None) or (trade.get("trade_id") if isinstance(trade, dict) else "?"))
        return None


class PostmortemPoller:
    """Background poller that sweeps recently closed SQL Trade rows idempotently."""

    def __init__(self) -> None:
        self._seen_ids: set[str] = set()
        self._task: Optional[asyncio.Task[None]] = None

    async def sweep_recent_closed(self, limit: int = 50) -> dict[str, int]:
        processed = 0
        skipped = 0
        try:
            import sqlalchemy as sa
            from backend.database.connection import SessionLocal
            from backend.database.models import Trade

            with SessionLocal() as db:
                rows = (
                    db.query(Trade)
                    .filter(
                        sa.and_(
                            Trade.status.in_(["closed", "exit"]),
                            Trade.pnl.isnot(None),
                        )
                    )
                    .order_by(Trade.closed_at.desc())
                    .limit(limit)
                    .all()
                )
                snapshots = [normalize_trade(r) for r in rows]

            for snap in snapshots:
                tid = snap["trade_id"]
                if tid in self._seen_ids:
                    skipped += 1
                    continue
                self._seen_ids.add(tid)
                await on_trade_closed(snap)
                processed += 1
        except Exception as e:
            logger.warning("PostmortemPoller sweep failed: %s", e)
        return {"processed": processed, "skipped": skipped}

    async def run_loop(self) -> None:
        if os.getenv("POSTMORTEM_ENABLED", "true").strip().lower() != "true":
            logger.info("PostmortemPoller disabled via POSTMORTEM_ENABLED")
            return
        interval_min = max(1, int(os.getenv("POSTMORTEM_POLL_INTERVAL_MIN", "15")))
        lookback = int(os.getenv("POSTMORTEM_POLL_LOOKBACK", "50"))
        logger.info("PostmortemPoller started (every %dm, lookback=%d)", interval_min, lookback)
        while True:
            try:
                await asyncio.sleep(interval_min * 60)
                await self.sweep_recent_closed(limit=lookback)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("PostmortemPoller loop error: %s", e)

    def start(self) -> None:
        if self._task is None or self._task.done():
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self.run_loop())
            except RuntimeError:
                pass


postmortem_poller = PostmortemPoller()
