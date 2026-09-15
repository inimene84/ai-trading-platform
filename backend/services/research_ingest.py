"""Decision ingestion into OpenSearch for QuantumTrade Pro Research Plane.

Captures all evaluation decisions (vetoes, holds, passes, and shadows)
fire-and-forget behind RESEARCH_INGEST_ENABLED.
Ingestion failures MUST never raise or disrupt live trading execution.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, Optional

from backend.services.opensearch_client import opensearch_client

logger = logging.getLogger(__name__)


def is_research_ingest_enabled() -> bool:
    return os.getenv("RESEARCH_INGEST_ENABLED", "false").strip().lower() in ("true", "1", "yes")


def format_decision_doc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Format and normalize a decision document for OpenSearch qt-decisions."""
    now_iso = datetime.now(timezone.utc).isoformat()
    doc = {
        "ts": payload.get("ts") or now_iso,
        "symbol": str(payload.get("symbol") or "UNKNOWN").upper(),
        "broker": str(payload.get("broker") or "binance").lower(),
        "mode": str(payload.get("mode") or "paper").lower(),
        "signal": str(payload.get("signal") or payload.get("direction") or "HOLD").upper(),
        "confidence": float(payload.get("confidence") or 0.0),
        "promotion_verdict": payload.get("promotion_verdict"),
        "promotion_reason": payload.get("promotion_reason"),
        "gate_id": payload.get("gate_id"),
        "reason": str(payload.get("reason") or ""),
        "shadow": bool(payload.get("shadow", False)),
        "rejected": bool(payload.get("rejected", False)),
        "prompt_version": payload.get("prompt_version"),
        "model_id": payload.get("model_id"),
    }
    return doc


async def ingest_decision(payload: Dict[str, Any]) -> None:
    """Ingest a decision record into qt-decisions. Fail-soft, swallows all errors."""
    if not is_research_ingest_enabled():
        return

    try:
        doc = format_decision_doc(payload)
        await opensearch_client.bulk_index("qt-decisions", [doc])
    except Exception as exc:
        logger.warning(f"Decision ingest failed (swallowed): {exc}")


def ingest_decision_sync(payload: Dict[str, Any]) -> None:
    """Fire-and-forget synchronous wrapper for ingest_decision."""
    if not is_research_ingest_enabled():
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(ingest_decision(payload))
    except RuntimeError:
        # No running event loop
        try:
            asyncio.run(ingest_decision(payload))
        except Exception as exc:
            logger.warning(f"Sync decision ingest failed: {exc}")
    except Exception as exc:
        logger.warning(f"Fire-and-forget decision ingest scheduling failed: {exc}")
