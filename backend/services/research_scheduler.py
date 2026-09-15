"""Dedicated Research Plane background scheduler for QuantumTrade Pro.

Handles periodic fetch and indexing into OpenSearch:
1. SearXNG news queries (fixed pack with >= 5s rate limiting).
2. Economic calendar high-impact events into qt-events.
3. MetaTrader 5 closed bars into qt-mt-bars.
4. Batch LLM sentiment labeling for news hits.

Active ONLY when RESEARCH_INGEST_ENABLED=true.
Fail-soft: does not disrupt live trading tick loops.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

from backend.services.opensearch_client import opensearch_client
from backend.services.searxng_client import searxng_client
from backend.services.mt_bridge_client import mt_bridge_client
from backend.services.llm_sentiment import classify_bundle, is_research_sentiment_enabled

logger = logging.getLogger(__name__)

QUERIES_FILE = Path(__file__).resolve().parent.parent / "data" / "research_queries.yaml"


def is_research_ingest_enabled() -> bool:
    return os.getenv("RESEARCH_INGEST_ENABLED", "false").strip().lower() in ("true", "1", "yes")


def get_searxng_interval_sec() -> int:
    try:
        return max(60, int(os.getenv("RESEARCH_SEARXNG_INTERVAL_SEC", "900")))
    except ValueError:
        return 900


def load_research_queries() -> List[Dict[str, str]]:
    """Load query pack from backend/data/research_queries.yaml."""
    if not QUERIES_FILE.exists():
        logger.warning(f"Research queries file not found at {QUERIES_FILE}")
        return []
    try:
        data = yaml.safe_load(QUERIES_FILE.read_text(encoding="utf-8"))
        return data.get("queries", [])
    except Exception as exc:
        logger.warning(f"Failed to parse {QUERIES_FILE}: {exc}")
        return []


async def run_searxng_news_sync() -> int:
    """Fetch news for all queries in research pack and bulk index to qt-news."""
    queries = load_research_queries()
    if not queries:
        return 0

    total_indexed = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for item in queries:
        q = item.get("query")
        symbol = item.get("symbol", "UNKNOWN")
        if not q:
            continue

        try:
            hits = await searxng_client.search(q, categories="news")
            if hits:
                docs = []
                headlines = []
                for h in hits:
                    docs.append({
                        "ts": now_iso,
                        "symbol": [symbol] if symbol else [],
                        "source": h.get("engine", "searxng"),
                        "title": h.get("title", ""),
                        "url": h.get("url", ""),
                        "snippet": h.get("snippet", ""),
                        "query": q,
                        "engine": h.get("engine", "searxng"),
                    })
                    if h.get("title"):
                        headlines.append(h["title"])

                # Optional batch LLM sentiment label (gated behind RESEARCH_SENTIMENT_ENABLED)
                if headlines and symbol and symbol != "UNKNOWN" and is_research_sentiment_enabled():
                    try:
                        sentiment_label = await classify_bundle(symbol, headlines[:10])
                        for d in docs:
                            d["sentiment"] = sentiment_label.model_dump()
                    except Exception as e:
                        logger.debug(f"Sentiment labeling skipped for {symbol}: {e}")

                count = await opensearch_client.bulk_index("qt-news", docs)
                total_indexed += count

            # Rate-limit: sleep at least 5s between queries to avoid getting banned
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Error indexing SearXNG query '{q}': {exc}")

    return total_indexed


async def run_calendar_events_sync() -> int:
    """Fetch high-impact calendar events and bulk index into qt-events."""
    try:
        from backend.services.calendar_service import calendar_service
        events = calendar_service.get_upcoming_events(hours_ahead=48, hours_behind=6, min_impact="HIGH")
        if not events:
            return 0

        docs = []
        for ev in events:
            t = ev.get("time_utc")
            ts_str = t.isoformat() if hasattr(t, "isoformat") else str(t)
            docs.append({
                "ts": ts_str,
                "currency": ev.get("currency", "").upper(),
                "impact": ev.get("impact", "HIGH").upper(),
                "title": ev.get("title", ""),
                "source": ev.get("source", "calendar_service"),
            })

        count = await opensearch_client.bulk_index("qt-events", docs)
        return count
    except Exception as exc:
        logger.warning(f"Error during calendar events research sync: {exc}")
        return 0


async def run_mt_bars_sync() -> int:
    """Fetch recent closed bars from MT5 bridge and bulk index into qt-mt-bars."""
    symbols = ["XAUUSD", "EURUSD", "GBPUSD"]
    timeframes = ["H1", "M15"]
    total_indexed = 0

    for sym in symbols:
        for tf in timeframes:
            try:
                bars = await mt_bridge_client.get_bars(symbol=sym, tf=tf, limit=50)
                if bars:
                    docs = []
                    for b in bars:
                        docs.append({
                            "ts": b.get("ts") or datetime.now(timezone.utc).isoformat(),
                            "symbol": sym.upper(),
                            "tf": tf.upper(),
                            "o": float(b.get("o") or 0.0),
                            "h": float(b.get("h") or 0.0),
                            "l": float(b.get("l") or 0.0),
                            "c": float(b.get("c") or 0.0),
                            "v": float(b.get("v") or 0.0),
                        })
                    count = await opensearch_client.bulk_index("qt-mt-bars", docs)
                    total_indexed += count
            except Exception as exc:
                logger.debug(f"Error fetching MT bars for {sym} {tf}: {exc}")

    return total_indexed


async def research_scheduler_loop() -> None:
    """Main continuous scheduler loop."""
    logger.info("Research scheduler worker starting...")
    while True:
        try:
            if is_research_ingest_enabled():
                logger.info("Running Research Plane ingestion cycle...")
                # 1. News sync
                news_count = await run_searxng_news_sync()
                # 2. Calendar sync
                cal_count = await run_calendar_events_sync()
                # 3. MT bars sync
                mt_count = await run_mt_bars_sync()
                logger.info(
                    f"Research Plane sync cycle finished: {news_count} news, "
                    f"{cal_count} events, {mt_count} MT bars indexed."
                )

            interval = get_searxng_interval_sec()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Research scheduler loop received cancellation.")
            break
        except Exception as exc:
            logger.warning(f"Unhandled error in research scheduler loop: {exc}")
            await asyncio.sleep(30.0)
