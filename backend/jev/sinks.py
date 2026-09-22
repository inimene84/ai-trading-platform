"""Best-effort sinks for Jev social output.

Dedup already lives in SQLite. Sentiment time series go to the existing
news-sentiment Influx bucket. Tweet embeddings go to the Qdrant collection
`jev-tweets` only when JEV_EMBED_TWEETS is on and an embedding provider
answers. A sink failure never changes the advisory result.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from backend.jev.config import jev_embed_tweets
from backend.services.influxdb_writer import InfluxDBWriter
from backend.services.qdrant_client import (
    AsyncQdrantClient,
    Distance,
    PointStruct,
    QDRANT_AVAILABLE,
    VectorParams,
)
from backend.utils.embeddings import generate_text_embedding

logger = logging.getLogger(__name__)

JEV_TWEET_COLLECTION = "jev-tweets"


def _polarity_direction(score: float) -> str:
    if score > 0.1:
        return "BULLISH"
    if score < -0.1:
        return "BEARISH"
    return "NEUTRAL"


async def persist_social(symbol: str, stats: dict[str, Any], posts: list[dict[str, Any]]) -> None:
    sample = int(stats.get("sample_size") or 0)
    if sample <= 0:
        return
    await _write_influx(symbol, stats)
    if jev_embed_tweets():
        await _write_qdrant(symbol, posts)


async def _write_influx(symbol: str, stats: dict[str, Any]) -> None:
    try:
        writer = InfluxDBWriter()
        if not writer._enabled:
            return
        polarity = float(stats.get("polarity_score") or 0.0)
        await writer.write_news_sentiment(
            symbol=symbol,
            sentiment_score=polarity,
            impact_score=min(1.0, abs(polarity)),
            source="jev-twitter",
            time_horizon="short",
            confidence=min(1.0, int(stats.get("sample_size") or 0) / 100.0),
            direction=_polarity_direction(polarity),
        )
    except Exception as exc:
        logger.warning("Jev sentiment Influx write skipped: %s", exc)


async def _write_qdrant(symbol: str, posts: list[dict[str, Any]]) -> None:
    if not QDRANT_AVAILABLE or AsyncQdrantClient is None:
        return

    url = os.getenv("QDRANT_URL", "").strip()
    if not url:
        return
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))
    client = AsyncQdrantClient(url=url, api_key=api_key or None, timeout=5.0)
    try:
        names = [item.name for item in (await client.get_collections()).collections]
        if JEV_TWEET_COLLECTION not in names:
            await client.create_collection(
                collection_name=JEV_TWEET_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        points: list[PointStruct] = []
        for post in posts[:10]:
            text = str(post.get("text") or "").strip()
            if not text:
                continue
            vector = await generate_text_embedding(text, vector_size=vector_size)
            if not vector:
                continue
            digest = hashlib.sha256(str(post.get("id") or text).encode("utf-8")).hexdigest()
            point_id = int(digest[:15], 16)
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "symbol": symbol,
                    "text": text[:500],
                    "author": post.get("author_username") or post.get("author"),
                    "source": "jev-twitter",
                },
            ))
        if points:
            await client.upsert(collection_name=JEV_TWEET_COLLECTION, points=points)
    except Exception as exc:
        logger.warning("Jev tweet Qdrant write skipped: %s", exc)
    finally:
        try:
            await client.close()
        except Exception:
            logger.debug("Jev Qdrant client close failed", exc_info=True)
