"""TwitterAPI.io ingestion with cursor pagination and early-stop dedup.

A missing key or a vendor error returns an empty sample. This module does
not synthesize posts. Fake sentiment must never reach the opinion layer.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from backend.jev.config import env_int, jev_cache_seconds, twitter_api_key
from backend.jev.store import TweetStore

logger = logging.getLogger(__name__)

TWITTER_API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{1,15}$")
ASSET_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "DOGE": "Dogecoin",
    "XRP": "Ripple",
    "ADA": "Cardano",
    "AVAX": "Avalanche",
    "BNB": "BNB",
    "LINK": "Chainlink",
    "UNI": "Uniswap",
}

_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def clear_twitter_cache() -> None:
    _cache.clear()


def _parse_timestamp(raw: str | None) -> int:
    if not raw:
        return int(time.time())
    try:
        if raw.endswith("Z") or "T" in raw:
            return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except (TypeError, ValueError, OverflowError):
        return int(time.time())


def build_query(symbol: str) -> str:
    if not SYMBOL_REGEX.match(symbol):
        raise ValueError(f"Invalid symbol for social query: {symbol!r}")
    name = ASSET_NAMES.get(symbol)
    if not name or name == symbol:
        return f"${symbol} lang:en -is:retweet min_faves:2"
    return f"(${symbol} OR {name}) lang:en -is:retweet min_faves:2"


def _normalize(raw: dict[str, Any]) -> dict[str, Any] | None:
    post_id = str(raw.get("id") or "").strip()
    if not post_id:
        return None
    author = raw.get("author") if isinstance(raw.get("author"), dict) else {}
    created = str(raw.get("createdAt") or raw.get("created_at") or "")
    return {
        "id": post_id,
        "text": str(raw.get("text") or "")[:1000],
        "created_at": created,
        "timestamp_epoch": _parse_timestamp(created or None),
        "likes": int(raw.get("likeCount") or raw.get("likes") or 0),
        "retweets": int(raw.get("retweetCount") or raw.get("retweets") or 0),
        "replies": int(raw.get("replyCount") or raw.get("replies") or 0),
        "author_username": author.get("userName") or author.get("username") or "anonymous",
    }


class TwitterIngestor:
    def __init__(
        self,
        api_key: str | None = None,
        store: TweetStore | None = None,
        transport: httpx.BaseTransport | None = None,
        cache_seconds: int | None = None,
    ) -> None:
        self.api_key = twitter_api_key() if api_key is None else api_key.strip()
        self.store = store or TweetStore()
        self._transport = transport
        self.cache_seconds = jev_cache_seconds() if cache_seconds is None else cache_seconds

    async def fetch(self, symbol: str, target_count: int | None = None) -> dict[str, Any]:
        asset = symbol.upper().replace("$", "").strip()
        target = target_count if target_count is not None else env_int("JEV_TWEET_TARGET", 100)
        target = max(1, min(target, 1000))
        if not SYMBOL_REGEX.match(asset):
            return self._empty(asset, "invalid symbol")

        cache_key = f"{asset}:{target}"
        cached = _cache.get(cache_key)
        if cached and cached[0] > time.time():
            payload = dict(cached[1])
            payload["cached"] = True
            return payload

        if not self.api_key:
            return self._empty(asset, "TWITTERAPI_KEY not configured")

        known = self.store.known_ids(asset)
        seen_ids = set(known)
        fresh: list[dict[str, Any]] = []
        cursor: str | None = None
        pages = 0
        early_stopped = False
        max_pages = min(35, max(5, (target // 15) + 3))
        headers = {"X-API-Key": self.api_key}
        try:
            query = build_query(asset)
        except ValueError as exc:
            return self._empty(asset, str(exc))

        try:
            async with httpx.AsyncClient(timeout=15.0, transport=self._transport) as client:
                while len(fresh) < target and pages < max_pages:
                    pages += 1
                    params: dict[str, str] = {"query": query, "queryType": "Latest"}
                    if cursor:
                        params["cursor"] = cursor
                    response = await client.get(TWITTER_API_URL, headers=headers, params=params)
                    if response.status_code != 200:
                        logger.warning("TwitterAPI.io status %s for %s", response.status_code, asset)
                        break
                    body = response.json()
                    batch = body.get("tweets") or body.get("data") or []
                    if not batch:
                        break
                    for raw in batch:
                        if not isinstance(raw, dict):
                            continue
                        normalized = _normalize(raw)
                        if normalized is None:
                            continue
                        if normalized["id"] in known:
                            early_stopped = True
                            break
                        if normalized["id"] in seen_ids:
                            continue
                        fresh.append(normalized)
                        seen_ids.add(normalized["id"])
                        if len(fresh) >= target:
                            break
                    if early_stopped:
                        break
                    cursor = body.get("next_cursor") or body.get("cursor")
                    if not cursor or cursor is True:
                        break
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Twitter ingestion failed for %s: %s", asset, exc)
            stored = self.store.recent(asset, target)
            return {
                "symbol": asset,
                "status": "unavailable",
                "reason": "twitter request failed",
                "tweets": stored,
                "count": len(stored),
                "newly_fetched_count": 0,
                "api_pages_called": pages,
                "early_stopped": early_stopped,
                "cached": False,
            }

        if fresh:
            self.store.save(asset, fresh)

        combined = list(fresh)
        if len(combined) < target:
            seen = {row["id"] for row in combined}
            for row in self.store.recent(asset, target * 2):
                if row["id"] in seen:
                    continue
                combined.append(row)
                seen.add(row["id"])
                if len(combined) >= target:
                    break

        result = {
            "symbol": asset,
            "status": "ok",
            "reason": "",
            "tweets": combined[:target],
            "count": min(len(combined), target),
            "newly_fetched_count": len(fresh),
            "api_pages_called": pages,
            "early_stopped": early_stopped,
            "cached": False,
        }
        if self.cache_seconds > 0:
            _cache[cache_key] = (time.time() + self.cache_seconds, result)
        return result

    @staticmethod
    def _empty(symbol: str, reason: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "status": "unavailable",
            "reason": reason,
            "tweets": [],
            "count": 0,
            "newly_fetched_count": 0,
            "api_pages_called": 0,
            "early_stopped": False,
            "cached": False,
        }
