"""Fail-soft SearXNG client for news search in QuantumTrade Pro Research Plane.

Fetches news headlines and article snippets from a self-hosted SearXNG instance.
Swallows network errors/timeouts and deduplicates results by URL hash.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SEARXNG_URL = "http://ai-trading-searxng:8080"


def url_hash(url: str) -> str:
    """Generate SHA256 hex digest for a URL."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


class SearXNGClient:
    def __init__(self, base_url: Optional[str] = None, default_timeout: float = 5.0):
        self.base_url = (base_url or os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)).rstrip("/")
        self.default_timeout = default_timeout

    async def ping(self, timeout: float = 3.0) -> bool:
        """Ping SearXNG instance health."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.get(f"{self.base_url}/healthz")
                if res.status_code == 200:
                    return True
                # Fallback check root
                root_res = await client.get(f"{self.base_url}/")
                return root_res.status_code == 200
        except Exception as exc:
            logger.warning(f"SearXNG ping failed ({self.base_url}): {exc}")
            return False

    async def search(
        self,
        q: str,
        categories: str = "news",
        timeout: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """Search SearXNG with JSON format and deduplicate hits by SHA256(url).
        
        Returns a list of dicts: {"title": ..., "url": ..., "snippet": ..., "engine": ..., "hash": ...}.
        Fail-soft: returns [] on error/timeout.
        """
        t = timeout if timeout is not None else self.default_timeout
        params = {
            "q": q,
            "categories": categories,
            "format": "json",
        }
        try:
            async with httpx.AsyncClient(timeout=t) as client:
                res = await client.get(f"{self.base_url}/search", params=params)
                if res.status_code != 200:
                    logger.warning(f"SearXNG search returned HTTP {res.status_code} for query '{q}'")
                    return []
                data = res.json()
                raw_results = data.get("results", [])
                
                deduped: List[Dict[str, str]] = []
                seen_hashes = set()

                for item in raw_results:
                    url = str(item.get("url") or "").strip()
                    if not url:
                        continue
                    h = url_hash(url)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    
                    title = str(item.get("title") or "").strip()
                    snippet = str(item.get("content") or item.get("snippet") or "").strip()
                    engine = str(item.get("engine") or "searxng").strip()

                    deduped.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "engine": engine,
                        "hash": h,
                    })

                return deduped
        except httpx.TimeoutException:
            logger.warning(f"SearXNG search timed out ({t}s) for query '{q}'")
            return []
        except Exception as exc:
            logger.warning(f"SearXNG search failed for query '{q}': {exc}")
            return []


searxng_client = SearXNGClient()
