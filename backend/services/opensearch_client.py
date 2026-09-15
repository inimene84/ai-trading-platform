"""Fail-soft OpenSearch client for the QuantumTrade Pro Research Plane.

All calls swallow connection errors and timeouts, returning None or []
so failures never raise into or disrupt the live trading loop.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OPENSEARCH_URL = "http://ai-trading-opensearch:9200"


class OpenSearchClient:
    def __init__(self, base_url: Optional[str] = None, default_timeout: float = 2.5):
        self.base_url = (base_url or os.getenv("OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL)).rstrip("/")
        self.default_timeout = default_timeout

    async def ping(self, timeout: float = 2.0) -> bool:
        """Ping OpenSearch cluster root. Returns True if healthy, False otherwise."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.get(f"{self.base_url}/")
                return res.status_code == 200
        except Exception as exc:
            logger.warning(f"OpenSearch ping failed ({self.base_url}): {exc}")
            return False

    async def ensure_index(self, name: str, mapping: Dict[str, Any], timeout: float = 5.0) -> bool:
        """Ensure an index exists with the provided mapping. Fail-soft."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                head_res = await client.head(f"{self.base_url}/{name}")
                if head_res.status_code == 200:
                    return True
                
                # Create index with mappings / settings
                put_res = await client.put(f"{self.base_url}/{name}", json=mapping)
                if put_res.status_code in (200, 201):
                    logger.info(f"Created OpenSearch index '{name}'")
                    return True
                else:
                    logger.warning(
                        f"Failed to create OpenSearch index '{name}': {put_res.status_code} {put_res.text}"
                    )
                    return False
        except Exception as exc:
            logger.warning(f"Error ensuring OpenSearch index '{name}': {exc}")
            return False

    async def bulk_index(self, index: str, docs: List[Dict[str, Any]], timeout: float = 5.0) -> int:
        """Bulk index documents using NDJSON format. Returns count of indexed docs or 0 on error."""
        if not docs:
            return 0

        lines: List[str] = []
        for doc in docs:
            action = {"index": {"_index": index}}
            lines.append(json.dumps(action))
            lines.append(json.dumps(doc))
        ndjson_body = "\n".join(lines) + "\n"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(
                    f"{self.base_url}/_bulk",
                    content=ndjson_body,
                    headers={"Content-Type": "application/x-ndjson"},
                )
                if res.status_code in (200, 201):
                    data = res.json()
                    items = data.get("items", [])
                    indexed = sum(1 for item in items if item.get("index", {}).get("status") in (200, 201))
                    return indexed
                logger.warning(f"OpenSearch bulk index failed on '{index}': {res.status_code} {res.text}")
                return 0
        except Exception as exc:
            logger.warning(f"Exception during OpenSearch bulk index on '{index}': {exc}")
            return 0

    async def search(
        self,
        index: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search OpenSearch index. Returns list of hit sources, or [] on error/timeout."""
        t = timeout if timeout is not None else self.default_timeout
        try:
            async with httpx.AsyncClient(timeout=t) as client:
                res = await client.post(f"{self.base_url}/{index}/_search", json=body)
                if res.status_code == 200:
                    hits = res.json().get("hits", {}).get("hits", [])
                    return [h.get("_source", {}) for h in hits]
                logger.warning(f"OpenSearch search returned {res.status_code} for '{index}'")
                return []
        except httpx.TimeoutException:
            logger.warning(f"OpenSearch search timed out ({t}s) for index '{index}'")
            return []
        except Exception as exc:
            logger.warning(f"OpenSearch search failed for index '{index}': {exc}")
            return []


opensearch_client = OpenSearchClient()
