"""Fail-soft MetaTrader 5 Bridge client for QuantumTrade Pro Research Plane.

Queries the MT5 bridge sidecar for bars and health.
Fail-soft: timeout 3s, returns [] on error, never raises into callers.
NEVER used as ACTIVE_BROKER or for live order routing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MT5_BRIDGE_URL = "http://mt5-bridge:8001"


class MTBridgeClient:
    def __init__(self, base_url: Optional[str] = None, default_timeout: float = 3.0):
        self.base_url = (base_url or os.getenv("MT5_BRIDGE_URL", DEFAULT_MT5_BRIDGE_URL)).rstrip("/")
        self.default_timeout = default_timeout

    async def ping(self, timeout: Optional[float] = None) -> bool:
        """Ping MT5 bridge health endpoint."""
        t = timeout if timeout is not None else self.default_timeout
        try:
            async with httpx.AsyncClient(timeout=t) as client:
                res = await client.get(f"{self.base_url}/health")
                return res.status_code == 200
        except Exception as exc:
            logger.debug(f"MT5 bridge health ping failed ({self.base_url}): {exc}")
            return False

    async def get_bars(
        self,
        symbol: str,
        tf: str = "H1",
        limit: int = 100,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch bars from MT5 bridge sidecar. Fail-soft, returns [] on failure."""
        t = timeout if timeout is not None else self.default_timeout
        params = {
            "symbol": symbol.upper(),
            "tf": tf.upper(),
            "limit": int(limit),
        }
        try:
            async with httpx.AsyncClient(timeout=t) as client:
                res = await client.get(f"{self.base_url}/bars", params=params)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return data.get("bars", [])
                logger.debug(f"MT5 bridge bars returned HTTP {res.status_code} for {symbol}")
                return []
        except Exception as exc:
            logger.debug(f"MT5 bridge bars request failed for {symbol}: {exc}")
            return []


mt_bridge_client = MTBridgeClient()
