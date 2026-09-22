"""Read-only Jev evaluation endpoint.

GET /jev/evaluate is advisory. It does not place orders. Admin auth matches
the rest of the sensitive API surface.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from backend.jev.service import evaluate_symbol
from backend.security import validate_admin_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jev", tags=["jev"])


@router.get("/evaluate")
async def evaluate(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=32, description="Asset or pair, e.g. BTC or BTCUSDT"),
    include_social: bool = Query(False, description="Pull cached social posts when a TwitterAPI key is set"),
):
    """Return trade stance, sentiment, squeeze risk, catalyst, and direction probabilities."""
    validate_admin_request(request)
    logger.info("Jev evaluate requested for %s social=%s", symbol, include_social)
    return await evaluate_symbol(symbol, include_social=include_social, fetch_bars=True)
