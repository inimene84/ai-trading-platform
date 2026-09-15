"""Minimal FastAPI stub for MetaTrader 5 Bridge sidecar.

Provides read-only market data contracts for MT5 FX/metals data.
Never sends live orders or acts as an execution broker.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="MT5 Bridge Stub", version="1.0.0")

TERMINAL_CONNECTED = os.getenv("MT5_TERMINAL_CONNECTED", "false").lower() in ("true", "1", "yes")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "mt5-bridge",
        "terminal_connected": TERMINAL_CONNECTED,
    }


@app.get("/bars")
async def get_bars(
    symbol: str = Query(..., description="Symbol (e.g. EURUSD, XAUUSD)"),
    tf: str = Query("H1", description="Timeframe (e.g. M15, H1)"),
    limit: int = Query(100, ge=1, le=1000),
) -> JSONResponse:
    """Return OHLCV bars.
    
    If no MetaTrader terminal is attached, returns an empty list with status: 'no_terminal'.
    """
    if not TERMINAL_CONNECTED:
        return JSONResponse(
            status_code=200,
            content={"status": "no_terminal", "symbol": symbol.upper(), "tf": tf.upper(), "bars": []},
            headers={"X-MT5-Status": "no_terminal"}
        )

    # If terminal is attached in production, this would read from memory or JSONL dump
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "symbol": symbol.upper(), "tf": tf.upper(), "bars": []},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
