"""Market data routes for Binance-native data plus n8n enrichment ingest."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Union

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.services.binance_market_data import binance_market_data
from backend.services.crypto_news_service import crypto_news_service
from backend.services.influxdb_writer import influx
from backend.services.multi_asset_bars import fetch_bars

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market-data", tags=["market-data"])
alerts_router = APIRouter(prefix="/api/alerts", tags=["alerts"])

# Last successful n8n writes — GET /technical (WF4) reads this cache.
_LAST_WRITES: dict[str, list[dict[str, Any]]] = {
    "on-chain": [],
    "macro": [],
    "technical": [],
    "divergence": [],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_items(payload: Union[dict, list, BaseModel]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload[:20]
    else:
        items = [payload]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, BaseModel):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _direction_from_label(value: Any, default: str = "NEUTRAL") -> str:
    raw = str(value or default).upper()
    if raw in {"RISK_ON", "BULLISH", "BUY", "WEAK_DOLLAR_BULLISH_FOR_BTC"}:
        return "BULLISH"
    if raw in {"RISK_OFF", "BEARISH", "SELL", "STRONG_DOLLAR_BEARISH_FOR_BTC"}:
        return "BEARISH"
    if raw in {"BULLISH", "BEARISH", "NEUTRAL", "BUY", "SELL"}:
        return raw
    return "NEUTRAL"


class OnChainPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = "BTCUSDT"
    source: str = "on-chain-whale"
    whale_sentiment: str = "NEUTRAL"
    whale_score: float = 0.0
    whale_impact: float = 0.0
    whale_volume_24h: float = 0.0
    exchange_net_flow: float = 0.0
    time_horizon: str = "medium"
    topics: str = ""


class MacroPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = "BTCUSDT"
    source: str = "macro-correlation"
    btc_price: float = 0.0
    sp500_price: float = 0.0
    gold_price: float = 0.0
    dxy: float = 0.0
    vix: float = 0.0
    risk_regime: str = "NEUTRAL"
    dollar_impact: str = "NEUTRAL"
    gold_btc_ratio: float = 0.0
    time_horizon: str = "short"
    topics: str = ""


class TechnicalPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = "BTCUSDT"
    source: str = "technical-divergence"
    rsi: float = 50.0
    macd_histogram: float = 0.0
    price_change_4h: float = 0.0
    funding_rate: float = 0.0
    rsi_divergence: str = "NONE"
    macd_divergence: str = "NONE"
    funding_divergence: str = "NONE"
    overall_signal: str = "NEUTRAL"
    confidence: float = 0.0
    time_horizon: str = "medium"
    topics: str = ""


class DivergencePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = "BTCUSDT"
    source: str = "n8n-divergence"
    sentiment_score: float = 0.0
    direction: str = "NEUTRAL"
    price_change: float = 0.0
    price_change_pct: Optional[float] = None
    signal: str = "NEUTRAL"
    confidence: float = 0.0
    reasoning: str = ""


@router.get("/funding-rates")
async def get_funding_rates():
    """Get funding rates for all 20 symbols."""
    try:
        rates = await binance_market_data.get_all_funding_rates()
        return JSONResponse(content={"status": "ok", "data": rates, "count": len(rates)})
    except Exception as e:
        logger.error(f"Funding rates error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/open-interest")
async def get_open_interest():
    """Get open interest for all 20 symbols."""
    try:
        oi = await binance_market_data.get_all_open_interest()
        return JSONResponse(content={"status": "ok", "data": oi, "count": len(oi)})
    except Exception as e:
        logger.error(f"Open interest error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/overview")
async def get_market_overview():
    """Get combined 24h tickers for all symbols."""
    try:
        tickers = await binance_market_data.get_all_tickers_24h()
        return JSONResponse(content={"status": "ok", "data": tickers, "count": len(tickers)})
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/liquidations/{symbol}")
async def get_liquidations(symbol: str):
    """Get recent liquidations for a symbol. Requires BINANCE_API_KEY."""
    try:
        liquidations = await binance_market_data.get_recent_liquidations(symbol.upper())
        return JSONResponse(content={
            "status": "ok",
            "symbol": symbol.upper(),
            "data": liquidations,
            "count": len(liquidations),
        })
    except Exception as e:
        logger.error(f"Liquidations error for {symbol}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/crypto-news")
async def get_crypto_news():
    """Get aggregated crypto news with sentiment."""
    try:
        summary = await crypto_news_service.get_market_summary()
        return JSONResponse(content={"status": "ok", "data": summary})
    except Exception as e:
        logger.error(f"Crypto news error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/fear-greed")
async def get_fear_greed():
    """Get current Fear & Greed index."""
    try:
        fng = await crypto_news_service.get_fear_greed()
        return JSONResponse(content={"status": "ok", "data": fng})
    except Exception as e:
        logger.error(f"Fear & Greed error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/bars")
async def get_bars(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Fetch OHLCV bars — crypto (Binance), forex/metals (cTrader), stocks/oil (yfinance)."""
    try:
        payload = await fetch_bars(symbol=symbol, timeframe=timeframe, limit=limit)
        return JSONResponse(content={
            "status": "ok",
            "symbol": payload["symbol"],
            "asset_class": payload["asset_class"],
            "source": payload["source"],
            "data": payload["data"],
            "count": len(payload["data"]),
        })
    except Exception as e:
        logger.error(f"Bars error for {symbol}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/trending")
async def get_trending():
    """Get trending coins from CoinGecko."""
    try:
        trending = await crypto_news_service.get_trending_coins()
        return JSONResponse(content={"status": "ok", "data": trending, "count": len(trending)})
    except Exception as e:
        logger.error(f"Trending error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


def _enrichment_response(kind: str, stored: list[dict[str, Any]]) -> dict[str, Any]:
    output = [{"json": row} for row in stored]
    return {
        "status": "stored",
        "kind": kind,
        "count": len(stored),
        "data": stored,
        "output": output,
        "timestamp": _now_iso(),
    }


@router.post("/on-chain")
async def receive_onchain(payload: Union[OnChainPayload, list[OnChainPayload]] = Body(...)):
    """Ingest on-chain / whale-activity rows from n8n WF1."""
    stored: list[dict[str, Any]] = []
    try:
        for item in _as_items(payload):
            symbol = str(item.get("symbol") or "BTCUSDT").upper()
            direction = _direction_from_label(item.get("whale_sentiment"))
            score = float(item.get("whale_score") or 0.0)
            await influx.write_onchain_signal(
                symbol=symbol,
                score=score,
                direction=direction,
                whale_notional=float(item.get("whale_volume_24h") or 0.0),
                oi_change_pct=float(item.get("exchange_net_flow") or 0.0),
                impact_score=float(item.get("whale_impact") or 0.0),
                source=str(item.get("source") or "on-chain-whale"),
            )
            row = {**item, "symbol": symbol, "direction": direction, "score": score}
            stored.append(row)
            logger.info(f"Stored on-chain {symbol}: score={score} direction={direction}")
        _LAST_WRITES["on-chain"] = stored
        return _enrichment_response("on-chain", stored)
    except Exception as e:
        logger.error(f"Failed to store on-chain signal: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/on-chain")
async def get_onchain():
    """Return the last ingested on-chain rows (empty until n8n WF1 writes)."""
    data = _LAST_WRITES["on-chain"]
    return {"status": "ok", "kind": "on-chain", "count": len(data), "data": data, "output": [{"json": r} for r in data]}


@router.post("/macro")
async def receive_macro(payload: Union[MacroPayload, list[MacroPayload]] = Body(...)):
    """Ingest macro / cross-asset rows from n8n WF2."""
    stored: list[dict[str, Any]] = []
    try:
        for item in _as_items(payload):
            symbol = str(item.get("symbol") or "BTCUSDT").upper()
            direction = _direction_from_label(item.get("risk_regime") or item.get("dollar_impact"))
            vix = float(item.get("vix") or 0.0)
            score = (vix - 20.0) / 20.0 if vix else 0.0
            await influx.write_macro_signal(
                symbol=symbol,
                score=score,
                direction=direction,
                btc_price=float(item.get("btc_price") or 0.0),
                sp500_price=float(item.get("sp500_price") or 0.0),
                gold_price=float(item.get("gold_price") or 0.0),
                dxy=float(item.get("dxy") or 0.0),
                vix=vix,
                gold_btc_ratio=float(item.get("gold_btc_ratio") or 0.0),
                risk_regime=str(item.get("risk_regime") or "NEUTRAL"),
                source=str(item.get("source") or "macro-correlation"),
            )
            row = {**item, "symbol": symbol, "direction": direction, "score": score}
            stored.append(row)
            logger.info(f"Stored macro {symbol}: regime={item.get('risk_regime')} vix={vix}")
        _LAST_WRITES["macro"] = stored
        return _enrichment_response("macro", stored)
    except Exception as e:
        logger.error(f"Failed to store macro signal: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/macro")
async def get_macro():
    """Return the last ingested macro row (empty until n8n WF2 writes)."""
    data = _LAST_WRITES["macro"]
    return {"status": "ok", "kind": "macro", "count": len(data), "data": data, "output": [{"json": r} for r in data]}


@router.post("/technical")
async def receive_technical(payload: Union[TechnicalPayload, list[TechnicalPayload]] = Body(...)):
    """Ingest technical-divergence rows from n8n WF3."""
    stored: list[dict[str, Any]] = []
    try:
        for item in _as_items(payload):
            symbol = str(item.get("symbol") or "BTCUSDT").upper()
            direction = _direction_from_label(item.get("overall_signal"))
            score = float(item.get("confidence") or 0.0)
            if direction == "BEARISH":
                score = -abs(score)
            elif direction == "NEUTRAL":
                score = 0.0
            await influx.write_technical_signal(
                symbol=symbol,
                score=score,
                direction=direction,
                rsi=float(item.get("rsi") or 50.0),
                macd_histogram=float(item.get("macd_histogram") or 0.0),
                price_change_pct=float(item.get("price_change_4h") or 0.0),
                funding_rate=float(item.get("funding_rate") or 0.0),
                confidence=float(item.get("confidence") or 0.0),
                source=str(item.get("source") or "technical-divergence"),
            )
            row = {**item, "symbol": symbol, "direction": direction, "score": score}
            stored.append(row)
            logger.info(
                f"Stored technical {symbol}: signal={item.get('overall_signal')} "
                f"rsi={item.get('rsi')} conf={item.get('confidence')}"
            )
        _LAST_WRITES["technical"] = stored
        return _enrichment_response("technical", stored)
    except Exception as e:
        logger.error(f"Failed to store technical signal: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


def _technical_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape WF4 expects: $json.output[].json.{symbol,sentiment_score,direction,price_change_pct}."""
    output: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "BTCUSDT").upper()
        sent_dir = str(row.get("overall_signal") or row.get("direction") or "NEUTRAL")
        if sent_dir == "BUY":
            sent_dir = "BULLISH"
        elif sent_dir == "SELL":
            sent_dir = "BEARISH"
        price_change = float(row.get("price_change_4h") or row.get("price_change_pct") or 0.0)
        # WF4 treats 0.02 as 2%. Stored price_change_4h is already percent; normalize.
        price_change_frac = price_change / 100.0 if abs(price_change) > 1 else price_change
        output.append({
            "json": {
                "symbol": symbol,
                "sentiment_score": float(row.get("score") or 0.0),
                "direction": sent_dir,
                "price_change_pct": price_change_frac,
                "overall_signal": row.get("overall_signal", "NEUTRAL"),
                "rsi": row.get("rsi"),
                "confidence": row.get("confidence", 0.0),
            }
        })
    return output


@router.get("/technical")
async def get_technical():
    """Last technical writes, plus a WF4-compatible `output` envelope."""
    data = list(_LAST_WRITES["technical"])
    return {
        "status": "ok",
        "kind": "technical",
        "count": len(data),
        "data": data,
        "output": _technical_output_rows(data),
    }


async def _store_divergence_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stored: list[dict[str, Any]] = []
    for item in items:
        symbol = str(item.get("symbol") or "BTCUSDT").upper()
        direction = _direction_from_label(item.get("direction") or item.get("signal"))
        price_change = item.get("price_change_pct")
        if price_change is None:
            price_change = item.get("price_change") or 0.0
        score = float(item.get("confidence") or item.get("sentiment_score") or 0.0)
        await influx.write_divergence_alert(
            symbol=symbol,
            score=score,
            direction=direction,
            sentiment_score=float(item.get("sentiment_score") or 0.0),
            price_change_pct=float(price_change or 0.0),
            confidence=float(item.get("confidence") or 0.0),
            signal=str(item.get("signal") or "NEUTRAL"),
            source=str(item.get("source") or "n8n-divergence"),
        )
        row = {**item, "symbol": symbol, "direction": direction, "score": score}
        stored.append(row)
        logger.info(
            f"Stored divergence {symbol}: signal={item.get('signal')} "
            f"direction={direction} conf={item.get('confidence')}"
        )
    _LAST_WRITES["divergence"] = stored
    return stored


@router.post("/divergence")
async def receive_divergence(payload: Union[DivergencePayload, list[DivergencePayload]] = Body(...)):
    """Ingest sentiment-vs-price divergence alerts from n8n WF4."""
    try:
        stored = await _store_divergence_items(_as_items(payload))
        return _enrichment_response("divergence", stored)
    except Exception as e:
        logger.error(f"Failed to store divergence alert: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/divergence")
async def get_divergence():
    data = _LAST_WRITES["divergence"]
    return {
        "status": "ok",
        "kind": "divergence",
        "count": len(data),
        "data": data,
        "output": [{"json": r} for r in data],
    }


@alerts_router.post("/divergence")
async def receive_divergence_alias(payload: Union[DivergencePayload, list[DivergencePayload]] = Body(...)):
    """Alias for existing n8n WF4 URL `/api/alerts/divergence`."""
    return await receive_divergence(payload)
