"""News and market data routes for the News & Data Panel."""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List

from backend.services.influxdb_sentiment_reader import sentiment_reader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["news"])

# ─── Simple in-memory cache ───────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes

def _cached(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None

def _set_cache(key: str, data):
    _cache[key] = {"ts": time.time(), "data": data}

# ─── Pydantic Models ───────────────────────────────────────────────────────────
class SentimentPayload(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score -1 to 1")
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Impact score 0 to 1")
    source: str = Field(default="rss", description="News source")
    time_horizon: str = Field(default="short", description="Time horizon: short/medium/long")
    topics: str = Field(default="", description="Comma-separated topics")

class NewsArchivePayload(BaseModel):
    title: str
    content: str
    source: str
    url: str
    published_at: str
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    symbols: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding (1536d)")

class NewsSearchQuery(BaseModel):
    q: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100)

# ─── Sentiment detection ──────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "rally", "surge", "bullish", "gains", "gain", "rises", "rise", "soar",
    "jump", "jumps", "record", "high", "recover", "recovery", "boom",
    "growth", "profit", "profits", "strong", "positive", "upward", "up",
    "breakout", "outperform", "beat", "beats", "upgrade", "buy",
    # Forex & macro
    "hawkish", "tighten", "rate-hike", "surplus", "expansion", "easing",
    "stimulus", "dovish-pivot", "resilient", "robust", "hiring",
}
NEGATIVE_WORDS = {
    "crash", "drop", "drops", "bearish", "loss", "losses", "fall", "falls",
    "plunge", "plunges", "sink", "sinks", "decline", "declines", "down",
    "fear", "risk", "warn", "warning", "sell", "downgrade", "weak",
    "negative", "low", "slump", "tumble", "tumbles", "panic", "collapse",
    # Forex & macro
    "dovish", "recession", "contraction", "default", "tariff", "sanctions",
    "inflation", "deficit", "stagflation", "layoffs", "unemployment",
}

def _detect_sentiment(text: str) -> str:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"

# ─── RSS Feed ─────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    # Crypto
    ("CoinDesk",       "https://feeds.feedburner.com/CoinDesk"),
    ("CoinTelegraph",  "https://cointelegraph.com/rss"),
    ("Reddit Crypto",  "https://www.reddit.com/r/CryptoCurrency/new/.rss"),
    # Forex & FX Markets
    ("ForexLive",      "https://www.forexlive.com/feed/"),
    ("FXStreet",       "https://www.fxstreet.com/rss"),
    ("DailyFX",        "https://www.dailyfx.com/feeds/all"),
    ("Reddit Forex",   "https://www.reddit.com/r/Forex/new/.rss"),
    # Financial Services & Macro
    ("Yahoo Finance",  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD,ETH-USD,EURUSD=X,GBPUSD=X,USDJPY=X,GC=F&region=US&lang=en-US"),
    ("Reuters Business","https://feeds.reuters.com/reuters/businessNews"),
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Investing.com",  "https://www.investing.com/rss/news.rss"),
]

async def _fetch_feed(source: str, url: str) -> list:
    """Fetch and parse a single RSS feed, return list of news items."""
    try:
        import feedparser
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            feed = await loop.run_in_executor(pool, feedparser.parse, url)

        items = []
        for entry in feed.entries[:15]:  # cap at 15 per source
            title   = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            # strip HTML tags simply
            import re
            summary = re.sub(r"<[^>]+>", " ", summary).strip()[:300]
            link    = getattr(entry, "link", "") or ""
            pub     = ""
            if hasattr(entry, "published"):
                pub = entry.published
            elif hasattr(entry, "updated"):
                pub = entry.updated

            sentiment = _detect_sentiment(title + " " + summary)
            items.append({
                "title":     title,
                "summary":   summary,
                "url":       link,
                "source":    source,
                "published": pub,
                "sentiment": sentiment,
            })
        return items
    except Exception as e:
        logger.warning(f"Feed {source} failed: {e}")
        return []


@router.get("/feed")
async def get_news_feed():
    """Return mixed crypto+market news from multiple RSS feeds."""
    cached = _cached("news_feed")
    if cached:
        return cached

    tasks = [_fetch_feed(src, url) for src, url in RSS_FEEDS]
    results = await asyncio.gather(*tasks)

    all_items = []
    for r in results:
        all_items.extend(r)

    # sort by published desc (best-effort string sort)
    all_items.sort(key=lambda x: x["published"], reverse=True)

    response = {
        "items":     all_items,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "count":     len(all_items),
    }
    _set_cache("news_feed", response)
    return response


# ─── Fear & Greed ─────────────────────────────────────────────────────────────
@router.get("/fear-greed")
async def get_fear_greed():
    """Return Fear & Greed index from alternative.me."""
    cached = _cached("fear_greed")
    if cached:
        return cached

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=7")
            data = resp.json()

        latest   = data["data"][0]
        history  = data["data"]
        response = {
            "value":                int(latest["value"]),
            "value_classification": latest["value_classification"],
            "timestamp":            latest["timestamp"],
            "history": [
                {
                    "value":                int(h["value"]),
                    "value_classification": h["value_classification"],
                    "timestamp":            h["timestamp"],
                }
                for h in history
            ],
        }
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        response = {
            "value": 50,
            "value_classification": "Neutral",
            "timestamp": str(int(time.time())),
            "history": [],
        }

    _set_cache("fear_greed", response)
    return response


# ─── Economic Calendar ────────────────────────────────────────────────────────
@router.get("/economic-calendar")
async def get_economic_calendar():
    """Return upcoming economic events from Forex Factory, with mock fallback."""
    cached = _cached("economic_calendar")
    if cached:
        return cached

    from datetime import date, timedelta
    import httpx

    try:
        # Fetch real live data from Forex Factory free JSON endpoint
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            raw_data = resp.json()
        
        events = []
        for d in raw_data:
            # Map Forex Factory structure to our app structure
            # Example: {"title": "OPEC-JMMC Meetings", "country": "All", "date": "2026-06-07T05:15:00-04:00", "impact": "Medium", "forecast": "", "previous": ""}
            try:
                dt = datetime.fromisoformat(d.get("date", "").replace("Z", "+00:00"))
                time_str = dt.strftime("%b %d %H:%M")
            except Exception:
                time_str = d.get("date", "")
            
            events.append({
                "time": time_str,
                "currency": d.get("country", "USD"),
                "event": d.get("title", ""),
                "impact": d.get("impact", "low").lower(),
                "forecast": d.get("forecast", ""),
                "previous": d.get("previous", ""),
            })
        
        response = {"events": events}
        _set_cache("economic_calendar", response)
        return response

    except Exception as e:
        logger.warning(f"Failed to fetch economic calendar, using mock fallback: {e}")
        # Realistic mock data fallback
        today = date.today()
        events = [
            {
                "time": f"{today.strftime('%b %d')} 08:30",
                "currency": "USD",
                "event": "Non-Farm Payrolls",
                "impact": "high",
                "forecast": "185K",
                "previous": "177K",
            },
            {
                "time": f"{today.strftime('%b %d')} 14:00",
                "currency": "USD",
                "event": "FOMC Meeting Minutes",
                "impact": "high",
                "forecast": "-",
                "previous": "-",
            },
            {
                "time": f"{(today + timedelta(days=1)).strftime('%b %d')} 08:30",
                "currency": "USD",
                "event": "CPI (YoY)",
                "impact": "high",
                "forecast": "3.1%",
                "previous": "3.2%",
            },
            {
                "time": f"{(today + timedelta(days=2)).strftime('%b %d')} 08:30",
                "currency": "USD",
                "event": "GDP (QoQ)",
                "impact": "high",
                "forecast": "2.4%",
                "previous": "3.2%",
            },
            {
                "time": f"{(today + timedelta(days=4)).strftime('%b %d')} 05:30",
                "currency": "GBP",
                "event": "BoE Interest Rate Decision",
                "impact": "high",
                "forecast": "4.50%",
                "previous": "4.75%",
            },
        ]
        response = {"events": events}
        _set_cache("economic_calendar", response)
        return response


# ─── Market Sentiment ─────────────────────────────────────────────────────────
@router.get("/market-sentiment")
async def get_market_sentiment():
    """Return bull/bear sentiment and top movers from yfinance."""
    cached = _cached("market_sentiment")
    if cached:
        return cached

    # Crypto + Forex majors + Commodities
    SYMBOLS = [
        # Crypto
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
        # Forex (yfinance uses =X suffix)
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X",
        # Commodities
        "GC=F",  # Gold futures
    ]
    top_movers = []
    positive_count = 0
    negative_count = 0
    neutral_count   = 0

    try:
        import yfinance as yf
        import concurrent.futures
        loop = asyncio.get_event_loop()

        def _fetch_tickers():
            tickers = yf.Tickers(" ".join(SYMBOLS))
            results = []
            for sym in SYMBOLS:
                try:
                    t = tickers.tickers[sym]
                    info = t.fast_info
                    price      = float(info.last_price or 0)
                    prev_close = float(info.previous_close or price)
                    if prev_close == 0:
                        continue
                    pct = ((price - prev_close) / prev_close) * 100
                    # Clean up display name
                    display_sym = sym
                    asset_class = "crypto"
                    if sym.endswith("=X"):
                        display_sym = sym.replace("=X", "")
                        # Format as EUR/USD style
                        if len(display_sym) == 6:
                            display_sym = display_sym[:3] + "/" + display_sym[3:]
                        asset_class = "forex"
                    elif sym.endswith("=F"):
                        display_sym = {"GC=F": "GOLD"}.get(sym, sym.replace("=F", ""))
                        asset_class = "commodity"
                    else:
                        display_sym = sym.replace("-USD", "")
                        asset_class = "crypto"
                    results.append({
                        "symbol": display_sym,
                        "price":  round(price, 4),
                        "change_pct": round(pct, 2),
                        "asset_class": asset_class,
                    })
                except Exception:
                    pass
            return results

        with concurrent.futures.ThreadPoolExecutor() as pool:
            top_movers = await loop.run_in_executor(pool, _fetch_tickers)

        for m in top_movers:
            if m["change_pct"] > 0:
                positive_count += 1
            elif m["change_pct"] < 0:
                negative_count += 1
            else:
                neutral_count += 1

    except Exception as e:
        logger.warning(f"Market sentiment (yfinance) failed: {e}, falling back to Binance")
        try:
            from backend.services.binance_market_data import binance_market_data
            tickers = await binance_market_data.get_all_tickers_24h(
                ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
            )
            top_movers = []
            for t in tickers:
                pct = t.get("priceChangePercent", 0.0)
                sym = t.get("symbol", "").replace("USDT", "").replace("USDC", "")
                top_movers.append({
                    "symbol": sym,
                    "price": round(t.get("lastPrice", 0), 4),
                    "change_pct": round(pct, 2),
                    "direction": "up" if pct >= 0 else "down",
                })
                if pct > 0:
                    positive_count += 1
                elif pct < 0:
                    negative_count += 1
                else:
                    neutral_count += 1
        except Exception as e2:
            logger.warning(f"Binance fallback also failed: {e2}")
            top_movers = []

    total = max(positive_count + negative_count + neutral_count, 1)
    response = {
        "bull_pct":    round((positive_count / total) * 100, 1),
        "bear_pct":    round((negative_count / total) * 100, 1),
        "neutral_pct": round((neutral_count  / total) * 100, 1),
        "top_movers":  sorted(top_movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:8],
        "total_tracked": total,
    }
    _set_cache("market_sentiment", response)
    return response


# ─── GET /api/news/sentiment — n8n WF4 reads latest Influx scores ─────────────
@router.get("/sentiment")
async def get_news_sentiment(
    lookback_minutes: int = Query(default=180, ge=1, le=1440),
):
    """Latest per-symbol sentiment for n8n enrichment (WF4) and dashboards."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    output = []
    data = []
    for symbol in symbols:
        try:
            row = sentiment_reader.get_sentiment(symbol, lookback_minutes=lookback_minutes) or {}
        except Exception as exc:
            logger.warning(f"Sentiment read failed for {symbol}: {exc}")
            row = {}
        direction = str(row.get("direction") or "NEUTRAL").upper()
        if direction == "BUY":
            direction = "BULLISH"
        elif direction == "SELL":
            direction = "BEARISH"
        item = {
            "symbol": symbol,
            "sentiment_score": float(row.get("sentiment_score") or 0.0),
            "direction": direction,
            "confidence": float(row.get("confidence") or 0.0),
            "price_change_pct": 0.0,
        }
        data.append(item)
        output.append({"json": item})
    return {
        "status": "ok",
        "count": len(data),
        "data": data,
        "output": output,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── POST /api/news/sentiment - n8n pushes here ───────────────────────────────
@router.post("/sentiment")
async def receive_news_sentiment(payload: SentimentPayload):
    """
    Receive AI-analyzed sentiment data from n8n workflow.
    Writes to InfluxDB news-sentiment bucket via existing influxdb_writer.
    """
    try:
        from backend.services.influxdb_writer import influx
        
        await influx.write_news_sentiment(
            symbol=payload.symbol,
            sentiment_score=payload.sentiment_score,
            impact_score=payload.impact_score,
            source=payload.source,
            time_horizon=payload.time_horizon,
            topics=payload.topics,
        )
        
        logger.info(f"Stored sentiment for {payload.symbol}: score={payload.sentiment_score}")
        return {
            "status": "stored",
            "symbol": payload.symbol,
            "sentiment_score": payload.sentiment_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to store sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── POST /api/news/archive - n8n pushes filtered news + embeddings ────────────
@router.post("/archive")
async def archive_news(payload: NewsArchivePayload):
    """
    Receive filtered news with embeddings from n8n workflow.
    Stores in Qdrant for vector search.
    """
    try:
        # Only store if embedding provided (AI-filtered)
        if not payload.embedding:
            logger.warning("Archive called without embedding - skipping storage")
            return {"status": "skipped", "reason": "no_embedding"}
        
        from backend.services.qdrant_client import qdrant
        
        point_id = await qdrant.store_news_article(
            title=payload.title,
            content=payload.content,
            source=payload.source,
            url=payload.url,
            published_at=payload.published_at,
            sentiment=payload.sentiment,
            symbols=payload.symbols,
            embedding=payload.embedding,
        )
        
        logger.info(f"Archived news: {payload.title[:50]}... -> Qdrant point {point_id}")
        return {
            "status": "archived",
            "qdrant_point_id": point_id,
            "title": payload.title[:100],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except ImportError:
        # Qdrant client not available yet - log but don't fail
        logger.warning("Qdrant client not initialized - archive endpoint unavailable")
        return {"status": "unavailable", "reason": "qdrant_not_configured"}
    except Exception as e:
        logger.error(f"Failed to archive news: {e}")
        err_msg = str(e).lower()
        if "connection" in err_msg or "timeout" in err_msg or "refused" in err_msg or "unreachable" in err_msg:
            raise HTTPException(status_code=503, detail="Qdrant vector database is unreachable")
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/news/status ────────────────────────────────────────────────────
@router.get("/status")
async def news_status():
    """News service status and Qdrant collection info."""
    from backend.services.qdrant_client import qdrant
    try:
        info = await qdrant.get_collection_info()
    except Exception:
        info = {}
    return {"status": "ok", "qdrant": info}


# ─── GET /api/news/archive/history - browse Qdrant n8n docs ─────────────────
@router.get("/archive/history")
async def archive_history(page: int = 1, limit: int = 20):
    """Return paginated list of n8n-archived news analyses from Qdrant."""
    from backend.services.qdrant_client import qdrant
    try:
        offset = (page - 1) * limit
        pts, _ = await qdrant._client.scroll(
            collection_name=qdrant.collection_name,
            limit=limit,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        ) if qdrant._client else ([], None)
        items = []
        for p in pts:
            pl = p.payload or {}
            content = pl.get("content") or pl.get("text") or ""
            items.append({
                "id": p.id,
                "content_preview": content[:200],
                "metadata": pl.get("metadata", {}),
            })
        return {"page": page, "limit": limit, "items": items, "count": len(items)}
    except Exception as e:
        return {"page": page, "limit": limit, "items": [], "count": 0, "error": str(e)}


# ─── GET /api/news/search - semantic search via Qdrant ───────────────────────
@router.get("/search")
async def search_news(q: str, limit: int = 10):
    """
    Search news archive via semantic similarity.
    """
    try:
        from backend.services.qdrant_client import qdrant
        
        # Generate embedding for query if no embedding provided
        results = await qdrant.search_news(q, limit=limit)
        return {
            "query": q,
            "results": results,
            "count": len(results),
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Qdrant client not configured")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/news/history - paginated historical news ─────────────────────────
@router.get("/history")
async def get_news_history(page: int = 1, limit: int = 20):
    """
    Return paginated historical news from Qdrant (non-semantic).
    """
    try:
        from backend.services.qdrant_client import qdrant
        
        results = await qdrant.get_news_history(page=page, limit=limit)
        return {
            "page": page,
            "limit": limit,
            "results": results,
            "count": len(results),
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Qdrant client not configured")
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Google Drive Workflow Stubs ────────────────────────────────────────────────
class GoogleDriveArchivePayload(BaseModel):
    """Payload for Google Drive workflow to archive historical news."""
    articles: List[NewsArchivePayload] = Field(default_factory=list)
    batch_id: Optional[str] = None
    source: str = Field(default="google_drive")


@router.post("/gdrive/archive")
async def archive_google_drive_news(payload: GoogleDriveArchivePayload):
    """
    Receive batch of archived news from Google Drive workflow.
    Stores in Qdrant for historical vector search.
    Only stores AI-filtered articles (those with embeddings).
    """
    try:
        from backend.services.qdrant_client import qdrant
        
        stored = []
        skipped = []
        
        for article in payload.articles:
            if not article.embedding:
                skipped.append({"title": article.title, "reason": "no_embedding"})
                continue
            
            point_id = await qdrant.store_news_article(
                title=article.title,
                content=article.content,
                source=article.source,
                url=article.url,
                published_at=article.published_at,
                sentiment=article.sentiment,
                symbols=article.symbols,
                embedding=article.embedding,
            )
            stored.append({"title": article.title[:100], "qdrant_point_id": point_id})
        
        return {
            "status": "processed",
            "batch_id": payload.batch_id,
            "stored_count": len(stored),
            "skipped_count": len(skipped),
            "stored": stored[:5],  # Preview first 5
            "skipped_preview": skipped[:5],
        }
    except ImportError:
        logger.warning("Qdrant client not initialized - gdrive archive endpoint unavailable")
        return {"status": "unavailable", "reason": "qdrant_not_configured"}
    except Exception as e:
        logger.error(f"Failed to archive Google Drive news: {e}")
        err_msg = str(e).lower()
        if "connection" in err_msg or "timeout" in err_msg or "refused" in err_msg or "unreachable" in err_msg:
            raise HTTPException(status_code=503, detail="Qdrant vector database is unreachable")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Native sentiment loop: status + manual trigger ───────────────────────────
@router.get("/sentiment-loop/status")
async def sentiment_loop_status():
    """Status of the repo-side native sentiment loop (replaces n8n pipeline)."""
    try:
        from backend.services.sentiment_loop import sentiment_loop
        return sentiment_loop.status()
    except Exception as e:
        return {"state": "unavailable", "error": str(e)}


@router.post("/sentiment-loop/run")
async def sentiment_loop_run(dry_run: bool = False):
    """Trigger one sentiment pass on demand.

    dry_run=true computes per-coin sentiment WITHOUT writing to InfluxDB —
    handy for verifying the pipeline produces all symbols before going live.
    """
    try:
        from backend.services.sentiment_loop import sentiment_loop
        summary = await sentiment_loop.run_once(dry_run=dry_run)
        return {"status": "ok", "summary": summary}
    except Exception as e:
        logger.error(f"Sentiment loop manual run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-alerts/status")
async def market_alerts_status():
    """Status of the native market alerts loop."""
    try:
        from backend.services.market_alerts import market_alerts_loop
        return market_alerts_loop.status()
    except Exception as e:
        return {"running": False, "error": str(e)}


@router.post("/market-alerts/run")
async def market_alerts_run(dry_run: bool = False, skip_telegram: bool = False):
    """Trigger one market alerts cycle on demand."""
    try:
        from backend.services.market_alerts import market_alerts_loop
        summary = await market_alerts_loop.run_once(dry_run=dry_run, skip_telegram=skip_telegram)
        return {"status": "ok", "dry_run": dry_run, "skip_telegram": skip_telegram, "summary": summary}
    except Exception as e:
        logger.error(f"Market alerts manual run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gdrive/status")
async def gdrive_archive_status():
    """Check Google Drive archive integration status."""
    try:
        from backend.services.qdrant_client import qdrant
        collection_info = await qdrant.get_collection_info()
        return {
            "status": "ready",
            "collection": collection_info.get("name"),
            "points_count": collection_info.get("points_count", 0),
            "vector_size": collection_info.get("vector_size", 1536),
        }
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}