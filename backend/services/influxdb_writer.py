"""
InfluxDB Writer Service
Writes trading metrics to InfluxDB buckets for Grafana dashboards.

Buckets:
  trading-raw      – OHLCV tick data (30d retention)
  trading-signals  – AI/strategy signals (90d retention)
  trading-orders   – Trades, fills, PnL (1y retention)
  trading-memory   – Agent state / inter-layer memory (unlimited)
  trading-system   – System health, latency, errors (7d retention)
  news-sentiment   – Processed news/sentiment from n8n (90d retention)
"""

import asyncio
import logging
import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class InfluxDBWriter:
    """Async InfluxDB v2 writer using line protocol over HTTP."""

    BUCKET_SIGNALS = "trading-signals"
    BUCKET_ORDERS  = "trading-orders"
    BUCKET_RAW     = "trading-raw"
    BUCKET_MEMORY  = "trading-memory"
    BUCKET_SYSTEM  = "trading-system"
    BUCKET_NEWS    = "news-sentiment"

    def __init__(self):
        self.url   = os.getenv("INFLUXDB_URL", "").rstrip("/")
        self.token = os.getenv("INFLUXDB_TOKEN", "")
        self.org   = os.getenv("INFLUXDB_ORG", "-")
        self._enabled = bool(self.token and self.url)
        if not self._enabled:
            logger.warning("InfluxDB token not set – metrics disabled")

    # ─────────────────────────── public API ───────────────────────────── #

    async def write_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        strategy: str = "combined",
        ai_used: bool = False,
        signal_id: Optional[int] = None,
    ) -> None:
        """Write a trading signal to trading-signals bucket."""
        tags = {
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy,
            "ai_used": str(ai_used).lower(),
        }
        fields: dict[str, Any] = {
            "confidence": float(confidence),
            "entry_price": float(entry_price),
        }
        if stop_loss is not None:
            fields["stop_loss"] = float(stop_loss)
        if take_profit is not None:
            fields["take_profit"] = float(take_profit)
        if signal_id is not None:
            fields["signal_id"] = int(signal_id)

        await self._write(self.BUCKET_SIGNALS, "trading_signal", tags, fields)

    async def write_trade(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        status: str,
        strategy: str = "combined",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        pnl: Optional[float] = None,
        trade_id: Optional[int] = None,
    ) -> None:
        """Write a trade/order event to trading-orders bucket."""
        tags = {
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy,
            "status": status,
        }
        fields: dict[str, Any] = {
            "quantity": float(quantity),
            "entry_price": float(entry_price),
        }
        if stop_loss is not None:
            fields["stop_loss"] = float(stop_loss)
        if take_profit is not None:
            fields["take_profit"] = float(take_profit)
        if pnl is not None:
            fields["pnl"] = float(pnl)
        if trade_id is not None:
            fields["trade_id"] = int(trade_id)

        await self._write(self.BUCKET_ORDERS, "trade", tags, fields)

    async def write_ohlcv(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timeframe: str = "5m",
        ts: Optional[int] = None,
    ) -> None:
        """Write OHLCV bar to trading-raw bucket."""
        tags = {"symbol": symbol, "timeframe": timeframe}
        fields: dict[str, Any] = {
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
        await self._write(self.BUCKET_RAW, "ohlcv", tags, fields, ts=ts)

    async def write_portfolio_snapshot(
        self,
        cash: float,
        equity: float,
        margin_used: float,
        open_positions: int,
        cycle: int,
    ) -> None:
        """Write portfolio snapshot to trading-orders bucket."""
        tags = {"source": "portfolio"}
        fields: dict[str, Any] = {
            "cash": float(cash),
            "equity": float(equity),
            "margin_used": float(margin_used),
            "open_positions": int(open_positions),
            "cycle": int(cycle),
        }
        await self._write(self.BUCKET_ORDERS, "portfolio_snapshot", tags, fields)

    async def write_system_health(
        self,
        cycle: int,
        symbols_scanned: int,
        signals_generated: int,
        trades_executed: int,
        cycle_duration_ms: float,
        errors: int = 0,
        state: str = "running",
    ) -> None:
        """Write system health metrics to trading-system bucket."""
        tags = {"state": state}
        fields: dict[str, Any] = {
            "cycle": int(cycle),
            "symbols_scanned": int(symbols_scanned),
            "signals_generated": int(signals_generated),
            "trades_executed": int(trades_executed),
            "cycle_duration_ms": float(cycle_duration_ms),
            "errors": int(errors),
        }
        await self._write(self.BUCKET_SYSTEM, "system_health", tags, fields)

    async def write_performance(
        self,
        equity: float,
        realized_pnl: float,
        win_rate: float,
        wins: int,
        losses: int,
        total_trades: int,
        drawdown_pct: float,
        open_positions: int,
        cycle: int = 0,
    ) -> None:
        """Write rolling performance metrics (win rate, drawdown, equity) to
        trading-system bucket so Grafana can chart bot performance over time."""
        fields: dict[str, Any] = {
            "equity": float(equity),
            "realized_pnl": float(realized_pnl),
            "win_rate": float(win_rate),
            "wins": int(wins),
            "losses": int(losses),
            "total_trades": int(total_trades),
            "drawdown_pct": float(drawdown_pct),
            "open_positions": int(open_positions),
            "cycle": int(cycle),
        }
        await self._write(self.BUCKET_SYSTEM, "performance", {}, fields)

    async def write_agent_state(
        self,
        agent: str,
        symbol: str,
        direction: str,
        confidence: float,
        reasoning: str = "",
        session_id: Optional[str] = None,
    ) -> None:
        """Write agent state to trading-memory bucket (inter-layer communication)."""
        tags = {"agent": agent, "symbol": symbol, "direction": direction}
        fields: dict[str, Any] = {
            "confidence": float(confidence),
            "reasoning_len": len(reasoning),
        }
        if session_id:
            fields["session_id"] = f'"{session_id}"'

        await self._write(self.BUCKET_MEMORY, "agent_state", tags, fields)

    async def write_news_sentiment(
        self,
        symbol: str,
        sentiment_score: float,
        impact_score: float,
        source: str = "newsapi",
        time_horizon: str = "short",
        topics: str = "",
        confidence: float = 0.0,
        direction: str = "NEUTRAL",
    ) -> None:
        """Write processed news sentiment to news-sentiment bucket.

        Includes `confidence` and `direction` tag so that the InfluxDB
        sentiment reader can read them directly instead of re-deriving.
        """
        # Direction tag must be one of the values the reader understands
        _valid = {"BULLISH", "BEARISH", "NEUTRAL", "BUY", "SELL"}
        direction_tag = direction.upper() if direction.upper() in _valid else "NEUTRAL"

        tags = {"symbol": symbol, "source": source, "time_horizon": time_horizon,
                "direction": direction_tag}
        fields: dict[str, Any] = {
            "sentiment_score": float(sentiment_score),
            "impact_score": float(impact_score),
            "confidence": float(confidence),
            "topics_len": len(topics),
        }
        await self._write(self.BUCKET_NEWS, "news_sentiment", tags, fields)

    async def write_global_sentiment(
        self,
        index_value: float,
        source: str = "alternative.me",
    ) -> None:
        """Write global macro sentiment (Fear & Greed) to news-sentiment bucket."""
        tags = {"source": source}
        fields: dict[str, Any] = {"index": float(index_value)}
        await self._write(self.BUCKET_NEWS, "global_sentiment", tags, fields)

    async def write_market_alert(
        self,
        symbol: str,
        alert_type: str,
        score: float,
        *,
        impact_score: Optional[float] = None,
        source: str = "native_loop",
    ) -> None:
        """Write market alert (trending/pump/dump) to news-sentiment bucket."""
        tags = {
            "symbol": symbol.upper(),
            "alert_type": alert_type.lower(),
            "source": source,
        }
        impact = float(impact_score if impact_score is not None else score)
        fields: dict[str, Any] = {
            "score": float(score),
            "impact_score": impact,
        }
        await self._write(self.BUCKET_NEWS, "market_alert", tags, fields)

    async def write_onchain_signal(
        self,
        symbol: str,
        score: float,
        direction: str = "NEUTRAL",
        whale_notional: float = 0.0,
        oi_change_pct: float = 0.0,
        funding_rate: float = 0.0,
        impact_score: float = 0.0,
        source: str = "n8n-onchain",
    ) -> None:
        """Write an on-chain / whale-activity signal to news-sentiment."""
        tags = {
            "symbol": symbol.upper(),
            "source": source,
            "direction": self._normalize_direction(direction),
        }
        fields: dict[str, Any] = {
            "score": float(score),
            "whale_notional": float(whale_notional),
            "oi_change_pct": float(oi_change_pct),
            "funding_rate": float(funding_rate),
            "impact_score": float(impact_score),
        }
        await self._write(self.BUCKET_NEWS, "onchain_signal", tags, fields)

    async def write_macro_signal(
        self,
        symbol: str = "BTCUSDT",
        score: float = 0.0,
        direction: str = "NEUTRAL",
        btc_price: float = 0.0,
        sp500_price: float = 0.0,
        gold_price: float = 0.0,
        dxy: float = 0.0,
        vix: float = 0.0,
        gold_btc_ratio: float = 0.0,
        risk_regime: str = "NEUTRAL",
        source: str = "n8n-macro",
    ) -> None:
        """Write a macro / cross-asset signal to news-sentiment."""
        tags = {
            "symbol": symbol.upper(),
            "source": source,
            "direction": self._normalize_direction(direction),
            "risk_regime": str(risk_regime or "NEUTRAL"),
        }
        fields: dict[str, Any] = {
            "score": float(score),
            "btc_price": float(btc_price),
            "sp500_price": float(sp500_price),
            "gold_price": float(gold_price),
            "dxy": float(dxy),
            "vix": float(vix),
            "gold_btc_ratio": float(gold_btc_ratio),
        }
        await self._write(self.BUCKET_NEWS, "macro_signal", tags, fields)

    async def write_technical_signal(
        self,
        symbol: str,
        score: float,
        direction: str = "NEUTRAL",
        rsi: float = 50.0,
        macd_histogram: float = 0.0,
        price_change_pct: float = 0.0,
        funding_rate: float = 0.0,
        confidence: float = 0.0,
        source: str = "n8n-technical",
    ) -> None:
        """Write a technical / indicator-divergence signal to news-sentiment."""
        tags = {
            "symbol": symbol.upper(),
            "source": source,
            "direction": self._normalize_direction(direction),
        }
        fields: dict[str, Any] = {
            "score": float(score),
            "rsi": float(rsi),
            "macd_histogram": float(macd_histogram),
            "price_change_pct": float(price_change_pct),
            "funding_rate": float(funding_rate),
            "confidence": float(confidence),
        }
        await self._write(self.BUCKET_NEWS, "technical_signal", tags, fields)

    async def write_divergence_alert(
        self,
        symbol: str,
        score: float,
        direction: str = "NEUTRAL",
        sentiment_score: float = 0.0,
        price_change_pct: float = 0.0,
        confidence: float = 0.0,
        signal: str = "NEUTRAL",
        source: str = "n8n-divergence",
    ) -> None:
        """Write a sentiment-vs-price divergence alert to news-sentiment."""
        tags = {
            "symbol": symbol.upper(),
            "source": source,
            "direction": self._normalize_direction(direction),
            "signal": str(signal or "NEUTRAL"),
        }
        fields: dict[str, Any] = {
            "score": float(score),
            "sentiment_score": float(sentiment_score),
            "price_change_pct": float(price_change_pct),
            "confidence": float(confidence),
        }
        await self._write(self.BUCKET_NEWS, "divergence_alert", tags, fields)

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        valid = {"BULLISH", "BEARISH", "NEUTRAL", "BUY", "SELL"}
        tag = (direction or "NEUTRAL").upper()
        return tag if tag in valid else "NEUTRAL"

    async def write_feed_snapshot(self, snapshot: dict) -> None:
        """Write unified-feed dashboard snapshot summary to trading-system bucket.

        Expects the /api/feed/overview payload ({as_of, crypto, equities,
        metals}); records per-class quote counts and stale counts. No-ops
        when InfluxDB is disabled (handled by _write).
        """
        def _quotes(key: str) -> list:
            return list(snapshot.get(key) or [])

        def _live(quotes: list) -> int:
            return sum(1 for q in quotes if q.get("price") is not None and not q.get("stale"))

        def _stale(quotes: list) -> int:
            return sum(1 for q in quotes if q.get("stale"))

        crypto, equities, metals = _quotes("crypto"), _quotes("equities"), _quotes("metals")
        fields: dict[str, Any] = {
            "crypto_count": int(_live(crypto)),
            "equities_count": int(_live(equities)),
            "metals_count": int(_live(metals)),
            "stale_count": int(_stale(crypto) + _stale(equities) + _stale(metals)),
        }
        await self._write(
            self.BUCKET_SYSTEM, "feed_snapshot",
            {"source": "feed_scheduler"}, fields,
        )

    # ─────────────────────────── internals ────────────────────────────── #

    @staticmethod
    def _escape_tag(v: str) -> str:
        return str(v).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    @staticmethod
    def _field_value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return f"{v}i"
        if isinstance(v, float):
            return f"{v:.6f}"
        # Already-formatted string field (e.g. quoted)
        if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
            return v
        return f'"{str(v)}"'

    def _build_line(self, measurement: str, tags: dict, fields: dict, ts: Optional[int] = None) -> str:
        tag_str = ",".join(
            f"{self._escape_tag(k)}={self._escape_tag(v)}"
            for k, v in sorted(tags.items())
        )
        field_str = ",".join(
            f"{k}={self._field_value(v)}"
            for k, v in fields.items()
        )
        # Line protocol: omit the comma entirely when there are no tags,
        # otherwise InfluxDB rejects "measurement, field=.." as a parse error.
        prefix = f"{measurement},{tag_str}" if tag_str else measurement
        line = f"{prefix} {field_str}"
        if ts is not None:
            line += f" {ts}"
        return line

    async def _write(self, bucket: str, measurement: str, tags: dict, fields: dict, ts: Optional[int] = None) -> None:
        if not self._enabled:
            return
        if not fields:
            return

        line = self._build_line(measurement, tags, fields, ts)
        url = f"{self.url}/api/v2/write"
        params = {"org": self.org, "bucket": bucket, "precision": "ns" if ts else "s"}
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "text/plain; charset=utf-8",
        }

        max_retries = 3
        backoff = 0.5  # start with 500ms
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, params=params, headers=headers, content=line)
                    if resp.status_code in (200, 204):
                        return
                    else:
                        logger.warning(
                            f"InfluxDB write error [{bucket}] (attempt {attempt + 1}/{max_retries}): "
                            f"{resp.status_code} {resp.text[:200]}"
                        )
            except Exception as exc:
                logger.warning(
                    f"InfluxDB write failed [{bucket}] (attempt {attempt + 1}/{max_retries}): {exc}"
                )
            
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2  # 0.5s -> 1s -> 2s

    async def write_binance_wallet(self, balance: float, available: float,
                                    equity: float, unrealized_pnl: float,
                                    margin_used: float) -> None:
        """Write Binance Futures wallet snapshot to trading-system bucket."""
        await self._write(
            bucket='trading-system',
            measurement='binance_wallet',
            tags={'broker': 'binance_futures', 'account': 'live'},
            fields={
                'balance': float(balance),
                'available': float(available),
                'equity': float(equity),
                'unrealized_pnl': float(unrealized_pnl),
                'margin_used': float(margin_used),
            }
        )

    async def write_binance_position(self, symbol: str, side: str,
                                      quantity: float, entry_price: float,
                                      unrealized_pnl: float, leverage: int,
                                      mark_price: float = 0.0,
                                      liquidation_price: float = 0.0) -> None:
        """Write a single Binance Futures open position to trading-system bucket."""
        await self._write(
            bucket='trading-system',
            measurement='binance_position',
            tags={'symbol': symbol, 'side': side, 'broker': 'binance_futures'},
            fields={
                'quantity': float(quantity),
                'entry_price': float(entry_price),
                'unrealized_pnl': float(unrealized_pnl),
                'leverage': float(leverage),
                'mark_price': float(mark_price),
                'liquidation_price': float(liquidation_price),
                'notional_value': float(quantity * (mark_price or entry_price)),
            }
        )


# Singleton instance
influx = InfluxDBWriter()
