"""
Opinion Layer — Multi-Agent Market Analysis Pipeline
====================================================
Integrates:
  1. ai-hedge-fund agents (technical, sentiment, risk, portfolio)
  2. Kronos foundation model forecast
  3. n8n social sentiment (X/Reddit/Discord/News)
  4. Market alerts (trending/pumps/whales)

Produces a unified TradingOpinion with direction, confidence, and reasoning.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import sqlalchemy as sa
from backend.database.connection import SessionLocal
from backend.database.models import Trade

from backend.services import kronos_service
from backend.services.influxdb_sentiment_reader import sentiment_reader
from backend.services.qdrant_client import qdrant
from backend.services.trade_memory import trade_memory
from backend.services.skill_miner import skill_miner
from backend.services.persona_adapter import run_all_personas, get_persona_weights, set_persona_weight
from backend.jev.opinion import evaluate_opinion, should_skip_personas

logger = logging.getLogger(__name__)

def _get_trade_memory(symbol: str, limit: int = 5) -> dict:
    try:
        with SessionLocal() as db:
            trades = db.query(Trade).filter(
                sa.and_(Trade.symbol == symbol, Trade.status == "closed")
            ).order_by(Trade.closed_at.desc()).limit(limit).all()
            if not trades:
                return {"count": 0, "summary": "No recent closed trades"}
            total_pnl = sum((t.pnl or 0) for t in trades)
            wins = sum(1 for t in trades if (t.pnl or 0) > 0)
            avg_pnl = total_pnl / len(trades)
            summary = (
                f"Last {len(trades)} trades: {wins}W/{len(trades)-wins}L, "
                f"Avg PnL: {avg_pnl:+.4f}, Total: {total_pnl:+.4f}"
            )
            return {"count": len(trades), "wins": wins, "losses": len(trades)-wins,
                    "avg_pnl": avg_pnl, "total_pnl": total_pnl, "summary": summary}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Trade memory query failed for {symbol}: {e}")
        return {"count": 0, "summary": f"Error: {e}"}


def _build_market_context(
    symbol: str,
    opinions: list,
    kronos_result: dict,
    metrics: Optional[dict],
) -> dict:
    """Assemble the market-context dict consumed by trade_memory.recall_similar
    from the opinions already computed this cycle. Best-effort: every field is
    optional and defaults to neutral inside the feature extractor."""
    metrics = metrics or {}
    ctx: dict = {"symbol": symbol}

    by_agent = {o.agent: o for o in opinions}

    tech = by_agent.get("technical_analyst")
    if tech is not None:
        md = tech.metadata or {}
        # Sub-signal dicts look like {"signal": "bullish", "confidence": 0.7}
        for key in ("trend", "momentum", "mean_reversion"):
            sub = md.get(key) or {}
            if isinstance(sub, dict) and sub.get("signal"):
                ctx[f"{key}_signal"] = sub["signal"]
        vol = md.get("volatility") or {}
        if isinstance(vol, dict) and isinstance(vol.get("confidence"), (int, float)):
            ctx["volatility"] = vol["confidence"]
        ctx["confidence"] = tech.confidence

    if kronos_result:
        ctx["kronos_change_pct"] = kronos_result.get("predicted_change_pct", 0.0)

    social = by_agent.get("social_sentiment")
    if social is not None:
        s = social.signal
        ctx["sentiment_score"] = 1.0 if s == "bullish" else (-1.0 if s == "bearish" else 0.0)

    # From metrics (set by the decision engine / loop)
    if "funding_rate" in metrics:
        ctx["funding_rate"] = metrics["funding_rate"]
    if "regime" in metrics:
        ctx["regime"] = metrics["regime"]
    if "rsi" in metrics:
        ctx["rsi"] = metrics["rsi"]

    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentOpinion:
    agent: str
    signal: str          # "bullish" | "bearish" | "neutral"
    confidence: float    # 0.0 – 1.0
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TradingOpinion:
    symbol: str
    direction: str       # "BUY" | "SELL" | "HOLD"
    confidence: float    # 0.0 – 1.0
    reasoning: str
    agent_opinions: List[AgentOpinion]
    kronos: dict = field(default_factory=dict)
    social: dict = field(default_factory=dict)
    alerts: List[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# Technical Analysis Adapter (uses existing hedge-fund agent logic)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_technical_opinion(bars: pd.DataFrame, symbol: str) -> AgentOpinion:
    """
    Compute technical opinion using the same logic as technical_analyst_agent
    but directly on pre-fetched Binance bars (no FINANCIAL_DATASETS_API_KEY needed).
    """
    try:
        # Ensure DataFrame has the right columns
        df = bars.copy()
        if 'close' not in df.columns and 'Close' in df.columns:
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

        # Import signal calculation functions from existing agent
        from backend.agents.technicals import (
            calculate_trend_signals, calculate_mean_reversion_signals,
            calculate_momentum_signals, calculate_volatility_signals,
            calculate_stat_arb_signals, weighted_signal_combination
        )

        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        trend = calculate_trend_signals(df)
        mean_rev = calculate_mean_reversion_signals(df)
        momentum = calculate_momentum_signals(df)
        volatility = calculate_volatility_signals(df)
        stat_arb = calculate_stat_arb_signals(df)

        combined = weighted_signal_combination({
            "trend": trend,
            "mean_reversion": mean_rev,
            "momentum": momentum,
            "volatility": volatility,
            "stat_arb": stat_arb,
        }, strategy_weights)

        sig = combined["signal"]
        conf = combined["confidence"]

        # Map to BUY/SELL/NEUTRAL for reasoning

        reasoning_parts = []
        for name, data in [
            ("Trend", trend), ("Mean Reversion", mean_rev),
            ("Momentum", momentum), ("Volatility", volatility),
            ("Stat Arb", stat_arb)
        ]:
            reasoning_parts.append(f"{name}: {data['signal']}({data['confidence']:.0%})")

        return AgentOpinion(
            agent="technical_analyst",
            signal=sig,
            confidence=conf,
            reasoning=f"Technical composite: {sig} | " + " | ".join(reasoning_parts),
            metadata={
                "trend": trend, "mean_reversion": mean_rev,
                "momentum": momentum, "volatility": volatility, "stat_arb": stat_arb
            }
        )
    except Exception as e:
        logger.warning(f"Technical opinion error for {symbol}: {e}")
        return AgentOpinion(
            agent="technical_analyst", signal="neutral", confidence=0.0,
            reasoning=f"Error: {e}"
        )


# NOTE: _run_kronos_opinion() was removed — Kronos is now handled inline in
# analyze_symbol() to avoid asyncio.run() inside an already-running event loop.


# ═══════════════════════════════════════════════════════════════════════════════
# Social Sentiment Opinion
# ═══════════════════════════════════════════════════════════════════════════════

def _run_social_opinion(symbol: str) -> AgentOpinion:
    """Get social sentiment opinion from n8n/InfluxDB."""
    try:
        sent = sentiment_reader.get_sentiment(symbol, lookback_minutes=60)
        if not sent:
            return AgentOpinion(
                agent="social_sentiment", signal="neutral", confidence=0.0,
                reasoning="No social sentiment data available"
            )

        direction = sent.get("direction", "NEUTRAL")
        signal_map = {"BUY": "bullish", "SELL": "bearish", "NEUTRAL": "neutral"}
        signal = signal_map.get(direction, "neutral")
        confidence = sent.get("confidence", 0.0)
        sources = sent.get("sources", {})

        def _fmt_src(k, v):
            try:
                return f"{k}({float(v):+.2f})"
            except (TypeError, ValueError):
                return f"{k}({v})"

        reasoning = (
            f"Social score {sent.get('sentiment_score', 0):+.3f} "
            f"from {sent.get('article_count', 0)} posts. "
            f"Sources: {', '.join(_fmt_src(k, v) for k, v in list(sources.items())[:3])}"
        )

        return AgentOpinion(
            agent="social_sentiment",
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata=sent
        )
    except Exception as e:
        logger.warning(f"Social opinion error for {symbol}: {e}")
        return AgentOpinion(
            agent="social_sentiment", signal="neutral", confidence=0.0,
            reasoning=f"Error: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Market Alert Opinion
# ═══════════════════════════════════════════════════════════════════════════════

def _run_alert_opinion(symbol: str) -> AgentOpinion:
    """Get market alert opinion (trending/pumps/whales)."""
    try:
        interval = int(os.getenv("MARKET_ALERTS_INTERVAL_MINUTES", "120"))
        lookback = int(os.getenv("MARKET_ALERTS_LOOKBACK_MINUTES", str(interval + 30)))
        alerts = sentiment_reader.get_market_alerts(symbol, lookback_minutes=lookback)
        if not alerts:
            return AgentOpinion(
                agent="market_alerts", signal="neutral", confidence=0.0,
                reasoning="No market alerts"
            )

        # Score alerts: pump = bullish, dump = bearish
        bullish_score = 0.0
        bearish_score = 0.0
        reasons = []
        for a in alerts[:5]:
            atype = a.get("alert_type", "").lower()
            score = a.get("score", 0)
            if "pump" in atype or "trending" in atype or "outflow" in atype:
                bullish_score += score
                reasons.append(f"{atype}(+{score})")
            elif "dump" in atype or "inflow" in atype or "crash" in atype:
                bearish_score += score
                reasons.append(f"{atype}(-{score})")

        if bullish_score > bearish_score * 1.5:
            signal = "bullish"
            confidence = min(bullish_score / 200, 1.0)
        elif bearish_score > bullish_score * 1.5:
            signal = "bearish"
            confidence = min(bearish_score / 200, 1.0)
        else:
            signal = "neutral"
            confidence = 0.0

        return AgentOpinion(
            agent="market_alerts",
            signal=signal,
            confidence=confidence,
            reasoning="Alerts: " + ", ".join(reasons) if reasons else "No directional alerts",
            metadata={"alerts": alerts}
        )
    except Exception as e:
        logger.warning(f"Alert opinion error for {symbol}: {e}")
        return AgentOpinion(
            agent="market_alerts", signal="neutral", confidence=0.0,
            reasoning=f"Error: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Macro Sentiment Opinion (global Fear & Greed from InfluxDB)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_macro_sentiment_opinion() -> AgentOpinion:
    """Global market sentiment (Fear & Greed) from InfluxDB / n8n.

    get_global_sentiment() already applies contrarian logic (extreme greed ->
    SELL, extreme fear -> BUY) and returns a 0-100 index.
    """
    try:
        g = sentiment_reader.get_global_sentiment(lookback_minutes=240)
        if not g:
            return AgentOpinion(
                agent="macro_sentiment", signal="neutral", confidence=0.0,
                reasoning="No global sentiment data",
            )
        signal_map = {"BUY": "bullish", "SELL": "bearish", "NEUTRAL": "neutral"}
        signal = signal_map.get(g.get("direction", "NEUTRAL"), "neutral")
        idx = g.get("index", 50.0)
        return AgentOpinion(
            agent="macro_sentiment",
            signal=signal,
            confidence=float(g.get("confidence", 0.0) or 0.0),
            reasoning=f"Fear&Greed index {idx:.0f} → {g.get('direction', 'NEUTRAL')} (contrarian)",
            metadata=g,
        )
    except Exception as e:
        logger.warning(f"Macro sentiment opinion error: {e}")
        return AgentOpinion(
            agent="macro_sentiment", signal="neutral", confidence=0.0,
            reasoning=f"Error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# News Archive Opinion (Qdrant vector store written by n8n)
# ═══════════════════════════════════════════════════════════════════════════════

_COIN_NAMES = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "bnb",
    "XRP": "xrp", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche",
    "DOT": "polkadot", "LINK": "chainlink", "LTC": "litecoin", "UNI": "uniswap",
    "POL": "polygon", "ATOM": "cosmos", "NEAR": "near",
}
_FOREX_KEYWORDS = {
    "EUR": "euro", "USD": "dollar", "GBP": "pound", "JPY": "yen",
    "AUD": "australian dollar", "CAD": "canadian dollar", "CHF": "franc", "NZD": "kiwi",
}
_METAL_KEYWORDS = {"XAU": "gold", "XAG": "silver", "GOLD": "gold", "SILVER": "silver"}
_OIL_KEYWORDS = {"USOIL": "crude oil", "UKOIL": "brent oil", "WTI": "wti oil", "BRENT": "brent"}
_STOCK_TICKERS = {"AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META", "SPY", "QQQ"}
_BULL_WORDS = (
    "bullish", "rally", "surge", "breakout", "accumulat", "inflow", "upgrade",
    "rebound", "optimism", "approval", "support holds", "uptrend", "buy ",
    "all-time high", "catalyst", "adoption",
)
_BEAR_WORDS = (
    "bearish", "crash", "dump", "breakdown", "sell-off", "selloff", "outflow",
    "downgrade", "resistance", "decline", "fear", "liquidation", "weakness",
    "lows", "correction", "downtrend", "plunge", "capitulation",
)


async def _run_news_archive_opinion(symbol: str) -> AgentOpinion:
    """Directional signal from the Qdrant news archive (multi-asset keywords)."""
    try:
        from backend.services.multi_asset_bars import classify_symbol
        asset_class = classify_symbol(symbol)
        base = symbol.replace("USDT", "").replace("USDC", "").replace("PERP", "").upper()
        keywords = [base.lower(), symbol.lower()]
        if base in _COIN_NAMES:
            keywords.append(_COIN_NAMES[base])
        for part in (base[:3], base[3:6]) if len(base) >= 6 else ():
            if part in _FOREX_KEYWORDS:
                keywords.append(_FOREX_KEYWORDS[part])
        for key, label in {**_METAL_KEYWORDS, **_OIL_KEYWORDS}.items():
            if key in base:
                keywords.append(label)
        if base in _STOCK_TICKERS:
            keywords.append(base.lower())
        if asset_class == "forex":
            keywords.extend(["forex", "central bank", "interest rate"])
        elif asset_class == "metal":
            keywords.extend(["gold", "silver", "precious metal"])
        elif asset_class == "oil":
            keywords.extend(["crude", "oil", "opec", "inventory"])
        elif asset_class == "stock":
            keywords.extend(["earnings", "equity", "stock market"])
        docs = await qdrant.search_content(keywords, limit=5)
        if not docs:
            return AgentOpinion(
                agent="news_archive", signal="neutral", confidence=0.0,
                reasoning="No matching news in archive",
            )
        text = " ".join(d.get("content", "") for d in docs).lower()
        bull = sum(text.count(w) for w in _BULL_WORDS)
        bear = sum(text.count(w) for w in _BEAR_WORDS)
        total = bull + bear
        if total == 0:
            return AgentOpinion(
                agent="news_archive", signal="neutral", confidence=0.0,
                reasoning=f"{len(docs)} articles, no directional language",
            )
        score = (bull - bear) / total
        if score > 0.15:
            signal = "bullish"
        elif score < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"
        coverage = min(len(docs), 5) / 5.0
        confidence = round(min(abs(score) * (0.5 + 0.5 * coverage), 1.0), 3)
        return AgentOpinion(
            agent="news_archive",
            signal=signal,
            confidence=confidence,
            reasoning=f"{len(docs)} news analyses: {bull} bullish / {bear} bearish terms (score={score:+.2f})",
            metadata={"articles": len(docs), "bull": bull, "bear": bear},
        )
    except Exception as e:
        logger.warning(f"News archive opinion error for {symbol}: {e}")
        return AgentOpinion(
            agent="news_archive", signal="neutral", confidence=0.0,
            reasoning=f"Error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Opinion Aggregator (Portfolio Manager logic, simplified)
# ═══════════════════════════════════════════════════════════════════════════════

# Agent weights loaded from config at module level.
# Falls back to hardcoded defaults if YAML config is missing.
def _load_agent_config() -> dict:
    """Load agent_config.yaml and return the raw dict."""
    import yaml as _yaml
    _cfg_path = Path(__file__).resolve().parent.parent / "agents" / "config" / "agent_config.yaml"
    try:
        with open(_cfg_path, "r", encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load agent_config.yaml: {e}. Using defaults.")
        return {}

_LOADED_CONFIG = _load_agent_config()
_AGENT_WEIGHTS = _LOADED_CONFIG.get("base_agents", {
    "technical_analyst": 0.25,
    "kronos_foundation": 0.20,
    "social_sentiment": 0.20,
    "market_alerts": 0.05,
    "macro_sentiment": 0.10,
    "news_archive": 0.12,
    # Learning loop (Track C+) — modest weights; tune via /trading/opinion/weights.
    "semantic_trade_memory": 0.10,
    "learned_skill": 0.10,
    # FINMEM Cognitive Agent (Stevens Institute of Technology / arXiv:2311.13743v2)
    "finmem_cognitive_agent": 0.20,
    # Jev typed-eval (TypeSafe). Advisory only; unused until JEV_ANALYSIS_ENABLED.
    "jev_analyst": 0.16,
})
_AGG_CFG = _LOADED_CONFIG.get("aggregation", {})
_BUY_THRESHOLD = float(_AGG_CFG.get("buy_threshold", 0.15))
_SELL_THRESHOLD = float(_AGG_CFG.get("sell_threshold", -0.15))
_CONF_SCORE_W = float(_AGG_CFG.get("confidence_score_weight", 0.6))
_CONF_CONVICTION_W = float(_AGG_CFG.get("confidence_conviction_weight", 0.4))

_SIGNAL_MAP = {
    "bullish": 1,
    "neutral": 0,
    "bearish": -1,
}


_LEARNING_AGENT_FLOORS = {
    "semantic_trade_memory": 0.08,
    "learned_skill": 0.08,
    "trade_memory": 0.04,
}
_AGENT_WEIGHT_FLOOR = 0.02
_AGENT_WEIGHT_CAP = 0.35


def _get_combined_weights() -> dict:
    """Merge base agent weights with persona weights; keep learning agents live."""
    combined = dict(_AGENT_WEIGHTS)
    combined.update(get_persona_weights())
    for agent, floor in _LEARNING_AGENT_FLOORS.items():
        current = float(combined.get(agent) or 0.0)
        combined[agent] = max(current, floor)
    for agent, weight in list(combined.items()):
        combined[agent] = min(max(float(weight or 0.0), 0.0), _AGENT_WEIGHT_CAP)
    return combined


def _aggregate_opinions(
    symbol: str,
    opinions: List[AgentOpinion],
    kronos_result: dict,
    social_result: dict,
    alerts: List[dict],
) -> TradingOpinion:
    """
    Weighted-vote aggregation with confidence scaling.
    Adapted from portfolio_management_agent logic.
    """
    weighted_sum = 0.0
    total_possible_weight = 0.0
    conviction_sum = 0.0   # sum of weighted confidence magnitudes (direction-agnostic)
    engaged_weight = 0.0   # weight of agents that actually have a non-neutral opinion
    opinion_lines = []

    for op in opinions:
        weight = _get_combined_weights().get(op.agent, 0.05)
        numeric = _SIGNAL_MAP.get(op.signal, 0)
        
        # Directional score: bullish pushes positive, bearish negative
        weighted_sum += numeric * weight * op.confidence
        # Conviction: how sure any agent is (regardless of direction)
        conviction_sum += weight * op.confidence
        total_possible_weight += weight
        # Track weight of engaged agents (conf > 0 or non-neutral signal)
        if op.confidence > 0.0 or op.signal != "neutral":
            engaged_weight += weight
        
        opinion_lines.append(
            f"  • {op.agent}: {op.signal.upper()} (conf={op.confidence:.2f}, weight={weight})"
        )

    if total_possible_weight > 0:
        final_score = weighted_sum / total_possible_weight
        # Average conviction: divide by engaged weight so abstaining agents
        # don't dilute the conviction of agents that actually have an opinion.
        # Falls back to total_possible_weight if nobody engaged (avoids div/0).
        conviction_denom = engaged_weight if engaged_weight > 0.0 else total_possible_weight
        avg_conviction = conviction_sum / conviction_denom
    else:
        final_score = 0.0
        avg_conviction = 0.0

    # Map score to direction (thresholds from config)
    if final_score > _BUY_THRESHOLD:
        direction = "BUY"
    elif final_score < _SELL_THRESHOLD:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Confidence = directional clarity blended with average conviction
    # abs(final_score) measures directional consensus; avg_conviction measures overall engagement
    # Weights from config (default: 60% directional + 40% conviction)
    confidence = min(_CONF_SCORE_W * abs(final_score) + _CONF_CONVICTION_W * avg_conviction, 1.0)

    # Build reasoning
    reasoning = (
        f"Opinion Layer consensus: {direction} (score={final_score:+.3f}, conf={confidence:.2f})\n\n"
        f"Agent votes:\n" + "\n".join(opinion_lines) + "\n\n"
        f"Kronos: {kronos_result.get('signal', 'N/A')} "
        f"({kronos_result.get('predicted_change_pct', 0):+.2f}%)\n"
        f"Social: {social_result.get('direction', 'N/A')} "
        f"score={social_result.get('sentiment_score', 0):+.3f}\n"
        f"Alerts: {len(alerts)} active"
    )

    return TradingOpinion(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        reasoning=reasoning,
        agent_opinions=opinions,
        kronos=kronos_result,
        social=social_result,
        alerts=alerts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def analyze_symbol(
    symbol: str,
    bars: list[dict],
    include_kronos: bool = True,
    include_social: bool = True,
    include_alerts: bool = True,
    include_personas: bool = True,
    include_macro: bool = True,
    include_news: bool = True,
    metrics: Optional[dict] = None,
) -> TradingOpinion:
    """
    Run the full Opinion Layer pipeline for a symbol.

    Args:
        symbol: Trading symbol (e.g. "BTCUSDT")
        bars: List of OHLCV dicts from Binance
        include_kronos: Whether to run Kronos forecast
        include_social: Whether to fetch n8n social sentiment
        include_alerts: Whether to fetch market alerts
        include_personas: Whether to run hedge-fund persona agents
        metrics: Optional extra metrics (funding_rate, open_interest, etc.)

    Returns:
        TradingOpinion with unified direction/confidence/reasoning
    """
    started = datetime.now(timezone.utc)
    logger.info(f"Opinion Layer starting for {symbol}")

    df = pd.DataFrame(bars)
    if df.empty:
        return TradingOpinion(
            symbol=symbol, direction="HOLD", confidence=0.0,
            reasoning="Insufficient bar data",
            agent_opinions=[],
        )

    # Ensure column names match what technical agent expects
    col_map = {}
    for c in df.columns:
        lower = c.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[c] = lower
    if col_map:
        df = df.rename(columns=col_map)

    # Run all opinion generators concurrently
    opinions: List[AgentOpinion] = []
    kronos_result: dict = {}
    social_result: dict = {}
    alerts: List[dict] = []

    # Technical (sync, runs in thread pool)
    tech_op = await asyncio.to_thread(_run_technical_opinion, df, symbol)
    opinions.append(tech_op)

    # Kronos (async)
    if include_kronos:
        try:
            kronos_result = await kronos_service.predict(df, symbol)
            k_sig = str(kronos_result.get("signal", "NEUTRAL")).lower()
            if k_sig in ("up", "buy", "bullish"):
                k_sig = "bullish"
            elif k_sig in ("down", "sell", "bearish"):
                k_sig = "bearish"
            else:
                k_sig = "neutral"
            opinions.append(AgentOpinion(
                agent="kronos_foundation",
                signal=k_sig,
                confidence=kronos_result.get("confidence", 0.0),
                reasoning=f"Kronos: {kronos_result.get('predicted_change_pct', 0):+.2f}%",
                metadata=kronos_result,
            ))
        except Exception as e:
            logger.warning(f"Kronos failed for {symbol}: {e}")
            opinions.append(AgentOpinion(
                agent="kronos_foundation", signal="neutral", confidence=0.0,
                reasoning=f"Error: {e}"
            ))

    # Social sentiment (sync, InfluxDB query)
    if include_social:
        try:
            social_op = await asyncio.to_thread(_run_social_opinion, symbol)
            opinions.append(social_op)
            social_result = social_op.metadata
        except Exception as e:
            logger.warning(f"Social sentiment failed for {symbol}: {e}")

    # Market alerts (sync, InfluxDB query)
    if include_alerts:
        try:
            alert_op = await asyncio.to_thread(_run_alert_opinion, symbol)
            opinions.append(alert_op)
            alerts = alert_op.metadata.get("alerts", [])
        except Exception as e:
            logger.warning(f"Market alerts failed for {symbol}: {e}")

    # Global macro sentiment — Fear & Greed (InfluxDB, written by n8n)
    if include_macro:
        try:
            macro_op = await asyncio.to_thread(_run_macro_sentiment_opinion)
            opinions.append(macro_op)
        except Exception as e:
            logger.warning(f"Macro sentiment failed for {symbol}: {e}")

    # News archive — semantic/keyword recall from Qdrant (written by n8n)
    if include_news:
        try:
            news_op = await _run_news_archive_opinion(symbol)
            opinions.append(news_op)
        except Exception as e:
            logger.warning(f"News archive failed for {symbol}: {e}")

    # Persona agents (async LLM calls, concurrent)
    # Trade memory injection
    trade_mem = _get_trade_memory(symbol)
    if not metrics:
        metrics = {}
    metrics["trade_memory"] = trade_mem
    if trade_mem.get("count", 0) > 0:
        opinions.append(AgentOpinion(
            agent="trade_memory",
            signal="neutral",
            confidence=min(0.3, trade_mem.get("count", 0) * 0.06),
            reasoning=trade_mem.get("summary", ""),
            metadata=trade_mem,
        ))

    # Semantic trade memory (Track C) — "what happened last time the market
    # looked like this?". Recall the most similar historical setups from Qdrant
    # and turn their realised outcomes into a directional bias.
    try:
        recall_ctx = _build_market_context(symbol, opinions, kronos_result, metrics)
        recall = await trade_memory.recall_similar(recall_ctx, symbol=symbol)
        metrics["semantic_trade_memory"] = recall.to_dict()
        if recall.samples >= trade_memory.min_samples and recall.confidence > 0:
            opinions.append(AgentOpinion(
                agent="semantic_trade_memory",
                signal=recall.signal,
                confidence=recall.confidence,
                reasoning=recall.reasoning,
                metadata=recall.to_dict(),
            ))
    except Exception as e:
        logger.warning(f"Semantic trade memory failed for {symbol}: {e}")
        recall_ctx = _build_market_context(symbol, opinions, kronos_result, metrics)

    # Learned strategy skill (skill miner) — match the live setup to the best
    # mined archetype and vote with its historical edge.
    try:
        skill = await asyncio.to_thread(skill_miner.match_skill, recall_ctx)
        if skill:
            metrics["learned_skill"] = {
                k: skill.get(k) for k in (
                    "name", "direction", "win_rate", "avg_pnl",
                    "edge_score", "sample_count", "similarity", "skill_key",
                )
            }
            # Confidence = edge_score scaled by match similarity, capped 0.5.
            conf = min(0.5, float(skill.get("edge_score", 0.0)) * float(skill.get("similarity", 0.0)))
            if conf > 0 and skill.get("direction") in ("bullish", "bearish"):
                opinions.append(AgentOpinion(
                    agent="learned_skill",
                    signal=skill["direction"],
                    confidence=round(conf, 4),
                    reasoning=(
                        f"Matched skill '{skill.get('name')}' "
                        f"(sim {skill.get('similarity')}, edge {skill.get('edge_score')}, "
                        f"{skill.get('sample_count')} trades, win {skill.get('win_rate')})"
                    ),
                    metadata=metrics["learned_skill"],
                ))
    except Exception as e:
        logger.warning(f"Learned skill match failed for {symbol}: {e}")

    # Jev is the cheap typed-eval alternative to the persona LLM fan-out.
    # It votes only on a validated response. Failures leave the existing
    # agents in place and never invent a buy.
    jev_result = None
    try:
        jev_result = await evaluate_opinion(symbol, bars, metrics)
    except Exception as e:
        logger.warning(f"Jev opinion failed for {symbol}: {e}")
        jev_result = None
    # Uncalibrated Jev output is logged, not weighted, unless JEV_INFLUENCE_BOOK
    # is on and the calibration / meta / conformal gates all pass.
    if (
        jev_result
        and jev_result.get("influence_book")
        and jev_result.get("signal") in {"bullish", "bearish", "neutral"}
        and not jev_result.get("vetoed")
    ):
        opinions.append(AgentOpinion(
            agent="jev_analyst",
            signal=str(jev_result["signal"]),
            confidence=float(jev_result.get("confidence") or 0.0),
            reasoning=str(jev_result.get("reason") or ""),
            metadata={
                "advisory": True,
                "action": jev_result.get("action"),
                "answers": jev_result.get("answers"),
                "vetoed": False,
            },
        ))
        logger.info(
            "[%s] Jev analyst: %s conf=%.2f",
            symbol,
            jev_result.get("signal"),
            float(jev_result.get("confidence") or 0.0),
        )

    jev_status = jev_result.get("status") if isinstance(jev_result, dict) else None
    if should_skip_personas(include_personas, jev_status):
        logger.info("[%s] Jev replaced persona LLM agents for this cycle", symbol)
    elif include_personas:
        try:
            persona_results = await run_all_personas(symbol, bars, metrics)
            for po in persona_results:
                opinions.append(AgentOpinion(
                    agent=po.persona,
                    signal=po.signal,
                    confidence=po.confidence,
                    reasoning=po.reasoning,
                    metadata={"persona": po.persona},
                ))
        except Exception as e:
            logger.error(f"Persona agents failed for {symbol}: {e}")

    # FINMEM Cognitive Agent (Stevens Institute of Technology / arXiv:2311.13743v2)
    # Stratified layered memory (shallow/intermediate/deep) + self-adaptive dynamic character
    if os.getenv("FINMEM_ENABLED", "false").lower() == "true":
        try:
            from backend.services.finmem_service import finmem_service
            finmem_dec = await finmem_service.immediate_reflect(symbol, bars, external_context=metrics)
            sig = "bullish" if finmem_dec.action == "BUY" else ("bearish" if finmem_dec.action == "SELL" else "neutral")
            opinions.append(AgentOpinion(
                agent="finmem_cognitive_agent",
                signal=sig,
                confidence=finmem_dec.confidence,
                reasoning=f"[{finmem_dec.risk_character.upper()}] {finmem_dec.reasoning}",
                metadata={
                    "action": finmem_dec.action,
                    "character": finmem_dec.risk_character,
                    "cited_ids": finmem_dec.cited_memory_ids,
                    "retrieved_count": finmem_dec.retrieved_memory_count,
                    "latency_ms": finmem_dec.latency_ms,
                },
            ))
            logger.info(f"[{symbol}] FINMEM Cognitive Agent: {finmem_dec.action} (conf={finmem_dec.confidence:.2f}, char={finmem_dec.risk_character})")
        except Exception as e:
            logger.warning(f"FINMEM opinion failed for {symbol}: {e}")

    # Aggregate
    opinion = _aggregate_opinions(symbol, opinions, kronos_result, social_result, alerts)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    # ── DEBUG: per-agent vote table emitted every cycle ──────────────────────
    sep = "─" * 55
    vote_lines = [f"┌{sep}┐", f"│  OPINION LAYER DEBUG — {symbol:<28}│",
                  f"│  Result: {opinion.direction:<6} conf={opinion.confidence:.2f}  score={opinion.confidence:.2f}  ({elapsed:.1f}s){'':>5}│",
                  f"├{sep}┤"]
    weights = _get_combined_weights()
    for op in opinion.agent_opinions:
        w = weights.get(op.agent, 0.05)
        bar = "▲" if op.signal == "bullish" else ("▼" if op.signal == "bearish" else "━")
        vote_lines.append(
            f"│  {bar} {op.agent:<24} {op.signal.upper():<8} conf={op.confidence:.2f}  w={w:.2f}  │"
        )
    vote_lines.append(f"├{sep}┤")
    vote_lines.append(f"│  Kronos: {kronos_result.get('signal','N/A'):<8} "
                       f"{kronos_result.get('predicted_change_pct',0):+.2f}%   "
                       f"Social: {social_result.get('direction','N/A'):<8} "
                       f"score={social_result.get('sentiment_score',0):+.3f}  │")
    vote_lines.append(f"│  Alerts active: {len(alerts):<4}  Trade memory: {opinion.agent_opinions and any(o.agent=='trade_memory' for o in opinion.agent_opinions)}{'':>14}│")
    vote_lines.append(f"└{sep}┘")
    logger.info("\n".join(vote_lines))
    # ─────────────────────────────────────────────────────────────────────────

    logger.info(
        f"Opinion Layer for {symbol}: {opinion.direction} "
        f"conf={opinion.confidence:.2f} ({elapsed:.1f}s, {len(opinions)} agents)"
    )
    return opinion


def register_agent_weight(agent_name: str, weight: float):
    """Dynamically adjust an agent's voting weight."""
    _AGENT_WEIGHTS[agent_name] = weight
    logger.info(f"Opinion Layer weight updated: {agent_name} = {weight}")


def register_persona_weight(persona_name: str, weight: float):
    """Dynamically adjust a persona agent's voting weight."""
    set_persona_weight(persona_name, weight)
