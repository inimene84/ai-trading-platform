"""
FINMEM: Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design
Reference: Stevens Institute of Technology (arXiv:2311.13743v2)

Core Architecture:
  1. Profiling Module:
     - Asset-specific domain knowledge base (sectors, tokenomics, historical context)
     - Dynamic Character Design with Self-Adaptive risk inclination (switches between
       Risk-Seeking and Risk-Averse based on short-term cumulative returns).
  2. Layered Long-Term Memory (Stratified processing in Qdrant):
     - Shallow Layer (Q=14d, alpha=0.900): Daily news, real-time sentiment, momentum
     - Intermediate Layer (Q=90d, alpha=0.967): Quarterly reports, ecosystem releases, mid-term trends
     - Deep Layer (Q=365d, alpha=0.988): Macro cycles, annual reports, and Extended Reflections
     - Stratified retrieval score: gamma = S_Recency + S_Relevancy + S_Importance
     - Access counter & Promotion: pivotal memories ascend to deeper layers; recency resets to 1.0
  3. Working Memory & Dual Reflection:
     - Observation: Market momentum, price action, open position state
     - Immediate Reflection: Top-K (K=5) from all 3 layers -> Action (BUY/SELL/HOLD) + Rationale + Cited IDs
     - Extended Reflection: Retrospective self-evolution stored into Deep Layer.
"""

from __future__ import annotations

import os
import math
import time
import uuid
import json
import hashlib
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from backend.utils.embeddings import generate_text_embedding
from backend.services.qdrant_client import (
    AsyncQdrantClient,
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    QDRANT_AVAILABLE,
)
from backend.llm.router import call_llm_resilient

logger = logging.getLogger(__name__)

# ── Hyperparameters from Paper (Section 3.2.2 & Equations 1-6) ───────────────
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_FINMEM", "finmem-memory")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

# Stability constants Q_l (in days)
Q_SHALLOW = 14.0
Q_INTERMEDIATE = 90.0
Q_DEEP = 365.0

# Importance degradation bases alpha_l (decay rates)
ALPHA_SHALLOW = 0.900
ALPHA_INTERMEDIATE = 0.967
ALPHA_DEEP = 0.988

# Cognitive working memory bandwidth: Top-K per layer (Paper Table 5 found K=5 optimal)
TOP_K_PER_LAYER = 5

# Memory purge thresholds
MIN_RECENCY_THRESHOLD = 0.05
MIN_IMPORTANCE_THRESHOLD = 5.0  # pre-scaled [0, 100]

# Promotion criteria (citations in profitable / impactful decisions)
PROMOTION_SHALLOW_TO_INTERMEDIATE = 4
PROMOTION_INTERMEDIATE_TO_DEEP = 8


# ─────────────────────────────────────────────────────────────────────────────
# 1. Profiling Module: Asset Knowledge & Dynamic Character
# ─────────────────────────────────────────────────────────────────────────────

ASSET_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "BTC-USDT": {
        "name": "Bitcoin (BTC)",
        "sectors": "Decentralized Settlement, Digital Gold, Store of Value",
        "description": "Bitcoin operates as the primary macro reserve asset in digital markets with fixed supply cap (21M). Its valuation is driven by ETF institutional inflows, global liquidity expansion, mining difficulty halving cycles, and macroeconomic risk sentiment.",
    },
    "ETH-USDT": {
        "name": "Ethereum (ETH)",
        "sectors": "Smart Contract Platform, Layer 1 Settlement, Proof-of-Stake",
        "description": "Ethereum serves as the global computational layer for decentralized finance (DeFi), real-world assets (RWA), and tokenized liquidity. Key drivers include staking yields, Layer 2 blob fee dynamics, network gas burn (EIP-1559), and institutional staking adoption.",
    },
    "SOL-USDT": {
        "name": "Solana (SOL)",
        "sectors": "High-Throughput Layer 1, Retail DEX, DePIN, Consumer Web3",
        "description": "Solana emphasizes monolithic parallel execution, sub-second latency, and ultra-low transaction fees. Valuation is influenced by on-chain retail trading volumes, active DEX liquidity, developer mindshare, and ecosystem fee generation.",
    },
}


def normalize_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "-")
    if "-" in s:
        return s
    for quote in ["USDT", "USDC", "BUSD", "USD"]:
        if s.endswith(quote):
            return f"{s[:-len(quote)]}-USDT"
    return s


@dataclass
class FinMemCharacter:
    """Dynamic character setting supporting Self-Adaptive risk inclination."""
    risk_mode: str  # "risk-seeking" | "risk-averse" | "self-adaptive"
    current_inclination: str  # "risk-seeking" or "risk-averse"
    rolling_returns: List[float] = field(default_factory=list)
    window_periods: int = 3

    def update_performance(self, return_pct: float):
        """Update recent performance history and dynamically adapt character."""
        self.rolling_returns.append(return_pct)
        if len(self.rolling_returns) > self.window_periods:
            self.rolling_returns.pop(0)

        cum_ret = sum(self.rolling_returns)
        # Self-Adaptive Rule (Paper Section 3.1 & Figure 1):
        # When cumulative return is positive -> risk-seeking
        # When cumulative return is negative -> risk-averse
        if self.risk_mode == "self-adaptive":
            if cum_ret < 0:
                self.current_inclination = "risk-averse"
            else:
                self.current_inclination = "risk-seeking"

    def get_prompt_preamble(self) -> str:
        """Construct prompt preamble reflecting active risk character (Paper Figure 1)."""
        if self.current_inclination == "risk-seeking":
            return (
                "Character Setting: Risk-Seeking Quantitative Trader.\n"
                "You are drawn to high-reward momentum opportunities in trading. Your strategy is bold and clear:\n"
                "- Chase the potential for substantial upside gains and momentum continuation.\n"
                "- Assertively exploit high-conviction trend breakouts, accepting measured volatility.\n"
                "- Avoid excessive passivity during bullish and directional market phases."
            )
        else:
            return (
                "Character Setting: Risk-Averse Quantitative Trader.\n"
                "You prioritize capital preservation, drawdown defense, and downside risk containment:\n"
                "- Prioritize trades with high margin of safety and tight risk-reward profiles.\n"
                "- Reject ambiguous or conflicting signals; favor defensive HOLD positioning during choppy regimes.\n"
                "- Protect accumulated gains and safeguard portfolio equity above all else."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Layered Long-Term Memory Module
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEvent:
    id: str
    symbol: str
    layer: str  # "shallow" | "intermediate" | "deep"
    content: str
    source_type: str  # "news" | "quarterly_filing" | "annual_filing" | "extended_reflection"
    timestamp: float
    base_importance: float  # [0, 100]
    access_count: int = 0
    recency_reset: bool = False
    positive_sentiment: float = 0.0
    neutral_sentiment: float = 1.0
    negative_sentiment: float = 0.0
    vector: Optional[List[float]] = None

    # Calculated retrieval metrics
    s_recency: float = 0.0
    s_relevancy: float = 0.0
    s_importance: float = 0.0
    gamma_score: float = 0.0


class FinMemLayeredMemory:
    """
    Stratified Long-Term Memory Store backed by Qdrant.
    Organizes information into Shallow, Intermediate, and Deep processing layers.
    """

    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "http://vps-qdrant:6333")
        self.api_key = os.getenv("QDRANT_API_KEY", "")
        self.collection = COLLECTION_NAME
        self.vector_size = VECTOR_SIZE

        if QDRANT_AVAILABLE:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key if self.api_key else None,
                timeout=8.0,
            )
        else:
            self._client = None
            logger.warning("Qdrant client not available; FINMEM memory degraded to in-memory fallback")
        
        # Local fallback cache
        self._local_memory: List[MemoryEvent] = []

    async def ensure_collection(self) -> bool:
        """Ensure collection exists in Qdrant with appropriate index."""
        if not self._client:
            return False
        try:
            cols = await self._client.get_collections()
            if not any(c.name == self.collection for c in cols.collections):
                await self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Initialized Qdrant collection: {self.collection}")
            return True
        except Exception as e:
            logger.warning(f"Error ensuring Qdrant collection {self.collection}: {e}")
            return False

    def _sample_base_importance(self, layer: str) -> float:
        """
        Sample base importance v_l^E according to piecewise probabilities (Paper Formula 4).
        Shallow: 80% 40, 15% 60, 5% 80
        Intermediate: 5% 40, 80% 60, 15% 80
        Deep: 5% 40, 15% 60, 80% 80
        """
        r = np.random.rand()
        if layer == "shallow":
            return 40.0 if r < 0.80 else (60.0 if r < 0.95 else 80.0)
        elif layer == "intermediate":
            return 40.0 if r < 0.05 else (60.0 if r < 0.85 else 80.0)
        else:
            return 40.0 if r < 0.05 else (60.0 if r < 0.20 else 80.0)

    async def store_memory(
        self,
        symbol: str,
        layer: str,
        content: str,
        source_type: str = "news",
        base_importance: Optional[float] = None,
        sentiments: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Store a summarized event into the specified long-term memory layer."""
        norm_sym = normalize_symbol(symbol)
        mem_id = str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        b_imp = base_importance if base_importance is not None else self._sample_base_importance(layer)
        sents = sentiments or {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

        # Generate embedding vector
        vec = await generate_text_embedding(content, vector_size=self.vector_size)
        if not vec:
            # Deterministic pseudo-embedding fallback
            # Deterministic seed only, not a security digest.
            h = int(hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            np.random.seed(h % (2**32))
            rnd = np.random.normal(0, 1, self.vector_size)
            vec = (rnd / np.linalg.norm(rnd)).tolist()

        event = MemoryEvent(
            id=mem_id,
            symbol=norm_sym,
            layer=layer,
            content=content,
            source_type=source_type,
            timestamp=ts,
            base_importance=b_imp,
            access_count=0,
            recency_reset=False,
            positive_sentiment=sents.get("positive", 0.0),
            neutral_sentiment=sents.get("neutral", 1.0),
            negative_sentiment=sents.get("negative", 0.0),
            vector=vec,
        )

        if self._client:
            try:
                await self.ensure_collection()
                point = PointStruct(
                    id=mem_id,
                    vector=vec,
                    payload={
                        "symbol": norm_sym,
                        "layer": layer,
                        "content": content,
                        "source_type": source_type,
                        "timestamp": ts,
                        "base_importance": b_imp,
                        "access_count": 0,
                        "recency_reset": False,
                        "sentiment": sents,
                    },
                )
                await self._client.upsert(collection_name=self.collection, points=[point])
            except Exception as e:
                logger.error(f"Failed to upsert memory event to Qdrant: {e}")
                self._local_memory.append(event)
        else:
            self._local_memory.append(event)

        return mem_id

    async def retrieve_stratified_memories(
        self,
        symbol: str,
        query_text: str,
        k: int = TOP_K_PER_LAYER,
        inquiry_time: Optional[float] = None,
    ) -> Dict[str, List[MemoryEvent]]:
        """
        Computes stratified retrieval score gamma_l^E = S_Recency + S_Relevancy + S_Importance
        and returns Top-K events for Shallow, Intermediate, and Deep layers.
        """
        norm_sym = normalize_symbol(symbol)
        now_ts = inquiry_time if inquiry_time is not None else time.time()

        # Query embedding
        q_vec = await generate_text_embedding(query_text, vector_size=self.vector_size)
        if not q_vec:
            q_vec = [0.0] * self.vector_size

        stratified_candidates: Dict[str, List[MemoryEvent]] = {
            "shallow": [],
            "intermediate": [],
            "deep": [],
        }

        if self._client:
            try:
                for layer_name in ["shallow", "intermediate", "deep"]:
                    filter_cond = Filter(
                        must=[
                            FieldCondition(key="layer", match=MatchValue(value=layer_name)),
                        ]
                    )
                    # Query top candidates from Qdrant by cosine similarity
                    resp = await self._client.query_points(
                        collection_name=self.collection,
                        query=q_vec,
                        query_filter=filter_cond,
                        limit=k * 4,
                        with_payload=True,
                    )
                    hits = getattr(resp, "points", resp) or []
                    for hit in hits:
                        p = hit.payload or {}
                        # Allow symbol match or global macro ('*')
                        hit_sym = p.get("symbol", "")
                        if hit_sym not in (norm_sym, "*", "ALL"):
                            continue
                        ev = MemoryEvent(
                            id=str(hit.id),
                            symbol=hit_sym,
                            layer=p.get("layer", layer_name),
                            content=p.get("content", ""),
                            source_type=p.get("source_type", "news"),
                            timestamp=float(p.get("timestamp", now_ts)),
                            base_importance=float(p.get("base_importance", 50.0)),
                            access_count=int(p.get("access_count", 0)),
                            recency_reset=bool(p.get("recency_reset", False)),
                            s_relevancy=float(hit.score),  # Cosine similarity
                        )
                        stratified_candidates[layer_name].append(ev)
            except Exception as e:
                logger.warning(f"Qdrant retrieval error, using local memory: {e}")

        # If Qdrant returned few/no items, supplement from local cache
        for ev in self._local_memory:
            if ev.symbol in (norm_sym, "*") and len(stratified_candidates[ev.layer]) < k * 4:
                # Approximate cosine similarity
                dot = sum(a * b for a, b in zip(ev.vector or [], q_vec)) if ev.vector else 0.5
                ev_copy = MemoryEvent(**asdict(ev))
                ev_copy.s_relevancy = max(0.0, min(1.0, dot))
                stratified_candidates[ev.layer].append(ev_copy)

        results: Dict[str, List[MemoryEvent]] = {"shallow": [], "intermediate": [], "deep": []}

        # Layer configuration mappings
        layer_params = {
            "shallow": (Q_SHALLOW, ALPHA_SHALLOW),
            "intermediate": (Q_INTERMEDIATE, ALPHA_INTERMEDIATE),
            "deep": (Q_DEEP, ALPHA_DEEP),
        }

        for layer_name, (q_l, alpha_l) in layer_params.items():
            scored_events: List[MemoryEvent] = []
            for ev in stratified_candidates[layer_name]:
                # 1. Delta time in days (delta^E = t_P - t_E)
                delta_days = max(0.0, (now_ts - ev.timestamp) / 86400.0)

                # 2. Recency score: exp(-delta / Q_l)
                # If recency was reset on promotion, boost recency
                if ev.recency_reset:
                    s_recency = 1.0
                else:
                    s_recency = float(math.exp(-delta_days / q_l))

                # 3. Importance score: v_l^E * (alpha_l)^delta_days, scaled to [0, 1]
                theta = float(math.pow(alpha_l, delta_days))
                raw_importance = ev.base_importance * theta
                s_importance = float(np.clip(raw_importance / 100.0, 0.0, 1.0))

                # Purge check: skip decaying noise (Paper Section 3.2.2)
                if s_recency < MIN_RECENCY_THRESHOLD or raw_importance < MIN_IMPORTANCE_THRESHOLD:
                    continue

                # 4. Total Information Retrieval Score (Equation 1)
                # gamma = S_Recency + S_Relevancy + S_Importance (scaled [0, 3])
                gamma = s_recency + ev.s_relevancy + s_importance

                ev.s_recency = round(s_recency, 4)
                ev.s_importance = round(s_importance, 4)
                ev.gamma_score = round(gamma, 4)
                scored_events.append(ev)

            # Sort descending by gamma_score and pick Top-K
            scored_events.sort(key=lambda x: x.gamma_score, reverse=True)
            results[layer_name] = scored_events[:k]

        return results

    async def record_citation_and_promote(self, cited_ids: List[str], was_profitable: bool = True):
        """
        Access Counter & Promotion Function (Paper Section 3.2.2):
        Pivotal events cited in successful trades gain +5 importance points.
        Frequently referenced events ascend to deeper layers, resetting recency to 1.0.
        """
        if not self._client or not cited_ids:
            return

        try:
            for mem_id in cited_ids:
                pts = await self._client.retrieve(collection_name=self.collection, ids=[mem_id])
                if not pts:
                    continue
                p = pts[0].payload or {}
                layer = p.get("layer", "shallow")
                access_count = int(p.get("access_count", 0)) + 1
                base_imp = float(p.get("base_importance", 50.0))

                # Bonus for profitable investment relevance
                if was_profitable:
                    base_imp = min(100.0, base_imp + 5.0)

                new_layer = layer
                recency_reset = p.get("recency_reset", False)

                # Promotion checks
                if layer == "shallow" and access_count >= PROMOTION_SHALLOW_TO_INTERMEDIATE:
                    new_layer = "intermediate"
                    recency_reset = True
                    logger.info(f"[FINMEM Memory Promotion] {mem_id} promoted: shallow -> intermediate")
                elif layer == "intermediate" and access_count >= PROMOTION_INTERMEDIATE_TO_DEEP:
                    new_layer = "deep"
                    recency_reset = True
                    logger.info(f"[FINMEM Memory Promotion] {mem_id} promoted: intermediate -> deep")

                p["access_count"] = access_count
                p["base_importance"] = base_imp
                p["layer"] = new_layer
                p["recency_reset"] = recency_reset

                await self._client.set_payload(
                    collection_name=self.collection,
                    payload=p,
                    points=[mem_id],
                )
        except Exception as e:
            logger.debug(f"Error updating memory citations: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Working Memory & Decision-Making Module
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FinMemDecision:
    symbol: str
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    reasoning: str
    risk_character: str
    cited_memory_ids: List[str]
    retrieved_memory_count: int
    momentum_3d: float
    latency_ms: float


class FinMemService:
    """
    Main FINMEM Autonomous Agent orchestrating Profiling, Layered Memory,
    and Dual Reflection for cryptocurrency trading.
    """

    def __init__(self):
        self.memory = FinMemLayeredMemory()
        self.characters: Dict[str, FinMemCharacter] = {}
        self._seed_default_memories_task = None

    def get_character(self, symbol: str) -> FinMemCharacter:
        norm_sym = normalize_symbol(symbol)
        if norm_sym not in self.characters:
            self.characters[norm_sym] = FinMemCharacter(
                risk_mode="self-adaptive",
                current_inclination="risk-seeking",
            )
        return self.characters[norm_sym]

    def observe(self, symbol: str, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Observation Operation (Paper Section 3.2.1):
        Gathers current market facts, closing price differences, and 3-period momentum.
        """
        if not bars:
            raise ValueError(f"Valid candle bars required for FINMEM observation of {symbol} (got empty bars) — fail closed")

        closes = [float(b["close"]) for b in bars if b.get("close") is not None]
        if not closes or closes[-1] <= 0:
            raise ValueError(f"Valid positive close price required for FINMEM observation of {symbol} — fail closed")
        latest_close = closes[-1]

        # 3-period momentum: log return over last 3 bars
        if len(closes) >= 4:
            p_curr = closes[-1]
            p_prev = closes[-4]
            momentum = math.log(max(1e-8, p_curr) / max(1e-8, p_prev)) * 100.0
        else:
            momentum = 0.0

        # Simple volatility
        returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))] if len(closes) > 1 else [0.0]
        vol = float(np.std(returns)) if len(returns) > 1 else 0.0

        return {
            "latest_close": latest_close,
            "momentum_3d": round(momentum, 3),
            "volatility": round(vol, 4),
            "bars_count": len(bars),
        }

    async def seed_domain_knowledge(self):
        """Seed foundational asset backgrounds and macro reflections into Qdrant."""
        for sym, data in ASSET_KNOWLEDGE_BASE.items():
            # Seed deep layer with fundamental profile
            await self.memory.store_memory(
                symbol=sym,
                layer="deep",
                content=f"{data['name']} Core Profile & Sector Mechanics:\nSectors: {data['sectors']}\n{data['description']}",
                source_type="annual_filing",
                base_importance=85.0,
            )
            # Seed intermediate layer with macro ecosystem baseline
            await self.memory.store_memory(
                symbol=sym,
                layer="intermediate",
                content=f"{sym} Ecosystem Baseline: High institutional liquidity tier, major Binance Perpetual pair with dynamic funding adjustments and deep orderbook support.",
                source_type="quarterly_filing",
                base_importance=65.0,
            )
        logger.info("[FINMEM] Seeded foundational domain knowledge across layers.")

    async def immediate_reflect(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        external_context: Optional[Dict[str, Any]] = None,
    ) -> FinMemDecision:
        """
        Immediate Reflection Operation (Paper Section 3.2.1a & Figure 3):
        Synthesizes active character, market observation facts, and Top-K stratified memories.
        Outputs investment decision (BUY, SELL, HOLD), confidence, reasoning, and cited IDs.
        """
        t0 = time.time()
        norm_sym = normalize_symbol(symbol)
        char = self.get_character(norm_sym)
        obs = self.observe(norm_sym, bars)

        # Asset domain profile
        asset_info = ASSET_KNOWLEDGE_BASE.get(norm_sym, {
            "name": norm_sym,
            "sectors": "Cryptocurrency Perpetual Futures",
            "description": "Digital asset with 24/7 continuous trading and leverage dynamics.",
        })

        # Inquiry context query for relevancy search
        query_text = f"Investment decision for {norm_sym}: price={obs['latest_close']}, momentum={obs['momentum_3d']}%"
        if external_context:
            query_text += f", sentiment={external_context.get('sentiment_score', 0)}"

        # Stratified Memory Retrieval (Top-K per layer)
        stratified = await self.memory.retrieve_stratified_memories(
            symbol=norm_sym,
            query_text=query_text,
            k=TOP_K_PER_LAYER,
            inquiry_time=time.time(),
        )

        shallow_events = stratified.get("shallow", [])
        inter_events = stratified.get("intermediate", [])
        deep_events = stratified.get("deep", [])
        total_retrieved = len(shallow_events) + len(inter_events) + len(deep_events)

        # Format memories for prompt
        def _format_events(events: List[MemoryEvent]) -> str:
            if not events:
                return "  (No active memory events in this layer)\n"
            lines = []
            for e in events:
                lines.append(f"  - [ID: {e.id[:8]}] (Gamma Score: {e.gamma_score:.2f}) {e.content}")
            return "\n".join(lines) + "\n"

        prompt = f"""
You are FINMEM, a state-of-the-art cognitive LLM trading agent with layered long-term memory.
Evaluate the current market situation and output your investment decision for {norm_sym}.

=== 1. AGENT PROFILING & CHARACTER ===
{char.get_prompt_preamble()}

Asset Profile: {asset_info['name']}
Primary Sectors: {asset_info['sectors']}
Historical Overview: {asset_info['description']}

=== 2. CURRENT MARKET OBSERVATION ===
- Current Close Price: ${obs['latest_close']:.2f}
- 3-Period Momentum: {obs['momentum_3d']:+.3f}%
- Return Volatility: {obs['volatility']:.4f}

=== 3. RETRIEVED LAYERED MEMORIES (TOP-K COGNITIVE BANDWIDTH) ===
SHALLOW PROCESSING LAYER (Daily Market News & Real-Time Sentiments):
{_format_events(shallow_events)}

INTERMEDIATE PROCESSING LAYER (Quarterly Reports & Multi-Week Ecosystem Trends):
{_format_events(inter_events)}

DEEP PROCESSING LAYER (Macro Cycles, Annual Reports & Past Extended Reflections):
{_format_events(deep_events)}

=== 4. TRADING INSTRUCTION ===
Synthesize the layered memory insights with current market momentum according to your active risk character.
Determine whether to BUY, SELL, or HOLD.
Reply ONLY with a valid JSON object matching this exact schema:
{{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": <float between 0.00 and 1.00>,
  "reasoning": "<concise 2-sentence rationale referencing key memory evidence>",
  "cited_memory_ids": ["<ID1>", "<ID2>"]
}}
"""
        # Call LLM via resilient router
        try:
            raw_res = await call_llm_resilient(
                task_type="premium_analysis",
                prompt=prompt,
                system="You are the FINMEM financial reasoning agent. You strictly adhere to the requested JSON schema.",
                temperature=0.4,
                max_tokens=600,
                response_json=True,
            )
            # Parse JSON
            cleaned = raw_res.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned)
            action = str(parsed.get("decision", "HOLD")).upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            conf = float(np.clip(float(parsed.get("confidence", 0.5)), 0.0, 1.0))
            reasoning = str(parsed.get("reasoning", "Synthesized layered memory consensus."))
            cited_ids = [str(cid) for cid in parsed.get("cited_memory_ids", [])]

        except Exception as e:
            logger.warning(f"FINMEM LLM reflection parse fallback: {e}")
            # Safe quantitative fallback based on momentum
            if obs["momentum_3d"] > 1.5:
                action = "BUY" if char.current_inclination == "risk-seeking" else "HOLD"
                conf = 0.55
            elif obs["momentum_3d"] < -1.5:
                action = "SELL"
                conf = 0.55
            else:
                action = "HOLD"
                conf = 0.50
            reasoning = f"Deterministic fallback based on momentum ({obs['momentum_3d']:+.2f}%)"
            cited_ids = []

        latency_ms = round((time.time() - t0) * 1000, 2)

        decision = FinMemDecision(
            symbol=norm_sym,
            action=action,
            confidence=round(conf, 4),
            reasoning=reasoning,
            risk_character=char.current_inclination,
            cited_memory_ids=cited_ids,
            retrieved_memory_count=total_retrieved,
            momentum_3d=obs["momentum_3d"],
            latency_ms=latency_ms,
        )

        return decision

    async def extended_reflect(
        self,
        symbol: str,
        trade_outcomes: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Extended Reflection Operation (Paper Section 3.2.1b):
        Reviews immediate reflection outcomes over an M-period horizon.
        Derives high-level self-critical wisdom and transmits it into the DEEP PROCESSING LAYER.
        """
        if not trade_outcomes:
            return None

        norm_sym = normalize_symbol(symbol)
        char = self.get_character(norm_sym)

        # Compute summary metrics
        total_pnl = sum(float(t.get("pnl", 0.0)) for t in trade_outcomes)
        wins = sum(1 for t in trade_outcomes if float(t.get("pnl", 0.0)) > 0)
        losses = len(trade_outcomes) - wins

        prompt = f"""
You are the self-evolution reflection core of FINMEM for {norm_sym}.
Review the recent trading execution history and synthesize an Extended Reflection for long-term retention:

Recent Executions: {len(trade_outcomes)} trades | {wins} Wins, {losses} Losses | Net PnL: {total_pnl:+.4f} USDT
Active Character: {char.current_inclination}

Tasks:
1. Identify which market signals or memory types correctly anticipated price action.
2. Identify which assumptions failed or suffered deceptive market noise.
3. Formulate a 2-sentence guiding heuristic that FINMEM must remember in its DEEP MEMORY for future market cycles.

Reply in 2 concise sentences of high-level quantitative trading wisdom.
"""
        try:
            reflection_text = await call_llm_resilient(
                task_type="general",
                prompt=prompt,
                system="You are FINMEM Extended Reflection Core. Provide distilled trading heuristics.",
                temperature=0.3,
                max_tokens=250,
            )
            reflection_clean = reflection_text.strip()
            
            # Ingest into Deep Processing Layer (Q_deep = 365 days, high importance)
            mem_id = await self.memory.store_memory(
                symbol=norm_sym,
                layer="deep",
                content=f"FINMEM Extended Reflection ({datetime.now(timezone.utc).strftime('%Y-%m-%d')}):\n{reflection_clean}",
                source_type="extended_reflection",
                base_importance=90.0,
            )
            logger.info(f"[FINMEM Extended Reflection] Stored deep memory {mem_id}: {reflection_clean[:80]}...")
            return mem_id
        except Exception as e:
            logger.warning(f"Failed to generate extended reflection: {e}")
            return None

    def record_trade_feedback(self, symbol: str, pnl: float, cited_ids: Optional[List[str]] = None):
        """Update character performance and promote pivotal memories upon trade close."""
        norm_sym = normalize_symbol(symbol)
        char = self.get_character(norm_sym)
        char.update_performance(pnl)

        if cited_ids:
            was_profitable = (pnl > 0)
            asyncio.create_task(self.memory.record_citation_and_promote(cited_ids, was_profitable))


# Singleton instance
finmem_service = FinMemService()
