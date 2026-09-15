"""
Persona Agent Adapter for Crypto
================================
Adapts ai-hedge-fund investor persona agents for crypto trading.
Each persona receives crypto-specific metrics (funding, OI, volatility,
exchange flows, recent returns) and responds in their signature style.

Persona prompts and weights are loaded from YAML config files in
``backend/agents/config/personas/``.  The LLM model is selected via
the centralised router in ``backend/llm/router.py``.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import time

from backend.llm.router import call_llm_resilient


logger = logging.getLogger(__name__)

# Reasoning prefix used when the LLM chain is down. Callers can detect a skipped
# enrichment without treating it as a real persona vote. Not cached.
_LLM_UNAVAILABLE_PREFIX = "LLM unavailable:"


# ═══════════════════════════════════════════════════════════════════════════════
# Persona Registry — loads from YAML config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonaConfig:
    """Single persona definition loaded from YAML."""
    id: str
    name: str
    style: str
    time_horizon: str
    risk_tolerance: str
    weight: float
    prompt_template: str


class PersonaRegistry:
    """
    Lazy-loading registry of persona configs from YAML files.
    Caches after first load.
    """
    _instance: Optional["PersonaRegistry"] = None
    _loaded: bool = False
    _personas: Dict[str, PersonaConfig] = {}
    _config_dir: Path = Path(__file__).resolve().parent.parent / "agents" / "config" / "personas"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        self._personas = {}
        if not self._config_dir.exists():
            logger.warning(f"Persona config dir not found: {self._config_dir}")
            self._loaded = True
            return
        for yaml_file in sorted(self._config_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not data or not isinstance(data, dict):
                    continue
                persona = PersonaConfig(
                    id=data.get("id", yaml_file.stem),
                    name=data.get("name", yaml_file.stem),
                    style=data.get("style", "unknown"),
                    time_horizon=data.get("time_horizon", "swing"),
                    risk_tolerance=data.get("risk_tolerance", "medium"),
                    weight=float(data.get("weight", 0.05)),
                    prompt_template=data.get("prompt_template", ""),
                )
                self._personas[persona.id] = persona
            except Exception as e:
                logger.warning(f"Failed to load persona config {yaml_file.name}: {e}")
        self._loaded = True
        logger.info(f"PersonaRegistry: loaded {len(self._personas)} personas from {self._config_dir}")

    def get(self, persona_id: str) -> Optional[PersonaConfig]:
        self._load()
        return self._personas.get(persona_id)

    def all(self) -> Dict[str, PersonaConfig]:
        self._load()
        return dict(self._personas)

    def weights(self) -> Dict[str, float]:
        self._load()
        return {pid: p.weight for pid, p in self._personas.items()}

    def prompts(self) -> Dict[str, str]:
        self._load()
        return {pid: p.prompt_template for pid, p in self._personas.items()}

    def active_ids(self) -> List[str]:
        """Return active persona IDs, respecting ACTIVE_PERSONAS env override."""
        self._load()
        env = os.getenv("ACTIVE_PERSONAS", "")
        if env:
            return [p.strip() for p in env.split(",") if p.strip() in self._personas]
        return list(self._personas.keys())

    def reload(self):
        """Force reload from disk (useful for hot-reloading)."""
        self._loaded = False
        self._load()


# Module-level singleton
_registry = PersonaRegistry()

# In-memory cache for persona opinions to reduce token costs (runs once per hour)
# Key: (symbol, persona_id), Value: (timestamp, PersonaOpinion)
_PERSONA_OPINION_CACHE: Dict[tuple[str, str], tuple[float, "PersonaOpinion"]] = {}


@dataclass
class PersonaOpinion:
    persona: str
    signal: str       # bullish | bearish | neutral
    confidence: float # 0-1
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════════════
# LLM caller — uses central router
# ═══════════════════════════════════════════════════════════════════════════════

async def _call_llm(system: str, user: str) -> dict:
    """Call the LLM and return parsed JSON with signal, confidence, reasoning."""
    try:
        res_str = await call_llm_resilient(
            task_type="persona_analysis",
            prompt=user,
            system=system,
            response_json=True
        )
        parsed = json.loads(res_str)
        return {
            "signal": parsed.get("signal", "neutral").lower(),
            "confidence": float(parsed.get("confidence", 0)) / 100.0,
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        logger.error(
            "Persona LLM call failed; skipping persona enrichment this cycle: %s",
            type(e).__name__,
        )
        return {
            "signal": "neutral",
            "confidence": 0.0,
            "reasoning": f"{_LLM_UNAVAILABLE_PREFIX} {type(e).__name__}",
            "_llm_failed": True,
        }



def _build_crypto_context(symbol: str, bars: list, metrics: dict) -> str:
    """Build a compact crypto context string from bars and metrics."""
    if not bars:
        return "No price data available."

    df_last = bars[-1]
    price = df_last.get("close", 0)
    vol_24h = sum(b.get("volume", 0) for b in bars[-24:]) if len(bars) >= 24 else sum(b.get("volume", 0) for b in bars)

    # Calculate returns
    returns = []
    for i in range(1, len(bars)):
        prev = bars[i - 1].get("close", 0)
        curr = bars[i].get("close", 0)
        if prev > 0:
            returns.append((curr - prev) / prev)
    vol_annual = 0.0
    if returns:
        import numpy as np
        vol_annual = float(np.std(returns) * np.sqrt(len(returns)) * 100)

    # Recent performance
    if len(bars) >= 24:
        ret_1d = (bars[-1]["close"] - bars[-24]["close"]) / bars[-24]["close"] * 100
    else:
        ret_1d = 0.0
    if len(bars) >= 7:
        ret_7d = (bars[-1]["close"] - bars[-7]["close"]) / bars[-7]["close"] * 100
    else:
        ret_7d = 0.0
    if len(bars) >= 30:
        ret_30d = (bars[-1]["close"] - bars[-30]["close"]) / bars[-30]["close"] * 100
    else:
        ret_30d = 0.0

    funding = metrics.get("funding_rate", "N/A")
    oi = metrics.get("open_interest", "N/A")
    oi_change = metrics.get("oi_change_24h", "N/A")

    lines = [
        f"Asset: {symbol}",
        f"Current Price: ${price:,.2f}" if price > 1 else f"Current Price: ${price:.6f}",
        f"24h Volume: {vol_24h:,.0f}",
        f"1d Return: {ret_1d:+.2f}%",
        f"7d Return: {ret_7d:+.2f}%",
        f"30d Return: {ret_30d:+.2f}%",
        f"Annualized Volatility: {vol_annual:.1f}%",
        f"Funding Rate: {funding}",
        f"Open Interest: {oi} (24h change: {oi_change})",
    ]

    # Add technical snapshot
    if len(bars) >= 20:
        closes = [b["close"] for b in bars]
        sma_20 = sum(closes[-20:]) / 20
        lines.append(f"Price vs SMA20: {((price - sma_20) / sma_20 * 100):+.2f}%")
    if len(bars) >= 50:
        sma_50 = sum([b["close"] for b in bars[-50:]]) / 50
        lines.append(f"Price vs SMA50: {((price - sma_50) / sma_50 * 100):+.2f}%")

    trade_mem = metrics.get("trade_memory", {})
    if trade_mem.get("count", 0) > 0:
        lines.append(f"Trade Memory: {trade_mem.get('summary', '')}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def run_persona(
    persona: str,
    symbol: str,
    bars: list,
    metrics: Optional[dict] = None,
) -> PersonaOpinion:
    """
    Run a single persona agent on crypto data.

    Args:
        persona: One of the persona IDs from the registry
        symbol: Trading symbol
        bars: List of OHLCV dicts
        metrics: Optional dict with funding_rate, open_interest, etc.

    Returns:
        PersonaOpinion with signal, confidence, reasoning
    """
    cache_key = (symbol, persona)
    now = time.time()
    if cache_key in _PERSONA_OPINION_CACHE:
        cached_time, cached_op = _PERSONA_OPINION_CACHE[cache_key]
        if now - cached_time < 3600:
            logger.info(f"Using cached opinion for persona {persona} on {symbol} (age={int(now - cached_time)}s)")
            return cached_op

    cfg = _registry.get(persona)
    if not cfg:
        return PersonaOpinion(
            persona=persona, signal="neutral", confidence=0.0,
            reasoning=f"Unknown persona: {persona}"
        )

    system = cfg.prompt_template

    metrics = metrics or {}
    context = _build_crypto_context(symbol, bars, metrics)

    user = (
        f"Analyze the following crypto asset and return a JSON object with exactly these keys:\n"
        f'  "signal": "bullish" | "bearish" | "neutral"\n'
        f'  "confidence": integer 0-100 (how sure you are)\n'
        f'  "reasoning": concise explanation in your character\'s voice\n\n'
        f"Context:\n{context}\n\n"
        f"Respond ONLY with valid JSON. No markdown."
    )

    result = await _call_llm(system, user)

    # Normalize signal
    sig = result.get("signal", "neutral")
    if sig not in ("bullish", "bearish", "neutral"):
        sig = "neutral"

    opinion = PersonaOpinion(
        persona=persona,
        signal=sig,
        confidence=min(max(result.get("confidence", 0.0), 0.0), 1.0),
        reasoning=result.get("reasoning", ""),
    )
    # Do not cache LLM outages — the next cycle should retry providers.
    if not result.get("_llm_failed"):
        _PERSONA_OPINION_CACHE[cache_key] = (now, opinion)
    return opinion


async def run_all_personas(
    symbol: str,
    bars: list,
    metrics: Optional[dict] = None,
    selected: Optional[List[str]] = None,
) -> List[PersonaOpinion]:
    """
    Run all (or selected) persona agents concurrently.

    Args:
        symbol: Trading symbol
        bars: OHLCV list
        metrics: Optional extra metrics dict
        selected: List of persona names to run (None = all active)

    Returns:
        List of PersonaOpinion
    """
    personas = selected or _registry.active_ids()
    try:
        enrichment_timeout = float(os.getenv("PERSONA_ENRICHMENT_TIMEOUT_SECONDS", "20"))
        if enrichment_timeout <= 0:
            enrichment_timeout = 20.0
    except (TypeError, ValueError):
        enrichment_timeout = 20.0
    tasks = [run_persona(p, symbol, bars, metrics) for p in personas]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=max(1.0, enrichment_timeout),
        )
    except asyncio.TimeoutError:
        logger.error(
            "Persona enrichment timed out after %.1fs; skipping so the trading loop can continue",
            enrichment_timeout,
        )
        return []

    opinions = []
    skipped = 0
    for persona, result in zip(personas, results):
        if isinstance(result, Exception):
            skipped += 1
            logger.error(
                "Persona %s failed; skipping enrichment for this agent: %s",
                persona,
                type(result).__name__,
            )
            continue
        if result.signal == "neutral" and str(result.reasoning).startswith(_LLM_UNAVAILABLE_PREFIX):
            skipped += 1
            continue
        opinions.append(result)

    if skipped and not opinions:
        logger.error(
            "All persona LLM calls failed; continuing trading cycle without persona enrichment"
        )
        return []
    if skipped:
        logger.warning(
            "Skipped %s/%s persona agents after LLM failure; using %s successful opinions",
            skipped,
            len(personas),
            len(opinions),
        )

    return opinions


def get_persona_weights() -> Dict[str, float]:
    """Get voting weights for persona agents from YAML config."""
    return _registry.weights()


def set_persona_weight(persona: str, weight: float):
    """Adjust a persona's voting weight at runtime."""
    cfg = _registry.get(persona)
    if cfg:
        cfg.weight = weight
        logger.info(f"Persona weight updated: {persona} = {weight}")
    else:
        logger.warning(f"Cannot set weight for unknown persona: {persona}")
