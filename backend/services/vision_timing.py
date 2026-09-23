"""
Optional Vision LLM Candlestick Timing Inspector (WP4)
======================================================
Secondary visual verification using LLM Vision.
Fires ONLY when:
  1. ENABLE_VISION_TIMING=true in env
  2. Position size exceeds VISION_MIN_NOTIONAL
  3. Heuristic timing score or Kronos forecast indicates ambiguity/risk

Dual-panel chart layout (1h structure + 15m entry timing).
"""

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

ENABLE_VISION = os.getenv("ENABLE_VISION_TIMING", "false").lower() == "true"
VISION_MIN_NOTIONAL = float(os.getenv("VISION_MIN_NOTIONAL", "500.0"))


async def evaluate_vision_timing_optional(
    bars: List[Dict[str, Any]],
    symbol: str,
    proposed_signal: str,
    notional_usd: float = 0.0,
    heuristic_risk: float = 0.0
) -> Optional[bool]:
    """
    Evaluates vision timing if enabled and conditions met.
    Returns True (approved), False (vetoed), or None (skipped/unavailable).
    """
    if not ENABLE_VISION:
        return None

    if notional_usd < VISION_MIN_NOTIONAL and heuristic_risk < 0.45:
        return None

    # Vision goes through the LiteLLM proxy (OpenAI-compatible); the default
    # alias "gemini-3-flash" is served by OpenRouter, so no Google key is needed.
    litellm_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY")
    if litellm_key:
        api_key = litellm_key
        base_url = os.getenv("LITELLM_BASE_URL", "http://litellm:4000/v1")
        vision_model = os.getenv("VISION_TIMING_MODEL", "gemini-3-flash")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        vision_model = os.getenv("VISION_TIMING_MODEL", "gpt-4o-mini")
    if not api_key:
        logger.debug("VisionTiming: No Vision API key configured — skipping")
        return None

    try:
        # Chart rendering attempt if mplfinance is installed
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Simple dual chart fallback rendering
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(f"Dual MTF Chart Timing — {symbol} ({proposed_signal})")

        df = pd.DataFrame(bars)
        ax1.plot(df["close"].tail(60).values, label="1H Close")
        ax1.set_title("1H Primary Structure")
        ax1.legend()

        ax2.plot(df["close"].tail(20).values, label="15M Entry", color="orange")
        ax2.set_title("15M Micro Entry")
        ax2.legend()

        chart_path = f"scratch_chart_{symbol}.png"
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close(fig)

        # Call vision model
        with open(chart_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            f"You are a quant trader verifying entry timing for a {proposed_signal} trade on {symbol}. "
            "Inspect the attached 1h/15m chart. Respond with JSON: {\"approved\": true/false, \"reason\": \"string\"}"
        )

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(model=vision_model, api_key=api_key, base_url=base_url, timeout=60)
        res = await llm.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            ])
        ])

        clean_text = res.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_text)
        approved = bool(parsed.get("approved", True))
        logger.info(f"VisionTiming: {symbol} {proposed_signal} -> {'APPROVED' if approved else 'VETOED'} ({parsed.get('reason')})")
        return approved

    except Exception as e:
        logger.warning(f"VisionTiming execution exception: {e}")
        return None
