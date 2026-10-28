"""
Combined Strategy — runs all strategies and votes.

Phase 2–3 allocators (SelfDrivingPortfolio, FINMEM engine, Almgren-Chriss,
VCP sidecar) are not imported here. Live path is the three technical
strategies + regime weights only.

v2 — Regime-Aware Weights
--------------------------
Previous bug: fixed weights (trend=0.50, mean_rev=0.20, breakout=0.30) applied
regardless of market conditions. When market transitioned to RANGING, trend_following
kept winning due to first-priority override, generating continuous false signals.

Fix:
  - Accepts `regime` parameter from MarketRegimeDetector
  - Weights shift dynamically based on regime:
      TRENDING  → trend=0.60, mean_rev=0.10, breakout=0.30
      RANGING   → mean_rev=0.60, trend=0.15, breakout=0.25
      VOLATILE  → breakout=0.65, trend=0.20, mean_rev=0.15
      BREAKOUT  → breakout=0.65, trend=0.25, mean_rev=0.10
  - Priority override threshold adapts to regime (RANGING blocks weak trend signals)
  - Kronos ML model signal now gates execution as a veto layer (in trading_loop.py)

Priority Logic:
  1. Strong single signal >= regime_threshold → pass through directly
  2. Consensus: weighted average when multiple strategies agree
  3. Moderate single signal (>= 0.40) → pass through
  4. No clear signal → NEUTRAL
"""

from __future__ import annotations

from backend.strategies.base import BaseStrategy, StrategySignal
from backend.strategies.trend_following import TrendFollowingStrategy
from backend.strategies.mean_reversion import MeanReversionStrategy
from backend.strategies.breakout import BreakoutStrategy

import json
from pathlib import Path

# Legacy module-level globals kept as fallback defaults if config is missing
_DEFAULT_WEIGHTS = {"trend_following": 0.40, "mean_reversion": 0.30, "breakout": 0.30}
_REGIME_STRONG_THRESHOLD: dict[str, float] = {
    "TRENDING": 0.52, "RANGING":  0.78, "VOLATILE": 0.62, "BREAKOUT": 0.55, "UNKNOWN":  0.55,
}
_REGIME_PRIORITY: dict[str, list[str]] = {
    "TRENDING": ["trend_following", "breakout", "mean_reversion"],
    "RANGING":  ["mean_reversion", "breakout", "trend_following"],
    "VOLATILE": ["breakout", "trend_following", "mean_reversion"],
    "BREAKOUT": ["breakout", "trend_following", "mean_reversion"],
    "UNKNOWN":  ["trend_following", "breakout", "mean_reversion"],
}


class CombinedStrategy(BaseStrategy):
    name = "combined"

    def __init__(self, weights: dict = None):
        self._trend = TrendFollowingStrategy()
        self._mean  = MeanReversionStrategy()
        self._break = BreakoutStrategy()
        
        # Load JSON config
        self._config = {}
        config_path = Path(__file__).parent / "config" / "combined.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to load combined.json: {e}")

        # weights here are the FALLBACK when no regime is passed
        if weights:
            self._default_weights = weights
        else:
            self._default_weights = self._config.get("DEFAULT_WEIGHTS", _DEFAULT_WEIGHTS)
        
        self._regime_thresholds = self._config.get("REGIME_STRONG_THRESHOLD", _REGIME_STRONG_THRESHOLD)
        self._regime_priority = self._config.get("REGIME_PRIORITY", _REGIME_PRIORITY)

    def generate_signal(
        self,
        symbol: str,
        bars: list[dict],
        regime: str = "UNKNOWN",
        regime_weights: dict | None = None,
        **kwargs,
    ) -> StrategySignal:
        """
        Generate a combined signal.

        Args:
            symbol:         Trading symbol
            bars:           OHLCV bar list
            regime:         Market regime string (TRENDING/RANGING/VOLATILE/BREAKOUT)
            regime_weights: Weight dict from MarketRegimeDetector, overrides default
        """
        # ── Resolve weights and thresholds for this regime ────────────────────
        weights        = regime_weights or self._default_weights
        strong_thresh  = self._regime_thresholds.get(regime, self._regime_thresholds.get("UNKNOWN", 0.55))
        priority_order = self._regime_priority.get(regime, self._regime_priority.get("UNKNOWN", ["trend_following", "breakout", "mean_reversion"]))

        sigs = {
            "trend_following": self._trend.generate_signal(symbol, bars),
            "mean_reversion":  self._mean.generate_signal(symbol, bars),
            "breakout":        self._break.generate_signal(symbol, bars),
        }

        price = bars[-1]["close"] if bars else 0

        # ── 1. Strong single signal check (regime-aware threshold & priority) ─
        for strat_name in priority_order:
            sig = sigs[strat_name]
            if sig.signal in ("BUY", "SELL") and sig.confidence >= strong_thresh:
                return StrategySignal(
                    symbol=symbol,
                    signal=sig.signal,
                    confidence=sig.confidence,
                    entry_price=price,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    strategy=self.name,
                    reasoning=(
                        f"STRONG[{regime}] {strat_name}: "
                        f"{sig.signal}({sig.confidence:.2f}) "
                        f"thresh={strong_thresh:.2f} | {sig.reasoning or ''}"
                    ),
                )

        # ── 2. Weighted consensus vote ────────────────────────────────────────
        buy_score  = 0.0
        sell_score = 0.0
        buy_reasons  = []
        sell_reasons = []

        for strat_name, sig in sigs.items():
            w = weights.get(strat_name, 0.33)
            if sig.signal == "BUY":
                buy_score  += w * sig.confidence
                buy_reasons.append(f"{strat_name.split('_')[0]}:BUY({sig.confidence:.2f})")
            elif sig.signal == "SELL":
                sell_score += w * sig.confidence
                sell_reasons.append(f"{strat_name.split('_')[0]}:SELL({sig.confidence:.2f})")

        buy_count  = sum(1 for s in sigs.values() if s.signal == "BUY")
        sell_count = sum(1 for s in sigs.values() if s.signal == "SELL")

        consensus_threshold = 0.20

        if buy_score > sell_score and buy_score >= consensus_threshold:
            best = max(
                [s for s in sigs.values() if s.signal == "BUY"],
                key=lambda s: s.confidence,
                default=sigs["trend_following"],
            )
            # Boost more when multiple strategies agree
            boost = 1.5 if buy_count >= 2 else 1.2
            conf  = min(buy_score * boost, 1.0)
            return StrategySignal(
                symbol=symbol, signal="BUY",
                confidence=conf,
                entry_price=price,
                stop_loss=best.stop_loss,
                take_profit=best.take_profit,
                strategy=self.name,
                reasoning=(
                    f"CONSENSUS[{regime}]({buy_count}): "
                    + " | ".join(buy_reasons)
                    + f" weights={weights}"
                ),
            )

        if sell_score > buy_score and sell_score >= consensus_threshold:
            best = max(
                [s for s in sigs.values() if s.signal == "SELL"],
                key=lambda s: s.confidence,
                default=sigs["trend_following"],
            )
            boost = 1.5 if sell_count >= 2 else 1.2
            conf  = min(sell_score * boost, 1.0)
            return StrategySignal(
                symbol=symbol, signal="SELL",
                confidence=conf,
                entry_price=price,
                stop_loss=best.stop_loss,
                take_profit=best.take_profit,
                strategy=self.name,
                reasoning=(
                    f"CONSENSUS[{regime}]({sell_count}): "
                    + " | ".join(sell_reasons)
                    + f" weights={weights}"
                ),
            )

        # ── 3. Moderate single signal passthrough (regime-aware priority) ─────
        for strat_name in priority_order:
            sig = sigs[strat_name]
            if sig.signal in ("BUY", "SELL") and sig.confidence >= 0.55:
                return StrategySignal(
                    symbol=symbol,
                    signal=sig.signal,
                    confidence=sig.confidence,
                    entry_price=price,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    strategy=self.name,
                    reasoning=(
                        f"MODERATE[{regime}] {strat_name}: "
                        f"{sig.signal}({sig.confidence:.2f})"
                    ),
                )

        # ── 4. No clear signal ────────────────────────────────────────────────
        # Surface best sub-strategy partial score on the dashboard (not a flat 5%
        # for every pair when setups are near-miss).
        partial_peak = max((s.confidence for s in sigs.values()), default=0.0)
        best_conf = max(buy_score, sell_score, partial_peak, 0.05)
        return StrategySignal(
            symbol=symbol, signal="NEUTRAL",
            confidence=best_conf,
            strategy=self.name,
            reasoning=(
                f"No clear signal [{regime}]: "
                f"BUY={buy_score:.2f} SELL={sell_score:.2f} "
                f"peak={partial_peak:.2f} weights={weights}"
            ),
        )


# ── Strategy factory ──────────────────────────────────────────────────────────
def get_strategy(name: str):
    from backend.strategies.scalping import ScalpingStrategy
    strategies = {
        "trend_following": TrendFollowingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
        "scalping": ScalpingStrategy,
        "combined": CombinedStrategy,
    }
    cls = strategies.get(name, CombinedStrategy)
    return cls()
