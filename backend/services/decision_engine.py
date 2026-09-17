import os
import logging
from typing import Optional, Dict, Any, List, Never, Tuple
from dataclasses import dataclass

from sqlalchemy import func

from backend.database.connection import SessionLocal
from backend.database.models import Trade
from backend.ml.promotion_service import (
    PromotionState,
    apply_live_signal_check,
    log_shadow_prediction,
    promotion_required,
    resolve_promotion,
)
from backend.services.equity_scope import risk_broker_name
from backend.services.jesse_bridge import jesse_bridge
from backend.services.jesse_ml_gates import (
    STRATEGY_PAYOFF_RATIO,
    calculate_fractional_kelly,
    clip_kelly_for_thin_book,
    empirical_payoff_ratio,
)
from backend.services.kronos_gate import apply_kronos_gate
from backend.services.opinion_layer import analyze_symbol as opinion_analyze
from backend.services.risk_config import RiskConfig
from backend.services.skill_miner import skill_miner
from backend.services.symbol_aliases import (
    is_opposing_side,
    normalize_side,
)
from backend.services.trading_mode import TradingMode, get_trading_mode, live_exchange_orders_allowed
from backend.services import kronos_service
from backend.strategies.combined import CombinedStrategy
from backend.strategies.market_regime import MarketRegimeDetector

logger = logging.getLogger(__name__)


def affordable_notional(
    equity: float,
    available: float,
    leverage: float,
    max_equity_mult: float,
    max_margin_use_pct: float,
) -> float:
    """Largest notional a book can take without blowing the margin check."""
    caps: list[float] = []
    if equity > 0 and max_equity_mult > 0:
        caps.append(equity * max_equity_mult)
    avail = available if available > 0 else equity
    lev = leverage if leverage > 0 else 1.0
    frac = max_margin_use_pct if max_margin_use_pct > 0 else 0.95
    if avail > 0:
        caps.append(avail * lev * frac)
    return min(caps) if caps else 0.0


def apply_hard_notional_caps(
    notional: float,
    *,
    trade_usdt: float,
    max_directional_usdt: float,
    leverage: float,
    live: bool,
    exchange_min_notional: float = 0.0,
) -> tuple[float, str | None]:
    """Clip a single order so it cannot breach configured USDT caps.

    Live per-order cap is trade_usdt * leverage when that is at least the
    venue minimum; otherwise only max_directional_exposure_usdt binds (so a
    $10 paper-style trade_usdt cannot veto Binance's $20 min notional).
    """
    if notional <= 0:
        return 0.0, "non-positive notional"
    caps: list[float] = []
    if max_directional_usdt and max_directional_usdt > 0:
        caps.append(float(max_directional_usdt))
    if live:
        lev = max(float(leverage or 1.0), 1.0)
        per_order = max(float(trade_usdt or 0.0), 0.0) * lev
        if per_order >= max(float(exchange_min_notional or 0.0), 0.0):
            caps.append(per_order)
    if not caps:
        return float(notional), None
    cap = min(caps)
    if exchange_min_notional > 0 and cap < exchange_min_notional:
        cap = float(exchange_min_notional)
    if notional > cap:
        logger.warning(
            "Hard notional cap: clipping $%.2f -> $%.2f (live=%s trade_usdt=%.2f dir_max=%.2f lev=%.1f)",
            notional, cap, live, trade_usdt, max_directional_usdt, leverage,
        )
        return cap, None
    return float(notional), None


def pyramid_price_improved(
    direction: str, current_price: float, last_layer_price: float, minimum: float,
) -> bool:
    """Pyramid only in the profitable direction by at least `minimum`."""
    if direction == "BUY":
        return current_price >= last_layer_price * (1 + minimum)
    if direction == "SELL":
        return current_price <= last_layer_price * (1 - minimum)
    return False


def pyramid_position_underwater(
    direction: str, entry_price: float, current_price: float,
) -> bool:
    """True when mark has moved against the open position vs its entry."""
    if not entry_price or not current_price:
        return False
    if direction == "BUY":
        return current_price < entry_price
    if direction == "SELL":
        return current_price > entry_price
    return False


def _reviewer_gate_fail_open() -> bool:
    """Whether an unexpected error in the risk-reviewer GATE may let a trade through.

    Mirrors risk_reviewer._reviewer_outage_fail_open: fail-open is fine for
    paper/backtest, but in LIVE mode an errored veto gate must block the trade
    instead of silently approving it. RISK_REVIEWER_FAIL_OPEN=true overrides.
    """
    if os.getenv("RISK_REVIEWER_FAIL_OPEN", "false").lower() == "true":
        return True
    return get_trading_mode() != TradingMode.LIVE


def _sentiment_filter_enabled() -> bool:
    """Optional news-sentiment veto/dampen on Decision Engine entries.

    Default OFF. Native sentiment_loop and n8n still write Influx/Qdrant for
    Grafana dashboards and the opinion layer's social weight; they do not
    gate size or entries unless SENTIMENT_FILTER_ENABLED=true. That is
    intentional — do not force-enable on live. Fail-closed entry filters
    (Kronos, Jesse, geometry, risk guard) stay independent of this flag.
    """
    return os.getenv("SENTIMENT_FILTER_ENABLED", "false").lower() == "true"



def _positive_price_level(level: Optional[float]) -> Optional[float]:
    """Treat 0 / negative / unparseable as 'no level' — never a valid stop or target."""
    if level is None:
        return None
    try:
        value = float(level)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def atr_from_bars(bars: List[Dict[str, Any]], fallback_price: float, periods: int = 14) -> float:
    """True-range ATR using aligned prev-close windows (avoids length mismatch on short series)."""
    if not bars:
        return fallback_price * 0.02
    window = bars[-(periods + 1):] if len(bars) >= periods + 1 else bars
    if len(window) < 2:
        return fallback_price * 0.02
    trs = []
    for i in range(1, len(window)):
        h = window[i]["high"]
        l_val = window[i]["low"]
        prev_c = window[i - 1]["close"]
        trs.append(max(h - l_val, abs(h - prev_c), abs(l_val - prev_c)))
    return sum(trs) / len(trs) if trs else fallback_price * 0.02


def compute_sl_tp_levels(
    bars: List[Dict[str, Any]],
    direction: str,
    entry_price: float,
    config: RiskConfig,
    signal_sl: Optional[float] = None,
    signal_tp: Optional[float] = None,
) -> tuple[float, float]:
    """ATR-based stop-loss and take-profit for an entry (shared by loop + manual orders)."""
    try:
        periods = int(getattr(config, "atr_period", 14) or 14)
        atr = atr_from_bars(bars, entry_price, periods=periods)
    except Exception:
        atr = entry_price * 0.02

    # 0.0 used to be treated as a real stop. min(0, entry - ATR) kept it, then
    # Binance stripped stop_loss<=0 as "missing" and the long sat naked.
    signal_sl = _positive_price_level(signal_sl)
    signal_tp = _positive_price_level(signal_tp)

    if direction == "BUY":
        sl = signal_sl if signal_sl is not None else (entry_price - (atr * config.sl_atr_mult))
        tp = signal_tp if signal_tp is not None else (entry_price + (atr * config.tp_atr_mult))
        sl = min(sl, entry_price - (atr * config.sl_atr_mult))
        tp = max(tp, entry_price + (atr * config.tp_atr_mult))
    else:
        sl = signal_sl if signal_sl is not None else (entry_price + (atr * config.sl_atr_mult))
        tp = signal_tp if signal_tp is not None else (entry_price - (atr * config.tp_atr_mult))
        sl = max(sl, entry_price + (atr * config.sl_atr_mult))
        tp = min(tp, entry_price - (atr * config.tp_atr_mult))
    return sl, tp


@dataclass
class Decision:
    action: str  # "BUY", "SELL", "HOLD", "CLOSE_LONG", "CLOSE_SHORT"
    symbol: str
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    is_pyramid: bool = False
    kelly_multiplier: Optional[float] = None

class DecisionEngine:
    def __init__(
        self,
        risk_config: RiskConfig,
        regime_detector: Optional[MarketRegimeDetector] = None,
    ):
        self.config = risk_config
        self.strategy = CombinedStrategy()
        # Injected by the trading loop so per-symbol history survives cycles.
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.enable_kronos = os.getenv("ENABLE_KRONOS", "true").lower() == "true"
        self.enable_jesse_ml = os.getenv("JESSE_ML_GATE_ENABLED", "true").lower() == "true" and getattr(self.config, "enable_jesse_ml", True)
        # Promotion contract is fail-closed when artifacts exist. None = legacy ML path.
        self.promotion_state: Optional[PromotionState] = resolve_promotion(self.config)
        # Snapshot of the most recent evaluation so the loop can persist a
        # signal row for EVERY symbol it scans (not just executed trades).
        self.last_evaluation: Dict[str, Any] = {}
        # Live account equity (set by the loop each cycle) for risk-based sizing.
        # 0 → fall back to fixed trade_usdt_amount notional.
        self.account_equity: float = 0.0
        # Available cash/margin and session leverage (paper 1x vs Binance 10x).
        self.account_available: float = 0.0
        self.account_leverage: float = 1.0

    def _record_eval(
        self,
        symbol,
        direction,
        confidence,
        reason,
        entry=None,
        sl=None,
        tp=None,
        approved=False,
        kelly_multiplier=None,
    ):
        self.last_evaluation = {
            "symbol": symbol,
            "direction": (direction or "HOLD").upper(),
            "confidence": float(confidence or 0.0),
            "reason": reason,
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "approved": approved,
            "kelly_multiplier": kelly_multiplier,
        }

    def _ranging_entries_permitted(self) -> bool:
        """Config + mode gate. Default is fully off — including paper.

        allow_ranging_entries unlocks the ranging *path*. Live still needs
        allow_ranging_in_live so a paper experiment cannot leak into production.
        """
        if not getattr(self.config, "allow_ranging_entries", False):
            return False
        if get_trading_mode() == TradingMode.LIVE and not getattr(
            self.config, "allow_ranging_in_live", False
        ):
            return False
        return True

    def _ranging_setup_matches(self, symbol: str, signal: Any) -> bool:
        """Allow ranging only for mean-reversion or a matching mined skill.

        Does not invent win-rate constants; uses CombinedStrategy's source
        name/reasoning and skill_miner.match_skill. Fail-closed on errors.
        """
        strat = str(getattr(signal, "strategy", "") or "").lower()
        reasoning = str(getattr(signal, "reasoning", "") or "").lower()
        if strat == "mean_reversion":
            return True
        if "mean_reversion" in reasoning or "mean_rev" in reasoning:
            return True
        return self._ranging_skill_match(symbol, signal)

    def _ranging_skill_match(self, symbol: str, signal: Any) -> bool:
        """True when skill miner has a positive-edge skill aligned with this setup."""
        try:
            direction = str(getattr(signal, "signal", None) or getattr(signal, "direction", "") or "").upper()
            ctx = {
                "symbol": symbol,
                "direction": direction,
                "regime": "RANGING",
                "mean_reversion_signal": direction,
            }
            skill = skill_miner.match_skill(ctx)
            if not skill:
                return False
            skill_dir = str(skill.get("direction") or "").lower()
            wanted = "bullish" if direction == "BUY" else "bearish" if direction == "SELL" else ""
            if wanted and skill_dir not in (wanted, "neutral"):
                return False
            name = str(skill.get("name") or "").lower()
            desc = str(skill.get("description") or "").lower()
            looks_ranging = any(
                token in name or token in desc
                for token in ("rang", "mean-rev", "mean_rev", "mean rev", "chop")
            )
            if not looks_ranging:
                summary = skill.get("feature_summary") or {}
                if isinstance(summary, dict):
                    try:
                        looks_ranging = abs(float(summary.get("regime", 1.0))) <= 0.25
                    except (TypeError, ValueError):
                        looks_ranging = "rang" in str(summary.get("regime", "")).lower()
            if not looks_ranging:
                return False
            edge = float(skill.get("edge_score") or 0.0)
            avg_pnl = float(skill.get("avg_pnl") or 0.0)
            return edge > 0.0 or avg_pnl > 0.0
        except Exception as err:
            logger.warning("[%s] ranging skill match failed (fail-closed): %s", symbol, err)
            return False

    def _ranging_entry_allowed(self, symbol: str, signal: Any) -> bool:
        return self._ranging_entries_permitted() and self._ranging_setup_matches(symbol, signal)

    @staticmethod
    def _position_field(position: Any, name: str, default: Any = None) -> Any:
        if position is None:
            return default
        if isinstance(position, dict):
            return position.get(name, default)
        return getattr(position, name, default)

    def _flatten_opposing_position(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        existing_position: Any,
    ) -> Optional[Decision]:
        """CLOSE_LONG / CLOSE_SHORT when Combined fires opposite the live leg.

        BTCUSDT / BTCUSDC are resolved by the loop before this is called.
        Does not open a reverse slot here — next cycle can enter if caps allow.
        """
        existing_side = normalize_side(self._position_field(existing_position, "direction"))
        live_symbol = str(self._position_field(existing_position, "symbol") or symbol or "").upper()
        try:
            qty = float(self._position_field(existing_position, "quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if existing_side not in ("BUY", "SELL") or qty <= 0:
            return None
        regime_result = self.regime_detector.detect(bars)
        signal = self.strategy.generate_signal(
            symbol,
            bars,
            regime=regime_result.regime,
            regime_weights=regime_result.weights(),
        )
        if not signal or signal.signal not in ("BUY", "SELL"):
            self._record_eval(
                symbol,
                existing_side,
                float(getattr(signal, "confidence", 0.0) or 0.0) if signal else 0.0,
                "existing position; no opposing signal",
            )
            return None
        if not is_opposing_side(signal.signal, existing_side):
            self._record_eval(
                symbol,
                existing_side,
                signal.confidence,
                "existing same-direction position — skip entry",
            )
            return None
        if signal.confidence < self.config.min_signal_strength:
            self._record_eval(
                symbol,
                signal.signal,
                signal.confidence,
                "opposing signal below threshold; hold existing",
            )
            return None
        action = "CLOSE_LONG" if existing_side == "BUY" else "CLOSE_SHORT"
        decision = Decision(
            action=action,
            symbol=live_symbol,
            quantity=qty,
            entry_price=float(bars[-1]["close"]),
            confidence=float(signal.confidence),
            reasoning=(
                f"opposing {signal.signal} flatten of {existing_side} {live_symbol}"
            ),
        )
        self._record_eval(
            live_symbol,
            action,
            signal.confidence,
            "opposing flatten",
            entry=decision.entry_price,
            approved=True,
        )
        logger.info(
            "[%s] opposing %s (conf=%.2f) flattens live %s %s qty=%s",
            symbol,
            signal.signal,
            signal.confidence,
            existing_side,
            live_symbol,
            qty,
        )
        return decision

    async def evaluate_symbol(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        existing_position: Optional[Any],  # DB Trade object or dict
        open_count: int,
        pyramid_layers: List[float],
        cooldown_active: bool,
        current_funding_rate: float = 0.0
    ) -> Optional[Decision]:
        """
        Evaluate market data and return a trading Decision.
        Does NOT execute trades or interact with the database.
        """
        decision = await self._evaluate_symbol_internal(
            symbol=symbol,
            bars=bars,
            existing_position=existing_position,
            open_count=open_count,
            pyramid_layers=pyramid_layers,
            cooldown_active=cooldown_active,
            current_funding_rate=current_funding_rate,
        )
        try:
            self._dispatch_research_ingest(symbol, decision)
        except Exception as exc:
            logger.warning(f"Swallowed decision ingest dispatch exception for {symbol}: {exc}")
        return decision

    def _dispatch_research_ingest(self, symbol: str, decision: Optional[Decision]) -> None:
        """Non-blocking fire-and-forget background task dispatch.

        Never awaits or blocks the trading loop event cycle.
        """
        try:
            from backend.services.research_ingest import ingest_decision, is_research_ingest_enabled
            if not is_research_ingest_enabled():
                return
            eval_data = getattr(self, "last_evaluation", None) or {}
            promo = getattr(self, "promotion_state", None)
            promo_verdict = getattr(promo, "verdict", None) or eval_data.get("promotion_verdict")
            promo_reason = getattr(promo, "reason", None) or eval_data.get("promotion_reason")
            is_shadow = bool(getattr(promo, "shadow", False))
            is_rejected = (decision is None) or not bool(eval_data.get("approved", False))

            gate_id = eval_data.get("gate_id")
            reason_str = str(eval_data.get("reason", ""))
            if not gate_id and is_rejected and reason_str:
                lower_reason = reason_str.lower()
                if "risk reviewer" in lower_reason:
                    gate_id = "RISK_REVIEWER"
                elif "event risk" in lower_reason:
                    gate_id = "EVENT_RISK_FILTER"
                elif "promotion" in lower_reason:
                    gate_id = "QTP_PROMOTION"
                elif "jesse" in lower_reason:
                    gate_id = "JESSE_ML"
                elif "max positions" in lower_reason:
                    gate_id = "MAX_POSITIONS"
                elif "pyramid" in lower_reason:
                    gate_id = "PYRAMID_GATE"
                elif "ranging" in lower_reason:
                    gate_id = "REGIME_RANGING"

            payload = {
                "symbol": symbol,
                "broker": "binance",
                "mode": str(get_trading_mode().value if hasattr(get_trading_mode(), "value") else get_trading_mode()),
                "signal": str(eval_data.get("direction", "HOLD")),
                "confidence": float(eval_data.get("confidence", 0.0)),
                "promotion_verdict": promo_verdict,
                "promotion_reason": promo_reason,
                "gate_id": gate_id,
                "reason": reason_str,
                "shadow": is_shadow,
                "rejected": is_rejected,
            }

            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(ingest_decision(payload))
            except RuntimeError:
                pass
        except Exception as exc:
            logger.warning(f"Error in decision ingest dispatch for {symbol}: {exc}")


    async def _evaluate_symbol_internal(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        existing_position: Optional[Any],  # DB Trade object or dict
        open_count: int,
        pyramid_layers: List[float],
        cooldown_active: bool,
        current_funding_rate: float = 0.0
    ) -> Optional[Decision]:

        if not bars or len(bars) < 50:
            self._record_eval(symbol, "HOLD", 0.0, "insufficient bars")
            return None
            
        self._record_eval(symbol, "HOLD", 0.0, "evaluating")
        
        # 1. Active position logic
        if existing_position:
            # Check pyramid
            if self.config.pyramid_mode:
                regime_result = self.regime_detector.detect(bars)
                if regime_result.regime == "RANGING":
                    logger.info(f"[{symbol}] Pyramiding blocked: market is in RANGING/CHOP regime.")
                    return None
                if len(pyramid_layers) < self.config.pyramid_max_layers:
                    cur_px = bars[-1]["close"]
                    entry_px = float(getattr(existing_position, "entry_price", 0) or 0)
                    direction = getattr(existing_position, "direction", None)
                    if (
                        getattr(self.config, "pyramid_block_underwater", True)
                        and direction
                        and pyramid_position_underwater(direction, entry_px, cur_px)
                    ):
                        self._record_eval(
                            symbol, direction, 0.0,
                            f"pyramid blocked: underwater vs entry {entry_px} (mark {cur_px})",
                        )
                        logger.info(
                            f"[{symbol}] Pyramiding blocked: position underwater "
                            f"(entry={entry_px}, mark={cur_px})."
                        )
                        return None
                    # Strategy signal for pyramid
                    signal = self.strategy.generate_signal(
                        symbol,
                        bars,
                        regime=regime_result.regime,
                        regime_weights=regime_result.weights()
                    )
                    if signal and signal.signal == existing_position.direction and signal.confidence >= self.config.min_signal_strength:
                        # Optional: require confidence to rise vs prior layer
                        if pyramid_layers and self.config.pyramid_min_conf_increase > 0:
                            # pyramid_layers length tracks layers; no per-layer conf stored —
                            # gate is best-effort on current signal only.
                            pass
                        # Optional price gate (PYRAMID_MIN_IMPROVEMENT=0 → add every cycle).
                        if pyramid_layers and self.config.pyramid_min_improvement > 0:
                            last_px = pyramid_layers[-1]
                            imp = self.config.pyramid_min_improvement
                            # LONG pyramid: add on strength (price higher).
                            # SHORT pyramid/DCA: add when price moved vs last layer.
                            blocked = not pyramid_price_improved(
                                signal.signal, cur_px, last_px, imp,
                            )
                            if blocked:
                                self._record_eval(
                                    symbol, signal.signal, signal.confidence,
                                    f"pyramid: price gate {imp:.1%} not met vs layer @ {last_px}",
                                )
                                return None
                        decision = self._create_entry_decision(
                            symbol, bars, signal, existing_position.direction, is_pyramid=True
                        )
                        if decision:
                            self._record_eval(
                                symbol, decision.action, decision.confidence, "pyramid entry",
                                entry=decision.entry_price, sl=decision.stop_loss,
                                tp=decision.take_profit, approved=True,
                            )
                        if decision and getattr(self.config, "use_risk_reviewer_llm", True):
                            try:
                                from backend.services.risk_reviewer import fetch_news_summary, review_trade_decision
                                news_summary = await fetch_news_summary(symbol)
                                approved, reasoning = await review_trade_decision(
                                    symbol=symbol,
                                    action=decision.action,
                                    quantity=decision.quantity,
                                    entry_price=decision.entry_price,
                                    stop_loss=decision.stop_loss,
                                    take_profit=decision.take_profit,
                                    confidence=decision.confidence,
                                    funding_rate=current_funding_rate,
                                    news_summary=news_summary
                                )
                                if not approved:
                                    logger.warning(f"[{symbol}] Pyramid add VETOED by Risk Reviewer: {reasoning}")
                                    return None
                                logger.info(f"[{symbol}] Pyramid add APPROVED by Risk Reviewer: {reasoning}")
                                decision.reasoning += f" | Risk Reviewer: {reasoning}"
                            except Exception as e:
                                logger.error(f"[{symbol}] Error in LLM Risk Reviewer gate for pyramid add: {e}")
                                if not _reviewer_gate_fail_open():
                                    self._record_eval(symbol, decision.action, decision.confidence,
                                                      "pyramid add blocked: risk reviewer gate errored (fail-closed in live)")
                                    return None
                        return decision
            flatten = self._flatten_opposing_position(symbol, bars, existing_position)
            if flatten:
                return flatten
            return None

        # 2. Cooldown check
        if cooldown_active:
            return None

        # 3. Strategy execution — detect regime and generate signal.
        #    Runs FIRST so the expensive Kronos/opinion calls below are only
        #    paid when there's an actual directional setup (most pairs are
        #    NEUTRAL and short-circuit here).
        regime_result = self.regime_detector.detect(bars)
        signal = self.strategy.generate_signal(
            symbol,
            bars,
            regime=regime_result.regime,
            regime_weights=regime_result.weights()
        )
        if not signal or signal.signal not in ["BUY", "SELL"]:
            self._record_eval(
                symbol,
                signal.signal if signal else "HOLD",
                signal.confidence if signal else 0.0,
                "no strategy signal",
            )
            return None

        # 3b. Early RANGING regime block — skip BEFORE paying for Kronos/LLM
        # unless the config-gated ranging path matches mean-reversion / a mined
        # skill. Default (flag off) preserves the historical hard block.
        if regime_result.regime == "RANGING":
            if not self._ranging_entry_allowed(symbol, signal):
                self._record_eval(
                    symbol, signal.signal, signal.confidence,
                    "RANGING regime: blocked early (saves Kronos/LLM cost)",
                )
                return None
            logger.info(
                f"[{symbol}] RANGING regime: allowing matching setup "
                f"(strategy={getattr(signal, 'strategy', '')})"
            )

        # Adjust signal confidence based on the perp funding rate
        # Funding rate units on Binance: 0.0001 = 0.01% per 8h.
        # Clamp the adjustment: unclamped, a routine 0.01% funding moved
        # confidence by ±0.10, systematically flipping marginal signals toward
        # shorts (measured live: BUY win rate 22.7% vs SELL 53%). Funding is a
        # carry-cost nudge, not a directional signal — cap its influence.
        funding_adj = current_funding_rate * 1000.0
        funding_cap = getattr(self.config, "funding_conf_adj_cap", 0.05)
        funding_adj = max(-funding_cap, min(funding_cap, funding_adj))
        if signal.signal == "SELL":
            # Boost shorts if positive funding (we get paid to hold)
            old_conf = signal.confidence
            signal.confidence = max(0.0, min(1.0, signal.confidence + funding_adj))
            if funding_adj != 0:
                logger.info(f"[{symbol}] SHORT confidence adjusted by funding rate ({current_funding_rate*100:.4f}%): {old_conf:.2f} -> {signal.confidence:.2f}")
        elif signal.signal == "BUY":
            # Dampen longs if positive funding (we pay to hold)
            old_conf = signal.confidence
            signal.confidence = max(0.0, min(1.0, signal.confidence - funding_adj))
            if funding_adj != 0:
                logger.info(f"[{symbol}] LONG confidence adjusted by funding rate ({current_funding_rate*100:.4f}%): {old_conf:.2f} -> {signal.confidence:.2f}")

        # Regime-aware confidence gate
        required_gate = self.config.min_signal_strength + (0.15 if regime_result.regime == "RANGING" else 0.0)
        if signal.confidence < required_gate:
            self._record_eval(
                symbol,
                signal.signal,
                signal.confidence,
                f"strategy confidence below threshold ({required_gate:.2f}) in {regime_result.regime} regime",
            )
            return None

        # Block BUYs if funding rate is over cap
        fr_cap = getattr(self.config, "funding_rate_cap", 0.0)
        if signal.signal == "BUY" and fr_cap > 0 and current_funding_rate > fr_cap:
            logger.info(f"[{symbol}] BUY signal blocked by Funding Rate Gate ({current_funding_rate*100:.3f}% > {fr_cap*100:.3f}%)")
            # Keep BUY so the shadow tracker can score the blocked long.
            self._record_eval(symbol, signal.signal, signal.confidence, "blocked by funding rate")
            return None

        # 4. Multi-model Pre-Execution Gating (Kronos Sidecar + Heuristic Timing Guard)
        if self.enable_kronos:
            kronos_result = {}
            if bars:
                try:
                    kronos_result = await kronos_service.predict(bars, symbol)
                except Exception as e:
                    logger.warning(f"Kronos prediction failed for {symbol}: {e}")
                    kronos_result = {}

            # Optional Vision timing verification if enabled
            vision_approved = None
            if bars:
                try:
                    from backend.services.vision_timing import evaluate_vision_timing_optional
                    vision_approved = await evaluate_vision_timing_optional(
                        bars=bars,
                        symbol=symbol,
                        proposed_signal=signal.signal,
                    )
                except Exception as e:
                    logger.debug(f"Vision timing check notice for {symbol}: {e}")

            # 4b. Apply Pre-Execution Gate (shadow-aware; FLIP removed)
            gate_result = apply_kronos_gate(
                strategy_signal=signal.signal,
                strategy_confidence=signal.confidence,
                kronos_result=kronos_result,
                bars=bars,
                vision_approved=vision_approved,
                symbol=symbol,
            )
            if gate_result.action == "veto":
                if gate_result.final_signal == "NEUTRAL":
                    logger.info(f"[{symbol}] PreExecutionGate ACTIVE VETO: {gate_result.reasoning}")
                    # Keep the intended BUY/SELL — HOLD made Kronos vetoes unscorable.
                    self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed: {gate_result.reasoning}")
                    return None
                logger.info(f"[{symbol}] PreExecutionGate SHADOW VETO (allowed): {gate_result.reasoning}")
                self._record_eval(
                    symbol, "SHADOW_VETO", signal.confidence, f"shadow_vetoed: {gate_result.reasoning}",
                )
            elif gate_result.action == "boost":
                logger.info(f"[{symbol}] PreExecutionGate BOOST: {gate_result.reasoning}")
                signal.confidence = gate_result.confidence
            elif gate_result.action == "dampen":
                logger.info(f"[{symbol}] PreExecutionGate DAMPEN: {gate_result.reasoning}")
                signal.confidence = gate_result.confidence
            elif gate_result.action == "pass":
                pass
            elif gate_result.action == "flip":
                # Legacy: FLIP removed — treat as active veto.
                logger.info(f"[{symbol}] PreExecutionGate FLIP→VETO (legacy): {gate_result.reasoning}")
                return None
            else:
                _unreachable: Never = gate_result.action  # type: ignore[assignment]
                raise AssertionError(f"Unhandled PreExecutionGate action: {_unreachable}")

            # Re-check after gate modification
            if signal.signal not in ["BUY", "SELL"] or signal.confidence < self.config.min_signal_strength:
                return None

        # 4d. Soft News Sentiment Gate (Forex/Macro & Crypto recency-weighted sentiment)
        if _sentiment_filter_enabled():
            try:
                from backend.services.news_sentiment_service import news_sentiment_service
                sent_data = news_sentiment_service.get_pair_sentiment(symbol)
                sent_score = sent_data.get("recency_weighted_score", 0.0)
                sent_conf = sent_data.get("confidence", 0.0)

                # Hard Veto on extreme divergence with high confidence
                if signal.signal == "BUY" and sent_score < -0.35 and sent_conf >= 0.45:
                    logger.info(f"[{symbol}] Sentiment Gate VETO: Bearish sentiment ({sent_score:+.2f}, conf={sent_conf:.2f}) blocks BUY")
                    self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by sentiment gate: {sent_score:+.2f}")
                    return None
                elif signal.signal == "SELL" and sent_score > 0.35 and sent_conf >= 0.45:
                    logger.info(f"[{symbol}] Sentiment Gate VETO: Bullish sentiment ({sent_score:+.2f}, conf={sent_conf:.2f}) blocks SELL")
                    self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by sentiment gate: {sent_score:+.2f}")
                    return None

                # Soft Dampen on moderate disagreement
                if signal.signal == "BUY" and sent_score < -0.15:
                    dampen = max(0.05, min(0.20, abs(sent_score) * 0.3))
                    signal.confidence = max(0.0, signal.confidence - dampen)
                elif signal.signal == "SELL" and sent_score > 0.15:
                    dampen = max(0.05, min(0.20, abs(sent_score) * 0.3))
                    signal.confidence = max(0.0, signal.confidence - dampen)
                elif (signal.signal == "BUY" and sent_score > 0.15) or (signal.signal == "SELL" and sent_score < -0.15):
                    # Slight alignment boost
                    signal.confidence = min(1.0, signal.confidence + 0.05)

                if signal.confidence < self.config.min_signal_strength:
                    self._record_eval(symbol, signal.signal, signal.confidence, "confidence reduced below threshold by sentiment gate")
                    return None
            except Exception as e:
                logger.warning(f"[{symbol}] Sentiment gate evaluation error (failing neutral): {e}")

        # 4e. Jesse Machine Learning Directional Consensus & Meta-Label Gate
        if self.enable_jesse_ml:
            promo = self.promotion_state
            if promo is not None:
                self.last_evaluation["promotion_verdict"] = promo.verdict
                self.last_evaluation["promotion_reason"] = promo.result.reason
            if promo is not None and promo.reject_model:
                if live_exchange_orders_allowed() or promotion_required():
                    self._record_eval(
                        symbol, signal.signal, signal.confidence,
                        f"vetoed by QTP promotion contract ({promo.verdict}: {promo.result.reason})",
                    )
                    return None
                logger.info("[%s] Promotion %s — skipping model in paper (Core Hub strategy only)", symbol, promo.verdict)
            else:
                try:
                    ml_res = await jesse_bridge.get_ml_prediction(symbol=symbol, timeframe="1h")
                    if ml_res.get("status") == "success":
                        if promo is not None and promo.shadow:
                            # SHADOW: load artifact, log preds, size=0 from this model.
                            log_shadow_prediction(promo, symbol, ml_res)
                            setattr(signal, "kelly_multiplier", 0.0)
                            self.last_evaluation["shadow_ml"] = {
                                "signal": ml_res.get("signal"),
                                "confidence": ml_res.get("confidence"),
                            }
                        else:
                            ml_sig = ml_res.get("signal")
                            ml_conf = ml_res.get("confidence", 0.0)
                            probs = ml_res.get("probabilities", {})
                            p_bullish = probs.get("bullish", 0.0)
                            p_bearish = probs.get("bearish", 0.0)
                            uncertainty = ml_res.get("uncertainty", "LOW")
                            gated = ml_res.get("gated", False)
                            kelly = ml_res.get("kelly", {})

                            if promo is not None and promo.promote:
                                live_check = apply_live_signal_check(promo, {
                                    "side": ml_sig,
                                    "p_win": ml_res.get("p_win"),
                                    "conformal_width": ml_res.get("conformal_width"),
                                    "costed_edge_bps": ml_res.get("costed_edge_bps"),
                                })
                                missing_telemetry = "missing live telemetry" in (live_check.reason or "")
                                if missing_telemetry and not live_exchange_orders_allowed():
                                    logger.warning(
                                        f"[{symbol}] Promoted-model telemetry missing in paper — fail-open"
                                    )
                                elif live_check.applied and not live_check.allowed:
                                    self._record_eval(
                                        symbol, signal.signal, signal.confidence,
                                        f"vetoed by promoted-model live four-number check ({live_check.reason})",
                                    )
                                    return None

                            # Conformal Uncertainty Veto: reject trades in high ambiguity or wide conformal sets
                            if uncertainty == "HIGH" or gated:
                                reason = ml_res.get("gated_reason") or f"Conformal uncertainty is HIGH (margin={ml_res.get('conformal_margin', 0):.3f})"
                                logger.info(f"[{symbol}] Jesse ML Conformal Uncertainty Gate VETO: {reason} blocks {signal.signal}")
                                self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by Jesse ML uncertainty gate ({reason})")
                                return None

                            # Hard Veto if ML model opposes candidate direction
                            if signal.signal == "BUY" and (p_bearish >= 0.50 or ml_sig == "SELL"):
                                logger.info(f"[{symbol}] Jesse ML Gate VETO: Bearish drift ({p_bearish*100:.1f}%, sig={ml_sig}) opposes BUY")
                                self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by Jesse ML gate (bearish prob: {p_bearish:.2f})")
                                return None
                            elif signal.signal == "SELL" and (p_bullish >= 0.50 or ml_sig == "BUY"):
                                logger.info(f"[{symbol}] Jesse ML Gate VETO: Bullish drift ({p_bullish*100:.1f}%, sig={ml_sig}) opposes SELL")
                                self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by Jesse ML gate (bullish prob: {p_bullish:.2f})")
                                return None

                            # Confidence boost on directional consensus
                            if (signal.signal == "BUY" and ml_sig == "BUY") or (signal.signal == "SELL" and ml_sig == "SELL"):
                                boost = min(0.12, ml_conf * 0.15)
                                signal.confidence = min(1.0, signal.confidence + boost)
                                logger.info(f"[{symbol}] Jesse ML Gate BOOST: Consensus {ml_sig} (+{boost:.2f} -> conf={signal.confidence:.2f})")
                            elif (signal.signal == "BUY" and ml_sig == "SELL") or (signal.signal == "SELL" and ml_sig == "BUY"):
                                dampen = min(0.15, ml_conf * 0.15)
                                signal.confidence = max(0.0, signal.confidence - dampen)
                                logger.info(f"[{symbol}] Jesse ML Gate DAMPEN: Disagreement {ml_sig} (-{dampen:.2f} -> conf={signal.confidence:.2f})")

                            # Capture Fractional Kelly using empirical payoff b (avg win / avg |loss|)
                            # when the active partition has enough closed trades; otherwise the
                            # live 5.5/1.75 ATR geometry (≈3.14) replaces the hardcoded b=2.0.
                            win_p = p_bullish if signal.signal == "BUY" else p_bearish
                            setattr(signal, "kelly_win_prob", float(win_p))
                            if "size_multiplier" in kelly:
                                setattr(signal, "kelly_multiplier", float(kelly["size_multiplier"]))
                                logger.info(
                                    f"[{symbol}] Jesse ML Calibrated Edge: Kelly multiplier="
                                    f"{kelly['size_multiplier']}x (fractional_kelly={kelly.get('fractional_kelly', 0):.4f})"
                                )

                            if signal.confidence < self.config.min_signal_strength:
                                self._record_eval(symbol, signal.signal, signal.confidence, "confidence reduced below threshold by Jesse ML gate")
                                return None
                    else:
                        err = ml_res.get("error") or "Jesse ML status not success"
                        # Only the bridge may classify a gap as no_model (200 / 4xx).
                        # Do not re-parse error text here — a 5xx traceback that
                        # mentions artifacts is still an outage and must veto live.
                        if ml_res.get("status") == "no_model":
                            if live_exchange_orders_allowed():
                                logger.error(
                                    f"[{symbol}] Jesse ML gate no_model in LIVE mode ({err}) — fail closed: vetoing {signal.signal}"
                                )
                                self._record_eval(
                                    symbol, signal.signal, signal.confidence,
                                    f"vetoed by Jesse ML no_model in LIVE mode ({err})",
                                )
                                return None
                            logger.warning(
                                f"[{symbol}] Jesse ML gate skipped — no deployable model ({err})"
                            )
                            setattr(signal, "jesse_ml_gap", err)
                        elif live_exchange_orders_allowed():
                            logger.error(f"[{symbol}] Jesse ML gate returned non-success in LIVE mode ({err}) — fail closed: vetoing {signal.signal}")
                            self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by Jesse ML error in LIVE mode ({err})")
                            return None
                        else:
                            logger.debug(f"[{symbol}] Jesse ML returned non-success in paper mode ({err}) — skipping gate")
                except Exception as e:
                    if live_exchange_orders_allowed():
                        logger.error(f"[{symbol}] Jesse ML gate failed in LIVE mode ({e}) — fail closed: vetoing {signal.signal}")
                        self._record_eval(symbol, signal.signal, signal.confidence, f"vetoed by Jesse ML failure in LIVE mode ({e})")
                        return None
                    logger.debug(f"[{symbol}] Jesse ML gate evaluation notice (fail-open in paper): {e}")

        # 5. AI Opinion Layer — multi-agent weighted consensus
        if self.config.enable_personas:
            try:
                opinion = await opinion_analyze(
                    symbol=symbol,
                    bars=bars,
                    # Kronos already ran above; reuse cache via sidecar client,
                    # but skip a second opinion-layer forecast call.
                    include_kronos=False,
                    include_social=True,
                    include_alerts=True,
                    include_personas=True,
                )
                if opinion:
                    self._record_eval(symbol, opinion.direction, opinion.confidence,
                                      "AI opinion evaluated")
                    # Capture FINMEM cited memory IDs for downstream attribution and promotion
                    for a_op in getattr(opinion, "agent_opinions", []):
                        if a_op.agent == "finmem_cognitive_agent":
                            cited = (a_op.metadata or {}).get("cited_ids") or []
                            if cited:
                                setattr(signal, "finmem_cited_ids", cited)
                if opinion and opinion.confidence < self.config.ai_analysis_threshold:
                    # Strategy can override weak AI if its own confidence is high enough
                    if signal.confidence > (self.config.ai_analysis_threshold + self.config.opinion_override_margin):
                        logger.info(f"[{symbol}] Strategy overrides weak AI opinion (strategy conf={signal.confidence:.2f})")
                    else:
                        logger.info(
                            f"[{symbol}] AI opinion too weak (conf={opinion.confidence:.2f} < {self.config.ai_analysis_threshold}), skipping"
                        )
                        self._record_eval(symbol, signal.signal, opinion.confidence,
                                          f"AI opinion too weak (<{self.config.ai_analysis_threshold})")
                        return None
            except Exception as e:
                logger.warning(f"[{symbol}] Opinion layer error: {e}")
                # Fallback: require higher strategy confidence if opinion layer fails
                if signal.confidence < self.config.min_signal_strength + 0.1:
                    return None

        # 6. Max positions check (flatten/reverse of an existing leg is not a new slot)
        max_positions_cap = getattr(self.config, "max_binance_positions", self.config.max_positions)
        if open_count >= max_positions_cap:
            self._record_eval(symbol, signal.signal, signal.confidence,
                              f"max positions reached ({max_positions_cap})")
            return None
            
        # 7. Create Decision
        decision = self._create_entry_decision(symbol, bars, signal, signal.signal, is_pyramid=False, regime=regime_result.regime)
        if not decision:
            return None

        # 8. Single LLM Risk Reviewer (Veto Gate)
        if getattr(self.config, "use_risk_reviewer_llm", True):
            try:
                from backend.services.risk_reviewer import fetch_news_summary, review_trade_decision
                news_summary = await fetch_news_summary(symbol)
                approved, reasoning = await review_trade_decision(
                    symbol=symbol,
                    action=decision.action,
                    quantity=decision.quantity,
                    entry_price=decision.entry_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    confidence=decision.confidence,
                    funding_rate=current_funding_rate,
                    news_summary=news_summary
                )
                if not approved:
                    logger.warning(f"[{symbol}] VETOED by Risk Reviewer: {reasoning}")
                    self._record_eval(symbol, decision.action, decision.confidence, f"vetoed by risk reviewer: {reasoning}")
                    return None
                else:
                    logger.info(f"[{symbol}] APPROVED by Risk Reviewer: {reasoning}")
                    decision.reasoning += f" | Risk Reviewer: {reasoning}"
            except Exception as e:
                logger.error(f"[{symbol}] Error in LLM Risk Reviewer gate: {e}")
                if not _reviewer_gate_fail_open():
                    self._record_eval(symbol, decision.action, decision.confidence,
                                      "entry blocked: risk reviewer gate errored (fail-closed in live)")
                    return None

        # 9. Event-Risk Filter (Macro Economic Event Gate)
        if decision:
            try:
                from backend.services.event_risk_filter import event_risk_filter
                risk_res = event_risk_filter.evaluate_order(
                    symbol=symbol,
                    proposed_quantity=decision.quantity,
                    proposed_direction=decision.action,
                )
                if not risk_res.approved:
                    logger.warning(f"[{symbol}] Entry VETOED by Event-Risk Filter: {risk_res.reason}")
                    self._record_eval(symbol, decision.action, decision.confidence, f"vetoed by event risk filter: {risk_res.reason}")
                    return None
                if risk_res.action == "reduce":
                    logger.info(f"[{symbol}] Quantity REDUCED by Event-Risk Filter: {decision.quantity} -> {risk_res.final_quantity}")
                    decision.quantity = risk_res.final_quantity
                    decision.reasoning += f" | EventRisk: {risk_res.reason}"
            except Exception as e:
                logger.error(f"[{symbol}] Error in Event-Risk Filter gate: {e}")
                if get_trading_mode() == TradingMode.LIVE:
                    self._record_eval(symbol, decision.action, decision.confidence, "entry blocked: event risk filter errored (fail-closed in live)")
                    return None

        if decision:
            self._record_eval(symbol, decision.action, decision.confidence, "entry decision",
                              entry=decision.entry_price, sl=decision.stop_loss,
                              tp=decision.take_profit, approved=True,
                              kelly_multiplier=decision.kelly_multiplier)
            if self.promotion_state is not None:
                self.last_evaluation["promotion_verdict"] = self.promotion_state.verdict
                if self.promotion_state.shadow and self.promotion_state.logged_predictions:
                    last_pred = self.promotion_state.logged_predictions[-1]
                    self.last_evaluation["shadow_ml"] = {
                        "signal": last_pred.get("signal"),
                        "confidence": last_pred.get("confidence"),
                    }
        return decision

    def _partition_pnl_stats(self) -> Tuple[int, float, float]:
        """Closed-trade count, average win, and average |loss| for the active book partition."""
        closed_count = 0
        avg_win = 0.0
        avg_loss_abs = 0.0
        try:
            db = SessionLocal()
            try:
                active_broker = risk_broker_name().lower()
                current_mode = get_trading_mode().value if hasattr(get_trading_mode(), "value") else str(get_trading_mode())
                closed_filter = Trade.status.in_(["closed", "exit"])
                q = db.query(Trade).filter(closed_filter)
                if active_broker == "ctrader":
                    q = q.filter(Trade.broker.in_(["ctrader", "ctrader:paper", "ic", "icmarkets"]))
                elif active_broker in {"binance", "binance_futures"}:
                    q = q.filter(Trade.broker.in_(["binance", "binance_futures", "binanceusdm"]))
                q = q.filter(Trade.mode == current_mode.lower())
                closed_count = q.with_entities(func.count(Trade.id)).scalar() or 0
                avg_win = q.filter(Trade.pnl > 0).with_entities(func.avg(Trade.pnl)).scalar() or 0.0
                avg_loss_abs = (
                    q.filter(Trade.pnl < 0).with_entities(func.avg(func.abs(Trade.pnl))).scalar() or 0.0
                )
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Could not query partition PnL stats: {e}")
        return int(closed_count), float(avg_win or 0.0), float(avg_loss_abs or 0.0)

    def _create_entry_decision(self, symbol: str, bars: List[Dict[str, Any]], signal: Any, direction: str, is_pyramid: bool, regime: str = "UNKNOWN") -> Optional[Decision]:
        current_price = bars[-1]["close"]
        entry_price = signal.entry_price or current_price

        # SL/TP first — needed for risk-based position sizing.
        sl, tp = compute_sl_tp_levels(
            bars, direction, entry_price, self.config,
            signal_sl=signal.stop_loss, signal_tp=signal.take_profit,
        )

        # ── Position sizing ──
        # Risk-based: size so a SL hit costs ~risk_per_trade_pct of equity,
        # modulated by the Fractional Kelly multiplier from calibrated ML.
        # Falls back to the fixed trade_usdt_amount notional when equity or the
        # SL distance aren't usable. Pyramid layers keep their fixed notional.
        trade_usdt = self.config.pyramid_usdt_per_layer if is_pyramid else self.config.trade_usdt_amount
        kelly_mult = getattr(signal, "kelly_multiplier", None)
        win_prob = getattr(signal, "kelly_win_prob", None)

        closed_count, avg_win, avg_loss_abs = self._partition_pnl_stats()
        payoff_b = empirical_payoff_ratio(avg_win, avg_loss_abs, closed_count)

        if win_prob is not None and float(win_prob) > 0:
            recomputed = calculate_fractional_kelly(float(win_prob), payoff_ratio=payoff_b)
            kelly_mult = float(recomputed["size_multiplier"])
            logger.info(
                f"[{symbol}] Empirical Kelly b={payoff_b:.2f} "
                f"(geometry fallback={STRATEGY_PAYOFF_RATIO:.2f}, closed={closed_count}) "
                f"-> multiplier={kelly_mult:.3f}x"
            )

        if kelly_mult is not None and kelly_mult > 0:
            kelly_mult, was_clipped = clip_kelly_for_thin_book(float(kelly_mult), closed_count)
            if was_clipped:
                logger.info(
                    f"[{symbol}] Kelly multiplier clipped to {kelly_mult:.2f}x "
                    f"(unconditional 1.0 cap; thin-book floor 0.25 when closed={closed_count})"
                )

        # Apply Kelly sizing to baseline notional for primary entries
        if not is_pyramid and kelly_mult is not None and kelly_mult > 0:
            trade_usdt = trade_usdt * float(kelly_mult)
            logger.info(f"[{symbol}] Base notional scaled by Fractional Kelly ({kelly_mult:.2f}x) -> {trade_usdt:.2f} USDT")

        notional = trade_usdt
        if (not is_pyramid and getattr(self.config, "equity_sizing_enabled", False)
                and self.account_equity > 0 and sl and entry_price > 0):
            per_unit_risk = abs(entry_price - sl)
            if per_unit_risk > 0:
                k_factor = float(kelly_mult) if (kelly_mult is not None and kelly_mult > 0) else 1.0
                risk_amount = self.account_equity * (self.config.risk_per_trade_pct * k_factor)
                qty_by_risk = risk_amount / per_unit_risk
                notional = max(trade_usdt, qty_by_risk * entry_price)
                max_affordable = affordable_notional(
                    self.account_equity,
                    self.account_available,
                    self.account_leverage,
                    self.config.max_trade_notional_equity_mult,
                    getattr(self.config, "max_trade_margin_use_pct", 0.95),
                )
                if max_affordable > 0:
                    notional = min(notional, max_affordable)
        
        # Block NEW entries in RANGING regime unless the config-gated ranging
        # path matched a mean-reversion / mined-skill setup. Pyramid adds stay
        # blocked in chop regardless of the flag.
        if regime == "RANGING" and not is_pyramid:
            if not self._ranging_entry_allowed(symbol, signal):
                logger.info(f"[{symbol}] RANGING regime: blocking new entry")
                return None

        # Floor at Binance MIN_NOTIONAL ($20 for most symbols, $100 for BTC)
        # BTC uses $100 flat to match Binance min notional requirement.
        _bn_min = 100.0 if 'BTC' in symbol else 20.0
        live = get_trading_mode() == TradingMode.LIVE
        notional, reject = apply_hard_notional_caps(
            notional,
            trade_usdt=float(self.config.trade_usdt_amount),
            max_directional_usdt=float(self.config.max_directional_exposure_usdt or 0),
            leverage=float(self.account_leverage or 1.0),
            live=live,
            exchange_min_notional=_bn_min if live else 0.0,
        )
        if reject:
            logger.warning(f"[{symbol}] REJECT notional cap: {reject}")
            self._record_eval(
                symbol, direction, getattr(signal, "confidence", 0.0),
                f"SKIP (notional cap): {reject}",
                entry=entry_price, sl=sl, tp=tp,
            )
            return None
        if not live:
            notional = max(notional, _bn_min)
        elif notional < _bn_min:
            notional = _bn_min
        quantity = notional / entry_price if entry_price > 0 else 0

        # Min-edge / fee-churn gate: reject trades whose *captured* move can't clear cost.
        if not self._passes_min_edge(symbol, entry_price, tp, quantity, bars):
            self._record_eval(
                symbol, direction, getattr(signal, "confidence", 0.0),
                "SKIP (min-edge): expected capture below round-trip cost",
                entry=entry_price, sl=sl, tp=tp,
            )
            return None

        applied_kelly = float(kelly_mult) if (kelly_mult is not None and kelly_mult > 0) else None
        if applied_kelly is not None:
            logger.info(f"[{symbol}] Applied Kelly multiplier recorded: {applied_kelly:.3f}x")

        return Decision(
            action=direction,
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            confidence=signal.confidence,
            reasoning=(
                f"Regime: {regime}"
                + (" | Jesse ML gate skipped (no deployable model)"
                   if getattr(signal, "jesse_ml_gap", None) else "")
                + (f" | kelly={applied_kelly:.3f}x" if applied_kelly is not None else "")
            ),
            is_pyramid=is_pyramid,
            kelly_multiplier=applied_kelly,
        )

    def _passes_min_edge(
        self,
        symbol: str,
        entry_price: float,
        tp: float,
        quantity: float,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Min-edge / fee-churn gate.

        A trade is only worth taking if its gross *expected captured* move
        clears a multiple of the round-trip cost (fees + slippage).

        When trailing is enabled, winners are typically scratched near the
        trail lock (activation − trail distance), not at full TP. Gate on
        that captured move so fee-unsafe trail scratches are rejected.
        When partial TP is enabled the captured ATR is the 50/50 blend of
        the partial level and the trail lock (0.5×1.0 + 0.5×1.2 = 1.1 ATR
        at default geometry).

        FAILS OPEN: any bad input / disabled config -> allow the trade.
        """
        try:
            mult = getattr(self.config, "min_edge_fee_mult", 0.0) or 0.0
            if mult <= 0:
                return True  # gate disabled
            if not entry_price or not quantity or tp is None:
                return True  # missing data -> don't block
            notional = entry_price * quantity
            if notional <= 0:
                return True

            tp_distance = abs(tp - entry_price)
            expected_move = tp_distance
            if getattr(self.config, "trailing_stop_enabled", False):
                atr = atr_from_bars(bars or [], entry_price)
                if atr <= 0:
                    atr = entry_price * 0.02
                activation = float(getattr(self.config, "trail_activation_atr", 0.0) or 0.0)
                trail_mult = float(getattr(self.config, "trail_atr_mult", 0.0) or 0.0)
                captured_atr = max(0.0, activation - trail_mult)
                if getattr(self.config, "partial_tp_enabled", False):
                    raw_pct = getattr(self.config, "partial_tp_close_pct", None)
                    close_pct = 0.5 if raw_pct is None else float(raw_pct)
                    close_pct = min(1.0, max(0.0, close_pct))
                    raw_partial = getattr(self.config, "partial_tp_atr_mult", None)
                    partial_mult = 1.0 if raw_partial is None else float(raw_partial)
                    captured_atr = (close_pct * partial_mult) + ((1.0 - close_pct) * captured_atr)
                expected_move = min(tp_distance, captured_atr * atr)

            gross_expected = expected_move * quantity
            roundtrip_cost = self.config.roundtrip_cost_rate * notional
            required = mult * roundtrip_cost
            if gross_expected < required:
                logger.info(
                    f"  [ {symbol} ] SKIP (min-edge): expected capture ${gross_expected:.4f} "
                    f"< {mult:.1f}x round-trip cost ${roundtrip_cost:.4f} "
                    f"(need >= ${required:.4f}; tp_move=${tp_distance * quantity:.4f})"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"  [ {symbol} ] min-edge gate error (allowing trade): {e}")
            return True
