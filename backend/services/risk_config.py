from pydantic_settings import BaseSettings
from pydantic import AliasChoices, Field as PydanticField

class RiskConfig(BaseSettings):
    """Centralized risk and trading configuration."""
    
    # Position management
    # 10x isolated liquidation is roughly a -9.5% underlying move; the
    # catastrophe backstop must trigger before liquidation, not after it.
    emergency_drawdown_pct: float = -8.0
    max_position_hold_hours: float = 72.0
    exit_opinion_threshold: float = 0.65
    min_position_hold_min: int = 20
    
    # Cooldowns
    sl_cooldown_minutes: int = 30
    opinion_close_cooldown_min: int = 30
    
    # Pyramid/DCA settings
    # Live week (Jul 11-18): 53 same-symbol multi-close clusters; stacked layers
    # stopped out together and amplified losses. Cap layers hard; never add
    # while the base position is underwater.
    pyramid_mode: bool = False
    pyramid_max_layers: int = PydanticField(
        default=2,
        validation_alias=AliasChoices("pyramid_max_layers", "PYRAMID_MAX_LAYERS"),
    )
    pyramid_atr_multiplier: float = 1.0
    pyramid_min_conf_increase: float = 0.05
    pyramid_min_improvement: float = 0.005
    pyramid_usdt_per_layer: float = 10.0
    pyramid_max_wallet_pct: float = 0.70
    # Block pyramid adds when mark is against the position vs entry.
    pyramid_block_underwater: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices(
            "pyramid_block_underwater", "PYRAMID_BLOCK_UNDERWATER"
        ),
    )
    
    # Risk Limits
    max_positions: int = PydanticField(
        default=20,
        validation_alias=AliasChoices("max_positions", "MAX_POSITIONS"),
    )
    max_binance_positions: int = PydanticField(
        default=10,
        validation_alias=AliasChoices(
            "max_binance_positions", "MAX_BINANCE_POSITIONS", "BINANCE_MAX_OPEN_POSITIONS"
        ),
    )
    max_ctrader_positions: int = PydanticField(
        default=10,
        validation_alias=AliasChoices(
            "max_ctrader_positions", "MAX_CTRADER_POSITIONS", "CTRADER_MAX_OPEN_POSITIONS"
        ),
    )
    max_directional_exposure_usdt: float = 500.0
    trade_usdt_amount: float = 10.0
    # ── P0 safety gates ──
    # Skip the whole entry pipeline when Binance availableBalance is below this
    # floor — prevents blind-firing orders that fail with "Insufficient margin"
    # and wasting Kronos/LLM calls on entries that can never fill.
    min_available_margin_usdt: float = PydanticField(
        default=5.0,
        validation_alias=AliasChoices("min_available_margin_usdt", "MIN_AVAILABLE_MARGIN_USDT"),
    )
    # Correlation cap: max open positions (distinct symbols) in the SAME
    # direction.
    max_same_direction_positions: int = PydanticField(
        default=10,
        validation_alias=AliasChoices("max_same_direction_positions", "MAX_SAME_DIRECTION_POSITIONS"),
    )
    # Portfolio-level direction cap: total same-direction NOTIONAL may reach
    # this multiple of equity. Notional is leveraged, so at 10x leverage a
    # 4.0 multiple = ~40% of equity committed as margin in one direction.
    # (The old hardcoded 0.8x-equity notional cap allowed only ~8% margin
    # utilization at 10x and blocked entries with just 2-4 open positions.)
    max_direction_notional_equity_mult: float = PydanticField(
        default=4.0,
        validation_alias=AliasChoices(
            "max_direction_notional_equity_mult", "MAX_DIRECTION_NOTIONAL_EQUITY_MULT"
        ),
    )
    funding_rate_cap: float = 0.0005  # 0.05%
    # Max confidence nudge (absolute) the funding rate may apply to a signal.
    # Unclamped funding adjustment (rate*1000) flipped marginal signals toward
    # shorts and produced a measured 22.7% BUY win rate vs 53% SELL.
    funding_conf_adj_cap: float = PydanticField(
        default=0.05,
        validation_alias=AliasChoices("funding_conf_adj_cap", "FUNDING_CONF_ADJ_CAP"),
    )
    # ── Position sizing ──
    # When enabled, size each entry so a stop-loss hit loses ~risk_per_trade_pct
    # of account equity (proper risk-based sizing) instead of a flat $ notional.
    # Falls back to trade_usdt_amount when equity/SL are unavailable.
    equity_sizing_enabled: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("equity_sizing_enabled", "EQUITY_SIZING_ENABLED"),
    )
    risk_per_trade_pct: float = PydanticField(
        default=0.01,  # risk 1% of equity per trade
        validation_alias=AliasChoices("risk_per_trade_pct", "RISK_PER_TRADE_PCT"),
    )
    # Hard cap on a single trade's notional as a multiple of equity (after
    # leverage). Keeps risk-based sizing from over-allocating on a tight stop.
    max_trade_notional_equity_mult: float = PydanticField(
        default=2.0,
        validation_alias=AliasChoices("max_trade_notional_equity_mult", "MAX_TRADE_NOTIONAL_EQUITY_MULT"),
    )
    # Never request more margin than this fraction of available cash/margin.
    # Stops paper 1x books from skipping a 1% risk entry when the stop is tight
    # (need $107k on a $100k book).
    max_trade_margin_use_pct: float = PydanticField(
        default=0.95,
        validation_alias=AliasChoices("max_trade_margin_use_pct", "MAX_TRADE_MARGIN_USE_PCT"),
    )
    kill_floor_usdt: float = PydanticField(
        default=65.0,
        validation_alias=AliasChoices("kill_floor_usdt", "TRADING_KILL_FLOOR_USDT"),
    )
    
    # Safety hard boundaries (Phase 7)
    max_position_risk_pct: float = 1.0
    # Env-tunable so the loop can be tested through a drawdown without a rebuild.
    # Set RISK_MAX_DRAWDOWN_PCT=99 in .env to effectively disable the drawdown
    # halt; default 20.0 preserves the original protection.
    max_portfolio_drawdown_pct: float = PydanticField(
        default=20.0,
        validation_alias=AliasChoices(
            "max_portfolio_drawdown_pct", "RISK_MAX_DRAWDOWN_PCT", "RISK_MAX_PORTFOLIO_DRAWDOWN_PCT"
        ),
    )
    max_daily_loss_pct: float = PydanticField(
        default=5.0,
        validation_alias=AliasChoices(
            "max_daily_loss_pct", "RISK_MAX_DAILY_LOSS_PCT"
        ),
    )
    max_open_positions: int = PydanticField(
        default=20,
        validation_alias=AliasChoices("max_open_positions", "MAX_OPEN_POSITIONS"),
    )
    
    # LLM Overhaul Toggles
    use_risk_reviewer_llm: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("use_risk_reviewer_llm", "USE_RISK_REVIEWER_LLM"),
    )
    enable_personas: bool = PydanticField(
        default=False,
        validation_alias=AliasChoices("enable_personas", "ENABLE_PERSONAS"),
    )
    enable_jesse_ml: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("enable_jesse_ml", "ENABLE_JESSE_ML", "JESSE_ML_GATE_ENABLED"),
    )
    
    # Signal thresholds (env-tunable so trade aggressiveness can change without a rebuild)
    min_signal_strength: float = PydanticField(
        default=0.45,
        validation_alias=AliasChoices("min_signal_strength", "MIN_SIGNAL_STRENGTH"),
    )
    ai_analysis_threshold: float = PydanticField(
        default=0.30,
        validation_alias=AliasChoices("ai_analysis_threshold", "AI_ANALYSIS_THRESHOLD"),
    )
    opinion_override_margin: float = 0.15
    opinion_entry_min: float = 0.25
    
    # SL/TP (env-tunable so the payoff geometry can be tuned without a rebuild)
    sl_atr_mult: float = PydanticField(
        default=1.75,
        validation_alias=AliasChoices("sl_atr_mult", "SL_ATR_MULT"),
    )
    tp_atr_mult: float = PydanticField(
        default=5.5,
        validation_alias=AliasChoices("tp_atr_mult", "TP_ATR_MULT"),
    )
    # Must match training Triple-Barrier ATR period (geometry.json locked field).
    atr_period: int = PydanticField(
        default=14,
        validation_alias=AliasChoices("atr_period", "ATR_PERIOD"),
    )

    # ── Min-edge gate (fee-churn killer) ──
    # 81-day history: 52% of trades were scratches (|PnL| <= $0.20) and
    # commission drag was -$193.54 on a +$64 net. Round-trip taker fee on
    # Binance Futures is ~2 * 0.04% = 0.08% of notional. We only take a
    # trade if its gross expected move to TP clears `min_edge_fee_mult`x the
    # round-trip cost (fees + slippage buffer). This kills tight-TP scratch
    # entries that can never out-earn their own fees. Set 0 to disable.
    taker_fee_rate: float = PydanticField(
        default=0.0004,
        validation_alias=AliasChoices("taker_fee_rate", "TAKER_FEE_RATE"),
    )
    slippage_rate: float = PydanticField(
        default=0.0002,
        validation_alias=AliasChoices("slippage_rate", "SLIPPAGE_RATE"),
    )
    min_edge_fee_mult: float = PydanticField(
        default=2.5,
        validation_alias=AliasChoices("min_edge_fee_mult", "MIN_EDGE_FEE_MULT"),
    )

    @property
    def roundtrip_cost_rate(self) -> float:
        """Round-trip cost as a fraction of notional: entry+exit fee + slippage both sides."""
        return 2.0 * (self.taker_fee_rate + self.slippage_rate)

    # ── Trailing stop (ratchet-only, ATR-based) ──
    # Solves the "trade just oscillates between SL and TP" problem: once a
    # position has moved `trail_activation_atr` x ATR into profit, the stop
    # starts following price by `trail_atr_mult` x ATR and only ever tightens
    # (never loosens). Chop that reverses gets stopped at a locked-in profit /
    # breakeven instead of round-tripping back to the original SL. The trailed
    # level is written into the Trade.stop_loss column, so the existing
    # _check_sl_tp close path enforces it — no separate order needed.
    # Set trailing_stop_enabled=false to disable entirely.
    trailing_stop_enabled: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("trailing_stop_enabled", "TRAILING_STOP_ENABLED"),
    )
    # ── Exchange-native trailing stop (Binance TRAILING_STOP_MARKET) ──
    # Placed at entry so the stop trails the market continuously on the exchange
    # (tick-by-tick) and keeps protecting profit even if the bot is offline —
    # unlike the software ratchet which only updates once per 15-min cycle.
    # When enabled, the per-cycle software ratchet is skipped to avoid two
    # competing trail mechanisms; the hard STOP_MARKET stays as a catastrophe floor.
    native_trailing_enabled: bool = PydanticField(
        # Disabled by default: a quantity-based native trail placed per pyramid
        # layer creates duplicate partial-coverage orders. The software ATR
        # trail now safely replaces the full closePosition STOP_MARKET.
        default=False,
        validation_alias=AliasChoices("native_trailing_enabled", "NATIVE_TRAILING_ENABLED"),
    )
    # Binance callbackRate: trail distance as a percent of price (0.1–5.0).
    trailing_callback_rate: float = PydanticField(
        default=1.0,
        validation_alias=AliasChoices("trailing_callback_rate", "TRAILING_CALLBACK_RATE"),
    )
    # Activate the trail only after price moves this fraction in favor (0 = immediate).
    trailing_activation_pct: float = PydanticField(
        default=0.005,
        validation_alias=AliasChoices("trailing_activation_pct", "TRAILING_ACTIVATION_PCT"),
    )
    # How far into profit (in ATR units) before the trail activates.
    # Live week: 69/72 wins closed near a raised SL and 0 hits of full TP —
    # early trail/STEP-TRAIL clipped winners while losers paid full risk
    # (avg win $0.34 vs avg loss $0.66). Activate closer to TP (2.5 ATR) so
    # winners have room to run.
    trail_activation_atr: float = PydanticField(
        default=2.0,
        validation_alias=AliasChoices("trail_activation_atr", "TRAIL_ACTIVATION_ATR"),
    )
    # Trailing distance behind the high-water mark, in ATR units.
    # MUST be < trail_activation_atr, otherwise the trail locks a LOSS at
    # activation and can only lock +1R once price reaches the TP itself — which
    # caps every realized winner below the 2.5R the min-edge gate sized for and
    # bleeds refunded margin (leak #4). At 0.8 with activation 2.0 the trail
    # locks ~+1.2 ATR once armed.
    trail_atr_mult: float = PydanticField(
        default=0.8,
        validation_alias=AliasChoices("trail_atr_mult", "TRAIL_ATR_MULT"),
    )
    # Early breakeven ratchet at 0.75× activation. Disabled by default — it was
    # the dominant "win near SL" path and destroyed payoff asymmetry.
    step_trail_enabled: bool = PydanticField(
        default=False,
        validation_alias=AliasChoices("step_trail_enabled", "STEP_TRAIL_ENABLED"),
    )

    # ── Partial take-profit ──
    # Solves the "TP never hit" problem (0% TP rate over 500 trades): the trail
    # chokes every winner before it reaches the full TP. With partial TP, we
    # close `partial_tp_close_pct` of the position at `partial_tp_atr_mult` x ATR
    # into profit, locking in real gains. The remainder rides the trail for runners.
    partial_tp_enabled: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("partial_tp_enabled", "PARTIAL_TP_ENABLED"),
    )
    partial_tp_atr_mult: float = PydanticField(
        default=1.0,  # close partial at 1× ATR profit
        validation_alias=AliasChoices("partial_tp_atr_mult", "PARTIAL_TP_ATR_MULT"),
    )
    partial_tp_close_pct: float = PydanticField(
        default=0.50,  # close 50% of position
        validation_alias=AliasChoices("partial_tp_close_pct", "PARTIAL_TP_CLOSE_PCT"),
    )
    # ── Symbol-quality gate (liquidity filter + blacklist) ──
    # Reject any symbol whose 24h quote volume (USDT) is below this floor.
    # Low-liquidity / new-listing meme symbols (SIREN, AIGENSYN, MAGMA, SPK ...)
    # produced the entire realized-PnL loss tail (~-$300) via wide spreads,
    # slippage and forced liquidations. Set 0 to disable the volume gate.
    min_24h_quote_volume_usdt: float = PydanticField(
        default=3_000_000.0,
        validation_alias=AliasChoices(
            "min_24h_quote_volume_usdt", "MIN_24H_QUOTE_VOLUME_USDT"
        ),
    )
    # Hard blacklist (comma-separated env). Always skipped for NEW entries
    # regardless of volume. Open positions on blacklisted symbols are still
    # managed (SL/TP/trail) so an in-flight leg is never abandoned.
    # Pre-seeded with confirmed weekly loss leaders (ADA/ARB/DOGE/APT).
    symbol_blacklist_raw: str = PydanticField(
        default=(
            "ADAUSDT,ARBUSDT,DOGEUSDT,APTUSDT,"
            "ADAUSDC,ARBUSDC,DOGEUSDC,APTUSDC"
        ),
        validation_alias=AliasChoices("symbol_blacklist", "SYMBOL_BLACKLIST"),
    )

    # ── New-bar gate ──
    # The decision pipeline consumes 1h bars but the loop runs every 15 min,
    # so the same bar was re-evaluated up to 4x — measured live as heavy churn
    # (~18 closes/day; trades held <1h were net losers at a 37.5% win rate
    # while >24h holds won 78.8%). When enabled, the entry/pyramid pipeline
    # runs only when a NEW bar has opened since the last evaluation; exits,
    # trailing and SL/TP management still run every cycle.
    eval_on_new_bar_only: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("eval_on_new_bar_only", "EVAL_ON_NEW_BAR_ONLY"),
    )

    # ── Per-symbol expectancy gate ──
    # Live measurement: AVAX/SOL/LINK/UNI/BNB/BTC bled consistently (18-42%
    # win rates) while ADA/DOGE/XRP were net positive. Block NEW entries on
    # symbols whose realized P&L over the lookback window is negative with a
    # meaningful sample. Rolling window, so a blocked symbol is retried after
    # its losses age out. Symbols with open positions are never blocked
    # (management must continue). Set enabled=false to disable.
    symbol_expectancy_gate_enabled: bool = PydanticField(
        default=True,
        validation_alias=AliasChoices("symbol_expectancy_gate_enabled", "SYMBOL_EXPECTANCY_GATE_ENABLED"),
    )
    symbol_expectancy_lookback_days: int = PydanticField(
        default=30,
        validation_alias=AliasChoices("symbol_expectancy_lookback_days", "SYMBOL_EXPECTANCY_LOOKBACK_DAYS"),
    )
    symbol_expectancy_min_trades: int = PydanticField(
        default=20,
        validation_alias=AliasChoices("symbol_expectancy_min_trades", "SYMBOL_EXPECTANCY_MIN_TRADES"),
    )

    # ── RANGING entries (config-gated; OFF by default) ──
    # Skill miner edges include Ranging→Bullish/Bearish and CombinedStrategy
    # already weights mean_reversion in RANGING, but the decision engine blocks
    # every new ranging entry at two points. Unlock only when this flag is on,
    # and only for mean-reversion / matching mined skills. Live trading still
    # requires allow_ranging_in_live — paper/backtest is the intended default.
    allow_ranging_entries: bool = PydanticField(
        default=False,
        validation_alias=AliasChoices("allow_ranging_entries", "ALLOW_RANGING_ENTRIES"),
    )
    allow_ranging_in_live: bool = PydanticField(
        default=False,
        validation_alias=AliasChoices("allow_ranging_in_live", "ALLOW_RANGING_IN_LIVE"),
    )

    @property
    def symbol_blacklist(self) -> set[str]:
        return {
            s.strip().upper()
            for s in self.symbol_blacklist_raw.split(",")
            if s.strip()
        }
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

# Global singleton
_risk_config = None

def get_risk_config() -> RiskConfig:
    global _risk_config
    if _risk_config is None:
        _risk_config = RiskConfig()
    return _risk_config


def refresh_risk_config() -> RiskConfig:
    """Force-reload configuration from environment variables."""
    global _risk_config
    # Reload dotenv to ensure we grab the latest changes from .env
    import os
    from dotenv import load_dotenv
    env_path = os.getenv("ENV_FILE_PATH", ".env")
    load_dotenv(env_path, override=True)
    _risk_config = RiskConfig()
    return _risk_config


# Re-export get_trading_mode from the canonical location so existing
# `from backend.services.risk_config import get_trading_mode` imports
# keep working without code changes in callers.
from backend.services.trading_mode import get_trading_mode  # noqa: E402, F401
