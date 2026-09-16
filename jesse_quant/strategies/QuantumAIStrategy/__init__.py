from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils


class QuantumAIStrategy(Strategy):
    """
    QuantumTrade Long-Only Trend Riding Strategy (for Crypto Bull Regimes)
    - Macro Filter: Price > 200 EMA (Trend alignment)
    - Entry: Pullback to 20/50 EMA with RSI dip (buying value in an uptrend)
    - Take Profit: 3.5 ATR (letting winners run)
    - Trailing Lock: Locks SL to break-even once 1.2 ATR profit is achieved
    - Zero Shorts: Eliminates counter-trend liquidation risk during bull markets
    """

    def __init__(self):
        super().__init__()
        self.current_sl = 0.0
        self.highest_price = 0.0
        self.bars_in_position = 0
        self.accrued_funding = 0.0

    def hyperparameters(self):
        return [
            {"name": "fast_ema", "type": int, "min": 10, "max": 30, "default": 20},
            {"name": "mid_ema", "type": int, "min": 40, "max": 60, "default": 50},
            {"name": "rsi_low", "type": int, "min": 32, "max": 45, "default": 38},
            {"name": "rsi_high", "type": int, "min": 48, "max": 60, "default": 54},
            {"name": "sl_atr_mult", "type": float, "min": 1.5, "max": 3.0, "step": 0.25, "default": 1.75},
            {"name": "tp_atr_mult", "type": float, "min": 3.0, "max": 6.0, "step": 0.5, "default": 5.5},
            {"name": "trail_be_mult", "type": float, "min": 1.2, "max": 2.5, "step": 0.2, "default": 1.8},
            {"name": "trail_dist_mult", "type": float, "min": 1.0, "max": 2.2, "step": 0.2, "default": 1.6},
        ]

    @property
    def ema_fast(self):
        period = self.hp["fast_ema"] if hasattr(self, "hp") and "fast_ema" in self.hp else 20
        return ta.ema(self.candles, period)

    @property
    def ema_mid(self):
        period = self.hp["mid_ema"] if hasattr(self, "hp") and "mid_ema" in self.hp else 50
        return ta.ema(self.candles, period)

    @property
    def ema_slow(self):
        return ta.ema(self.candles, 200)

    @property
    def rsi(self):
        return ta.rsi(self.candles, 14)

    @property
    def atr(self):
        return ta.atr(self.candles, 14)

    def should_long(self) -> bool:
        # Strong bull trend: Price > EMA 200 and EMA 50 > EMA 200
        is_bull_regime = self.price > self.ema_slow and self.ema_mid > self.ema_slow
        
        # Pullback into EMA 20/50 value zone with RSI dipping between 38 and 54
        is_pullback = self.price <= self.ema_fast * 1.005 and self.price >= self.ema_mid * 0.99
        rsi_low = self.hp["rsi_low"] if hasattr(self, "hp") and "rsi_low" in self.hp else 38
        rsi_high = self.hp["rsi_high"] if hasattr(self, "hp") and "rsi_high" in self.hp else 54
        is_rsi_dip = rsi_low <= self.rsi <= rsi_high
        
        # Bullish confirmation: Green candle closing near its high
        candle_bullish = self.close > self.open

        return is_bull_regime and is_pullback and is_rsi_dip and candle_bullish

    def should_short(self) -> bool:
        return False

    def should_cancel_entry(self) -> bool:
        return True

    def dynamic_slippage_pct(self) -> float:
        """Dynamic ATR-scaled slippage model: Base 0.03% scaled by current volatility."""
        avg_atr = ta.ema(self.candles, 100)
        vol_mult = (self.atr / avg_atr) if avg_atr > 0 else 1.0
        # Scale between 0.03% and 0.15% based on volatility spikes
        return float(min(0.0015, max(0.0003, 0.0003 * vol_mult)))

    def go_long(self):
        qty = utils.size_to_qty(self.balance * 0.35, self.price, fee_rate=self.fee_rate)
        sl_mult = self.hp["sl_atr_mult"] if hasattr(self, "hp") and "sl_atr_mult" in self.hp else 1.75
        tp_mult = self.hp["tp_atr_mult"] if hasattr(self, "hp") and "tp_atr_mult" in self.hp else 5.5

        # Account for dynamic execution slippage on entry
        slippage = self.dynamic_slippage_pct()
        entry_price = self.price * (1.0 + slippage)

        sl = entry_price - (sl_mult * self.atr)
        tp = entry_price + (tp_mult * self.atr)

        self.current_sl = sl
        self.highest_price = entry_price
        self.bars_in_position = 0
        self.accrued_funding = 0.0
        self.buy = qty, entry_price
        self.stop_loss = qty, sl
        self.take_profit = qty, tp

    def go_short(self):
        pass

    def update_position(self):
        if not self.is_long:
            self.bars_in_position = 0
            return

        self.bars_in_position += 1
        self.highest_price = max(self.highest_price, self.price)
        gain = self.highest_price - self.position.entry_price

        be_mult = self.hp["trail_be_mult"] if hasattr(self, "hp") and "trail_be_mult" in self.hp else 1.8
        trail_dist = self.hp["trail_dist_mult"] if hasattr(self, "hp") and "trail_dist_mult" in self.hp else 1.6

        # Perpetual funding: 0.04% every 8 hours. Walk the stop up by the
        # funding paid so carry cost is in the backtest path, not just fees.
        if self.bars_in_position > 0 and self.bars_in_position % 8 == 0:
            funding_per_unit = self.price * 0.0004
            self.accrued_funding += funding_per_unit * abs(self.position.qty)
            funded_sl = self.current_sl + funding_per_unit
            if funded_sl > self.current_sl:
                self.current_sl = funded_sl
                self.stop_loss = self.position.qty, funded_sl

        # 1. Break-even ratchet once price achieves be_mult * ATR
        if gain >= be_mult * self.atr:
            be_sl = self.position.entry_price + (0.5 * self.atr)
            if be_sl > self.current_sl:
                self.current_sl = be_sl
                self.stop_loss = self.position.qty, be_sl

        # 2. Chandelier trailing stop once in strong profit (> 2.8 ATR)
        if gain >= 2.8 * self.atr:
            trail_sl = self.highest_price - (trail_dist * self.atr)
            if trail_sl > self.current_sl:
                self.current_sl = trail_sl
                self.stop_loss = self.position.qty, trail_sl
