import os
import logging
from typing import List, Dict
from collections import defaultdict

from backend.services.binance_market_data import BinanceMarketDataService
from backend.services.decision_engine import DecisionEngine
from backend.services.risk_config import RiskConfig

logger = logging.getLogger(__name__)


class CryptoBacktestEngine:
    def __init__(
        self,
        symbols: List[str],
        interval: str = '15m',
        initial_capital: float = 10000.0,
        limit: int = 1500,
        cost_aware: bool = True,
    ):
        self.symbols = symbols
        self.interval = interval
        self.initial_capital = initial_capital
        self.limit = limit
        self.cost_aware = cost_aware

        self.market_data = BinanceMarketDataService()
        self.risk_config = RiskConfig()

        # Disable slow LLM API calls for backtesting to prevent massive delays/costs
        os.environ["ENABLE_PERSONAS"] = "false"
        self.decision_engine = DecisionEngine(self.risk_config)
        self.decision_engine.account_equity = initial_capital

        self.positions = {}  # symbol -> {'entry_price', 'qty', 'direction', 'sl', 'tp', 'entry_time', ...}
        self.cash = initial_capital
        self.trade_history = []

    def _per_leg_cost_rate(self) -> float:
        if not self.cost_aware:
            return 0.0
        rt_rate = float(getattr(self.risk_config, "roundtrip_cost_rate", 0.0) or 0.0)
        return max(0.0, rt_rate / 2.0)

    @staticmethod
    def _compute_atr(history: List[Dict], fallback_price: float) -> float:
        try:
            highs = [b["high"] for b in history[-15:]]
            lows = [b["low"] for b in history[-15:]]
            closes = [b["close"] for b in history[-16:-1]]
            trs = []
            for h, l_val, c in zip(highs, lows, closes):
                trs.append(max(h - l_val, abs(h - c), abs(l_val - c)))
            atr = sum(trs) / len(trs) if trs else 0.0
        except Exception:
            atr = fallback_price * 0.02
        if atr <= 0:
            atr = fallback_price * 0.02
        return atr

    async def fetch_data(self) -> Dict[str, List[Dict]]:
        logger.info(f"Fetching {self.limit} historical {self.interval} klines for {self.symbols}...")
        data = {}
        for sym in self.symbols:
            klines = await self.market_data.get_klines(sym, self.interval, self.limit)
            if klines:
                data[sym] = klines
            else:
                logger.warning(f"No data returned for {sym}")
        return data

    def _execute_trade(self, symbol: str, direction: str, qty: float, price: float, sl: float, tp: float, timestamp: str):
        notional = qty * price
        entry_fee = notional * self._per_leg_cost_rate()

        self.positions[symbol] = {
            'direction': direction,
            'qty': qty,
            'initial_qty': qty,
            'entry_price': price,
            'sl': sl,
            'tp': tp,
            'entry_time': timestamp,
            'partial_tp_done': False,
            'entry_fee_remaining': entry_fee,
        }
        logger.info(
            f"[{timestamp}] OPEN {direction} {symbol}: {qty:.4f} @ ${price:.4f} "
            f"(Notional: ${notional:.2f}, Fee: ${entry_fee:.4f}) SL={sl:.4f} TP={tp:.4f}"
        )

    def _record_realized_leg(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        entry_time: str,
        exit_time: str,
        reason: str,
        entry_fee_share: float,
    ) -> float:
        if direction == 'BUY':
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        exit_fee = (exit_price * qty) * self._per_leg_cost_rate()
        fees = entry_fee_share + exit_fee
        net_pnl = gross_pnl - fees

        self.cash += net_pnl
        self.decision_engine.account_equity = self.cash

        trade = {
            'symbol': symbol,
            'direction': direction,
            'qty': qty,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'gross_pnl': gross_pnl,
            'fees': fees,
            'pnl': net_pnl,
            'reason': reason,
        }
        self.trade_history.append(trade)
        logger.info(
            f"[{exit_time}] CLOSE {direction} {symbol}: {qty:.4f} @ ${exit_price:.4f} | "
            f"Gross: ${gross_pnl:.2f} | Fees: ${fees:.4f} | Net PnL: ${net_pnl:.2f} | "
            f"Reason: {reason} | Cash: ${self.cash:.2f}"
        )
        return net_pnl

    def _close_trade(self, symbol: str, exit_price: float, timestamp: str, reason: str):
        pos = self.positions.pop(symbol)
        self._record_realized_leg(
            symbol=symbol,
            direction=pos['direction'],
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            qty=pos['qty'],
            entry_time=pos['entry_time'],
            exit_time=timestamp,
            reason=reason,
            entry_fee_share=float(pos.get('entry_fee_remaining', 0.0)),
        )

    def _ratchet_stop_to_be_fees(self, pos: Dict, mark_price: float) -> bool:
        entry_px = float(pos['entry_price'])
        if entry_px <= 0:
            return False
        cost_rate = max(0.0, float(getattr(self.risk_config, "roundtrip_cost_rate", 0.0) or 0.0))
        fee_offset = entry_px * cost_rate
        direction = pos['direction']
        old_sl = pos.get('sl')

        if direction == 'BUY':
            candidate = entry_px + fee_offset
            floor = float(old_sl) if old_sl is not None else float("-inf")
            if candidate <= floor or candidate >= mark_price:
                return False
            pos['sl'] = candidate
            return True
        elif direction == 'SELL':
            candidate = entry_px - fee_offset
            ceiling = float(old_sl) if old_sl is not None else float("inf")
            if candidate >= ceiling or candidate <= mark_price:
                return False
            pos['sl'] = candidate
            return True
        return False

    def _check_exits(self, symbol: str, candle: Dict, history: List[Dict]):
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        direction = pos['direction']
        sl = pos['sl']
        tp = pos['tp']

        high = candle['high']
        low = candle['low']
        close = candle['close']
        timestamp = candle['date']

        # 1. Check intrabar SL/TP hits against the stop/target active entering this bar
        if direction == 'BUY':
            if low <= sl:
                self._close_trade(symbol, sl, timestamp, "SL")
                return
            if tp is not None and tp > 0 and tp < 99999.0 and high >= tp:
                self._close_trade(symbol, tp, timestamp, "TP")
                return
        elif direction == 'SELL':
            if high >= sl:
                self._close_trade(symbol, sl, timestamp, "SL")
                return
            if tp is not None and tp > 0 and tp < 99999.0 and low <= tp:
                self._close_trade(symbol, tp, timestamp, "TP")
                return

        cfg = self.risk_config
        atr = self._compute_atr(history, close)

        # 2. Partial TP + breakeven-plus-fees stop ratchet at bar close (matching live PartialTPManager)
        if getattr(cfg, "partial_tp_enabled", False) and not pos.get('partial_tp_done', False):
            partial_dist = float(getattr(cfg, "partial_tp_atr_mult", 1.0)) * atr
            close_pct = float(getattr(cfg, "partial_tp_close_pct", 0.5))
            if 0.0 < close_pct < 1.0 and partial_dist > 0:
                if direction == 'BUY':
                    profit_dist = close - pos['entry_price']
                else:
                    profit_dist = pos['entry_price'] - close
                if profit_dist >= partial_dist:
                    close_qty = pos['qty'] * close_pct
                    if close_qty > 0:
                        fee_share = float(pos.get('entry_fee_remaining', 0.0)) * close_pct
                        pos['qty'] -= close_qty
                        pos['entry_fee_remaining'] = max(0.0, float(pos.get('entry_fee_remaining', 0.0)) - fee_share)
                        pos['partial_tp_done'] = True
                        self._record_realized_leg(
                            symbol=symbol,
                            direction=direction,
                            entry_price=pos['entry_price'],
                            exit_price=close,
                            qty=close_qty,
                            entry_time=pos['entry_time'],
                            exit_time=timestamp,
                            reason="PARTIAL_TP",
                            entry_fee_share=fee_share,
                        )
                        if self._ratchet_stop_to_be_fees(pos, close):
                            sl = pos['sl']

        # 3. Trailing stop at bar close (matching live TrailingStopManager: step_trail gated by step_trail_enabled; no mark clamping)
        if getattr(cfg, "trailing_stop_enabled", False):
            activation_dist = cfg.trail_activation_atr * atr
            trail_dist = cfg.trail_atr_mult * atr
            step_trail = getattr(cfg, "step_trail_enabled", False)

            if direction == 'BUY':
                hw = pos.get('hw', max(pos['entry_price'], close))
                hw = max(hw, close)
                pos['hw'] = hw

                if hw - pos['entry_price'] < activation_dist:
                    if step_trail and hw - pos['entry_price'] >= (activation_dist * 0.75):
                        fee_offset = trail_dist * 0.1
                        candidate = pos['entry_price'] + fee_offset
                        if candidate > sl and candidate < close:
                            sl = candidate
                            pos['sl'] = sl
                else:
                    candidate = hw - trail_dist
                    if candidate < close and candidate > sl:
                        sl = candidate
                        pos['sl'] = sl
            else:  # SELL
                lw = pos.get('lw', min(pos['entry_price'], close))
                lw = min(lw, close)
                pos['lw'] = lw

                if pos['entry_price'] - lw < activation_dist:
                    if step_trail and pos['entry_price'] - lw >= (activation_dist * 0.75):
                        fee_offset = trail_dist * 0.1
                        candidate = pos['entry_price'] - fee_offset
                        if candidate < sl and candidate > close:
                            sl = candidate
                            pos['sl'] = sl
                else:
                    candidate = lw + trail_dist
                    if candidate > close and candidate < sl:
                        sl = candidate
                        pos['sl'] = sl

    async def run(self):
        data_by_sym = await self.fetch_data()

        # Group candles by timestamp to simulate tick-by-tick
        timeline = defaultdict(dict)
        for sym, candles in data_by_sym.items():
            for i, candle in enumerate(candles):
                timeline[candle['date']][sym] = (candle, i)  # Store index for slicing history

        sorted_times = sorted(timeline.keys())

        for t in sorted_times:
            tick_data = timeline[t]

            # 1. Check open positions for exits using current candle High/Low
            for sym, (candle, idx) in tick_data.items():
                history = data_by_sym[sym][:idx + 1]
                self._check_exits(sym, candle, history)

            # 2. Evaluate strategies for new entries
            for sym, (candle, idx) in tick_data.items():
                if sym in self.positions:
                    continue  # Already in position

                # We need at least 50 bars of history
                if idx < 50:
                    continue

                # Slice history up to current candle
                history = data_by_sym[sym][:idx + 1]

                decision = await self.decision_engine.evaluate_symbol(
                    symbol=sym,
                    bars=history,
                    existing_position=None,
                    open_count=len(self.positions),
                    pyramid_layers=[],
                    cooldown_active=False,
                )

                if decision and decision.action in ["BUY", "SELL"]:
                    sl = decision.stop_loss
                    tp = decision.take_profit
                    qty = decision.quantity
                    price = decision.entry_price

                    if qty > 0:
                        self._execute_trade(sym, decision.action, qty, price, sl, tp, t)

        # End of backtest: Force close any remaining open positions
        for sym in list(self.positions.keys()):
            last_candle = data_by_sym[sym][-1]
            self._close_trade(sym, last_candle['close'], last_candle['date'], "END_OF_TEST")

        return self._generate_report()

    def _generate_report(self) -> Dict:
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] <= 0]
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        total_gross_pnl = sum(t.get('gross_pnl', t['pnl']) for t in self.trade_history)
        total_fees = sum(t.get('fees', 0.0) for t in self.trade_history)

        logger.info("\n========== BACKTEST RESULTS ==========")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Capital:   ${self.cash:.2f}")
        logger.info(f"Gross PnL:       ${total_gross_pnl:.2f}")
        logger.info(f"Total Fees:      ${total_fees:.2f}")
        logger.info(f"Net PnL:         ${total_pnl:.2f} ({(total_pnl / self.initial_capital) * 100:.2f}%)")
        logger.info(f"Total Trades:    {len(self.trade_history)}")
        logger.info(f"Win Rate:        {win_rate * 100:.1f}% ({len(wins)}W / {len(losses)}L)")
        logger.info("======================================")

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.cash,
            'total_gross_pnl': total_gross_pnl,
            'total_fees': total_fees,
            'total_pnl': total_pnl,
            'cost_aware': self.cost_aware,
            'roundtrip_cost_rate': float(getattr(self.risk_config, "roundtrip_cost_rate", 0.0) or 0.0) if self.cost_aware else 0.0,
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'trades': self.trade_history,
        }
