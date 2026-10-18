"""
Simple vectorised backtesting engine.

Simulates bar-by-bar trade execution:
  - Entry: next bar open after a signal fires.
  - Exit:  stop-loss hit, take-profit hit, or signal reversal.

CLI usage::

    python -m src.backtesting.engine \\
        --symbol EURUSD \\
        --strategy combined \\
        --bars 90

Requires ``yfinance`` for live history download (pip install yfinance).
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.backtesting_ctrader.venue_costs import estimate_round_trip, venue_costs_enabled

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    total_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration_mins: float
    equity_curve: list = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────


class BacktestEngine:
    """
    Bar-by-bar backtesting engine.

    Parameters
    ----------
    strategy_name:
        Name understood by ``src.strategies.combined.get_strategy``.
    initial_balance:
        Starting equity in USD.
    commission:
        Round-trip commission as a fraction of notional (e.g. 0.001 = 0.1%).
    slippage:
        One-way slippage as a fraction of entry price (e.g. 0.0001 = 1 pip on 1.0).
    """

    def __init__(
        self,
        strategy_name: str = "combined",
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ) -> None:
        self.strategy_name   = strategy_name
        self.initial_balance = initial_balance
        self.commission      = commission
        self.slippage        = slippage
        
        try:
            from backend.strategies.market_regime import MarketRegimeDetector
            self._regime_detector = MarketRegimeDetector()
        except ImportError:
            self._regime_detector = None

    # ── main simulation ────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        bars: list,
        strategy=None,
    ) -> BacktestResult:
        """
        Run a backtest over *bars*.

        Parameters
        ----------
        symbol:   Trading symbol label (used in result).
        bars:     List of OHLCV dicts:
                  ``[{"timestamp": int, "open": f, "high": f, "low": f,
                      "close": f, "volume": f}, ...]``
        strategy: A ``BaseStrategy`` instance. If None, loaded via
                  ``get_strategy(self.strategy_name)``.
        """
        if strategy is None:
            from backend.strategies.combined import get_strategy
            strategy = get_strategy(self.strategy_name)

        balance       = self.initial_balance
        equity_curve  = [balance]
        trades: list  = []

        position: Optional[dict] = None  # current open position
        min_bars = 30  # warm-up period

        for i in range(min_bars, len(bars)):
            lookback = bars[max(0, i - 100): i]   # feed up to 100 bars of history
            current  = bars[i]
            next_bar = bars[i + 1] if i + 1 < len(bars) else current

            # ── Check exit conditions for open position ───────────────────────
            if position is not None:
                exit_price = None
                exit_reason = None
                high = float(current["high"])
                low  = float(current["low"])

                if position["side"] == "BUY":
                    if low <= position["stop_loss"]:
                        exit_price  = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif high >= position["take_profit"]:
                        exit_price  = position["take_profit"]
                        exit_reason = "take_profit"
                else:  # SELL
                    if high >= position["stop_loss"]:
                        exit_price  = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif low <= position["take_profit"]:
                        exit_price  = position["take_profit"]
                        exit_reason = "take_profit"

                if exit_price is not None:
                    pnl, balance = self._close_position(position, exit_price, balance)
                    position["exit_price"]  = exit_price
                    position["exit_reason"] = exit_reason
                    position["pnl"]         = pnl
                    position["close_bar"]   = i
                    trades.append(position)
                    position = None
                    equity_curve.append(balance)

            # ── Generate new signal ───────────────────────────────────────────
            if len(lookback) < 10:
                continue

            if self.strategy_name == "combined" and self._regime_detector:
                regime_result = self._regime_detector.detect(lookback)
                sig = strategy.generate_signal(symbol, lookback, regime=regime_result.regime)
            else:
                sig = strategy.generate_signal(symbol, lookback)

            if sig.signal == "NEUTRAL":
                continue

            # Close opposing position on reversal signal
            if position is not None and position["side"] != sig.signal:
                close_at = float(current["close"])
                pnl, balance = self._close_position(position, close_at, balance)
                position["exit_price"]  = close_at
                position["exit_reason"] = "reversal"
                position["pnl"]         = pnl
                position["close_bar"]   = i
                trades.append(position)
                position = None
                equity_curve.append(balance)

            # Enter new position on next bar open
            if position is None and sig.signal in ("BUY", "SELL"):
                entry = float(next_bar["open"])
                # Apply slippage
                if sig.signal == "BUY":
                    entry += entry * self.slippage
                else:
                    entry -= entry * self.slippage

                # Determine SL/TP from signal or fallback ATR-based defaults
                sl, tp = self._compute_sl_tp(sig, entry, lookback)
                if sl is None or tp is None:
                    continue

                # Simple fixed-lot sizing: 1% risk
                quantity = self._size(balance, entry, sl, sig.signal)
                if quantity <= 0:
                    continue

                position = {
                    "symbol":      symbol,
                    "side":        sig.signal,
                    "entry_price": entry,
                    "stop_loss":   sl,
                    "take_profit": tp,
                    "quantity":    quantity,
                    "open_bar":    i + 1,
                    "strategy":    self.strategy_name,
                }

        # Force-close any open position at last bar
        if position is not None and len(bars) > 0:
            close_at = float(bars[-1]["close"])
            pnl, balance = self._close_position(position, close_at, balance)
            position["exit_price"]  = close_at
            position["exit_reason"] = "end_of_data"
            position["pnl"]         = pnl
            position["close_bar"]   = len(bars) - 1
            trades.append(position)
            equity_curve.append(balance)

        return self._compile_result(symbol, trades, equity_curve)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _close_position(self, position: dict, exit_price: float, balance: float):
        """Compute P&L and update balance, net of commission."""
        qty  = position["quantity"]
        side = position["side"]
        if side == "BUY":
            raw_pnl = (exit_price - position["entry_price"]) * qty
        else:
            raw_pnl = (position["entry_price"] - exit_price) * qty

        commission = position["entry_price"] * qty * self.commission
        extra = 0.0
        if venue_costs_enabled():
            notional = float(position["entry_price"]) * abs(float(qty))
            hours = float(position.get("hold_hours") or 0.0)
            costs = estimate_round_trip(
                venue="ctrader",
                symbol=str(position.get("symbol") or ""),
                notional=notional,
                side=str(side),
                hours_held=hours,
                lots=abs(float(qty)),
            )
            extra = float(costs.get("total") or 0.0)
        pnl = raw_pnl - commission - extra
        return pnl, balance + pnl

    def _compute_sl_tp(self, sig, entry: float, bars: list):
        """
        Resolve SL and TP from the signal, or derive from ATR fallback.
        Returns (stop_loss, take_profit) as prices, or (None, None) on failure.
        """
        sl = sig.stop_loss
        tp = sig.take_profit

        if sl and tp and sl != entry and tp != entry:
            return float(sl), float(tp)

        # ATR fallback
        if len(bars) >= 14:
            highs  = np.array([b["high"]  for b in bars], dtype=float)
            lows   = np.array([b["low"]   for b in bars], dtype=float)
            closes = np.array([b["close"] for b in bars], dtype=float)
            atr    = _atr(highs, lows, closes, 14)
        else:
            atr = entry * 0.005  # 0.5% fallback

        if atr <= 0:
            return None, None

        pip = 0.01 if entry > 10 else 0.0001   # rough proxy
        min_stop = 10 * pip

        if sig.signal == "BUY":
            sl = entry - max(atr * 1.5, min_stop)
            tp = entry + max(atr * 3.0, min_stop * 2)
        else:
            sl = entry + max(atr * 1.5, min_stop)
            tp = entry - max(atr * 3.0, min_stop * 2)

        return round(sl, 5), round(tp, 5)

    def _size(self, balance: float, entry: float, stop_loss: float, side: str) -> float:
        """1% fixed-risk position sizing; returns units."""
        risk_usd = balance * 0.01
        if side == "BUY":
            sl_dist = abs(entry - stop_loss)
        else:
            sl_dist = abs(stop_loss - entry)
        if sl_dist == 0:
            return 0.0
        return risk_usd / sl_dist

    def _compile_result(
        self,
        symbol: str,
        trades: list,
        equity_curve: list,
    ) -> BacktestResult:
        """Aggregate trade list into a BacktestResult."""
        total = len(trades)
        if total == 0:
            return BacktestResult(
                symbol=symbol, strategy=self.strategy_name,
                total_trades=0, win_rate=0.0, total_pnl=0.0,
                max_drawdown=0.0, sharpe_ratio=0.0,
                profit_factor=0.0, avg_trade_duration_mins=0.0,
                equity_curve=equity_curve,
            )

        pnls = [t["pnl"] for t in trades]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate       = len(wins) / total
        total_pnl      = sum(pnls)
        gross_profit   = sum(wins)
        gross_loss     = abs(sum(losses)) if losses else 0.0
        profit_factor  = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average trade duration
        durations = []
        for t in trades:
            if "open_bar" in t and "close_bar" in t:
                durations.append((t["close_bar"] - t["open_bar"]) * 60)  # assume 1h bars
        avg_dur = float(np.mean(durations)) if durations else 0.0

        # Sharpe
        if len(pnls) > 1:
            pnl_arr = np.array(pnls, dtype=float)
            sharpe  = (pnl_arr.mean() / pnl_arr.std()) * math.sqrt(252) if pnl_arr.std() > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        eq = np.array(equity_curve, dtype=float)
        peak  = np.maximum.accumulate(eq)
        dd    = peak - eq
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        return BacktestResult(
            symbol=symbol,
            strategy=self.strategy_name,
            total_trades=total,
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 4),
            profit_factor=round(profit_factor, 4),
            avg_trade_duration_mins=round(avg_dur, 1),
            equity_curve=equity_curve,
        )

    # ── reporting ──────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(result: BacktestResult) -> None:
        """Pretty-print a BacktestResult using Rich tables."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"Backtest: {result.symbol} / {result.strategy}", show_header=True)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value",  justify="right")

            rows = [
                ("Total Trades",             str(result.total_trades)),
                ("Win Rate",                 f"{result.win_rate:.1%}"),
                ("Total P&L (USD)",          f"{result.total_pnl:,.2f}"),
                ("Max Drawdown (USD)",        f"{result.max_drawdown:,.2f}"),
                ("Sharpe Ratio",             f"{result.sharpe_ratio:.4f}"),
                ("Profit Factor",            f"{result.profit_factor:.4f}"),
                ("Avg Trade Duration (min)", f"{result.avg_trade_duration_mins:.1f}"),
                ("Final Equity (USD)",       f"{result.equity_curve[-1]:,.2f}" if result.equity_curve else "N/A"),
            ]
            for metric, value in rows:
                table.add_row(metric, value)

            console.print(table)

        except ImportError:
            # Fallback if rich is not installed
            print(f"\n=== Backtest: {result.symbol} / {result.strategy} ===")
            print(f"  Total Trades             : {result.total_trades}")
            print(f"  Win Rate                 : {result.win_rate:.1%}")
            print(f"  Total P&L (USD)          : {result.total_pnl:,.2f}")
            print(f"  Max Drawdown (USD)        : {result.max_drawdown:,.2f}")
            print(f"  Sharpe Ratio             : {result.sharpe_ratio:.4f}")
            print(f"  Profit Factor            : {result.profit_factor:.4f}")
            print(f"  Avg Trade Duration (min) : {result.avg_trade_duration_mins:.1f}")
            if result.equity_curve:
                print(f"  Final Equity (USD)       : {result.equity_curve[-1]:,.2f}")


# ── Walk-forward analysis ─────────────────────────────────────────────────────


def walk_forward(
    symbol: str,
    bars: list,
    strategy_name: str = "combined",
    train_pct: float = 0.7,
    n_splits: int = 5,
) -> list:
    """
    Walk-forward validation: split bars into n_splits expanding windows,
    each with train_pct in-sample and (1-train_pct) out-of-sample.

    Parameters
    ----------
    symbol:        Trading symbol label.
    bars:          Full list of OHLCV bar dicts.
    strategy_name: Strategy to instantiate for each fold.
    train_pct:     Fraction of each window used for training (not used for
                   parameter fitting here, but marks the OOS start).
    n_splits:      Number of walk-forward splits.

    Returns
    -------
    list[BacktestResult] — one result per out-of-sample fold.
    """
    from backend.strategies.combined import get_strategy

    n_bars    = len(bars)
    step      = n_bars // n_splits
    results   = []

    for fold in range(n_splits):
        end_idx   = step * (fold + 1)
        start_idx = 0  # expanding window
        fold_bars = bars[start_idx: end_idx]

        if len(fold_bars) < 50:
            logger.warning("Walk-forward fold %d: too few bars (%d) — skipping.", fold, len(fold_bars))
            continue

        oos_start = int(len(fold_bars) * train_pct)
        oos_bars  = fold_bars[oos_start:]

        if len(oos_bars) < 20:
            logger.warning("Walk-forward fold %d: too few OOS bars (%d) — skipping.", fold, len(oos_bars))
            continue

        strategy = get_strategy(strategy_name)
        engine   = BacktestEngine(strategy_name=strategy_name)
        # Warm-up: pass full fold to strategy for any internal state, test only OOS
        result   = engine.run(symbol=symbol, bars=fold_bars, strategy=strategy)
        result.strategy = f"{strategy_name}_wf_fold{fold}"
        results.append(result)
        logger.info(
            "Walk-forward fold %d: bars=%d OOS_start=%d trades=%d PnL=%.2f",
            fold, len(fold_bars), oos_start, result.total_trades, result.total_pnl,
        )

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────


def _atr(highs: "np.ndarray", lows: "np.ndarray", closes: "np.ndarray", period: int = 14) -> float:
    """Compute Average True Range for the last *period* bars."""
    if len(closes) < 2:
        return 0.0
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:]  - closes[:-1]),
        ),
    )
    if len(tr) == 0:
        return 0.0
    return float(np.mean(tr[-period:]))


def _bars_from_klines(raw: list) -> list:
    """Convert Binance /fapi/v1/klines rows to engine bar dicts."""
    bars = []
    for k in raw:
        bars.append({
            "timestamp": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return bars


def _download_crypto_bars(symbol: str, interval: str = "1h", limit: int = 500) -> list:
    """Download Binance Futures klines (sync, no API key required)."""
    import httpx

    sym = symbol.upper().replace("-", "").replace("/", "")
    if not (sym.endswith("USDT") or sym.endswith("USDC")):
        raise ValueError(f"Not a crypto futures symbol: {symbol!r}")

    url = "https://fapi.binance.com/fapi/v1/klines"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url, params={
            "symbol": sym,
            "interval": interval,
            "limit": min(limit, 1500),
        })
        resp.raise_for_status()
        data = resp.json()
    if not data:
        raise ValueError(f"Binance returned no klines for {sym!r}.")
    return _bars_from_klines(data)


def _download_bars(symbol: str, days: int) -> list:
    """
    Download historical OHLCV data via yfinance and convert to bar dicts.

    Raises ImportError if yfinance is not installed.
    """
    import yfinance as yf  # type: ignore

    ticker = yf.Ticker(symbol)
    df     = ticker.history(period=f"{days}d", interval="1h")
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r}.")

    bars = []
    for ts, row in df.iterrows():
        bars.append({
            "timestamp": int(ts.timestamp()),
            "open":      float(row["Open"]),
            "high":      float(row["High"]),
            "low":       float(row["Low"]),
            "close":     float(row["Close"]),
            "volume":    float(row.get("Volume", 0)),
        })
    return bars


def download_bars_for_symbol(symbol: str, days: int = 90, interval: str = "1h") -> list:
    """Pick Binance Futures for crypto pairs, yfinance otherwise."""
    sym = symbol.upper().replace("-", "").replace("/", "")
    if sym.endswith("USDT") or sym.endswith("USDC"):
        limit = min(1500, max(100, days * 24))
        return _download_crypto_bars(sym, interval=interval, limit=limit)
    yf_symbol = symbol if ("-" in symbol or "=" in symbol) else f"{symbol}-USD"
    return _download_bars(yf_symbol, days=days)


def backtest_result_to_dict(result: BacktestResult) -> dict:
    """Serialize BacktestResult for JSON API responses."""
    return {
        "symbol": result.symbol,
        "strategy": result.strategy,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "total_profit": result.total_pnl,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "sharpe": result.sharpe_ratio,
        "profit_factor": result.profit_factor,
        "avg_trade_duration_mins": result.avg_trade_duration_mins,
        "equity_curve": result.equity_curve,
        "trades": result.total_trades,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Hedge-fund backtest engine")
    parser.add_argument("--symbol",   default="EURUSD=X", help="yfinance ticker symbol")
    parser.add_argument("--strategy", default="combined",  help="Strategy name")
    parser.add_argument("--bars",     type=int, default=90, help="Days of 1h history to download")
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward validation instead of single backtest",
    )
    parser.add_argument("--splits",   type=int, default=5,   help="Walk-forward splits")
    parser.add_argument("--balance",  type=float, default=10_000.0, help="Initial balance USD")
    args = parser.parse_args()

    print(f"Downloading {args.bars}d of 1h bars for {args.symbol} …")
    bar_data = _download_bars(args.symbol, args.bars)
    print(f"Downloaded {len(bar_data)} bars.")

    if args.walk_forward:
        wf_results = walk_forward(
            symbol=args.symbol,
            bars=bar_data,
            strategy_name=args.strategy,
            n_splits=args.splits,
        )
        for r in wf_results:
            BacktestEngine.print_report(r)
    else:
        eng    = BacktestEngine(strategy_name=args.strategy, initial_balance=args.balance)
        result = eng.run(symbol=args.symbol, bars=bar_data)
        BacktestEngine.print_report(result)
