import os
from enum import Enum

_DEFAULT_PAPER_BALANCE = 100_000.0


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


def is_shadow_mode() -> bool:
    return get_trading_mode() == TradingMode.SHADOW


def is_paper_mode() -> bool:
    return get_trading_mode() == TradingMode.PAPER


def is_live_mode() -> bool:
    return get_trading_mode() == TradingMode.LIVE


def is_backtest_mode() -> bool:
    return get_trading_mode() == TradingMode.BACKTEST


def get_trading_mode() -> TradingMode:
    """Resolve trading mode (backtest | shadow | paper | live) based on TRADING_MODE env var.
    Falls back to PAPER_TRADING env configuration for backwards compatibility.
    """
    raw = os.getenv("TRADING_MODE", "").lower().strip()
    if raw in {m.value for m in TradingMode}:
        return TradingMode(raw)

    if os.getenv("SHADOW_MODE", "false").lower().strip() == "true":
        return TradingMode.SHADOW

    # Check legacy PAPER_TRADING flag if TRADING_MODE is unset
    paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
    dry_run = os.getenv("DRY_RUN_ALL", "true").lower() == "true"

    return TradingMode.PAPER if (paper_trading or dry_run) else TradingMode.LIVE


def paper_leverage_for_broker(broker: str) -> float:
    """Leverage used by the paper book for a broker session.

    Binance paper simulates USDT-M futures, so it follows BINANCE_LEVERAGE
    (default 10x). A 1x paper book rejects risk-based sizes that a tight
    stop produces (1% equity risk / ~0.9% stop ≈ 1.07x notional).
    """
    name = (broker or "").strip().lower()
    if name in {"binance_futures", "binance"}:
        raw = os.getenv("BINANCE_LEVERAGE", "10")
    elif name == "ctrader":
        raw = os.getenv("CTRADER_PAPER_LEVERAGE", "1")
    else:
        raw = os.getenv("PAPER_LEVERAGE", "1")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return value if value > 0 else 1.0


def paper_reported_equity(cash: float, margin_used: float = 0.0) -> float:
    """Paper NAV for kill switch, snapshots, and the drawdown gate.

    Margin is reserved buying power, not equity. Adding it to cash
    double-counts open notionals: a 1x book with $100k cash and $98k
    LINK/LTC margin reported a fake $198k peak; after 10x the same book
    reported $133k and the 20% drawdown gate halted a flat paper session.
    ``margin_used`` is accepted so call sites stay explicit.
    """
    _ = margin_used
    return float(cash or 0.0)


def paper_starting_balance() -> float:
    """Simulated cash for local paper mode (not a Binance wallet).

    PAPER_BALANCE env, else $100_000. Used when no in-memory / last paper
    book is available, and as the reset floor after a wiped paper book.
    """
    raw = os.getenv("PAPER_BALANCE", str(_DEFAULT_PAPER_BALANCE))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PAPER_BALANCE
    return value if value > 0 else _DEFAULT_PAPER_BALANCE


BINANCE_PAPER_SESSION_ID = "binance_paper"
BINANCE_LIVE_SESSION_ID = "binance_futures_live"


def binance_paper_parallel_enabled() -> bool:
    """Run a local Binance paper book beside a live/demo cTrader session.

    TRADING_MODE is process-wide, so paper crypto cannot be enabled by flipping
    that flag without also blocking cTrader. This opt-in keeps FX orders on
    cTrader and routes the crypto loop / smart-order crypto path through
    PaperTradingEngine. It never sends Binance private API orders.
    """
    return os.getenv("BINANCE_PAPER_PARALLEL", "false").strip().lower() == "true"


def live_exchange_orders_allowed() -> bool:
    """Whether any exchange adapter may send live orders.

    Single source of truth: only TRADING_MODE=live (or the legacy
    PAPER_TRADING/DRY_RUN_ALL fallback resolving to live) allows them.
    Unset TRADING_MODE with default PAPER_TRADING=true refuses them.
    """
    return get_trading_mode() == TradingMode.LIVE


def live_binance_orders_allowed() -> bool:
    """Whether BinanceFuturesService may hit the live/testnet private API."""
    if binance_paper_parallel_enabled():
        return False
    return live_exchange_orders_allowed()


def binance_order_session_id() -> str | None:
    """UnifiedTrading session for crypto place_order.

    Dual-live keeps the default session on cTrader. Passing None into
    place_order therefore sends AVAXUSDT (etc.) to Spotware, which rejects
    them as unknown. Always pick the Binance session explicitly.
    """
    if binance_paper_parallel_enabled():
        return BINANCE_PAPER_SESSION_ID
    if live_binance_orders_allowed():
        return BINANCE_LIVE_SESSION_ID
    return None


def live_ctrader_orders_allowed() -> bool:
    """Whether CTraderService may send ProtoOANewOrder / close / amend."""
    return live_exchange_orders_allowed()
