import logging
import os
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from backend.services.risk_config import RiskConfig
from backend.services.equity_scope import (
    ctrader_money_mode,
    drawdown_suppressed_by_sandbox,
    is_split_book,
    risk_broker_name,
)
from backend.services.trading_mode import TradingMode, get_trading_mode
from backend.database.models import Trade, PortfolioSnapshot
from backend.services.ledger import is_binance_paper_fill


def _snapshot_risk_equity(snapshot: PortfolioSnapshot | None) -> float:
    """Equity the drawdown / daily-loss gates should see.

    Paper snapshots used to store ``total_value = cash + margin_used``.
    That is reserved buying power, not NAV. Detect that shape from the
    row itself so a later TRADING_MODE=live flip cannot revive a fake
    $198k peak against a $133k current on a flat $100k book.
    """
    if snapshot is None:
        return 0.0
    cash = float(getattr(snapshot, "cash", 0.0) or 0.0)
    total_value = float(getattr(snapshot, "total_value", 0.0) or 0.0)
    positions_value = float(getattr(snapshot, "positions_value", 0.0) or 0.0)
    broker = (getattr(snapshot, "broker", None) or "").lower()
    mode = (getattr(snapshot, "mode", None) or "").lower()
    # Live USDT-M notional >> NAV. The paper inflation heuristic would
    # rewrite a live Binance snapshot to ``cash`` and trip drawdown.
    # Unlabeled / cTrader rows can still be leftover paper $198k books.
    if mode == "live" and "binance" in broker:
        return total_value
    inflated = total_value - cash
    if cash > 0 and inflated > 0 and positions_value >= inflated * 0.9:
        return cash
    if get_trading_mode() == TradingMode.PAPER and cash > 0:
        return cash
    return total_value


def _filter_snapshots_for_risk(rows: list[PortfolioSnapshot]) -> list[PortfolioSnapshot]:
    """Scope snapshot rows to the risk broker book (live-cash when split).

    In live mode, strictly enforce mode='live' and match broker/account_id.
    Never fallback to paper or unpartitioned rows in live mode.
    Demo cTrader rows are ignored when the risk book is live Binance.
    """
    active = risk_broker_name() if is_split_book() else _active_broker_name()
    aliases = {
        "ctrader": {"ctrader", "ctrader:paper", "ic", "icmarkets"},
        "binance_futures": {"binance_futures", "binance", "binanceusdm"},
    }
    wanted = aliases.get(active, {active})
    current_mode = get_trading_mode().value if hasattr(get_trading_mode(), "value") else str(get_trading_mode())
    is_live = (current_mode.lower() == "live")
    active_account = os.getenv("CTRADER_ACCOUNT_ID") if active.startswith("ctrader") else None
    split = is_split_book()
    skip_broker_filter = (not split) and active in {"all", "dual", "both", "*"}

    scoped = []
    for r in rows:
        row_broker = (getattr(r, "broker", None) or "").lower()
        row_mode = (getattr(r, "mode", None) or "").lower()
        row_account = getattr(r, "account_id", None)

        if is_live:
            # In live mode, strict matching: mode must be explicitly 'live', never null or paper
            if row_mode != "live":
                continue
            if split and not row_broker:
                continue
            if row_broker and not skip_broker_filter and row_broker not in wanted:
                continue
            if active_account and row_account and str(row_account).strip() != str(active_account).strip():
                continue
            scoped.append(r)
        else:
            # In paper/backtest mode:
            if row_mode and row_mode != current_mode.lower() and row_mode != "paper":
                continue
            if row_broker and not skip_broker_filter and row_broker not in wanted:
                continue
            if active_account and row_account and str(row_account).strip() != str(active_account).strip():
                continue
            scoped.append(r)

    if is_live:
        return scoped
    return scoped if scoped else rows


def latest_snapshot_for_risk(db: Session) -> PortfolioSnapshot | None:
    """Newest snapshot that belongs to the risk book (live-cash when split).

    Trading-loop kill/drawdown must not key off a demo cTrader row while
    Binance is the live-cash book. Bounded query — never load the full table.
    """
    q = (
        db.query(PortfolioSnapshot)
        .filter(PortfolioSnapshot.total_value > 0)
        .order_by(PortfolioSnapshot.timestamp.desc())
    )
    if is_split_book():
        risk = risk_broker_name()
        if risk == "binance_futures":
            q = q.filter(
                PortfolioSnapshot.broker.in_(("binance_futures", "binance", "binanceusdm"))
            )
        elif str(risk).startswith("ctrader"):
            q = q.filter(
                PortfolioSnapshot.broker.in_(("ctrader", "ctrader:paper", "ic", "icmarkets"))
            )
        if get_trading_mode() == TradingMode.LIVE:
            q = q.filter(PortfolioSnapshot.mode == "live")
        return q.first()
    raw_rows = q.limit(200).all()
    scoped = _filter_snapshots_for_risk(raw_rows)
    return scoped[0] if scoped else None


def _peak_risk_equity(db: Session, window_start: datetime, current_value: float) -> float:
    """Max de-poisoned snapshot equity in the lookback window.

    Scopes snapshots to the active broker book and mode to prevent paper/live
    cross-contamination.
    """
    raw_rows = (
        db.query(PortfolioSnapshot)
        .filter(PortfolioSnapshot.timestamp >= window_start)
        .all()
    )
    rows = _filter_snapshots_for_risk(raw_rows)
    values = [_snapshot_risk_equity(row) for row in rows]
    values = [v for v in values if v > 0]
    cur = float(current_value or 0.0)
    if cur > 0 and values:
        # Drop peaks more than 5x current NAV — fallback against legacy unpartitioned rows.
        sane = [v for v in values if v <= cur * 5.0]
        if sane:
            return max(sane)
        return cur
    if values:
        return max(values)
    fallback_rows = _filter_snapshots_for_risk(db.query(PortfolioSnapshot).all())
    fallback = [_snapshot_risk_equity(row) for row in fallback_rows]
    fallback = [v for v in fallback if v > 0]
    if cur > 0 and fallback:
        sane = [v for v in fallback if v <= cur * 5.0]
        if sane:
            return max(sane)
        return cur
    return max(fallback) if fallback else current_value

logger = logging.getLogger(__name__)

# Drawdown peak is computed over a rolling window, NOT all-time.
PEAK_LOOKBACK_HOURS = float(os.getenv("RISK_PEAK_LOOKBACK_HOURS", "72"))


def _sane_snapshot_equity(db: Session, current_value: float, *, since: datetime | None = None) -> float | None:
    """Pick a snapshot equity comparable to the active book (ignore cross-broker / paper rows)."""
    cur = float(current_value or 0.0)
    q = db.query(PortfolioSnapshot)
    if since is not None:
        q = q.filter(PortfolioSnapshot.timestamp >= since)
    raw_rows = q.order_by(PortfolioSnapshot.timestamp.asc()).all()
    rows = _filter_snapshots_for_risk(raw_rows)
    vals = []
    for row in rows:
        v = _snapshot_risk_equity(row)
        if v <= 0:
            continue
        if cur > 0 and v > cur * 5.0:
            continue
        vals.append(v)
    if vals:
        return vals[0] if since is not None else vals[-1]
    if since is not None:
        return None
    return cur if cur > 0 else None

class RiskBreach(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _is_fx_trade(trade: Trade) -> bool:
    """cTrader FX/metal rows store quantity in lots, not coin units."""
    broker = str(getattr(trade, "broker", "") or getattr(trade, "exchange", "") or "").lower()
    if broker in {"ctrader", "ctrader:paper", "icmarkets", "ic"}:
        return True
    symbol = str(getattr(trade, "symbol", "") or "").upper().replace("/", "").replace("_", "")
    if len(symbol) == 6 and symbol.isalpha():
        return True
    return False


def _fx_notional_usdt(symbol: str, lots: float, price: float) -> float:
    """Approximate USD notional for 1 standard lot = 100_000 units of base.

    quantity on cTrader trades is lots (0.01 = micro). Crypto-style
    price*qty understates FX by ~1e5 and must not be used for the gate.
    """
    sym = str(symbol or "").upper().replace("/", "").replace("_", "")
    lots = abs(float(lots or 0.0))
    px = abs(float(price or 0.0))
    if lots <= 0:
        return 0.0
    units = lots * 100_000.0
    if len(sym) != 6:
        return units * px if px > 0 else units
    base, quote = sym[:3], sym[3:]
    if base == "USD":
        return units  # USDXXX: base notional is already USD
    if quote == "USD":
        return units * (px if px > 0 else 1.0)  # XXXUSD
    if quote == "JPY":
        # Convert JPY notional to USD with price if USDJPY-like, else ~150.
        jpy_notional = units * (px if px > 0 else 150.0)
        usdjpy = px if base == "USD" and px > 50 else float(os.getenv("RISK_USDJPY_FALLBACK", "150"))
        return jpy_notional / max(usdjpy, 1.0)
    # Other crosses: treat quote*units as quote-ccy, rough USD via price if small.
    if px > 0 and px < 20:
        return units * px
    return units


def _trade_exposure_usdt(trade: Trade) -> float:
    price = getattr(trade, "entry_price", None) or getattr(trade, "price", None)
    qty = getattr(trade, "quantity", None)
    if price is None or qty is None:
        logger.warning(
            "[RISK GUARD] Open trade %s missing price/qty; excluded from exposure sum",
            getattr(trade, "symbol", "?"),
        )
        return 0.0
    try:
        px = float(price)
        q = float(qty)
    except (TypeError, ValueError):
        return 0.0
    if _is_fx_trade(trade):
        notional = _fx_notional_usdt(str(getattr(trade, "symbol", "") or ""), q, px)
        # Cap compares against equity*mult (~4x). Full FX notional is leveraged
        # buying power; use margin-equivalent exposure so a $1k demo with micros
        # is not always halted. Override with RISK_FX_LEVERAGE.
        lev = float(os.getenv("RISK_FX_LEVERAGE", os.getenv("CTRADER_LEVERAGE", "30")) or 30)
        lev = max(lev, 1.0)
        return notional / lev
    return abs(px * q)


def _active_broker_name() -> str:
    return (os.getenv("ACTIVE_BROKER", "ctrader") or "ctrader").strip().lower()


def _trades_for_risk(open_trades: list[Trade]) -> list[Trade]:
    """Only count the risk-broker book for exposure / position caps.

    Split-book (cTrader demo + Binance live) must count live Binance fills,
    not the demo FX book. Dual-live (ACTIVE_BROKER=dual) still counts both
    live-cash books. Stale paper rows are still skipped in live mode.
    """
    active = risk_broker_name() if is_split_book() else _active_broker_name()
    if active in {"all", "dual", "both", "*"} or not active:
        return list(open_trades)
    aliases = {
        "ctrader": {"ctrader", "ctrader:paper", "ic", "icmarkets"},
        "binance_futures": {"binance_futures", "binance", "binanceusdm"},
    }
    wanted = aliases.get(active, {active})
    scoped = []
    live_mode = get_trading_mode() == TradingMode.LIVE
    split = is_split_book()
    for t in open_trades:
        if live_mode:
            if is_binance_paper_fill(t):
                continue
        broker = str(getattr(t, "broker", "") or getattr(t, "exchange", "") or "").lower()
        if not broker:
            if live_mode and split:
                continue
            scoped.append(t)
            continue
        if broker in wanted:
            scoped.append(t)
    return scoped


def _directional_exposure_usdt(open_trades: list[Trade]) -> float:
    """Sum absolute exposure of open trades on the active broker book.

    Crypto: |entry_price * quantity|.
    FX/cTrader: lot notional in USD / RISK_FX_LEVERAGE (margin-equivalent).
    """
    total = 0.0
    for t in open_trades:
        total += _trade_exposure_usdt(t)
    return total


def is_drawdown_suppressed_in_testing() -> bool:
    """True when running in sandbox/testing mode and drawdown halt should be suppressed.

    Checks:
    - Explicit DISABLE_DRAWDOWN_IN_TESTING=true or TESTING_MODE=true
    - cTrader sandbox or paper mode (unless ENFORCE_SANDBOX_DRAWDOWN=true)
    - TRADING_MODE is paper or backtest (unless ENFORCE_TESTING_DRAWDOWN=true)
    """
    if os.getenv("DISABLE_DRAWDOWN_IN_TESTING", "false").lower() == "true":
        return True
    if os.getenv("TESTING_MODE", "false").lower() == "true":
        return True
    if drawdown_suppressed_by_sandbox():
        return True
    if get_trading_mode() in (TradingMode.PAPER, TradingMode.BACKTEST) and os.getenv("ENFORCE_TESTING_DRAWDOWN", "false").lower() != "true":
        return True
    return False


def enforce_risk_limits(
    db: Session,
    cfg: RiskConfig,
    open_trades: list[Trade],
    latest_snapshot: PortfolioSnapshot | None,
):
    """Enforces active trading safety guardrails.

    Raises RiskBreach if any boundary (drawdown, daily loss, max open
    positions, directional exposure) is violated. Designed to FAIL CLOSED:
    the caller must treat *any* exception escaping this function as a reason
    to skip the cycle, never to trade unguarded.
    """
    # The DISABLE_RISK_GUARD escape hatch is a foot-gun on a live account: one
    # stray env var would silently strip every protection from real money.
    # It is only honored in PAPER mode now — in LIVE mode it is ignored and a
    # loud error is logged so a misconfiguration cannot disable live guards.
    if os.getenv("DISABLE_RISK_GUARD", "false").lower() == "true":
        if get_trading_mode() == TradingMode.PAPER:
            logger.warning(
                "Risk guard disabled via DISABLE_RISK_GUARD (allowed: PAPER mode)."
            )
            return
        logger.error(
            "DISABLE_RISK_GUARD=true IGNORED in LIVE mode — risk guard stays ACTIVE."
        )

    # 1. Max open positions. Use max_positions (the single source of truth for
    #    the concurrent-position cap); max_open_positions is kept only as a
    #    looser legacy ceiling, so enforce the stricter of the two.
    #
    #    Count DISTINCT SYMBOLS, not raw trade rows. With pyramid DCA each entry
    #    layer is its own Trade row (e.g. 4 AVAX layers = 1 position), so
    #    len(open_trades) over-counts and would false-trigger this breach. This
    #    matches how the trading loop itself defines an open position:
    #    func.count(func.distinct(Trade.symbol)).
    risk_trades = _trades_for_risk(open_trades)
    open_symbols = {
        s for t in risk_trades if (s := getattr(t, "symbol", None))
    }
    position_count = len(open_symbols)
    position_cap = min(cfg.max_positions, cfg.max_open_positions)
    if position_count > position_cap:
        raise RiskBreach(
            f"Max open positions exceeded: {position_count} > {position_cap} "
            f"(symbols: {sorted(open_symbols)})"
        )

    # 2. Max directional exposure (notional / FX margin-equivalent).
    # When equity_sizing_enabled is True, calculate dynamic cap based on equity * max_direction_notional_equity_mult.
    effective_exposure_cap = cfg.max_directional_exposure_usdt
    if getattr(cfg, "equity_sizing_enabled", False):
        equity = 0.0
        if latest_snapshot and (
            latest_snapshot.total_value > 0 or float(getattr(latest_snapshot, "cash", 0) or 0) > 0
        ):
            equity = _snapshot_risk_equity(latest_snapshot)
        else:
            if get_trading_mode() == TradingMode.PAPER:
                equity = 100000.0
        if equity > 0:
            mult = getattr(cfg, "max_direction_notional_equity_mult", 4.0)
            effective_exposure_cap = max(cfg.max_directional_exposure_usdt, equity * mult)

    if effective_exposure_cap > 0:
        exposure = _directional_exposure_usdt(risk_trades)
        if exposure > effective_exposure_cap:
            msg = (
                f"Max directional exposure exceeded: ${exposure:.2f} > "
                f"${effective_exposure_cap:.2f}"
            )
            # Paper oversized fills must not freeze SL/TP management. Live still
            # fail-closes the cycle.
            if get_trading_mode() == TradingMode.PAPER:
                logger.warning("[RISK GUARD] %s — paper mode continues so exits still run", msg)
            else:
                raise RiskBreach(msg)

    if latest_snapshot:
        current_value = _snapshot_risk_equity(latest_snapshot)
        # If the latest snapshot is still a paper $100k row while the live
        # broker book is ~$1k, prefer the live cTrader equity when available.
        try:
            # Demo cTrader equity must not replace a live Binance snapshot.
            if (
                _active_broker_name().startswith("ctrader")
                and ctrader_money_mode() == "live_cash"
                and not is_split_book()
            ):
                from backend.services.ctrader_service import ctrader_service
                live_eq = float(getattr(ctrader_service, "equity", 0) or 0)
                if live_eq > 0 and (current_value <= 0 or current_value > live_eq * 5.0):
                    current_value = live_eq
        except Exception:
            pass

        # 3. Max portfolio drawdown — peak over a ROLLING window (see module note).
        #    Disable entirely by setting max_portfolio_drawdown_pct >= 100
        #    (e.g. RISK_MAX_DRAWDOWN_PCT=99 in .env) — useful for testing the
        #    loop through an existing drawdown. >=100 means "no drawdown can ever
        #    exceed it", so we skip the query work and never raise.
        if cfg.max_portfolio_drawdown_pct < 100:
            window_start = datetime.now(timezone.utc) - timedelta(hours=PEAK_LOOKBACK_HOURS)
            peak_value = _peak_risk_equity(db, window_start, current_value)
            if current_value < peak_value:
                drawdown_pct = ((peak_value - current_value) / peak_value) * 100
                if drawdown_pct > cfg.max_portfolio_drawdown_pct:
                    msg = (
                        f"Max portfolio drawdown exceeded: {drawdown_pct:.2f}% > {cfg.max_portfolio_drawdown_pct}% "
                        f"(Peak: ${peak_value:.2f}, Current: ${current_value:.2f})"
                    )
                    # Allow disabling drawdown halt when testing or in sandbox
                    if is_drawdown_suppressed_in_testing():
                        logger.warning(
                            f"[RISK GUARD] %s — testing/sandbox active: drawdown halt suppressed", msg
                        )
                    else:
                        raise RiskBreach(msg)

        # 4. Max daily loss — relative to the first snapshot of today (UTC).
        #    Disable entirely with max_daily_loss_pct >= 100
        #    (e.g. RISK_MAX_DAILY_LOSS_PCT=100 in .env) — mirrors the drawdown
        #    sentinel so the loop can be tested through a down day.
        if cfg.max_daily_loss_pct < 100:
            start_of_today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            # Prefer a same-scale snapshot (demo ~$1k), not leftover paper ~$100k.
            start_value = _sane_snapshot_equity(db, current_value, since=start_of_today)
            if start_value is None or (current_value > 0 and start_value > current_value * 5.0):
                start_value = _sane_snapshot_equity(db, current_value, since=None)
            if start_value is None:
                logger.warning(
                    "[RISK GUARD] No baseline snapshot for daily-loss check; skipping (cold start)."
                )

            if start_value and current_value < start_value:
                daily_loss_pct = ((start_value - current_value) / start_value) * 100
                if daily_loss_pct > cfg.max_daily_loss_pct:
                    raise RiskBreach(
                        f"Max daily loss exceeded: {daily_loss_pct:.2f}% > {cfg.max_daily_loss_pct}% "
                        f"(Daily Start: ${start_value:.2f}, Current: ${current_value:.2f})"
                    )
