"""
Automated Trading Loop Service
Runs as a background asyncio task, scanning markets and generating signals/trades.
"""

import asyncio
import os
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

import yfinance as yf
from dotenv import load_dotenv

from backend.database.connection import SessionLocal
from backend.database.models import TradingSignal, Trade, PortfolioSnapshot, PaperPortfolio
from backend.services.ctrader_service import ctrader_broker
from backend.services.binance_futures_service import binance_futures_broker
from backend.services.influxdb_writer import influx
from backend.services.binance_market_data import binance_market_data
from backend.services.binance_websocket import binance_ws
from backend.services.multi_provider_data import multi_provider_data

from backend.strategies.market_regime import MarketRegimeDetector
from backend.llm.router import LLMChainExhausted
from backend.services.decision_engine import DecisionEngine
from backend.services.equity_scope import (
    compose_balance_payload,
    effective_risk_scope,
    is_split_book,
    risk_broker_name,
)
from backend.services.market_hours import is_venue_open
from backend.services.unified_trading import UnifiedTrading, UnifiedOrder, OrderSide, OrderType
from backend.services.position_manager import get_position_manager
from backend.services.sentry_state import get_trading_status, is_trading_allowed
from backend.services.trading_mode import (
    BINANCE_PAPER_SESSION_ID,
    TradingMode,
    binance_order_session_id,
    binance_paper_parallel_enabled,
    get_trading_mode,
    live_binance_orders_allowed,
    paper_leverage_for_broker,
    paper_reported_equity,
    paper_starting_balance,
)
from backend.services.trading_loop_helpers import (
    EmergencyExitManager,
    BrokerPositionSyncService,
    TrailingStopManager,
    PartialTPManager,
    ExchangeProtectionManager,
    PerformanceMetricsWriter,
    remove_closed_pyramid_layer,
    trades_for_direction_cap,
    is_ctrader_trade,
)
from backend.services.symbol_aliases import (
    first_matching_open,
    is_flatten_action,
    perp_alias_set,
    same_crypto_perp_leg,
)

load_dotenv()

# ── Broker selector ──────────────────────────────────────────────────────────
DEFAULT_ACTIVE_BROKER = "ctrader"


def get_active_broker_name() -> str:
    """Name of the active broker. Single source of truth for broker resolution."""
    return os.getenv("ACTIVE_BROKER", DEFAULT_ACTIVE_BROKER).strip() or DEFAULT_ACTIVE_BROKER


def get_active_broker():
    """Dynamically resolve the active broker based on environment."""
    if get_active_broker_name() == "binance_futures":
        return binance_futures_broker
    return ctrader_broker

logger = structlog.get_logger(__name__)


def negative_expectancy_symbols(rows: list[tuple], min_trades: int) -> set[str]:
    """Symbols whose realized P&L is negative on a meaningful sample.

    `rows` are (symbol, trade_count, pnl_sum) aggregates over the lookback
    window (NULL-pnl closes already excluded by the caller's query).
    """
    blocked = set()
    for symbol, n, pnl_sum in rows:
        if n is None or pnl_sum is None:
            continue
        if n >= min_trades and pnl_sum < 0:
            blocked.add(str(symbol).upper())
    return blocked


class TradingLoopService:
    """Background trading loop that scans markets and generates paper trades."""

    def __init__(self):
        from backend.services.risk_config import get_risk_config
        self.risk_config = get_risk_config()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._state = "stopped"  # stopped, running, error
        self._error: Optional[str] = None
        self._interval_minutes = 15
        env_syms = os.getenv('TRADING_SYMBOLS', '')
        # Default universe omits ADA/ARB/DOGE/APT — confirmed weekly loss leaders.
        self._symbols = [s.strip() for s in env_syms.split(',') if s.strip()] if env_syms else [
            'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'POLUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'NEARUSDT',
            'OPUSDT', 'INJUSDT', 'BTCUSDT',
        ]
        self._strategy_name = "combined"
        self._last_cycle: Optional[str] = None
        self._next_cycle: Optional[str] = None
        self._cycle_count = 0
        # Per-symbol regime detectors injected into each cycle's DecisionEngine.
        # The detector's 5-bar transition smoothing depends on _history
        # persisting across calls; a fresh DecisionEngine per cycle used to
        # wipe it, making smoothing dead code. Keyed by symbol so histories
        # don't mix across instruments.
        self._regime_detectors: Dict[str, MarketRegimeDetector] = {}
        self._unified_trading = None  # Will be set to singleton in start()
        self._pyramid_layers = {}
        self._execution_lock = asyncio.Lock()
        # Cooldown tracker: symbol → datetime when SL/emergency exit fired
        # Prevents immediate re-entry after a losing close (30 min cooldown)
        self._sl_cooldown: dict = {}
        # P0 margin gate state (set each cycle from broker balance)
        self._entries_blocked = False
        self._entries_blocked_reason = ""
        self._cycle_available = 0.0
        self._cycle_margin_used = 0.0
        # Trailing-stop high/low-water marks, keyed by trade id → extreme price
        # reached since entry (max for longs, min for shorts). In-memory only;
        # reconstructed lazily from entry/current price on first sight.
        self._high_water: dict = {}
        self._last_digest_date = None
        # Cycle-level cache for exchange positions (populated once per cycle,
        # avoids calling get_positions() per symbol which returns ALL positions)
        self._cycle_positions: list = []
        # Semaphore: stagger symbol processing to bound Binance API bursts.
        # Default concurrency=1 with SYMBOL_BATCH_PAUSE_SEC=45 → one coin every ~45s
        # (20 symbols ≈ 15+ min per cycle). Env-tunable without rebuild.
        self._symbol_semaphore = asyncio.Semaphore(
            max(1, int(os.getenv("SYMBOL_CONCURRENCY", "1")))
        )
        # New-bar gate: symbol → open-time of the last bar the entry pipeline
        # evaluated. The loop cycles every 15 min on 1h bars, so without this
        # the same bar was re-decided up to 4x (churn + redundant LLM cost).
        self._last_eval_bar: dict = {}

    def _entry_bar_interval(self) -> str:
        """OHLCV timeframe for the entry pipeline.

        The new-bar gate allows one evaluation per bar. A 5-minute loop on
        1h candles therefore looks idle for ~60 minutes after the first
        cycle. Match the bar size to the loop interval so Trading Control
        actually emits signals on the cadence the operator set.
        """
        mins = int(getattr(self, "_interval_minutes", 15) or 15)
        if mins <= 5:
            return "5m"
        if mins <= 15:
            return "15m"
        return "1h"

    @staticmethod
    def _bar_interval_max_age_sec(interval: str) -> float:
        seconds = {"5m": 300.0, "15m": 900.0, "1h": 3600.0}.get(interval, 3600.0)
        override = os.getenv("WS_CANDLE_MAX_AGE_SEC")
        if override:
            try:
                return float(override)
            except (TypeError, ValueError):
                pass
        return 2.5 * seconds

    @staticmethod
    def _ctrader_period_for_interval(interval: str) -> str:
        mapping = {"5m": "M5", "15m": "M15", "1h": "H1"}
        return mapping.get(interval, "H1")

    def _regime_detector_for(self, symbol: str) -> MarketRegimeDetector:
        """Reuse one detector per symbol so regime smoothing can accumulate."""
        det = self._regime_detectors.get(symbol)
        if det is None:
            det = MarketRegimeDetector()
            self._regime_detectors[symbol] = det
        return det

    @staticmethod
    def _bar_open_age_seconds(bar: dict) -> Optional[float]:
        """Seconds since bar open time, or None if unparseable."""
        raw = bar.get("date") if bar else None
        if not raw:
            return None
        try:
            if isinstance(raw, datetime):
                dt = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).total_seconds()
        except Exception:
            return None

    def _detector_for(self, symbol: str) -> MarketRegimeDetector:
        """Return the persistent per-symbol regime detector (creates on first use)."""
        detector = self._regime_detectors.get(symbol)
        if detector is None:
            detector = MarketRegimeDetector()
            self._regime_detectors[symbol] = detector
        return detector

    def _should_evaluate_bar(self, symbol: str, bars: list[dict]) -> bool:
        """Entry pipeline runs at most once per bar (exits still run every cycle).

        Returns True when the latest bar is new since the last evaluation for
        this symbol (and marks it evaluated). Fails open when bars carry no
        timestamp or the gate is disabled.
        """
        if not getattr(self.risk_config, "eval_on_new_bar_only", True):
            return True
        last_bar_ts = bars[-1].get("date") if bars else None
        if not last_bar_ts:
            return True  # no timestamp info — don't block trading on it
        if self._last_eval_bar.get(symbol) == last_bar_ts:
            logger.info(
                f"  [{symbol}] new-bar gate: skip entry pipeline "
                f"(already evaluated bar {last_bar_ts})"
            )
            return False
        self._last_eval_bar[symbol] = last_bar_ts
        return True

    @staticmethod
    def _is_live_binance() -> bool:
        """True when crypto fills/sync must talk to the live Binance book.

        Dual-broker live (ACTIVE_BROKER=ctrader + TRADING_MODE=live) still
        owns USDT-M rows on Binance unless BINANCE_PAPER_PARALLEL is on.
        """
        return live_binance_orders_allowed()

    @staticmethod
    def _crypto_session_id() -> str | None:
        """Paper-parallel crypto fills use a dedicated session, not live cTrader.

        Live dual-broker must NOT return the Binance live session here — callers
        treat a non-None value as 'use the paper book / paper leverage'.
        """
        if binance_paper_parallel_enabled() and not TradingLoopService._is_live_binance():
            return BINANCE_PAPER_SESSION_ID
        return None

    @staticmethod
    def _binance_order_session_id() -> str | None:
        """Session id for crypto UnifiedTrading.place_order (paper or live)."""
        return binance_order_session_id()

    @staticmethod
    def _crypto_broker_name() -> str:
        if TradingLoopService._crypto_session_id() or TradingLoopService._is_live_binance():
            return "binance_futures"
        return get_active_broker_name()

    @staticmethod
    def _paper_balance_payload(cash: float, equity: float, margin_used: float = 0.0) -> dict:
        return {
            "balance": cash,
            "available": cash,
            "equity": equity,
            "margin_used": margin_used,
            "broker": "paper_trading",
        }

    def _last_paper_book_from_db(self) -> dict | None:
        """Most recent persisted paper book, if it still has cash."""
        try:
            db = SessionLocal()
            try:
                row = db.query(PaperPortfolio).order_by(PaperPortfolio.id.desc()).first()
                if row is None:
                    return None
                cash = float(row.cash or 0.0)
                if cash <= 0:
                    return None
                margin = float(row.margin_used or 0.0)
                return self._paper_balance_payload(
                    cash, paper_reported_equity(cash, margin), margin
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Could not load last paper book: {e}")
            return None

    def _get_effective_balance(self) -> dict:
        """Paper book in paper mode; live broker only when TRADING_MODE=live.

        Never reads the live Binance futures wallet in paper mode — a $0
        live account must not trip the kill switch. Prefer the in-memory
        paper portfolio, then the last DB paper book, then PAPER_BALANCE.
        """
        paper_sid = self._crypto_session_id()
        if get_trading_mode() != TradingMode.LIVE or paper_sid:
            try:
                pf = UnifiedTrading().get_paper_portfolio(paper_sid)
                if pf:
                    cash = float(pf.get("cash", 0.0) or 0.0)
                    margin = float(pf.get("margin_used", 0.0) or 0.0)
                    equity = float(pf.get("equity") or paper_reported_equity(cash, margin) or 0.0)
                    if equity > 0:
                        return self._paper_balance_payload(cash, equity, margin)
            except Exception as e:
                logger.warning(f"Could not fetch paper portfolio balance: {e}")
            last_book = self._last_paper_book_from_db()
            if last_book:
                return last_book
            starting = paper_starting_balance()
            return self._paper_balance_payload(starting, starting, 0.0)
        books: list[dict] = []
        try:
            if hasattr(ctrader_broker, "get_balance"):
                ct = ctrader_broker.get_balance() or {}
                if ct:
                    books.append({**ct, "broker": ct.get("broker") or "ctrader"})
        except Exception as exc:
            logger.warning("cTrader get_balance failed: %s", exc)
            books.append({"broker": "ctrader", "error": type(exc).__name__})

        if (
            self._is_live_binance()
            or get_active_broker_name() == "binance_futures"
            or risk_broker_name() == "binance_futures"
        ):
            try:
                bn = binance_futures_broker.get_balance(
                    raise_on_error=bool(
                        self._is_live_binance()
                        or risk_broker_name() == "binance_futures"
                    )
                ) or {}
                if bn:
                    books.append({**bn, "broker": bn.get("broker") or "binance_futures"})
            except Exception as exc:
                logger.warning("Binance get_balance failed (not treating as $0): %s", exc)
                books.append({"broker": "binance_futures", "error": type(exc).__name__})

        payload = compose_balance_payload(books)
        if not books:
            payload["error"] = payload.get("error") or "no_live_broker_balance"
        return payload

    def _kill_switch_action(self, equity: float, balance: dict) -> str:
        """Return 'halt', 'block_entries', or 'ok'.

        Paper/backtest never halt the loop: simulated cash is not a live
        funding failure, and live Binance $0 must not stop paper trading.
        """
        if get_trading_mode() != TradingMode.LIVE:
            return "ok"
        # A failed wallet fetch must not be treated as $0 equity (that used to
        # permanently halt the loop on any Binance blip).
        if balance.get("error"):
            return "block_entries"
        if equity <= 0 and "equity" not in balance and "balance" not in balance:
            return "block_entries"
        if equity <= float(self.risk_config.kill_floor_usdt):
            return "halt"
        return "ok"

    def _paper_cycle_positions(self) -> list:
        """Map the local paper book to the loop's exchange-position shape."""
        try:
            positions = UnifiedTrading().get_paper_positions(self._crypto_session_id())
            out = []
            for p in positions:
                qty = float(getattr(p, "quantity", 0) or 0)
                if qty <= 0:
                    continue
                side = (getattr(p, "side", "") or "").lower()
                out.append({
                    "symbol": p.symbol,
                    "side": "BUY" if side in ("long", "buy") else "SELL",
                    "quantity": qty,
                    "entry_price": float(getattr(p, "avg_price", 0) or 0),
                    "unrealized_pnl": float(getattr(p, "unrealized_pnl", 0) or 0),
                    "mark_price": float(getattr(p, "current_price", 0) or 0),
                })
            return out
        except Exception as e:
            logger.warning(f"Could not cache paper positions: {e}")
            return []

    @property
    def status(self) -> dict:
        balance_info = self._get_effective_balance()
        allowed = is_trading_allowed()
        effective_state = self._state
        if self._running:
            if not allowed:
                effective_state = "halted"
            elif self._state == "halted":
                effective_state = "running"
        return {
            "state": effective_state,
            "running": self._running,
            "interval_minutes": self._interval_minutes,
            "symbols": self._symbols,
            "strategy": self._strategy_name,
            "last_cycle": self._last_cycle,
            "next_cycle": self._next_cycle,
            "cycle_count": self._cycle_count,
            "error": self._error,
            "cash": float(balance_info.get("available", balance_info.get("balance", 0.0))),
            "equity": float(balance_info.get("equity", balance_info.get("balance", 0.0))),
            "margin_used": float(balance_info.get("margin_used", 0.0)),
            "equity_scope": balance_info.get("equity_scope") or effective_risk_scope(),
            "split_book": bool(balance_info.get("split_book") or is_split_book()),
            "risk_broker": balance_info.get("risk_broker") or risk_broker_name(),
            "ctrader_env": balance_info.get("ctrader_env"),
            "binance_env": balance_info.get("binance_env"),
            "split_book_warning": balance_info.get("split_book_warning"),
            "books": balance_info.get("books"),
            "trading_allowed": allowed,
            "trading_status": get_trading_status().value,
        }

    async def start(
        self,
        interval_minutes: int = 15,
        symbols: list[str] | None = None,
        strategy: str = "combined",
    ):
        """Start the trading loop."""
        if self._running:
            return {"message": "Trading loop is already running"}

        self._interval_minutes = interval_minutes
        if symbols:
            self._symbols = symbols
        self._strategy_name = strategy
        self._running = True
        self._state = "running"
        self._error = None
        self._unified_trading = UnifiedTrading()  # Set singleton instance
        sid = self._crypto_session_id()
        if sid:
            if not self._unified_trading.get_session(sid):
                self._unified_trading.init_session(
                    "binance_futures",
                    mode="paper",
                    paper_balance=paper_starting_balance(),
                    leverage=paper_leverage_for_broker("binance_futures"),
                    session_id=sid,
                )
            self._unified_trading.set_default_session(sid)
            logger.info("Trading loop crypto fills → Binance paper session %s", sid)
        else:
            live_sid = self._binance_order_session_id()
            if live_sid:
                if not self._unified_trading.get_session(live_sid):
                    self._unified_trading.init_session(
                        "binance_futures",
                        mode="live",
                        leverage=paper_leverage_for_broker("binance_futures"),
                        session_id=live_sid,
                    )
                logger.info(
                    "Trading loop crypto fills → Binance live session %s "
                    "(default session unchanged)",
                    live_sid,
                )

        mode = get_trading_mode()
        logger.info(f"TradingLoopService starting in mode={mode.value.upper()}")
        
        # Reconstruct pyramid layers from DB
        self._reconstruct_pyramid_layers()
        self._reconstruct_cooldowns()

        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Trading loop started: interval={interval_minutes}m, "
            f"symbols={self._symbols}, strategy={strategy}"
        )
        return {"message": "Trading loop started", "status": self.status}

    async def run(
        self,
        interval_minutes: int = 15,
        symbols: list[str] | None = None,
        strategy: str = "combined",
    ):
        """Supervisor entry: start the loop and await it so crashes restart."""
        await self.start(
            interval_minutes=interval_minutes,
            symbols=symbols,
            strategy=strategy,
        )
        if self._task is not None:
            await self._task

    def _reconstruct_cooldowns(self):
        """Repopulate SL cooldowns from recently-closed trades so a restart
        doesn't immediately re-enter a symbol that just stopped out."""
        db = SessionLocal()
        try:
            window_min = max(
                getattr(self.risk_config, "sl_cooldown_minutes", 30),
                getattr(self.risk_config, "opinion_close_cooldown_min", 30),
            )
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
            recent = db.query(Trade).filter(
                Trade.status == "closed", Trade.closed_at.isnot(None)
            ).order_by(Trade.closed_at.desc()).limit(50).all()
            restored = 0
            for t in recent:
                closed = t.closed_at
                if closed and closed.tzinfo is None:
                    closed = closed.replace(tzinfo=timezone.utc)
                if closed and closed >= cutoff and t.symbol not in self._sl_cooldown:
                    self._sl_cooldown[t.symbol] = closed
                    restored += 1
            if restored:
                logger.info(f"Restored {restored} SL cooldown(s) from recent closed trades")
        except Exception as e:
            logger.warning(f"Failed to reconstruct cooldowns: {e}")
        finally:
            db.close()

    @staticmethod
    def _pyramid_prices_from_trades(trades: list) -> dict[str, list[float]]:
        """Rebuild pyramid add prices per symbol (base entry is NOT a layer)."""
        by_symbol: dict[str, list] = {}
        for t in trades:
            by_symbol.setdefault(t.symbol, []).append(t)

        layers: dict[str, list[float]] = {}
        for symbol, sym_trades in by_symbol.items():
            sym_trades_sorted = sorted(sym_trades, key=lambda x: x.id or 0)
            base_id = sym_trades_sorted[0].id if sym_trades_sorted else None
            pyramid_prices: list[float] = []
            for t in sym_trades_sorted:
                notes = t.notes or ""
                if "pyramid_layer" in notes:
                    if t.entry_price is not None:
                        pyramid_prices.append(float(t.entry_price))
                    continue
                # Legacy rows without notes: first open trade is base, rest are adds
                if len(sym_trades_sorted) > 1 and t.id != base_id:
                    if t.entry_price is not None:
                        pyramid_prices.append(float(t.entry_price))
            if pyramid_prices:
                layers[symbol] = pyramid_prices
        return layers

    def _reconstruct_pyramid_layers(self):
        """Populate self._pyramid_layers from open pyramid-add trades in DB."""
        db = SessionLocal()
        try:
            open_trades = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
            self._pyramid_layers = self._pyramid_prices_from_trades(open_trades)
            if self._pyramid_layers:
                logger.info(
                    "Reconstructed pyramid layers for symbols: "
                    f"{ {k: len(v) for k, v in self._pyramid_layers.items()} }"
                )
        except Exception as e:
            logger.warning(f"Failed to reconstruct pyramid layers: {e}")
        finally:
            db.close()

    async def _sync_positions_with_broker(self, db):
        """Sync DB trades with actual broker positions.
        Uses the module-level binance_futures_broker singleton — avoids creating
        a new BinanceFuturesService() (which calls futures_exchange_info() in __init__).
        Paper mode never reconciles against the live Binance book.
        """
        if self._is_live_binance():
            await BrokerPositionSyncService.sync_positions(
                db=db,
                broker=binance_futures_broker,
                pyramid_layers=self._pyramid_layers,
                sl_cooldown=self._sl_cooldown,
                broker_name="binance_futures",
            )
        try:
            from backend.services.ctrader_service import ctrader_broker
            if ctrader_broker.has_credentials():
                await BrokerPositionSyncService.sync_positions(
                    db=db,
                    broker=ctrader_broker,
                    pyramid_layers=self._pyramid_layers,
                    sl_cooldown=self._sl_cooldown,
                    broker_name="ctrader",
                )
        except Exception as _ct_e:
            logger.warning("cTrader position sync during trading loop failed: %s", _ct_e)

    async def stop(self):
        """Stop the trading loop."""
        self._running = False
        self._state = "stopped"
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._next_cycle = None
        logger.info("Trading loop stopped")
        return {"message": "Trading loop stopped", "status": self.status}

    async def _loop(self):
        """Main loop: run cycles at the configured interval."""
        try:
            while self._running:
                try:
                    await self._run_cycle()
                except Exception as e:
                    logger.error(f"Trading cycle error: {e}")
                    self._error = str(e)
                    self._state = "error"

                if not self._running:
                    break

                next_time = datetime.now(timezone.utc).timestamp() + (
                    self._interval_minutes * 60
                )
                self._next_cycle = datetime.fromtimestamp(
                    next_time, tz=timezone.utc
                ).isoformat()

                # Sleep in small increments so we can cancel quickly
                for _ in range(self._interval_minutes * 60):
                    if not self._running:
                        break
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self._state = "stopped"

    async def _run_cycle(self):
        """Execute one trading cycle across all symbols in parallel."""
        if not is_trading_allowed():
            sentry_status = get_trading_status().value
            logger.warning(
                f"[SENTRY] Trading halted ({sentry_status}) — skipping cycle"
            )
            self._state = "halted"
            return {
                "signals": 0,
                "trades": 0,
                "sentry_halted": True,
                "trading_status": sentry_status,
            }
        if self._state == "halted":
            self._state = "running"

        # 0. Refresh RiskConfig from environment variables / .env
        from backend.services.risk_config import refresh_risk_config
        self.risk_config = refresh_risk_config()

        self._cycle_count += 1
        
        # UTC rollover daily digest trigger
        current_date_utc = datetime.now(timezone.utc).date()
        if self._last_digest_date is None:
            self._last_digest_date = current_date_utc
        elif current_date_utc > self._last_digest_date:
            self._last_digest_date = current_date_utc
            try:
                from backend.services.daily_digest import run_daily_digest
                asyncio.create_task(run_daily_digest())
                logger.info("Daily performance digest task queued successfully.")
            except Exception as e:
                logger.error(f"Failed to trigger automated daily digest: {e}")

        # Bind the cycle number to all structured logs emitted in this cycle
        structlog.contextvars.bind_contextvars(cycle=self._cycle_count)
        
        self._last_cycle = datetime.now(timezone.utc).isoformat()
        _cycle_start = datetime.now(timezone.utc).timestamp()
        self._state = "running"

        # ── RISK GUARD GATEKEEPER ──────────────────────────────────────────
        from backend.services.risk_guard import (
            enforce_risk_limits,
            latest_snapshot_for_risk,
            RiskBreach,
        )
        db_risk = SessionLocal()
        try:
            open_trades = db_risk.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
            latest_snapshot = latest_snapshot_for_risk(db_risk)
            enforce_risk_limits(db_risk, self.risk_config, open_trades, latest_snapshot)
        except RiskBreach as rb:
            logger.critical(
                f"[RISK BREACH] {rb.reason}. LOOP STOPPED. "
                f"\u26a0\ufe0f Manual restart required after resolving the breach."
            )
            self._state = "error"
            self._running = False
            self._error = f"RISK_BREACH: {rb.reason}"
            return {"signals": 0, "trades": 0, "risk_breach": True}
        except Exception as e:
            # FAIL CLOSED: a transient DB error / bad snapshot in the guard must
            # NOT let an unguarded trading cycle proceed. Previously this branch
            # only logged and fell through to order placement — meaning any
            # hiccup in the risk check silently disabled all risk enforcement
            # for that cycle. Skip the entire cycle instead.
            logger.error(f"[RISK GUARD] Failed to check limits — skipping cycle (fail-closed): {e}")
            db_risk.rollback()
            self._state = "idle"
            return {"signals": 0, "trades": 0, "risk_guard_error": True}
        finally:
            db_risk.close()
        # ─────────────────────────────────────────────────────────────────────

        # ── C8: ACCOUNT-LEVEL KILL SWITCH ──────────────────────────────────
        # Snapshot account balance ONCE per cycle — reused for kill switch and
        # margin gates (was 2x futures_account() per cycle).
        kill_switch_verified = True
        try:
            self._cycle_balance = self._get_effective_balance()
        except Exception as _be:
            logger.warning(f"Could not fetch balance for cycle: {_be}")
            self._cycle_balance = {}
            kill_switch_verified = False
            logger.error(
                "[KILL SWITCH] Equity unavailable; new entries will be blocked"
            )

        _bal = self._cycle_balance
        try:
            self._cycle_equity = float(_bal.get("equity", _bal.get("balance", 0.0)) or 0.0)
            self._cycle_available = float(_bal.get("available", 0.0) or 0.0)
            self._cycle_margin_used = float(_bal.get("margin_used", 0.0) or 0.0)
        except Exception as _be:
            logger.warning(f"Could not parse balance: {_be}")
            self._cycle_equity = 0.0
            self._cycle_available = 0.0
            self._cycle_margin_used = 0.0

        if kill_switch_verified:
            try:
                _kill_floor = self.risk_config.kill_floor_usdt
                _equity = self._cycle_equity
                _ks_action = self._kill_switch_action(_equity, _bal)
                if _ks_action == "block_entries":
                    kill_switch_verified = False
                    logger.error(
                        "[KILL SWITCH] Equity unavailable; new entries will be blocked"
                    )
                elif _ks_action == "halt":
                    logger.critical(
                        f"[KILL SWITCH] Equity ${_equity:.2f} <= floor ${_kill_floor:.2f} — LOOP STOPPED. "
                        f"\u26a0\ufe0f Manual restart required after depositing funds above ${_kill_floor:.0f}."
                    )
                    try:
                        await influx._write(
                            influx.BUCKET_SYSTEM, 'system_event',
                            {'event': 'kill_switch', 'reason': 'equity_below_floor'},
                            {'equity': float(_equity), 'kill_floor': float(_kill_floor)}
                        )
                    except Exception:
                        pass
                    self._state = "stopped"
                    self._running = False
                    self._error = "KILL_SWITCH_TRIGGERED"
                    return {"signals": 0, "trades": 0, "kill_switch": True}
            except Exception as _ks_e:
                kill_switch_verified = False
                logger.error(
                    f"[KILL SWITCH] Equity check failed; new entries will be blocked: {_ks_e}"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── STEP 0: Emergency Position Manager BEFORE broker sync ──────────
        # Live only: paper fills are not on Binance, so a live snapshot would
        # be empty/$0 and must not drive closes or API traffic.
        _exits_triggered = 0
        if self._is_live_binance():
            db_pre = SessionLocal()
            try:
                _exits_triggered = await EmergencyExitManager.run_emergency_exits(
                    db=db_pre,
                    broker=binance_futures_broker,
                    pyramid_layers=self._pyramid_layers,
                    sl_cooldown=self._sl_cooldown,
                )
            except Exception as e:
                logger.error(f"Pre-sync PM outer error: {e}")
            finally:
                db_pre.close()
        self._error = None

        logger.info(
            f"=== Trading Cycle #{self._cycle_count} at {self._last_cycle} (Parallel) ==="
        )

        min_confidence = self.risk_config.min_signal_strength
        ai_threshold = self.risk_config.ai_analysis_threshold
        max_positions = self.risk_config.max_positions

        # ── P0 margin gates: decide ONCE per cycle whether new entries are
        # affordable, instead of letting every symbol fail at the broker.
        self._entries_blocked = not kill_switch_verified
        self._entries_blocked_reason = (
            "kill-switch equity could not be verified"
            if not kill_switch_verified else ""
        )
        _min_avail = self.risk_config.min_available_margin_usdt
        _wallet_cap = self.risk_config.pyramid_max_wallet_pct
        if not kill_switch_verified:
            pass  # preserve the fail-closed reason above
        elif self._cycle_available < _min_avail:
            self._entries_blocked = True
            self._entries_blocked_reason = (
                f"available margin ${self._cycle_available:.2f} < floor ${_min_avail:.2f}"
            )
        elif (self._cycle_equity > 0
              and self._cycle_margin_used >= self._cycle_equity * _wallet_cap):
            self._entries_blocked = True
            self._entries_blocked_reason = (
                f"margin used ${self._cycle_margin_used:.2f} >= "
                f"{_wallet_cap:.0%} of equity ${self._cycle_equity:.2f}"
            )
        if self._entries_blocked:
            logger.warning(
                f"[MARGIN GATE] New entries + pyramid adds BLOCKED this cycle: "
                f"{self._entries_blocked_reason}. Exits/SL/TP still active."
            )
        # Pyramid DCA config
        self._pyramid_mode = self.risk_config.pyramid_mode
        self._pyramid_max_layers = self.risk_config.pyramid_max_layers
        # CRITICAL: directional exposure cap (correlation risk)
        self._max_directional_exposure_usdt = self.risk_config.max_directional_exposure_usdt
        # Pyramid layer separation
        self._pyramid_atr_multiplier = self.risk_config.pyramid_atr_multiplier
        self._pyramid_min_conf_increase = self.risk_config.pyramid_min_conf_increase
        self._pyramid_usdt_per_layer = self.risk_config.pyramid_usdt_per_layer
        self._pyramid_max_wallet_pct = self.risk_config.pyramid_max_wallet_pct
        # Tighter pyramid: only add if price improves by at least 0.5% vs last layer
        self._pyramid_min_improvement = self.risk_config.pyramid_min_improvement
        self._sl_atr_mult = self.risk_config.sl_atr_mult
        self._tp_atr_mult = self.risk_config.tp_atr_mult
        
        db = SessionLocal()
        try:
            # Sync DB trades with actual broker positions
            await self._sync_positions_with_broker(db)
            
            # Count unique symbols with open positions (including filled)
            from sqlalchemy import func
            self._open_count = db.query(func.count(func.distinct(Trade.symbol))).filter(
                Trade.status.in_(["open", "filled"]),
                (Trade.broker != "ctrader") | (Trade.broker.is_(None))
            ).scalar() or 0
            self._total_open_count = db.query(func.count(func.distinct(Trade.symbol))).filter(
                Trade.status.in_(["open", "filled"])
            ).scalar() or 0
        finally:
            db.close()

        # ── Cache all exchange positions ONCE per cycle ──────────────────
        # get_positions() returns ALL symbols; caching here avoids redundant
        # per-symbol API calls in _process_symbol() (was the #1 rate-limit cause).
        # Paper mode reads the local book — never the live futures wallet.
        if self._is_live_binance():
            try:
                self._cycle_positions = binance_futures_broker.get_positions(raise_on_error=True)
            except Exception as _pos_e:
                logger.warning(f"Could not cache exchange positions: {_pos_e}")
                self._cycle_positions = []
        else:
            self._cycle_positions = self._paper_cycle_positions()

        # ── STEP 1A: Position Manager — Review ALL open positions for exits ──
        _exits_triggered = 0
        try:
            pm = get_position_manager()
            db_check = SessionLocal()
            try:
                open_trades = db_check.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
                logger.info(f"  [POSITION MGR] Reviewing {len(open_trades)} open positions for exits...")
                for trade in open_trades:
                    try:
                        if is_ctrader_trade(trade):
                            continue
                        trade_bars = await self._fetch_bars(trade.symbol)
                        if not trade_bars or len(trade_bars) < 10:
                            continue
                        curr_price = trade_bars[-1]["close"]
                        # Prefer the live mark over a (possibly hours-old) candle close.
                        fsym = binance_futures_broker._to_futures_symbol(trade.symbol)
                        for p in self._cycle_positions:
                            if p.get("symbol") == (fsym or trade.symbol):
                                mark = float(p.get("mark_price") or 0)
                                if mark > 0:
                                    curr_price = mark
                                break
                        # Fetch funding rate for position exit review
                        try:
                            fr_data = await binance_market_data.get_funding_rate(trade.symbol)
                            current_funding_rate = float(fr_data.get('fundingRate', 0)) if fr_data else 0.0
                        except Exception:
                            current_funding_rate = 0.0

                        from backend.services.opinion_layer import analyze_symbol as analyze_opinion
                        exit_opinion = await pm.analyze_open_position(
                            symbol=trade.symbol,
                            trade={
                                "entry_price": trade.entry_price,
                                "direction": trade.direction,
                                "quantity": trade.quantity,
                                "opened_at": trade.timestamp.isoformat() if trade.timestamp else None,
                                "stop_loss": trade.stop_loss,
                                "take_profit": trade.take_profit,
                            },
                            bars=trade_bars,
                            current_price=curr_price,
                            opinion_layer_fn=analyze_opinion,
                            current_funding_rate=current_funding_rate,
                        )
                        if exit_opinion.exit:
                            logger.warning(
                                f"  [POSITION MGR] EXIT {trade.symbol}: {exit_opinion.reasoning}"
                            )
                            # Close position immediately
                            ut = UnifiedTrading()
                            close_side = OrderSide.SELL if trade.direction == "BUY" else OrderSide.BUY
                            res = ut.place_order(UnifiedOrder(
                                symbol=trade.symbol,
                                side=close_side,
                                order_type=OrderType.MARKET,
                                quantity=trade.quantity,
                                price=float(curr_price or 0),
                                reduce_only=True,
                            ), session_id=self._binance_order_session_id())
                            if res.success:
                                trade.exit_price = res.filled_price or curr_price
                                if res.realized_pnl is not None:
                                    trade.pnl = float(res.realized_pnl) - res.commission
                                else:
                                    if trade.direction == "BUY":
                                        gross = (trade.exit_price - trade.entry_price) * trade.quantity
                                    else:
                                        gross = (trade.entry_price - trade.exit_price) * trade.quantity
                                    trade.pnl = gross - res.commission
                                trade.status = "closed"
                                trade.closed_at = datetime.now(timezone.utc)
                                trade.notes = (trade.notes or "") + f" | AI EXIT: {exit_opinion.reasoning[:120]}"
                                db_check.add(trade)
                                if self._pyramid_mode:
                                    remove_closed_pyramid_layer(self._pyramid_layers, trade)
                                _exits_triggered += 1
                                logger.info(f"  -> {trade.symbol} CLOSED via Position Manager. PnL={trade.pnl:+.2f}")
                            else:
                                # already_flat = success (exchange SL already fired)
                                if hasattr(res, 'message') and ('-2022' in str(res.message) or 'already' in str(res.message).lower()):
                                    # Verify position is actually flat on the exchange to prevent quantity drift
                                    try:
                                        pos_qty = 0.0
                                        for p in self._cycle_positions:
                                            if p.get('symbol') == binance_futures_broker._to_futures_symbol(trade.symbol):
                                                pos_qty = float(p.get('quantity', 0))
                                                break
                                        if pos_qty == 0:
                                            logger.info(f"  -> {trade.symbol} verified flat on exchange — marking DB closed")
                                            trade.status = 'closed'
                                            trade.closed_at = datetime.now(timezone.utc)
                                            trade.notes = (trade.notes or '') + ' | AI-EXIT: verified flat on exchange'
                                            db_check.add(trade)
                                            if self._pyramid_mode:
                                                remove_closed_pyramid_layer(self._pyramid_layers, trade)
                                        else:
                                            logger.warning(
                                                f"  -> [DRIFT GUARD] {trade.symbol} got -2022 but exchange position still exists (qty={pos_qty})! "
                                                "Not marking DB trade closed."
                                            )
                                            trade.notes = (trade.notes or '') + f' | DRIFT GUARD: got -2022 but exchange qty={pos_qty}'
                                            db_check.add(trade)
                                    except Exception as verify_err:
                                        logger.error(f"Failed to verify flat state for {trade.symbol} after -2022: {verify_err}")
                                    _exits_triggered += 1
                                else:
                                    logger.error(f"  -> FAILED to close {trade.symbol}: {res.message}")
                                    trade.notes = (trade.notes or "") + f" | AI EXIT FAILED: {res.message}"
                                    db_check.add(trade)
                    except Exception as e:
                        logger.error(f"  Position review error for {trade.symbol}: {e}")
                if _exits_triggered > 0:
                    db_check.commit()
                    logger.info(f"  Position Manager: {_exits_triggered} positions closed this cycle.")
                    # Refresh open count after exits
                    self._open_count = db_check.query(func.count(func.distinct(Trade.symbol))).filter(
                        Trade.status.in_(["open", "filled"]),
                        (Trade.broker != "ctrader") | (Trade.broker.is_(None))
                    ).scalar() or 0
                    self._total_open_count = db_check.query(func.count(func.distinct(Trade.symbol))).filter(
                        Trade.status.in_(["open", "filled"])
                    ).scalar() or 0
            except Exception as e:
                logger.error(f"Position Manager cycle error: {e}")
                db_check.rollback()
            finally:
                db_check.close()
        except Exception as e:
            logger.error(f"Position Manager init error: {e}")

        # ── MARGIN-AWARE GATEKEEPER ───────────────────────────────────────
        # Skip symbol evaluation if available balance is below minimum layer cost ($5)
        # to avoid API cost, LLM/Kronos latency, and log/DB spam.
        # Uses the cached _cycle_balance from above (no redundant API call).
        if self._cycle_available < 5.0:
            logger.warning(f"[MARGIN GATE] Insufficient available balance (${self._cycle_available:.2f} < $5.00). Skipping symbol evaluation.")

            # Write a single ACCOUNT-level signal to DB
            db_sig = SessionLocal()
            try:
                db_sig.add(TradingSignal(
                    symbol="ACCOUNT",
                    strategy=self._strategy_name,
                    direction="HOLD",
                    confidence=0.0,
                    status="evaluated",
                    reasoning=f"Insufficient margin: available={self._cycle_available:.4f} < 5.0"
                ))
                db_sig.commit()
            except Exception as _se:
                logger.warning(f"Failed to write margin gate signal: {_se}")
                db_sig.rollback()
            finally:
                db_sig.close()

        # Always process symbols, even when entries are margin-blocked. The
        # per-symbol path sees `_entries_blocked` and skips strategy/entry work,
        # but still runs exchange protection restore and trailing maintenance.
        # Previously a transient balance failure (available=0) skipped every
        # symbol task and suspended protection maintenance for the full cycle.
        # Merge open-position symbols into the scan set so a tighter universe
        # (e.g. recovery majors-only) never abandons an in-flight leg's SL/TP.
        manage_symbols = list(self._symbols)
        try:
            db_open = SessionLocal()
            try:
                open_syms = [
                    s for (s,) in db_open.query(Trade.symbol)
                    .filter(Trade.status.in_(["open", "filled"]))
                    .distinct()
                    .all()
                    if s
                ]
            finally:
                db_open.close()
            known = {s.upper() for s in manage_symbols}
            for s in open_syms:
                if s.upper() not in known:
                    manage_symbols.append(s)
                    known.add(s.upper())
        except Exception as e:
            logger.warning(f"  open-symbol merge failed ({e}); using configured universe only")

        # ── SYMBOL-QUALITY GATE: drop blacklisted + illiquid symbols ──
        tradeable = await self._filter_tradeable_symbols(manage_symbols)
        results = await self._run_symbol_batches(
            tradeable, min_confidence, ai_threshold, max_positions,
        )

        _signals_generated = 0
        _trades_executed = 0
        _errors = 0

        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Symbol task error: {res}")
                _errors += 1
            elif isinstance(res, dict):
                _signals_generated += res.get("signals", 0)
                _trades_executed += res.get("trades", 0)

        # Save portfolio snapshot and system health
        db = SessionLocal()
        try:
            self._save_portfolio_snapshot(db)
            
            _cycle_ms = (datetime.now(timezone.utc).timestamp() - _cycle_start) * 1000
            await influx.write_system_health(
                cycle=self._cycle_count,
                symbols_scanned=len(self._symbols),
                signals_generated=_signals_generated,
                trades_executed=_trades_executed,
                cycle_duration_ms=_cycle_ms,
                errors=_errors,
                state=self._state,
            )

            # Rolling performance metrics (win rate / drawdown / equity)
            try:
                await self._write_performance_metrics(db)
            except Exception as _pe:
                logger.warning(f"performance metrics write failed: {_pe}")
            
            # Reuse cached cycle balance — no redundant API call
            bal = getattr(self, '_cycle_balance', None) or {}
            await influx.write_portfolio_snapshot(
                cash=bal.get("available", 0.0),
                equity=bal.get("equity", 0.0),
                margin_used=bal.get("margin_used", 0.0),
                open_positions=self._open_count,
                cycle=self._cycle_count,
            )
            
            if os.getenv("ACTIVE_BROKER") == 'binance_futures':
                try:
                    await influx.write_binance_wallet(
                        balance=bal.get('balance', 0.0),
                        available=bal.get('available', 0.0),
                        equity=bal.get('equity', 0.0),
                        unrealized_pnl=bal.get('unrealized_pnl', 0.0),
                        margin_used=bal.get('margin_used', 0.0),
                    )
                    positions = self._cycle_positions
                    for pos in positions:
                        await influx.write_binance_position(
                            symbol=pos['symbol'],
                            side=pos['side'],
                            quantity=pos['quantity'],
                            entry_price=pos['entry_price'],
                            unrealized_pnl=pos['unrealized_pnl'],
                            leverage=pos.get('leverage', 10),
                            mark_price=pos.get('mark_price', 0.0),
                            liquidation_price=pos.get('liquidation_price', 0.0),
                        )
                except Exception as _bf_e:
                    logger.warning(f'Binance InfluxDB write error: {_bf_e}')
        finally:
            db.close()

    @staticmethod
    def _chunk_symbols(symbols: list[str], batch_size: int) -> list[list[str]]:
        """Split symbols into fixed-size batches for rate-limit-friendly scanning."""
        size = max(1, batch_size)
        return [symbols[i:i + size] for i in range(0, len(symbols), size)]

    async def _run_symbol_batches(
        self,
        symbols: list[str],
        min_confidence: float,
        ai_threshold: float,
        max_positions: int,
    ) -> list:
        """Process symbols sequentially with pauses to stay under Binance weight limits."""
        batch_size = max(1, int(os.getenv("SYMBOL_BATCH_SIZE", "1")))
        pause_sec = max(0.0, float(os.getenv("SYMBOL_BATCH_PAUSE_SEC", "45")))
        batches = self._chunk_symbols(symbols, batch_size)
        concurrency = max(1, int(os.getenv("SYMBOL_CONCURRENCY", "1")))
        logger.info(
            f"Symbol scan: {len(symbols)} symbols in {len(batches)} rotation(s) "
            f"(batch_size={batch_size}, pause={pause_sec}s, concurrency={concurrency})"
        )

        all_results: list = []
        for batch_idx, batch in enumerate(batches):
            if batch_idx > 0 and pause_sec > 0:
                logger.info(
                    f"Symbol rotation pause {pause_sec:.0f}s before "
                    f"{batch_idx + 1}/{len(batches)} ({', '.join(batch)})"
                )
                await asyncio.sleep(pause_sec)
            tasks = [
                self._process_symbol(symbol, min_confidence, ai_threshold, max_positions)
                for symbol in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(batch_results)
        return all_results

    async def _process_symbol(self, symbol: str, min_confidence: float, ai_threshold: float, max_positions: int):
        """Analyze a single symbol and execute a trade if conditions are met."""
        # Stagger: at most N symbols processed concurrently (set in __init__)
        async with self._symbol_semaphore:
            return await self._process_symbol_inner(
                symbol, min_confidence, ai_threshold, max_positions,
            )

    async def _process_symbol_inner(self, symbol: str, min_confidence: float, ai_threshold: float, max_positions: int):
        """Inner implementation — runs under the symbol semaphore."""
        import structlog
        structlog.contextvars.bind_contextvars(symbol=symbol)
        from backend.database.connection import SessionLocal
        from backend.database.models import Trade, TradingSignal
        from datetime import datetime, timezone
        import asyncio

        db = SessionLocal()
        _signals = 0
        _trades = 0
        reserved_open_slot = False
        open_slot_committed = False

        try:
            # 1. Fetch existing position (DB + live exchange — hedge mode can drift)
            # USDT/USDC perps are one book: BTCUSDC must see a live BTCUSDT long.
            alias_syms = perp_alias_set(symbol)
            existing = first_matching_open(
                db.query(Trade).filter(
                    Trade.symbol.in_(alias_syms),
                    Trade.status.in_(["open", "filled"]),
                ).all(),
                symbol,
            )

            # Use cycle-level cached positions instead of per-symbol API call.
            # get_positions() returns ALL symbols; filtering is cheap.
            exchange_legs = [
                p for p in self._cycle_positions
                if same_crypto_perp_leg(p.get("symbol"), symbol)
                and float(p.get("quantity") or 0) > 0
            ]

            if exchange_legs and not existing:
                sides = [p.get("side") for p in exchange_legs]
                logger.warning(
                    f"  [ {symbol} ] Exchange has open leg(s) {sides} but no DB trade — "
                    "blocking new entry until reconciled"
                )
                bars = await self._fetch_bars(symbol)
                if bars:
                    self._check_sl_tp(db, symbol, bars)
                    db.commit()
                return {"signals": 0, "trades": 0}

            # Block opposite-leg hedges: exchange already has a position on this symbol.
            if exchange_legs and existing and len(exchange_legs) >= 1:
                live_sides = {p.get("side") for p in exchange_legs}
                if len(live_sides) > 1:
                    logger.warning(
                        f"  [ {symbol} ] Hedge legs detected on exchange {sorted(live_sides)} — "
                        "skipping new entries"
                    )
                    bars = await self._fetch_bars(symbol)
                    if bars:
                        self._check_sl_tp(db, symbol, bars)
                        db.commit()
                    return {"signals": 0, "trades": 0}

            # 2. Fetch bars
            bars = await self._fetch_bars(symbol)
            if not bars or len(bars) < 50:
                return {"signals": 0, "trades": 0}

            if not self._is_market_open(symbol):
                return {"signals": 0, "trades": 0}

            # 3. Check cooldown
            cooldown_active = False
            if symbol in self._sl_cooldown:
                elapsed = (datetime.now(timezone.utc) - self._sl_cooldown[symbol]).total_seconds() / 60
                if elapsed < getattr(self.risk_config, "opinion_close_cooldown_min", 30):
                    cooldown_active = True
                else:
                    del self._sl_cooldown[symbol]

            # 3b. Margin gate — when the cycle pre-check found no affordable
            # margin, skip the whole entry pipeline (strategy + Kronos + LLM)
            # for this symbol. Exits and SL/TP management still run below.
            if getattr(self, "_entries_blocked", False) and not existing:
                self._check_sl_tp(db, symbol, bars)
                db.commit()
                return {"signals": 0, "trades": 0}

            # 3b2. Blacklisted symbols with an open leg: manage exits only —
            # no new entry and no pyramid add.
            if symbol.upper() in self.risk_config.symbol_blacklist:
                self._check_sl_tp(db, symbol, bars)
                db.commit()
                return {"signals": 0, "trades": 0}

            # 3c. New-bar gate — one evaluation per bar. Bar size follows
            # loop interval (5m/15m/1h). SL/TP still run each cycle.
            if not self._should_evaluate_bar(symbol, bars):
                self._check_sl_tp(db, symbol, bars)
                db.commit()
                return {"signals": 0, "trades": 0}

            # Fetch funding rate before evaluation to allow gating
            try:
                from backend.services.binance_market_data import get_binance_market_data
                bmd = get_binance_market_data()
                fr_data = await bmd.get_funding_rate(symbol)
                current_funding_rate = float(fr_data.get('fundingRate', 0)) if fr_data else 0.0
            except Exception:
                current_funding_rate = 0.0

            # 4. Evaluate using Decision Engine
            decision_engine = DecisionEngine(
                self.risk_config,
                regime_detector=self._regime_detector_for(symbol),
            )
            decision_engine.account_equity = getattr(self, "_cycle_equity", 0.0)
            decision_engine.account_available = getattr(self, "_cycle_available", 0.0)
            if get_trading_mode() == TradingMode.PAPER or self._crypto_session_id():
                decision_engine.account_leverage = paper_leverage_for_broker(
                    self._crypto_broker_name()
                )
            else:
                try:
                    decision_engine.account_leverage = float(
                        os.getenv("BINANCE_LEVERAGE", "10") or 10
                    )
                except (TypeError, ValueError):
                    decision_engine.account_leverage = 10.0
                if decision_engine.account_leverage <= 0:
                    decision_engine.account_leverage = 10.0
            try:
                decision = await decision_engine.evaluate_symbol(
                    symbol=symbol,
                    bars=bars,
                    existing_position=existing,
                    open_count=self._open_count,
                    pyramid_layers=self._pyramid_layers.get(symbol, []),
                    cooldown_active=cooldown_active,
                    current_funding_rate=current_funding_rate
                )
            except LLMChainExhausted:
                logger.error(
                    "  [ %s ] LLM chain exhausted; continuing cycle without LLM-dependent entry",
                    symbol,
                )
                decision = None

            ev = getattr(decision_engine, "last_evaluation", None) or {}
            signal_status = "evaluated"
            signal_reason = ev.get("reason", "")
            order_result = None
            # Keep decision metadata for signal persist even if the order fails.
            decision_snapshot = decision

            if decision:
                _signals = 1
                signal_reason = "pyramid entry" if decision.is_pyramid else "entry decision"
                # Write signal to InfluxDB for Grafana dashboards
                try:
                    await influx.write_signal(
                        symbol=symbol,
                        direction=decision.action,
                        confidence=getattr(decision, "confidence", 0.0),
                        entry_price=decision.entry_price,
                        stop_loss=decision.stop_loss,
                        take_profit=decision.take_profit,
                        strategy=self._strategy_name,
                        ai_used=True,
                    )
                except Exception as _ie:
                    logger.warning(f"  [ {symbol} ] InfluxDB write_signal failed: {_ie}")

                # Check directional exposure + correlation limits (if not pyramid)
                # Flatten/reverse of an existing BTC (etc.) leg is not a new slot.
                _new_notional = decision.quantity * decision.entry_price
                if is_flatten_action(decision.action):
                    signal_status = "approved"
                    signal_reason = "opposing flatten"
                elif not decision.is_pyramid:
                    all_open = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
                    scoped_open = trades_for_direction_cap(all_open, risk_broker_name())
                    _long_notional = sum(
                        t.quantity * (t.entry_price or 0.0)
                        for t in scoped_open if t.direction == "BUY"
                    )
                    _short_notional = sum(
                        t.quantity * (t.entry_price or 0.0)
                        for t in scoped_open if t.direction == "SELL"
                    )
                    # Correlation cap: distinct symbols already open in this direction
                    _same_dir_syms = {t.symbol for t in scoped_open if t.direction == decision.action}
                    _dir_cap = self.risk_config.max_same_direction_positions

                    if len(_same_dir_syms) >= _dir_cap:
                        logger.warning(
                            f"  [ {symbol} ] {decision.action} blocked: already "
                            f"{len(_same_dir_syms)} {decision.action} positions (cap {_dir_cap}) "
                            f"— correlated-exposure guard"
                        )
                        signal_status = "rejected"
                        signal_reason = (
                            f"{signal_reason} | same-direction cap "
                            f"({len(_same_dir_syms)}/{_dir_cap} {decision.action})"
                        )
                        decision = None
                    elif not getattr(self.risk_config, "equity_sizing_enabled", False) and decision.action == "BUY" and (_long_notional + _new_notional > self.risk_config.max_directional_exposure_usdt):
                        logger.warning(f"  [ {symbol} ] BUY blocked: LONG exposure cap reached")
                        signal_status = "rejected"
                        signal_reason = f"{signal_reason} | LONG exposure cap reached"
                        decision = None
                    elif decision.action == "BUY" and getattr(self, "_cycle_equity", 0.0) > 0 and (
                        _long_notional + _new_notional
                        > self._cycle_equity * self.risk_config.max_direction_notional_equity_mult
                    ):
                        _mult = self.risk_config.max_direction_notional_equity_mult
                        logger.warning(
                            f"  [ {symbol} ] BUY blocked: LONG notional "
                            f"${_long_notional + _new_notional:.0f} > {_mult}x equity "
                            f"(${self._cycle_equity * _mult:.0f}) — direction notional cap"
                        )
                        signal_status = "rejected"
                        signal_reason = f"{signal_reason} | LONG direction notional cap ({_mult}x equity)"
                        decision = None
                    elif not getattr(self.risk_config, "equity_sizing_enabled", False) and decision.action == "SELL" and (_short_notional + _new_notional > self.risk_config.max_directional_exposure_usdt):
                        logger.warning(f"  [ {symbol} ] SELL blocked: SHORT exposure cap reached")
                        signal_status = "rejected"
                        signal_reason = f"{signal_reason} | SHORT exposure cap reached"
                        decision = None
                    elif decision.action == "SELL" and getattr(self, "_cycle_equity", 0.0) > 0 and (
                        _short_notional + _new_notional
                        > self._cycle_equity * self.risk_config.max_direction_notional_equity_mult
                    ):
                        _mult = self.risk_config.max_direction_notional_equity_mult
                        logger.warning(
                            f"  [ {symbol} ] SELL blocked: SHORT notional "
                            f"${_short_notional + _new_notional:.0f} > {_mult}x equity "
                            f"(${self._cycle_equity * _mult:.0f}) — direction notional cap"
                        )
                        signal_status = "rejected"
                        signal_reason = f"{signal_reason} | SHORT direction notional cap ({_mult}x equity)"
                        decision = None
                    else:
                        signal_status = "approved"

            if decision:
                # 5. Execute Order — serialized across the parallel symbol tasks
                #    so concurrent BUYs can't both pass the max-positions check
                #    and overshoot the cap (race on shared self._open_count).
                async with self._execution_lock:
                    # Re-check the position cap inside the lock with the freshest count
                    max_binance_cap = getattr(self.risk_config, "max_binance_positions", 10)
                    max_total_cap = self.risk_config.max_positions
                    total_open = getattr(self, "_total_open_count", self._open_count)
                    if (
                        not existing
                        and not is_flatten_action(getattr(decision, "action", ""))
                        and (self._open_count >= max_binance_cap or total_open >= max_total_cap)
                    ):
                        cap_msg = f"binance cap ({max_binance_cap})" if self._open_count >= max_binance_cap else f"total cap ({max_total_cap})"
                        logger.warning(
                            f"  [ {symbol} ] entry skipped: max positions "
                            f"({cap_msg}) reached at execution time"
                        )
                        signal_status = "rejected"
                        signal_reason = (
                            f"{signal_reason} | max positions "
                            f"({cap_msg}) at execution time"
                        )
                        decision = None

                    flatten_now = is_flatten_action(getattr(decision, "action", ""))
                    if flatten_now and existing:
                        try:
                            db.refresh(existing)
                        except Exception:
                            pass
                        if getattr(existing, "status", "") not in ("open", "filled"):
                            logger.info(
                                f"  [ {symbol} ] flatten skipped: {existing.symbol} already {existing.status}"
                            )
                            signal_status = "evaluated"
                            signal_reason = "opposing flatten already closed this cycle"
                            decision = None
                            flatten_now = False
                    if getattr(self, "_entries_blocked", False) and not flatten_now:
                        logger.warning(
                            f"  [ {symbol} ] entry skipped: margin gate blocked this cycle"
                        )
                        signal_status = "rejected"
                        signal_reason = f"{signal_reason} | margin gate"
                        decision = None

                    if decision:
                        ut = UnifiedTrading()
                        flatten_now = is_flatten_action(decision.action)
                        close_symbol = (
                            getattr(existing, "symbol", None) or decision.symbol or symbol
                        )
                        if flatten_now:
                            order_side = (
                                OrderSide.SELL if decision.action == "CLOSE_LONG" else OrderSide.BUY
                            )
                            live_qty = sum(
                                float(p.get("quantity") or 0)
                                for p in exchange_legs
                                if float(p.get("quantity") or 0) > 0
                            )
                            flatten_qty = float(
                                live_qty
                                or getattr(existing, "quantity", None)
                                or decision.quantity
                                or 0
                            )
                            logger.info(
                                f"  [ {symbol} ] Flatten {decision.action} "
                                f"{close_symbol} qty={flatten_qty:.6f} (reduce-only)"
                            )
                            order = UnifiedOrder(
                                symbol=close_symbol,
                                side=order_side,
                                order_type=OrderType.MARKET,
                                quantity=flatten_qty,
                                price=float(decision.entry_price or 0),
                                reduce_only=True,
                            )
                        else:
                            order_side = OrderSide.BUY if decision.action == "BUY" else OrderSide.SELL
                            logger.info(
                                f"  [ {symbol} ] Attempting {decision.action} order: "
                                f"qty={decision.quantity:.6f} @ {decision.entry_price} | "
                                f"SL={decision.stop_loss} TP={decision.take_profit}"
                            )
                            order = UnifiedOrder(
                                symbol=symbol, side=order_side, quantity=decision.quantity,
                                price=decision.entry_price,
                                stop_loss=decision.stop_loss, take_profit=decision.take_profit,
                                is_pyramid=decision.is_pyramid,
                            )

                        order_result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: ut.place_order(order, session_id=self._binance_order_session_id())
                        )
                        if order_result.success and not existing and not flatten_now:
                            self._open_count += 1
                            self._total_open_count = getattr(self, "_total_open_count", 0) + 1
                            reserved_open_slot = True
                        if order_result.success and flatten_now:
                            self._open_count = max(0, self._open_count - 1)
                            self._total_open_count = max(
                                0, getattr(self, "_total_open_count", 1) - 1
                            )
                if decision and order_result and order_result.success:
                    _trades = 1
                    filled_px = float(order_result.filled_price or decision.entry_price)
                    filled_qty = float(order_result.filled_qty or decision.quantity)
                    signal_status = "executed"
                    signal_reason = f"{signal_reason} | filled {order_result.order_id}"
                    logger.info(f"  [ {symbol} ] SUCCESS: {order_result.order_id} filled @ {filled_px}")

                    if is_flatten_action(decision.action) and existing:
                        existing.exit_price = filled_px
                        if getattr(order_result, "realized_pnl", None) is not None:
                            existing.pnl = float(order_result.realized_pnl) - float(
                                getattr(order_result, "commission", 0) or 0
                            )
                        elif existing.entry_price:
                            if existing.direction == "BUY":
                                existing.pnl = (filled_px - existing.entry_price) * existing.quantity
                            else:
                                existing.pnl = (existing.entry_price - filled_px) * existing.quantity
                        existing.status = "closed"
                        existing.closed_at = datetime.now(timezone.utc)
                        existing.notes = (existing.notes or "") + (
                            f" | OPPOSING FLATTEN {decision.action} "
                            f"conf={decision.confidence:.2f}"
                        )
                        db.add(existing)
                        if self._pyramid_mode:
                            remove_closed_pyramid_layer(self._pyramid_layers, existing)
                        db.commit()
                    else:
                        if decision.is_pyramid:
                            self._pyramid_layers.setdefault(symbol, []).append(filled_px)

                        from backend.services.ledger import binance_position_key, fill_mode_from_order

                        fill_mode = fill_mode_from_order(
                            getattr(order_result, "mode", None),
                            get_trading_mode().value,
                        )
                        crypto_broker = self._crypto_broker_name()
                        trade = Trade(
                            symbol=symbol, direction=decision.action, quantity=filled_qty,
                            entry_price=filled_px, status="open",
                            strategy=self._strategy_name,
                            binance_order_id=order_result.order_id,
                            stop_loss=decision.stop_loss, take_profit=decision.take_profit,
                            notes=f"pyramid_layer_{len(self._pyramid_layers.get(symbol, []))}" if decision.is_pyramid else None,
                            broker=crypto_broker,
                            exchange=crypto_broker,
                            mode=fill_mode,
                            broker_position_id=(
                                binance_position_key(symbol, decision.action)
                                if fill_mode == "live" and crypto_broker == "binance_futures"
                                else None
                            ),
                        )
                        db.add(trade)
                        db.commit()
                        open_slot_committed = True

                        # Write to InfluxDB for Grafana dashboards
                        try:
                            await influx.write_trade(
                                symbol=symbol, direction=decision.action,
                                quantity=filled_qty, entry_price=filled_px,
                                status="open", strategy=self._strategy_name, pnl=0.0,
                            )
                        except Exception as _ie:
                            logger.warning(f"  [ {symbol} ] InfluxDB write_trade failed: {_ie}")
                elif decision and order_result:
                    signal_status = "skipped"
                    fail_msg = order_result.message or "unknown"
                    if "position_already_open" in fail_msg:
                        signal_reason = (
                            f"{signal_reason} | duplicate blocked: "
                            "position already open on Binance (no second entry)"
                        )
                    else:
                        signal_reason = f"{signal_reason} | order failed: {fail_msg}"
                    logger.warning(f"  [ {symbol} ] FAILED: {fail_msg}")

            # Persist evaluation AFTER order attempt so status reflects Binance reality.
            # Prefer the Decision object over last_evaluation — pyramid re-entries used
            # to leave last_evaluation stuck at "evaluating" (HOLD 0% in the UI).
            try:
                if ev or decision_snapshot:
                    dec = decision_snapshot
                    sig_direction = (
                        dec.action if dec else ev.get("direction", "HOLD")
                    )
                    sig_confidence = (
                        float(getattr(dec, "confidence", 0.0))
                        if dec else float(ev.get("confidence", 0.0))
                    )
                    db.add(TradingSignal(
                        symbol=symbol,
                        strategy=self._strategy_name,
                        direction=sig_direction,
                        confidence=sig_confidence,
                        entry_price=(
                            dec.entry_price if dec else ev.get("entry_price")
                        ),
                        stop_loss=(
                            dec.stop_loss if dec else ev.get("stop_loss")
                        ),
                        take_profit=(
                            dec.take_profit if dec else ev.get("take_profit")
                        ),
                        status=signal_status,
                        reasoning=signal_reason,
                    ))
                    db.commit()
                    try:
                        await influx.write_agent_state(
                            agent="decision_engine",
                            symbol=symbol,
                            direction=ev.get("direction", "HOLD"),
                            confidence=float(ev.get("confidence", 0.0)),
                            reasoning=signal_reason,
                        )
                    except Exception:
                        pass
            except Exception as _se:
                logger.warning(f"  [ {symbol} ] signal persist failed: {_se}")
                db.rollback()

            # SL/TP Check
            self._check_sl_tp(db, symbol, bars)
            db.commit()

            return {"signals": _signals, "trades": _trades}

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            db.rollback()
            if reserved_open_slot and not open_slot_committed:
                # Undo the in-cycle reservation if SQL persistence failed.
                self._open_count = max(0, self._open_count - 1)
                self._total_open_count = max(0, getattr(self, "_total_open_count", 1) - 1)
            return {"signals": 0, "trades": 0}
        finally:
            db.close()
            
    async def _write_performance_metrics(self, db):
        """Compute and emit rolling win-rate / drawdown / equity to InfluxDB."""
        await PerformanceMetricsWriter.write_metrics(
            db=db,
            cycle_equity=float(getattr(self, "_cycle_equity", 0.0) or 0.0),
            open_count=self._open_count,
            cycle_count=self._cycle_count,
        )

    def _save_portfolio_snapshot(self, db):
        """Save portfolio snapshot with real broker balance."""
        try:
            # Get open trades from DB
            open_trades = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
            
            # Get balance - paper book in paper mode, live broker in live mode.
            # Never fall back to $0 in paper (that previously poisoned snapshots
            # and the next cycle's kill switch).
            balance_info = self._get_effective_balance()
            real_cash = float(balance_info.get("available", balance_info.get("balance", 0.0)) or 0.0)
            real_equity = float(balance_info.get("equity", real_cash) or 0.0)
            if balance_info.get("broker") == "paper_trading":
                logger.info(f"  Paper portfolio balance: ${real_cash:,.2f}")
            
            # Compute realized P&L from all closed trades
            from sqlalchemy import func as _func
            snapshot_broker = risk_broker_name()
            risk_open = trades_for_direction_cap(open_trades, snapshot_broker)
            closed_q = db.query(_func.coalesce(_func.sum(Trade.pnl), 0.0)).filter(
                Trade.status == "closed"
            )
            if snapshot_broker == "binance_futures":
                closed_q = closed_q.filter(Trade.broker != "ctrader")
            elif str(snapshot_broker).startswith("ctrader"):
                closed_q = closed_q.filter(Trade.broker == "ctrader")
            realized_pnl = closed_q.scalar() or 0.0

            positions_val = sum(
                (t.quantity or 0) * (t.entry_price or 0)
                for t in risk_open
            )
            distinct_open_symbols = len({t.symbol for t in risk_open if t.symbol})

            # Save snapshot
            from backend.services.trading_mode import get_trading_mode
            current_mode = get_trading_mode().value if hasattr(get_trading_mode(), "value") else str(get_trading_mode())

            snapshot = PortfolioSnapshot(
                total_value=real_equity,
                cash=real_cash,
                positions_value=round(positions_val, 4),
                total_pnl=round(float(realized_pnl), 4),
                open_positions=distinct_open_symbols,
                cycle_number=self._cycle_count,
                broker=snapshot_broker,
                account_id=(
                    os.getenv("CTRADER_ACCOUNT_ID")
                    if snapshot_broker.startswith("ctrader")
                    else "binance_main"
                ),
                mode=current_mode,
            )
            db.add(snapshot)
            db.commit()
            
            logger.info(
                f"  Portfolio: cash=${real_cash:,.2f}, equity=${real_equity:,.2f}, "
                f"open_symbols={distinct_open_symbols}"
            )
        except Exception as e:
            logger.warning(f"Failed to save portfolio snapshot: {e}")
            db.rollback()

    def _apply_partial_tp(self, db, symbol: str, bars: list[dict]):
        """Close a portion of the position at partial_tp_atr_mult × ATR profit (paper)."""
        PartialTPManager.apply_partial_tp(
            db=db,
            symbol=symbol,
            bars=bars,
            risk_config=self.risk_config,
            strategy_name=self._strategy_name,
        )

    def _apply_partial_tp_live(self, db, symbol: str, bars: list[dict]):
        """Live: one exchange-sized reduce-only partial (safe with pyramid rows)."""
        PartialTPManager.apply_partial_tp_live(
            db=db,
            symbol=symbol,
            bars=bars,
            risk_config=self.risk_config,
            broker=binance_futures_broker,
        )

    def _apply_trailing_stop(self, db, symbol: str, bars: list[dict]):
        """Ratchet-only ATR trailing stop."""
        TrailingStopManager.apply_trailing_stop(
            db=db,
            symbol=symbol,
            bars=bars,
            high_water=self._high_water,
            risk_config=self.risk_config,
            broker=binance_futures_broker,
        )

    def _ensure_exchange_protection(self, db, symbol: str):
        """Re-place missing Binance SL/TP from DB levels (fail-safe, never cancels first)."""
        ExchangeProtectionManager.ensure_exchange_protection(
            db=db,
            symbol=symbol,
            broker=binance_futures_broker,
        )

    def _check_sl_tp(self, db, symbol: str, bars: list[dict]):
        """Check stop-loss and take-profit for open positions."""
        if not bars:
            return
        self._ensure_exchange_protection(db, symbol)
        # Ratchet the trailing stop up/down BEFORE evaluating the crossing,
        # so a profit-locked stop can fire in the same cycle.
        self._apply_trailing_stop(db, symbol, bars)
        # Live Binance: exchange SL/TP + _sync_positions_with_broker are authoritative
        # for full closes. Symbol-aggregated partial TP is safe (one reduce-only
        # sized from exchange qty); per-DB-row partials remain paper-only.
        if self._is_live_binance():
            self._apply_partial_tp_live(db, symbol, bars)
            return
        # Partial take-profit: close a portion at 1× ATR profit (paper / non-live)
        self._apply_partial_tp(db, symbol, bars)
        current_price = bars[-1]["close"]
        trades = (
            db.query(Trade)
            .filter(Trade.symbol == symbol, Trade.status.in_(["open", "filled"]))
            .all()
        )
        for trade in trades:
            if is_ctrader_trade(trade):
                continue

            hit = False
            if trade.direction == "BUY":
                if trade.stop_loss and current_price <= trade.stop_loss:
                    hit = True
                    trade.notes = (trade.notes or "") + " | SL hit"
                elif trade.take_profit and current_price >= trade.take_profit:
                    hit = True
                    trade.notes = (trade.notes or "") + " | TP hit"
            else:  # SHORT
                if trade.stop_loss and current_price >= trade.stop_loss:
                    hit = True
                    trade.notes = (trade.notes or "") + " | SL hit (short)"
                elif trade.take_profit and current_price <= trade.take_profit:
                    hit = True
                    trade.notes = (trade.notes or "") + " | TP hit (short)"

            if hit:
                # Send close order via UnifiedTrading BEFORE updating DB
                ut = UnifiedTrading()
                close_side = OrderSide.SELL if trade.direction == "BUY" else OrderSide.BUY
                res = ut.place_order(UnifiedOrder(
                    symbol=trade.symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=trade.quantity,
                    price=float(current_price or 0),
                    reduce_only=True
                ), session_id=self._binance_order_session_id())

                if res.success:
                    trade.exit_price = res.filled_price or current_price
                    if res.realized_pnl is not None:
                        pnl = float(res.realized_pnl) - res.commission
                    else:
                        if trade.direction == "BUY":
                            gross = (trade.exit_price - trade.entry_price) * trade.quantity
                        else:
                            gross = (trade.entry_price - trade.exit_price) * trade.quantity
                        pnl = gross - res.commission
                    trade.pnl = pnl
                    trade.status = "closed"
                    trade.closed_at = datetime.now(timezone.utc)
                    trade.notes = (trade.notes or "") + f" | Closed via SL/TP ({res.mode})"
                    if "SL hit" in (trade.notes or "") or (pnl is not None and pnl < 0):
                        try:
                            from backend.services.signal_candidate_engine import signal_candidate_engine
                            signal_candidate_engine.set_symbol_cooldown(symbol)
                        except Exception as cd_err:
                            logger.debug("Failed to set symbol cooldown on %s: %s", symbol, cd_err)
                    self._high_water.pop(trade.id, None)
                    if self._pyramid_mode:
                        remove_closed_pyramid_layer(self._pyramid_layers, trade)
                    logger.info(
                        f"  -> {symbol} position closed (SL/TP), PnL={pnl:+.2f}"
                    )
                else:
                    logger.error(f"  -> Failed to close {symbol} via SL/TP: {res.message}")
                    trade.notes = (trade.notes or "") + f" | SL/TP close FAILED: {res.message}"

    def _is_market_open(self, symbol: str) -> bool:
        """Crypto is 24/7; FX/metals follow Sun 21:00–Fri 21:00 UTC."""
        return is_venue_open(symbol)

    @staticmethod
    def _to_yfinance_symbol(symbol: str) -> str:
        """Convert Binance-native or forex symbols to yfinance format."""
        s = symbol.upper().strip()
        if s.endswith('USDT') or s.endswith('USDC') or s.endswith('BUSD'):
            return s[:-4] + '-USD'
        if len(s) == 6 and not s.endswith('=X'):
            return f"{s}=X"
        return s

    async def _filter_tradeable_symbols(self, symbols: list[str]) -> list[str]:
        """Apply the symbol-quality gate: hard blacklist + 24h liquidity floor.

        Low-liquidity / new-listing symbols produced the entire realized-PnL
        loss tail (wide spreads, slippage, forced liquidations). One batch
        ticker call per cycle (cached) decides what is tradeable.
        Fails OPEN: if the volume snapshot is unavailable we keep all symbols
        rather than halting trading.
        """
        blacklist = self.risk_config.symbol_blacklist
        min_vol = float(self.risk_config.min_24h_quote_volume_usdt or 0)

        # Open positions must keep flowing through management even if blacklisted
        # for new entries (never abandon an in-flight SL/TP/trail).
        open_symbols: set[str] = set()
        try:
            db = SessionLocal()
            try:
                open_symbols = {
                    s.upper()
                    for (s,) in db.query(Trade.symbol)
                    .filter(Trade.status.in_(["open", "filled"]))
                    .distinct()
                    .all()
                }
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"  [SYMBOL GATE] open-symbol lookup failed ({e}); blacklist applies fully")

        # 1) Hard blacklist for NEW entries; keep open legs for management.
        candidates = []
        blacklisted_skipped = []
        blacklisted_managed = []
        for s in symbols:
            su = s.upper()
            if su in blacklist and su not in open_symbols:
                blacklisted_skipped.append(s)
            elif su in blacklist and su in open_symbols:
                blacklisted_managed.append(s)
                candidates.append(s)
            else:
                candidates.append(s)
        if blacklisted_skipped:
            logger.warning(f"  [SYMBOL GATE] blacklisted (skipped): {blacklisted_skipped}")
        if blacklisted_managed:
            logger.warning(
                f"  [SYMBOL GATE] blacklisted but open — manage only: {blacklisted_managed}"
            )

        # 1a) Binance USDT-M/USDC-M exchangeInfo: drop unlisted/halted perps
        # before evaluation so we never send them to place_order.
        listed, unlisted = [], []
        for s in candidates:
            su = s.upper()
            status = binance_futures_broker.futures_listing_status(s)
            if status is False and su not in open_symbols:
                unlisted.append(s)
            else:
                listed.append(s)
        if unlisted:
            logger.warning(
                "  [SYMBOL GATE] not TRADING on Binance futures (skipped): %s",
                unlisted,
            )
        candidates = listed

        # 1b) Per-symbol expectancy gate: skip symbols that measurably bleed.
        candidates = self._apply_expectancy_gate(candidates)

        if min_vol <= 0 or not candidates:
            return candidates

        # 2) 24h quote-volume liquidity floor (single batch call, cached)
        try:
            tickers = await binance_market_data.get_all_tickers_24h(candidates)
            vol_by_sym = {
                t.get("symbol", "").upper(): float(t.get("quoteVolume", 0) or 0)
                for t in (tickers or [])
            }
        except Exception as e:
            logger.warning(
                f"  [SYMBOL GATE] volume snapshot failed ({e}); "
                f"failing OPEN — keeping {len(candidates)} symbols"
            )
            return candidates

        if not vol_by_sym:
            logger.warning("  [SYMBOL GATE] empty volume snapshot; failing OPEN")
            return candidates

        passed, rejected = [], []
        for s in candidates:
            su = s.upper()
            v = vol_by_sym.get(su)
            # Unknown to Binance (delisted / FX / metal): reject new entries,
            # but never drop an already-open leg from management.
            if v is None:
                if su in open_symbols:
                    passed.append(s)
                else:
                    rejected.append((s, "no-ticker"))
                continue
            if v >= min_vol:
                passed.append(s)
            elif su in open_symbols:
                passed.append(s)
            else:
                rejected.append((s, f"{v/1e6:.1f}M<{min_vol/1e6:.0f}M"))
        if rejected:
            logger.warning(
                f"  [SYMBOL GATE] illiquid (skipped): "
                f"{[f'{s}({why})' for s, why in rejected]}"
            )
        logger.info(
            f"  [SYMBOL GATE] {len(passed)}/{len(symbols)} symbols pass "
            f"(min 24h vol ${min_vol/1e6:.0f}M)"
        )
        return passed

    def _apply_expectancy_gate(self, candidates: list[str]) -> list[str]:
        """Skip NEW entries on symbols with proven negative expectancy.

        Aggregates realized P&L per symbol over a rolling lookback window and
        blocks symbols with >= min_trades closes and net-negative P&L. Symbols
        with open positions are never blocked (management must continue), and
        losses age out of the rolling window so blocked symbols get retried.
        Fails OPEN on any DB error.
        """
        if not getattr(self.risk_config, "symbol_expectancy_gate_enabled", False):
            return candidates
        if not candidates:
            return candidates
        try:
            from sqlalchemy import func
            lookback_days = int(self.risk_config.symbol_expectancy_lookback_days)
            min_trades = int(self.risk_config.symbol_expectancy_min_trades)
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            db = SessionLocal()
            try:
                rows = (
                    db.query(Trade.symbol, func.count(Trade.id), func.sum(Trade.pnl))
                    .filter(
                        Trade.status == "closed",
                        Trade.pnl.isnot(None),
                        Trade.closed_at >= cutoff,
                    )
                    .group_by(Trade.symbol)
                    .all()
                )
                open_symbols = {
                    s.upper() for (s,) in db.query(Trade.symbol)
                    .filter(Trade.status.in_(["open", "filled"])).distinct().all()
                }
            finally:
                db.close()

            blocked = negative_expectancy_symbols(rows, min_trades) - open_symbols
            if not blocked:
                return candidates
            kept = [s for s in candidates if s.upper() not in blocked]
            skipped = [s for s in candidates if s.upper() in blocked]
            if skipped:
                logger.info(
                    f"  [SYMBOL GATE] negative expectancy over {lookback_days}d "
                    f"(skipped): {skipped}"
                )
            return kept
        except Exception as e:
            logger.warning(f"  [SYMBOL GATE] expectancy gate error (failing open): {e}")
            return candidates

    async def _fetch_bars(self, symbol: str) -> list[dict]:
        """Fetch OHLCV bars. Order: 1. WS Cache (0 REST calls) -> 2. Binance REST -> 3. CCXT Futures (Bybit/OKX/KuCoin) -> 4. yfinance (equities)."""
        interval = self._entry_bar_interval()
        # 1. Check WebSocket in-memory candle ring buffer (zero network latency / zero REST requests).
        #    Reject stale rings: futures kline WS can connect yet deliver no closed
        #    candles (seen live), which freezes bars[-1] and permanently trips the
        #    new-bar gate after the first evaluation.
        try:
            ws_bars = binance_ws.get_candle_history(symbol, interval=interval, limit=1500)
            if ws_bars and len(ws_bars) >= 50:
                age = self._bar_open_age_seconds(ws_bars[-1])
                max_age = self._bar_interval_max_age_sec(interval)
                if age is None or age <= max_age:
                    logger.info(
                        f"  [{symbol}] WS candle cache hit: {len(ws_bars)} bars ({interval})"
                    )
                    return ws_bars
                logger.warning(
                    f"  [{symbol}] WS candle cache STALE "
                    f"(last bar age={age/3600:.1f}h > {max_age/3600:.1f}h) — falling back to REST"
                )
        except Exception as e:
            logger.debug(f"  [{symbol}] WS candle cache read exception: {e}")

        # 1.5 For cTrader/forex symbols, fetch cTrader trendbars
        if not symbol.upper().endswith(("USDT", "USDC", "BUSD")):
            try:
                from backend.services.ctrader_service import ctrader_service
                if ctrader_service.has_credentials():
                    ct_period = self._ctrader_period_for_interval(interval)
                    ct_bars = await asyncio.to_thread(
                        ctrader_service.get_trendbars, symbol, ct_period, count=100
                    )
                    if ct_bars and len(ct_bars) >= 10:
                        logger.info(f"  [{symbol}] cTrader trendbars hit: {len(ct_bars)} bars")
                        return ct_bars
            except Exception as _ct_e:
                logger.debug(f"  [{symbol}] cTrader trendbar fetch error: {_ct_e}")

        # 2. Try Binance Futures REST API (cached with TTL)
        try:
            bars = await binance_market_data.get_klines(
                symbol, interval=interval, limit=1500
            )
            if bars and len(bars) >= 50:
                logger.info(f"  [{symbol}] Binance klines: {len(bars)} bars ({interval})")
                # Keep the WS ring honest so a recovered stream can resume later.
                try:
                    binance_ws.seed_candles(symbol, interval, bars)
                except Exception:
                    pass
                return bars
        except Exception as e:
            logger.warning(f"  [{symbol}] Binance klines failed: {e}")

        # 3. Fallback to alternative crypto futures venues (Bybit / OKX / KuCoin)
        try:
            mp_bars = await multi_provider_data.get_ohlcv(symbol, interval=interval, limit=500)
            if mp_bars and len(mp_bars) >= 50:
                logger.info(f"  [{symbol}] Multi-provider CCXT fallback: {len(mp_bars)} bars")
                return mp_bars
        except Exception as e:
            logger.warning(f"  [{symbol}] Multi-provider fallback failed: {e}")

        # 4. Fallback to yfinance (equities ONLY — skipped for crypto if symbol ends in USDT/USDC)
        if not symbol.upper().endswith(("USDT", "USDC")):
            return await asyncio.to_thread(self._fetch_bars_yfinance, symbol)

        logger.error(f"  [{symbol}] All OHLCV bar fetch sources exhausted")
        return []



    def _fetch_bars_yfinance(self, symbol: str) -> list[dict]:
        """Fallback: Fetch OHLCV bars via yfinance."""
        yf_symbol = self._to_yfinance_symbol(symbol)
        logger.info(f"  [{symbol}] fetching yfinance data as '{yf_symbol}'")
        ticker = yf.Ticker(yf_symbol)
        interval = self._entry_bar_interval()
        period = "5d" if interval == "5m" else "1mo" if interval == "15m" else "3mo"
        hist = ticker.history(period=period, interval=interval)
        bars = []
        for i, row in hist.iterrows():
            bars.append({
                "date": i.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        return bars

    async def _fetch_binance_extra(self, symbol: str) -> dict:
        """Fetch additional Binance market data: funding rate, OI, 24h ticker."""
        result = {}
        try:
            funding, oi, ticker = await asyncio.gather(
                binance_market_data.get_funding_rate(symbol),
                binance_market_data.get_open_interest(symbol),
                binance_market_data.get_ticker_24h(symbol),
                return_exceptions=True,
            )
            if isinstance(funding, dict):
                result['funding_rate'] = funding
            if isinstance(oi, dict):
                result['open_interest'] = oi
            if isinstance(ticker, dict):
                result['ticker_24h'] = ticker
        except Exception as e:
            logger.warning(f"  [{symbol}] Binance extra data error: {e}")
        return result

    def _run_strategy(
        self,
        symbol: str,
        bars: list[dict],
        regime: str = "UNKNOWN",
        regime_weights: dict | None = None,
    ):
        """Run the strategy on the given bars with regime-aware weights.

        Args:
            symbol:         Trading symbol
            bars:           OHLCV bar list
            regime:         Market regime string from MarketRegimeDetector
            regime_weights: Weight dict from MarketRegimeDetector
        """
        from backend.strategies.combined import CombinedStrategy
        strategy = CombinedStrategy()
        return strategy.generate_signal(
            symbol, bars,
            regime=regime,
            regime_weights=regime_weights,
        )

    def _get_current_price(self, bars: list[dict]) -> Optional[float]:
        """Get current price from bars."""
        if bars:
            return bars[-1]["close"]
        return None


# Singleton instance
trading_loop = TradingLoopService()
