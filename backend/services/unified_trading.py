"""
Unified Trading Router + Paper Trading Engine
Extracted and ported from FinceptTerminal C++ (UnifiedTrading.cpp + PaperTrading.cpp)

This module provides:
  1. UnifiedTrading   — Singleton router that switches between paper/live modes
  2. PaperTradingEngine — Full simulated exchange with fill engine, fees, margin
  3. BrokerProtocol   — Abstract interface for live brokers (drop-in for existing ones)

Usage:
    from backend.services.unified_trading import UnifiedTrading, UnifiedOrder, OrderSide, OrderType

    ut = UnifiedTrading()
    ut.register_broker("binance_futures", binance_futures_broker)
    ut.init_session("binance_futures", mode="paper", paper_balance=100_000.0)

    resp = ut.place_order(UnifiedOrder(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.01
    ))
    print(resp.success, resp.order_id, resp.message)
"""
from __future__ import annotations
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from backend.brokers import IBroker
from backend.services.portfolio import create_portfolio
from backend.services.trading_mode import paper_leverage_for_broker, paper_reported_equity

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DB Persistence helpers (lazy import to avoid circular deps)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_db():
    from backend.database.connection import SessionLocal
    return SessionLocal()


def _db_now():
    return datetime.now(timezone.utc)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared Types
# ═══════════════════════════════════════════════════════════════════════════════

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ProductType(Enum):
    INTRADAY = "intraday"
    DELIVERY = "delivery"
    MARGIN = "margin"
    COVER_ORDER = "cover_order"
    BRACKET_ORDER = "bracket_order"


@dataclass
class UnifiedOrder:
    symbol: str
    exchange: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    product_type: ProductType = ProductType.INTRADAY
    validity: str = "DAY"
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reduce_only: bool = False
    is_pyramid: bool = False


@dataclass
class UnifiedOrderResponse:
    success: bool
    order_id: str
    message: str
    mode: str = ""           # "paper" or "live"
    filled_price: Optional[float] = None
    filled_qty: Optional[float] = None
    commission: float = 0.0
    realized_pnl: Optional[float] = None


@dataclass
class BrokerPosition:
    symbol: str
    side: str          # "long" | "short"
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class PaperTrade:
    id: str
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    fee: float
    pnl: float
    timestamp: str


@dataclass
class TradingSession:
    broker: str
    mode: str = "paper"   # "live" or "paper"
    paper_portfolio_id: Optional[str] = None
    is_connected: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Trading Engine
# ═══════════════════════════════════════════════════════════════════════════════

class PaperTradingEngine:
    """
    Simulated exchange with full order lifecycle, fill engine, fee model,
    margin checks, and P&L tracking.

    Ported from Fincept PaperTrading.cpp (lines 42-364).
    """

    def __init__(self):
        self._portfolios: Dict[str, dict] = {}
        self._orders: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._order_counter = 0

    # ── Portfolio ───────────────────────────────────────────────────────────

    def create_portfolio(self, name: str, balance: float, currency: str = "USD",
                         leverage: float = 1.0, margin_mode: str = "cross",
                         fee_rate: float = 0.001, exchange: str = "") -> str:
        if not name:
            raise ValueError("Portfolio name cannot be empty")
        if balance <= 0:
            raise ValueError("Balance must be positive")
        if not (0 <= fee_rate <= 1):
            raise ValueError("fee_rate must be between 0 and 1")

        pid = str(uuid.uuid4())[:12]
        pf = create_portfolio(
            initial_cash=balance,
            margin_requirement=1.0 / leverage,
            tickers=[],
            portfolio_positions=None
        )
        pf["_meta"] = {
            "id": pid, "name": name, "currency": currency,
            "leverage": leverage, "margin_mode": margin_mode,
            "fee_rate": fee_rate, "exchange": exchange,
            "initial_balance": balance, "created_at": _db_now(),
        }
        pf["orders"] = []
        pf["trades"] = []
        with self._lock:
            self._portfolios[pid] = pf
        # ── Persist to Postgres ──
        db = None
        try:
            db = _get_db()
            from backend.database.models import PaperPortfolio as PP
            db.add(PP(
                portfolio_id=pid, name=name, broker=exchange or "paper",
                currency=currency, leverage=leverage, margin_mode=margin_mode,
                fee_rate=fee_rate, initial_balance=balance, cash=balance,
            ))
            db.commit()
        except Exception as e:
            if db is not None:
                db.rollback()
            logger.warning(f"Paper portfolio DB persist failed: {e}")
        finally:
            if db is not None:
                db.close()
        logger.info(f"Paper portfolio created: {name} (id={pid}, balance={balance}, leverage={leverage}x)")
        return pid

    def find_or_create_portfolio(self, name: str, balance: float, currency: str = "USD",
                                 leverage: float = 1.0, margin_mode: str = "cross",
                                 fee_rate: float = 0.001, exchange: str = "") -> str:
        """Reuse an in-memory or persisted paper book so restarts do not reset PnL."""
        with self._lock:
            for pid, pf in self._portfolios.items():
                meta = pf.get("_meta") or {}
                if meta.get("name") == name:
                    self._sync_portfolio_leverage(pid, pf, leverage)
                    return pid
        db = None
        try:
            db = _get_db()
            from backend.database.models import PaperPortfolio as PP
            row = (
                db.query(PP)
                .filter(PP.name == name)
                .order_by(PP.id.desc())
                .first()
            )
            if row:
                pid = row.portfolio_id
                with self._lock:
                    if pid not in self._portfolios:
                        pf = create_portfolio(
                            initial_cash=float(row.cash or balance),
                            margin_requirement=1.0 / max(float(row.leverage or leverage), 1e-9),
                            tickers=[],
                            portfolio_positions=None,
                        )
                        pf["_meta"] = {
                            "id": pid, "name": row.name, "currency": row.currency or currency,
                            "leverage": float(row.leverage or leverage),
                            "margin_mode": row.margin_mode or margin_mode,
                            "fee_rate": float(row.fee_rate or fee_rate),
                            "exchange": row.broker or exchange,
                            "initial_balance": float(row.initial_balance or balance),
                            "created_at": _db_now(),
                        }
                        pf["orders"] = []
                        pf["trades"] = []
                        pf["cash"] = float(row.cash or balance)
                        pf["margin_used"] = float(row.margin_used or 0.0)
                        self._portfolios[pid] = pf
                    self._sync_portfolio_leverage(pid, self._portfolios[pid], leverage)
                return pid
        except Exception as e:
            logger.warning(f"Paper portfolio lookup failed: {e}")
        finally:
            if db is not None:
                db.close()
        return self.create_portfolio(
            name=name, balance=balance, currency=currency, leverage=leverage,
            margin_mode=margin_mode, fee_rate=fee_rate, exchange=exchange,
        )

    def _recompute_margin_used(self, pf: dict) -> float:
        meta = pf.get("_meta") or {}
        leverage = max(float(meta.get("leverage") or 1.0), 1e-9)
        used = 0.0
        for pos in pf.get("positions", {}).values():
            used += (float(pos.get("long") or 0) * float(pos.get("long_cost_basis") or 0)) / leverage
            used += (float(pos.get("short") or 0) * float(pos.get("short_cost_basis") or 0)) / leverage
        pf["margin_used"] = used
        return used

    def _sync_portfolio_leverage(self, pid: str, pf: dict, leverage: float) -> None:
        """Keep a reused paper book on the session leverage (e.g. Binance 10x)."""
        lev = max(float(leverage), 1e-9)
        meta = pf.setdefault("_meta", {})
        current = float(meta.get("leverage") or 1.0)
        if abs(current - lev) < 1e-9:
            return
        meta["leverage"] = lev
        self._recompute_margin_used(pf)
        db = None
        try:
            db = _get_db()
            from backend.database.models import PaperPortfolio as PP
            row = db.query(PP).filter(PP.portfolio_id == pid).first()
            if row:
                row.leverage = lev
                row.margin_used = float(pf.get("margin_used") or 0.0)
                db.commit()
        except Exception as e:
            if db is not None:
                db.rollback()
            logger.warning(f"Paper leverage persist failed: {e}")
        finally:
            if db is not None:
                db.close()
        logger.info("Paper portfolio %s leverage %.4gx -> %.4gx", pid, current, lev)

    def get_portfolio(self, portfolio_id: str) -> dict:
        with self._lock:
            if portfolio_id not in self._portfolios:
                raise ValueError(f"Portfolio not found: {portfolio_id}")
            return self._clone_pf(self._portfolios[portfolio_id])

    def list_portfolios(self) -> List[dict]:
        with self._lock:
            return [self._clone_pf(p) for p in self._portfolios.values()]

    def reset_portfolio(self, portfolio_id: str) -> dict:
        with self._lock:
            pf = self._portfolios.get(portfolio_id)
            if not pf:
                raise ValueError(f"Portfolio not found: {portfolio_id}")
            meta = pf["_meta"]
            pf["cash"] = meta["initial_balance"]
            pf["margin_used"] = 0.0
            pf["positions"] = {}
            pf["realized_gains"] = {}
            pf["orders"] = []
            pf["trades"] = []
        return self.get_portfolio(portfolio_id)

    def delete_portfolio(self, portfolio_id: str):
        with self._lock:
            self._portfolios.pop(portfolio_id, None)

    # ── Orders ──────────────────────────────────────────────────────────────

    def place_order(self, portfolio_id: str, order: UnifiedOrder) -> UnifiedOrderResponse:
        with self._lock:
            pf = self._portfolios.get(portfolio_id)
            if not pf:
                return UnifiedOrderResponse(False, "", "Paper portfolio not found", "paper")

            meta = pf["_meta"]
            self._order_counter += 1
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            oid = f"paper_{ts_ms}_{self._order_counter:04d}_{uuid.uuid4().hex[:4]}"

            # Validation
            if order.order_type == OrderType.LIMIT and order.price <= 0:
                return UnifiedOrderResponse(False, oid, "Limit order requires price > 0", "paper")
            if order.quantity <= 0:
                return UnifiedOrderResponse(False, oid, "Quantity must be positive", "paper")

            # Margin check (skip for reduce_only)
            if not order.reduce_only:
                _ensure_ticker(pf, order.symbol)
                pos = pf["positions"].get(order.symbol, {})
                opposite_side = "short" if order.side == OrderSide.BUY else "long"
                opposite_qty = pos.get(opposite_side, 0)
                net_new = max(0.0, order.quantity - opposite_qty)

                if net_new > 0:
                    ref = order.price if order.price > 0 else order.stop_price
                    if ref <= 0 and order.order_type == OrderType.MARKET:
                        # Try to use last filled price as reference if available
                        last_fills = [t.price for t in pf["trades"] if t.symbol == order.symbol]
                        if not last_fills:
                            return UnifiedOrderResponse(
                                False, oid,
                                "Paper market entry requires an explicit price or prior market fill",
                                "paper",
                            )
                        ref = last_fills[-1]
                    leverage = max(float(meta.get("leverage") or 1.0), 1e-9)
                    required = net_new * ref / leverage
                    available = float(pf["cash"]) - float(pf.get("margin_used") or 0.0)
                    if required > available > 0 and ref > 0:
                        max_qty = opposite_qty + (available * leverage / ref) * 0.99
                        if max_qty > opposite_qty:
                            logger.warning(
                                "Paper order clipped %s qty %.6f -> %.6f "
                                "(need %.2f have %.2f @ %.4gx)",
                                order.symbol, order.quantity, max_qty,
                                required, available, leverage,
                            )
                            order.quantity = max_qty
                            net_new = max(0.0, order.quantity - opposite_qty)
                            required = net_new * ref / leverage
                    if required > available:
                        return UnifiedOrderResponse(False, oid,
                            f"Insufficient margin: need {required:.2f}, have {available:.2f}", "paper")

            rec = {
                "id": oid, "portfolio_id": portfolio_id,
                "symbol": order.symbol, "side": order.side.value,
                "order_type": order.order_type.value, "quantity": order.quantity,
                "price": order.price, "stop_price": order.stop_price,
                "filled_qty": 0.0, "avg_price": 0.0, "status": "pending",
                "reduce_only": order.reduce_only, "created_at": _now(),
                "fee_rate": meta["fee_rate"],
            }
            pf["orders"].append(rec)
            self._orders[oid] = rec

            # Persist order to DB
            db = None
            try:
                db = _get_db()
                from backend.database.models import PaperOrder as PO
                db.add(PO(
                    order_id=oid, portfolio_id=portfolio_id,
                    symbol=order.symbol, side=order.side.value,
                    order_type=order.order_type.value, quantity=order.quantity,
                    price=order.price, stop_price=order.stop_price,
                    status="pending", reduce_only=order.reduce_only,
                ))
                db.commit()
            except Exception as e:
                if db is not None:
                    db.rollback()
                logger.warning(f"Paper order DB persist failed: {e}")
            finally:
                if db is not None:
                    db.close()

            # Auto-fill market orders
            if order.order_type == OrderType.MARKET:
                fill_price = order.price
                if fill_price <= 0:
                    try:
                        from backend.services.binance_futures_service import binance_futures_broker
                        ep = binance_futures_broker.get_exit_price(order.symbol)
                        if ep and float(ep) > 0:
                            fill_price = float(ep)
                    except Exception:
                        pass
                if fill_price <= 0:
                    # Never invent a fill at cost basis — that zeroes paper PnL.
                    rec["status"] = "rejected"
                    return UnifiedOrderResponse(
                        False, oid, "Paper market order requires an explicit fill price",
                        "paper",
                    )
                self._fill_order_locked(pf, rec, fill_price)
                return UnifiedOrderResponse(
                    True, oid, f"Paper market order filled @ {fill_price}", "paper",
                    filled_price=fill_price, filled_qty=order.quantity
                )

            return UnifiedOrderResponse(True, oid, "Paper order placed (pending)", "paper")

    def cancel_order(self, order_id: str) -> UnifiedOrderResponse:
        with self._lock:
            rec = self._orders.get(order_id)
            if not rec:
                return UnifiedOrderResponse(False, order_id, "Order not found", "paper")
            if rec["status"] in ("filled", "cancelled"):
                return UnifiedOrderResponse(False, order_id, f"Order already {rec['status']}", "paper")
            rec["status"] = "cancelled"
        # Persist cancel
        db = None
        try:
            db = _get_db()
            from backend.database.models import PaperOrder as PO
            db_order = db.query(PO).filter(PO.order_id == order_id).first()
            if db_order:
                db_order.status = "cancelled"
                db.commit()
        except Exception as e:
            if db is not None:
                db.rollback()
            logger.warning(f"Paper cancel DB persist failed: {e}")
        finally:
            if db is not None:
                db.close()
        return UnifiedOrderResponse(True, order_id, "Order cancelled", "paper")

    def get_orders(self, portfolio_id: str, status: str = "") -> List[dict]:
        with self._lock:
            pf = self._portfolios.get(portfolio_id)
            if not pf:
                return []
            orders = pf.get("orders", [])
            if status:
                return [o.copy() for o in orders if o["status"] == status]
            return [o.copy() for o in orders]

    # ── Fill Engine (CORE) ──────────────────────────────────────────────────

    def _fill_order_locked(self, pf: dict, order: dict, fill_price: float,
                           fill_qty: Optional[float] = None) -> PaperTrade:
        """
        Core fill engine ported from Fincept PaperTrading.cpp lines 188-333.
        MUST be called with self._lock held.
        """
        if fill_price <= 0:
            raise ValueError("Invalid fill price")

        qty = fill_qty or (order["quantity"] - order["filled_qty"])
        if qty <= 0:
            raise ValueError("Nothing left to fill")

        meta = pf["_meta"]
        fee_rate = meta["fee_rate"]
        fee = qty * fill_price * fee_rate
        now = _now()

        sym = order["symbol"]
        _ensure_ticker(pf, sym)
        pos = pf["positions"][sym]

        position_side = "long" if order["side"] == "buy" else "short"
        opposite_side = "short" if order["side"] == "buy" else "long"
        pnl = 0.0
        had_opposite = False

        # 1. Close opposite position
        if pos.get(opposite_side, 0) > 0:
            had_opposite = True
            close_qty = min(qty, pos[opposite_side])
            if opposite_side == "long":
                pnl = (fill_price - pos["long_cost_basis"]) * close_qty
            else:
                pnl = (pos["short_cost_basis"] - fill_price) * close_qty

            remaining = pos[opposite_side] - close_qty
            if remaining <= 0:
                pos[opposite_side] = 0
                pos[f"{opposite_side}_cost_basis"] = 0.0
                pos[f"{opposite_side}_margin_used"] = 0.0
            else:
                pos[opposite_side] = remaining
                # keep same cost basis for remainder

            # Flip remaining qty to new position side
            if qty > close_qty:
                new_qty = qty - close_qty
                old_qty = pos.get(position_side, 0)
                total = old_qty + new_qty
                if total > 0:
                    old_cost = pos.get(f"{position_side}_cost_basis", 0.0) * old_qty
                    pos[f"{position_side}_cost_basis"] = (old_cost + fill_price * new_qty) / total
                pos[position_side] = total

        if not had_opposite:
            # 2. Same-side averaging or new position
            old_qty = pos.get(position_side, 0)
            total = old_qty + qty
            if total > 0:
                old_cost = pos.get(f"{position_side}_cost_basis", 0.0) * old_qty
                pos[f"{position_side}_cost_basis"] = (old_cost + fill_price * qty) / total
            pos[position_side] = total

        # 3. Update balance (deduct fee only on closing fills) and margin
        balance_change = pnl - (fee if had_opposite else 0)
        pf["cash"] += balance_change
        self._recompute_margin_used(pf)

        # 4. Update order
        new_filled = order["filled_qty"] + qty
        order["filled_qty"] = new_filled
        prev_avg = order["avg_price"]
        order["avg_price"] = (prev_avg * (new_filled - qty) + fill_price * qty) / new_filled if new_filled > 0 else fill_price
        order["status"] = "filled" if new_filled >= order["quantity"] else "partial"
        if order["status"] == "filled":
            order["filled_at"] = now

        # 5. Record trade
        trade = PaperTrade(
            id=f"trade_{uuid.uuid4().hex[:8]}",
            order_id=order["id"],
            symbol=sym,
            side=order["side"],
            price=fill_price,
            quantity=qty,
            fee=fee,
            pnl=pnl,
            timestamp=now,
        )
        pf["trades"].append(trade)

        # ── Persist fill to DB ──
        db = None
        try:
            db = _get_db()
            from backend.database.models import PaperTrade as PT, PaperOrder as PO, PaperPortfolio as PP
            db.add(PT(
                trade_id=trade.id, order_id=trade.order_id, portfolio_id=order["portfolio_id"],
                symbol=sym, side=order["side"], price=fill_price,
                quantity=qty, fee=fee, pnl=pnl,
            ))
            # Update order status
            db_order = db.query(PO).filter(PO.order_id == order["id"]).first()
            if db_order:
                db_order.status = order["status"]
                db_order.filled_qty = order["filled_qty"]
                db_order.avg_price = order["avg_price"]
                if order["status"] == "filled":
                    db_order.filled_at = _db_now()
            # Update portfolio cash
            db_pf = db.query(PP).filter(PP.portfolio_id == order["portfolio_id"]).first()
            if db_pf:
                db_pf.cash = pf["cash"]
                db_pf.margin_used = pf.get("margin_used", 0.0)
            db.commit()
        except Exception as e:
            if db is not None:
                db.rollback()
            logger.warning(f"Paper fill DB persist failed: {e}")
        finally:
            if db is not None:
                db.close()

        logger.info(
            f"Paper fill: {sym} {order['side']} {qty} @ {fill_price:.4f} "
            f"pnl={pnl:.4f} fee={fee:.4f} cash={pf['cash']:.2f}"
        )
        return trade

    def fill_order(self, order_id: str, fill_price: float, fill_qty: Optional[float] = None) -> PaperTrade:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order not found: {order_id}")
            pf = self._portfolios.get(order["portfolio_id"])
            if not pf:
                raise ValueError(f"Portfolio not found: {order['portfolio_id']}")
            return self._fill_order_locked(pf, order, fill_price, fill_qty)

    # ── Positions ───────────────────────────────────────────────────────────

    def get_positions(self, portfolio_id: str) -> List[BrokerPosition]:
        with self._lock:
            pf = self._portfolios.get(portfolio_id)
            if not pf:
                return []
            result = []
            for sym, pos in pf.get("positions", {}).items():
                if pos.get("long", 0) > 0:
                    result.append(BrokerPosition(
                        symbol=sym, side="long", quantity=pos["long"],
                        avg_price=pos.get("long_cost_basis", 0.0)
                    ))
                if pos.get("short", 0) > 0:
                    result.append(BrokerPosition(
                        symbol=sym, side="short", quantity=pos["short"],
                        avg_price=pos.get("short_cost_basis", 0.0)
                    ))
            return result

    def get_stats(self, portfolio_id: str) -> dict:
        with self._lock:
            pf = self._portfolios.get(portfolio_id)
            if not pf:
                return {}
            trades = pf.get("trades", [])
            if not trades:
                return {"total_trades": 0, "total_pnl": 0.0, "win_rate": 0.0}

            winning = sum(1 for t in trades if t.pnl > 0)
            total_pnl = sum(t.pnl for t in trades)
            return {
                "total_trades": len(trades),
                "winning_trades": winning,
                "losing_trades": len(trades) - winning,
                "win_rate": winning / len(trades),
                "total_pnl": total_pnl,
                "total_fees": sum(t.fee for t in trades),
                "cash": pf["cash"],
                "initial_balance": pf["_meta"]["initial_balance"],
            }

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clone_pf(pf: dict) -> dict:
        """Shallow clone for external reads."""
        cash = float(pf.get("cash", 0.0) or 0.0)
        margin_used = float(pf.get("margin_used", 0.0) or 0.0)
        # Margin is reserved buying power. Adding it to cash inflated paper
        # NAV by open notional (or notional/leverage) and tripped drawdown.
        equity = paper_reported_equity(cash, margin_used)
        return {
            "cash": cash,
            "equity": equity,
            "margin_used": margin_used,
            "positions": {k: dict(v) for k, v in pf.get("positions", {}).items()},
            "meta": dict(pf["_meta"]),
            "order_count": len(pf.get("orders", [])),
            "trade_count": len(pf.get("trades", [])),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Trading Router
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedTrading:
    """
    Singleton order router. THE ONLY entry point for placing orders.
    Supports multiple concurrent sessions (paper + live simultaneously).
    Ports Fincept UnifiedTrading.cpp (lines 14-305) + MultiAccount (C).
    """

    _instance: Optional[UnifiedTrading] = None
    _lock = threading.Lock()

    def __new__(cls) -> UnifiedTrading:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._sessions: Dict[str, TradingSession] = {}
                    obj._default_session_id: Optional[str] = None
                    obj._paper = PaperTradingEngine()
                    obj._brokers: Dict[str, IBroker] = {}
                    obj._session_lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    def register_broker(self, name: str, broker: IBroker):
        """Register a live broker adapter."""
        self._brokers[name] = broker
        logger.info(f"Registered broker: {name}")

    def init_session(self, broker: str, mode: str = "paper",
                     paper_balance: float = 100_000.0,
                     currency: str = "USD", leverage: float = 1.0,
                     session_id: Optional[str] = None) -> TradingSession:
        """Initialize a trading session (paper or live). Returns session."""
        sid = session_id or f"{broker}_{mode}"
        with self._session_lock:
            existing = self._sessions.get(sid)
            if existing:
                return existing
            session = TradingSession(broker=broker, mode=mode)
            if mode == "paper":
                session_leverage = leverage
                if abs(float(leverage) - 1.0) < 1e-9:
                    session_leverage = paper_leverage_for_broker(broker)
                pid = self._paper.find_or_create_portfolio(
                    name=f"{broker} Paper Trading",
                    balance=paper_balance,
                    currency=currency,
                    leverage=session_leverage,
                    exchange=broker,
                )
                session.paper_portfolio_id = pid
            self._sessions[sid] = session
            if self._default_session_id is None:
                self._default_session_id = sid
            logger.info(f"Session init: {sid} -> {broker} / {mode}")
            return session

    def get_session(self, session_id: Optional[str] = None) -> Optional[TradingSession]:
        with self._session_lock:
            if session_id:
                return self._sessions.get(session_id)
            return self._sessions.get(self._default_session_id)

    def set_default_session(self, session_id: str):
        with self._session_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session not found: {session_id}")
            self._default_session_id = session_id

    def list_sessions(self) -> List[dict]:
        with self._session_lock:
            return [
                {"id": sid, "broker": s.broker, "mode": s.mode,
                 "paper_portfolio_id": s.paper_portfolio_id}
                for sid, s in self._sessions.items()
            ]

    def switch_mode(self, mode: str, session_id: Optional[str] = None) -> TradingSession:
        """Switch session between paper and live."""
        with self._session_lock:
            sid = session_id or self._default_session_id
            if not sid or sid not in self._sessions:
                raise RuntimeError("No session. Call init_session first.")
            sess = self._sessions[sid]
            sess.mode = mode
            if mode == "paper" and not sess.paper_portfolio_id:
                pid = self._paper.find_or_create_portfolio(
                    name=f"{sess.broker} Paper Trading",
                    balance=100_000.0,
                    exchange=sess.broker,
                )
                sess.paper_portfolio_id = pid
            return sess

    # ── Order Routing ───────────────────────────────────────────────────────

    def place_order(self, order: UnifiedOrder,
                    session_id: Optional[str] = None) -> UnifiedOrderResponse:
        with self._session_lock:
            sid = session_id or self._default_session_id
            session = self._sessions.get(sid)
        if not session:
            return UnifiedOrderResponse(False, "", "No active session. Call init_session first.", "")

        # Safeguard 1: Kill switch / sentry halt check
        from backend.services.sentry_state import is_trading_allowed, read_state
        if not is_trading_allowed():
            sentry_info = read_state()
            halt_reason = sentry_info.get("reason") or "Kill switch or sentry halt is engaged"
            logger.warning(
                f"[SAFETY GUARD] Trading blocked by kill switch / sentry: {halt_reason} "
                f"for {order.symbol} {order.side.value if hasattr(order.side, 'value') else order.side}"
            )
            return UnifiedOrderResponse(False, "", f"Trading blocked: {halt_reason}", session.mode)

        # Safeguard 2: Global trading mode check
        from backend.services.trading_mode import get_trading_mode, TradingMode
        global_mode = get_trading_mode()

        if global_mode == TradingMode.SHADOW:
            logger.info(
                f"[SAFETY GUARD] Global TRADING_MODE=SHADOW. Order for {order.symbol} "
                f"blocked from live and paper broker execution."
            )
            return UnifiedOrderResponse(False, "", "Trading blocked: global TRADING_MODE is SHADOW", "shadow")
        
        if session.mode == "live" and global_mode != TradingMode.LIVE:
            logger.warning(
                f"[SAFETY GUARD] Session mode is LIVE, but global TRADING_MODE={global_mode.value.upper()}. "
                f"Forcing {order.symbol} order into paper execution."
            )
            if not session.paper_portfolio_id:
                pid = self._paper.find_or_create_portfolio(
                    name="Safety Fallback Paper Portfolio",
                    balance=100_000.0,
                )
                session.paper_portfolio_id = pid
            return self._paper.place_order(session.paper_portfolio_id, order)

        if session.mode == "paper":
            if not session.paper_portfolio_id:
                return UnifiedOrderResponse(False, "", "No paper portfolio", "paper")
            return self._paper.place_order(session.paper_portfolio_id, order)

        # Live mode
        broker = self._brokers.get(session.broker)
        if not broker:
            return UnifiedOrderResponse(False, "", f"Broker not registered: {session.broker}", "live")

        try:
            # Broker close contract uses the ORIGINAL POSITION direction, while
            # UnifiedOrder.side is the actual order side. Callers correctly send
            # SELL to close a LONG and BUY to close a SHORT, so recover the
            # position direction here and explicitly mark the action as close.
            # Without this, BinanceFuturesService flips the side a second time
            # and targets the opposite hedge leg (-2022 / false "already flat").
            if order.reduce_only:
                broker_dir = "SELL" if order.side == OrderSide.BUY else "BUY"
                broker_action = "close"
            else:
                broker_dir = "BUY" if order.side == OrderSide.BUY else "SELL"
                broker_action = "open"
            sl = order.stop_loss if order.stop_loss and order.stop_loss > 0 else None
            tp = order.take_profit if order.take_profit and order.take_profit > 0 else None
            result = broker.place_order(
                symbol=order.symbol,
                direction=broker_dir,
                action=broker_action,
                quantity=order.quantity,
                price=order.price,
                stop_loss=sl,
                take_profit=tp,
                reduce_only=order.reduce_only,
                is_pyramid=order.is_pyramid,
            )
            status = result.get("status", "error")
            # already_flat is an idempotent successful close: the exchange has
            # no matching leg left (usually because SL/TP won the race).
            ok = status in ("simulated", "sent", "filled", "already_flat")
            message = (
                result.get("message")
                or result.get("reason")
                or result.get("error")
                or ("Live order placed" if ok else f"Order {status}")
            )
            return UnifiedOrderResponse(
                success=ok,
                order_id=result.get("order_id", ""),
                message=message,
                mode="live",
                filled_price=result.get("filled_price"),
                filled_qty=result.get("quantity"),
                commission=float(result.get("commission") or 0.0),
                realized_pnl=result.get("realized_pnl"),
            )
        except Exception as e:
            logger.exception("Live order failed")
            return UnifiedOrderResponse(False, "", f"Live order error: {e}", "live")

    def cancel_order(self, order_id: str,
                     session_id: Optional[str] = None) -> UnifiedOrderResponse:
        with self._session_lock:
            sid = session_id or self._default_session_id
            session = self._sessions.get(sid)
        if not session:
            return UnifiedOrderResponse(False, "", "No session", "")

        if session.mode == "paper":
            return self._paper.cancel_order(order_id)

        broker = self._brokers.get(session.broker)
        if broker and hasattr(broker, "cancel_order"):
            try:
                result = broker.cancel_order(order_id)
                ok = result.get("success", False)
                return UnifiedOrderResponse(ok, order_id, result.get("message", ""), "live")
            except Exception as e:
                return UnifiedOrderResponse(False, order_id, str(e), "live")
        return UnifiedOrderResponse(False, order_id, "Cancel not supported by broker", "live")

    # ── Paper helpers (default session) ────────────────────────────────────────────

    def _default_pf_id(self) -> Optional[str]:
        sess = self.get_session()
        return sess.paper_portfolio_id if sess else None

    def get_paper_portfolio(self, session_id: Optional[str] = None) -> Optional[dict]:
        sess = self.get_session(session_id)
        if sess and sess.paper_portfolio_id:
            return self._paper.get_portfolio(sess.paper_portfolio_id)
        return None

    def get_paper_positions(self, session_id: Optional[str] = None) -> List[BrokerPosition]:
        sess = self.get_session(session_id)
        if sess and sess.paper_portfolio_id:
            return self._paper.get_positions(sess.paper_portfolio_id)
        return []

    def get_paper_stats(self, session_id: Optional[str] = None) -> dict:
        sess = self.get_session(session_id)
        if sess and sess.paper_portfolio_id:
            return self._paper.get_stats(sess.paper_portfolio_id)
        return {}

    def get_paper_orders(self, status: str = "",
                         session_id: Optional[str] = None) -> List[dict]:
        sess = self.get_session(session_id)
        if sess and sess.paper_portfolio_id:
            return self._paper.get_orders(sess.paper_portfolio_id, status)
        return []

    def get_positions(self, session_id: Optional[str] = None) -> List[BrokerPosition]:
        """Get open positions for the current session (paper or live)."""
        with self._session_lock:
            sid = session_id or self._default_session_id
            session = self._sessions.get(sid)
        if not session:
            return []

        if session.mode == "paper":
            return self._paper.get_positions(session.paper_portfolio_id)

        # Live mode
        broker = self._brokers.get(session.broker)
        if not broker:
            return []

        try:
            if hasattr(broker, "get_positions"):
                raw_pos = broker.get_positions()
                # Map raw broker positions to Unified BrokerPosition
                return [
                    BrokerPosition(
                        symbol=p.get("symbol", ""),
                        side=p.get("side", "long").lower(),
                        quantity=float(p.get("quantity", 0.0)),
                        avg_price=float(p.get("entry_price", 0.0)),
                        current_price=float(p.get("mark_price", 0.0)),
                        unrealized_pnl=float(p.get("unrealized_pnl", 0.0)),
                    )
                    for p in raw_pos
                    if abs(float(p.get("quantity", 0.0))) > 0
                ]
        except Exception as e:
            logger.error(f"Failed to get live positions: {e}")
        
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_ticker(pf: dict, symbol: str):
    if symbol not in pf["positions"]:
        pf["positions"][symbol] = {
            "long": 0, "short": 0,
            "long_cost_basis": 0.0, "short_cost_basis": 0.0,
            "short_margin_used": 0.0,
        }


# Convenience module-level singleton accessor
trading_router = UnifiedTrading()
