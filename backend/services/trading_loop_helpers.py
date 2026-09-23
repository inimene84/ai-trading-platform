"""
Trading Loop Helper Services
=============================
Decoupled manager classes for TradingLoopService.
Implements Emergency Exits, Broker Sync, Trailing Stops, Partial TP,
Exchange Protection, and Performance Metrics to clean up the main loop orchestrator.
"""

import logging
import os
import asyncio
from datetime import datetime, timezone

from sqlalchemy import or_

from backend.database.models import Trade, PortfolioSnapshot
from backend.services.influxdb_writer import influx
from backend.services.ledger import (
    ORPHAN_STATUS,
    binance_position_key,
    has_live_venue_id,
    is_binance_paper_fill,
    is_binance_position_key,
)
from backend.services.decision_engine import atr_from_bars
from backend.services.multi_asset_bars import classify_symbol
from backend.services.unified_trading import UnifiedTrading, UnifiedOrder, OrderSide, OrderType
from backend.services.trading_mode import binance_order_session_id, live_binance_orders_allowed
from backend.services.position_manager import get_position_manager
from backend.services.symbol_aliases import stop_on_correct_side

logger = logging.getLogger(__name__)

# An externally-closed perp position's exit fill is always near the entry
# (SL/TP sit within a few ATR; even liquidation at 10x is ~±10%). Anything
# further off is a bad price source, not a real fill — e.g. 35 historical
# ARBUSDT rows were closed at a bogus 0.00075 vs ~$0.08 entries, fabricating
# +$856 of phantom P&L that poisoned every stat and the AI feedback loop.
MAX_EXIT_PRICE_DEVIATION = 0.5


def is_plausible_exit_price(entry_price, exit_price) -> bool:
    """True when exit_price is a believable close for a position opened at entry_price."""
    if not exit_price or exit_price <= 0 or not entry_price or entry_price <= 0:
        return False
    return abs(exit_price - entry_price) / entry_price <= MAX_EXIT_PRICE_DEVIATION


def is_ctrader_symbol(symbol: str) -> bool:
    """True for FX/metal pairs that must never be sent to Binance Futures."""
    return classify_symbol(str(symbol or "")) in ("forex", "metal")


def is_ctrader_trade(trade) -> bool:
    """Detect a cTrader-owned row even when broker was not persisted."""
    from backend.services.ledger import is_binance_position_key

    broker_raw = getattr(trade, "broker", None) or getattr(trade, "exchange", None)
    broker = broker_raw.lower() if isinstance(broker_raw, str) else ""
    if "binance" in broker:
        return False
    if broker == "ctrader" or broker in {"ic", "icmarkets"}:
        return True
    position_id = getattr(trade, "broker_position_id", None)
    if isinstance(position_id, str) and position_id.strip():
        return not is_binance_position_key(position_id)
    if isinstance(position_id, int) and position_id:
        return True
    symbol = getattr(trade, "symbol", "")
    return is_ctrader_symbol(symbol if isinstance(symbol, str) else "")
def trades_for_direction_cap(all_open: list, broker_name: str) -> list:
    """Scope same-direction / exposure caps to one broker.

    cTrader FX rows (EURUSD, GBPUSD, …) must not consume Binance correlation
    budget — they are a separate book with their own risk limits.
    """
    name = (broker_name or "").strip().lower()
    if name == "binance_futures":
        return [
            t for t in all_open
            if (getattr(t, "broker", None) or "binance_futures") != "ctrader"
        ]
    if name == "ctrader":
        return [t for t in all_open if getattr(t, "broker", None) == "ctrader"]
    return list(all_open)


def remove_closed_pyramid_layer(pyramid_layers: dict, trade) -> None:
    """Remove only the closed pyramid layer; never reset sibling history."""
    layers = pyramid_layers.get(trade.symbol)
    if not layers:
        return
    notes = (getattr(trade, "notes", None) or "").lower()
    if "pyramid" not in notes:
        return  # closing a base row must not erase still-open add layers
    entry = float(getattr(trade, "entry_price", 0) or 0)
    for index, layer_price in enumerate(layers):
        if abs(float(layer_price) - entry) <= max(abs(entry), 1.0) * 1e-9:
            layers.pop(index)
            break
    if not layers:
        pyramid_layers.pop(trade.symbol, None)


class EmergencyExitManager:
    """Manages Step 0 pre-sync emergency exit checks."""

    @staticmethod
    async def run_emergency_exits(
        db,
        broker,
        pyramid_layers: dict,
        sl_cooldown: dict,
    ) -> int:
        exits_triggered = 0
        try:
            pm = get_position_manager()
            pre_open = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
            if not pre_open:
                return 0

            logger.info(f"  [POSITION MGR] Pre-sync review: {len(pre_open)} open positions")
            
            # Fetch live mark prices and qtys from broker
            live_prices = {}
            live_qty = {}
            try:
                for bp in broker.get_positions():
                    live_prices[bp['symbol']] = float(bp.get('mark_price') or bp.get('entry_price') or 0)
                    live_qty[bp['symbol']] = abs(float(bp.get('quantity') or bp.get('positionAmt') or 0))
            except Exception as _bfs_err:
                logger.error(f"  [STEP 0] Price fetch error: {_bfs_err}")
                return 0

            ut_pre = UnifiedTrading()
            for trade in pre_open:
                try:
                    if is_ctrader_trade(trade):
                        continue  # cTrader legs are owned by ctrader sync/poller, not Binance
                    live_px = live_prices.get(trade.symbol, 0.0)
                    if live_px <= 0:
                        continue  # can't check without a price
                    entry_px = trade.entry_price or 0.0
                    if not entry_px:
                        continue
                    direction = trade.direction or "BUY"
                    if direction == "BUY":
                        pnl_pct = ((live_px - entry_px) / entry_px) * 100
                    else:
                        pnl_pct = ((entry_px - live_px) / entry_px) * 100
                        
                    emergency_threshold = pm.emergency_drawdown_pct
                    if pnl_pct <= emergency_threshold:
                        logger.warning(
                            f"  [EMERGENCY EXIT] {trade.symbol}: PnL={pnl_pct:.1f}% "
                            f"<= threshold {emergency_threshold:.1f}% — FORCE CLOSING"
                        )
                        close_side = OrderSide.SELL if direction == "BUY" else OrderSide.BUY
                        res = ut_pre.place_order(UnifiedOrder(
                            symbol=trade.symbol,
                            side=close_side,
                            order_type=OrderType.MARKET,
                            quantity=trade.quantity,
                            price=float(live_px or 0),
                            reduce_only=True,
                        ), session_id=binance_order_session_id())
                        if res.success and res.filled_price:
                            trade.exit_price = res.filled_price
                            if res.realized_pnl is not None:
                                trade.pnl = float(res.realized_pnl) - res.commission
                            else:
                                if direction == "BUY":
                                    gross = (trade.exit_price - entry_px) * trade.quantity
                                else:
                                    gross = (entry_px - trade.exit_price) * trade.quantity
                                trade.pnl = gross - res.commission
                            trade.status = "closed"
                            trade.closed_at = datetime.now(timezone.utc)
                            trade.notes = (trade.notes or "") + f" | EMERGENCY EXIT: PnL={pnl_pct:.1f}%"
                            db.add(trade)
                            remove_closed_pyramid_layer(pyramid_layers, trade)
                            sl_cooldown[trade.symbol] = datetime.now(timezone.utc)
                            exits_triggered += 1
                            logger.warning(f"  [EMERGENCY EXIT] {trade.symbol} CLOSED. PnL={pnl_pct:.1f}%")
                        elif res.success or (
                            hasattr(res, 'message') and (
                                '-2022' in str(res.message) or 'already' in str(res.message).lower()
                            )
                        ):
                            # Re-fetch after the close. The pre-loop live_qty
                            # map is stale (it still has the size that triggered
                            # the exit) and would permanently leave the row open.
                            try:
                                remaining = 0.0
                                for bp in broker.get_positions():
                                    if bp.get("symbol") == trade.symbol:
                                        remaining = abs(float(bp.get("quantity") or 0))
                                        break
                            except Exception as verify_err:
                                logger.error(
                                    f"  [EMERGENCY EXIT] {trade.symbol} already_flat "
                                    f"unverified ({verify_err}); leaving DB open"
                                )
                                continue
                            if remaining and remaining > 0:
                                logger.error(
                                    f"  [EMERGENCY EXIT] {trade.symbol} claimed flat but "
                                    f"live qty {remaining} remains; leaving DB open"
                                )
                                continue
                            logger.info(f"  [EMERGENCY EXIT] {trade.symbol} already flat on exchange — marking DB closed")
                            trade.status = 'closed'
                            trade.closed_at = datetime.now(timezone.utc)
                            trade.exit_price = None
                            trade.pnl = None
                            trade.notes = (trade.notes or '') + ' | EMERGENCY-EXIT: already flat on exchange'
                            db.add(trade)
                            remove_closed_pyramid_layer(pyramid_layers, trade)
                            sl_cooldown[trade.symbol] = datetime.now(timezone.utc)
                            exits_triggered += 1
                        else:
                            logger.error(f"  [EMERGENCY EXIT] {trade.symbol} FAILED: {res.message}")
                except Exception as e:
                    logger.error(f"  [EMERGENCY EXIT] {trade.symbol} error: {e}")
            if exits_triggered > 0:
                db.commit()
        except Exception as e:
            logger.error(f"Pre-sync PM outer error: {e}")
        return exits_triggered


class BrokerPositionSyncService:
    """Syncs DB open trades with live broker state, reconciling external closures."""

    @staticmethod
    async def sync_positions(
        db,
        broker,
        pyramid_layers: dict,
        sl_cooldown: dict,
        broker_name: str = "binance_futures",
    ) -> int:
        updated = 0
        try:
            broker_raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: broker.get_positions(raise_on_error=True)
            )
            if broker_name == "ctrader":
                from backend.services.ctrader_trade_sync import reconcile_ctrader_positions
                res = reconcile_ctrader_positions(db, live_positions=broker_raw, broker=broker)
                return res.get("closed", 0) + res.get("updated", 0)

            broker_symbols = {
                bp['symbol'] for bp in broker_raw
                if float(bp.get('quantity') or bp.get('positionAmt') or 0) != 0
            }

            # Only reconcile rows that belong to the broker being synced. A
            # cTrader forex row is invisible to a Binance snapshot, so syncing
            # across brokers would close live FX positions as phantom orphans.
            owner = Trade.broker == broker_name
            if broker_name == "binance_futures":
                owner = or_(owner, Trade.broker.is_(None))  # legacy rows predate the column
            db_trades = (
                db.query(Trade)
                .filter(Trade.status.in_(["open", "filled"]), owner)
                .all()
            )
            # An empty exchange snapshot after raise_on_error succeeded means
            # the book is flat. Rows with numeric venue IDs stay open so a
            # permissions/testnet glitch cannot cancel protective orders.
            # Synthetic hedge-mode keys (BTCUSDT:LONG) and paper rows are
            # closed/quarantined so phantom size never blocks new entries.
            if db_trades and not broker_raw:
                logger.error(
                    "Broker returned an empty positions snapshot while DB has "
                    f"{len(db_trades)} open trade row(s); closing synthetic-key "
                    "ghosts, quarantining paper/unverified rows, refusing bulk "
                    "close for numeric venue IDs"
                )
                quarantined = 0
                closed = 0
                now = datetime.now(timezone.utc)
                exit_price_cache: dict = {}
                for t in db_trades:
                    pid = getattr(t, "broker_position_id", None)
                    if is_binance_paper_fill(t) or (
                        not is_binance_position_key(pid) and not has_live_venue_id(t)
                    ):
                        t.status = ORPHAN_STATUS
                        t.closed_at = now
                        t.exit_price = None
                        t.pnl = None
                        t.notes = (t.notes or "") + (
                            " | Quarantined: absent from live Binance book "
                            "(paper/unverified ledger row)"
                        )
                        remove_closed_pyramid_layer(pyramid_layers, t)
                        sl_cooldown[t.symbol] = now
                        quarantined += 1
                        updated += 1
                        continue
                    if has_live_venue_id(t) and not is_binance_position_key(pid):
                        continue
                    if not is_binance_position_key(pid):
                        continue
                    logger.info(
                        f"  [ {t.symbol} ] Synthetic-key ghost absent from flat "
                        "exchange book — marking closed in DB"
                    )
                    if t.symbol not in exit_price_cache:
                        exit_price_cache[t.symbol] = await asyncio.get_event_loop().run_in_executor(
                            None, broker.get_exit_price, t.symbol
                        )
                    exit_px = exit_price_cache[t.symbol]
                    t.status = "closed"
                    t.closed_at = now
                    if exit_px and t.entry_price and t.quantity and is_plausible_exit_price(t.entry_price, exit_px):
                        t.exit_price = exit_px
                        if str(t.direction).upper() == "BUY":
                            t.pnl = round((exit_px - t.entry_price) * t.quantity, 4)
                        else:
                            t.pnl = round((t.entry_price - exit_px) * t.quantity, 4)
                        t.notes = (t.notes or "") + " | Closed externally (flat book sync)"
                    else:
                        t.exit_price = None
                        t.pnl = None
                        t.notes = (t.notes or "") + (
                            f" | Closed externally (flat book sync; exit unavailable raw={exit_px})"
                        )
                    remove_closed_pyramid_layer(pyramid_layers, t)
                    sl_cooldown[t.symbol] = now
                    closed += 1
                    updated += 1
                if updated:
                    db.commit()
                    logger.warning(
                        f"Broker sync on empty snapshot: closed={closed}, "
                        f"quarantined={quarantined} ghost Binance row(s)"
                    )
                return updated

            # Adopt exchange legs that have no SQL row (manual entry, lost
            # response/DB commit, or restored DB). One aggregate row per symbol
            # gives the loop ownership instead of blocking that symbol forever.
            db_keys = {
                binance_position_key(t.symbol, t.direction)
                for t in db_trades
                if not is_binance_paper_fill(t)
            }
            adopted = 0
            for bp in broker_raw:
                symbol = bp.get("symbol")
                qty = float(bp.get("quantity") or bp.get("positionAmt") or 0)
                direction_hint = (bp.get("side") or "BUY").upper()
                live_key = binance_position_key(symbol or "", direction_hint)
                if not symbol or qty <= 0 or live_key in db_keys:
                    continue
                entry = float(bp.get("entry_price") or bp.get("entryPrice") or 0)
                if entry <= 0:
                    logger.critical(
                        f"  [ {symbol} ] orphan exchange leg has no entry price; "
                        "cannot adopt safely"
                    )
                    continue
                direction = (bp.get("side") or "BUY").upper()
                if direction == "LONG":
                    direction = "BUY"
                elif direction == "SHORT":
                    direction = "SELL"

                # Recover live protective levels where available.
                stop_loss = take_profit = None
                try:
                    orders = broker.get_open_orders(raise_on_error=True)
                    close_side = "SELL" if direction == "BUY" else "BUY"
                    relevant = [
                        o for o in orders
                        if o.get("symbol") == symbol and o.get("side") == close_side
                    ]
                    sl_levels = [
                        float(o.get("price") or 0) for o in relevant
                        if "STOP" in (o.get("type") or "")
                        and "TAKE_PROFIT" not in (o.get("type") or "")
                        and float(o.get("price") or 0) > 0
                    ]
                    tp_levels = [
                        float(o.get("price") or 0) for o in relevant
                        if "TAKE_PROFIT" in (o.get("type") or "")
                        and float(o.get("price") or 0) > 0
                    ]
                    stop_loss = sl_levels[0] if sl_levels else None
                    take_profit = tp_levels[0] if tp_levels else None
                except Exception as order_error:
                    logger.error(
                        f"  [ {symbol} ] could not inspect orphan protection: {order_error}"
                    )

                # If the orphan is naked, install conservative fallback
                # protection before adopting it into normal management.
                if not stop_loss:
                    fallback_dist = entry * 0.03
                    stop_loss = (
                        entry - fallback_dist
                        if direction == "BUY"
                        else entry + fallback_dist
                    )
                if not take_profit:
                    fallback_tp = entry * 0.04
                    take_profit = (
                        entry + fallback_tp
                        if direction == "BUY"
                        else entry - fallback_tp
                    )
                protection = broker.ensure_protective_orders(
                    symbol, direction, stop_loss, take_profit,
                )
                if protection.get("status") == "error":
                    logger.critical(
                        f"  [ {symbol} ] orphan adoption refused: protection failed "
                        f"({protection.get('message')})"
                    )
                    continue

                orphan = Trade(
                    symbol=symbol,
                    direction=direction,
                    quantity=qty,
                    entry_price=entry,
                    filled_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status="filled",
                    strategy="exchange_reconciliation",
                    notes="Adopted orphan exchange position into DB",
                    exchange="binance_futures",
                    broker="binance_futures",
                    broker_position_id=binance_position_key(symbol, direction),
                    mode="live",
                )
                db.add(orphan)
                db_trades.append(orphan)
                db_keys.add(live_key)
                adopted += 1
                logger.critical(
                    f"  [ {symbol} ] adopted orphan exchange leg: "
                    f"{direction} qty={qty} entry={entry}"
                )

            exit_price_cache: dict = {}
            cancelled_orphans: set = set()

            for t in db_trades:
                if t.symbol in broker_symbols:
                    continue
                if is_binance_paper_fill(t):
                    t.status = ORPHAN_STATUS
                    t.closed_at = datetime.now(timezone.utc)
                    t.exit_price = None
                    t.pnl = None
                    t.notes = (t.notes or "") + (
                        " | Quarantined: paper ledger row missing from live Binance book"
                    )
                    remove_closed_pyramid_layer(pyramid_layers, t)
                    sl_cooldown[t.symbol] = datetime.now(timezone.utc)
                    updated += 1
                    continue

                logger.info(f"  [ {t.symbol} ] Position not found in broker, marking as closed in DB.")

                if t.symbol not in exit_price_cache:
                    exit_price_cache[t.symbol] = await asyncio.get_event_loop().run_in_executor(
                        None, broker.get_exit_price, t.symbol
                    )
                exit_px = exit_price_cache[t.symbol]

                t.status = "closed"
                t.closed_at = datetime.now(timezone.utc)
                if exit_px and t.entry_price and t.quantity and is_plausible_exit_price(t.entry_price, exit_px):
                    t.exit_price = exit_px
                    if str(t.direction).upper() == "BUY":
                        t.pnl = round((exit_px - t.entry_price) * t.quantity, 4)
                    else:
                        t.pnl = round((t.entry_price - exit_px) * t.quantity, 4)
                    t.notes = (t.notes or "") + " | Closed externally (sync)"
                else:
                    # Bad or missing exit price: record the close honestly as
                    # unknown P&L rather than fabricating a number from a
                    # corrupt price (which previously produced phantom gains).
                    if exit_px:
                        logger.error(
                            f"  [ {t.symbol} ] implausible exit price {exit_px} "
                            f"vs entry {t.entry_price} — recording close with unknown P&L"
                        )
                    t.exit_price = None
                    t.pnl = None
                    t.notes = (t.notes or "") + (
                        f" | Closed externally (sync; exit price unavailable/implausible raw={exit_px})"
                    )

                if t.symbol not in cancelled_orphans:
                    try:
                        # Only strip orders when the exchange leg is verified flat.
                        # Never cancel protective SL/TP for symbols that still have
                        # a live position (pyramid layers may close one row at a time).
                        if t.symbol not in broker_symbols:
                            broker.cancel_all_orders(t.symbol)
                        cancelled_orphans.add(t.symbol)
                    except Exception as _ce:
                        logger.warning(f"  [ {t.symbol} ] orphan order cleanup failed: {_ce}")

                remove_closed_pyramid_layer(pyramid_layers, t)
                sl_cooldown[t.symbol] = datetime.now(timezone.utc)
                updated += 1

            if updated > 0 or adopted > 0:
                db.commit()
                logger.info(
                    f"Broker sync: closed={updated}, adopted_orphans={adopted}"
                )
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to sync positions with broker: {e}")
        return updated + adopted


class TrailingStopManager:
    """Manages ATR trailing stops and step trailing, syncing trailed levels to broker."""

    @staticmethod
    def apply_trailing_stop(
        db,
        symbol: str,
        bars: list[dict],
        high_water: dict,
        risk_config,
        broker,
    ) -> None:
        if not getattr(risk_config, "trailing_stop_enabled", False):
            return
        if getattr(risk_config, "native_trailing_enabled", False):
            return
        if not bars or len(bars) < 16:
            return

        try:
            current_price = bars[-1]["close"]
            atr = atr_from_bars(bars, current_price)
            if atr <= 0:
                atr = current_price * 0.02

            activation_dist = risk_config.trail_activation_atr * atr
            trail_dist = risk_config.trail_atr_mult * atr

            trades = db.query(Trade).filter(Trade.symbol == symbol, Trade.status.in_(["open", "filled"])).all()
            for trade in trades:
                if not trade.entry_price:
                    continue

                step_trail = bool(getattr(risk_config, "step_trail_enabled", False))

                if trade.direction == "BUY":
                    hw = high_water.get(trade.id, max(trade.entry_price, current_price))
                    hw = max(hw, current_price)
                    high_water[trade.id] = hw

                    if hw - trade.entry_price < activation_dist:
                        # Optional early BE ratchet. Off by default — it clipped
                        # winners before TP and produced avg-win << avg-loss.
                        if step_trail and hw - trade.entry_price >= (activation_dist * 0.75):
                            fee_offset = trail_dist * 0.1
                            candidate = trade.entry_price + fee_offset
                            old_stop = trade.stop_loss if trade.stop_loss is not None else float("-inf")
                            if candidate > old_stop and stop_on_correct_side(
                                trade.direction, candidate, mark=current_price, entry=trade.entry_price
                            ):
                                trade.stop_loss = candidate
                                logger.info(f"  [ {symbol} ] STEP-TRAIL ↑ stop to BE+fees {candidate:.6f} (hw={hw:.6f} >= 0.75 activation)")
                                TrailingStopManager._sync_exchange_stop(trade, candidate, broker, mark=current_price)
                        continue

                    candidate = hw - trail_dist
                    # Never clamp to mark — that parked stops on price and
                    # scratched winners on the next tick. Skip wrong-side candidates.
                    if candidate >= current_price:
                        continue
                    old_stop = trade.stop_loss if trade.stop_loss is not None else float("-inf")
                    if candidate > old_stop and stop_on_correct_side(
                        trade.direction, candidate, mark=current_price, entry=trade.entry_price
                    ):
                        trade.stop_loss = candidate
                        logger.info(f"  [ {symbol} ] TRAIL ↑ stop {old_stop if old_stop != float('-inf') else 'None'} -> {candidate:.6f}")
                        TrailingStopManager._sync_exchange_stop(trade, candidate, broker, mark=current_price)
                else:  # SHORT
                    lw = high_water.get(trade.id, min(trade.entry_price, current_price))
                    lw = min(lw, current_price)
                    high_water[trade.id] = lw

                    if trade.entry_price - lw < activation_dist:
                        if step_trail and trade.entry_price - lw >= (activation_dist * 0.75):
                            fee_offset = trail_dist * 0.1
                            candidate = trade.entry_price - fee_offset
                            old_stop = trade.stop_loss if trade.stop_loss is not None else float("inf")
                            if candidate < old_stop and stop_on_correct_side(
                                trade.direction, candidate, mark=current_price, entry=trade.entry_price
                            ):
                                trade.stop_loss = candidate
                                logger.info(f"  [ {symbol} ] STEP-TRAIL ↓ stop to BE+fees {candidate:.6f} (lw={lw:.6f} <= 0.75 activation)")
                                TrailingStopManager._sync_exchange_stop(trade, candidate, broker, mark=current_price)
                        continue

                    candidate = lw + trail_dist
                    # Never clamp to mark — skip wrong-side candidates instead.
                    if candidate <= current_price:
                        continue
                    old_stop = trade.stop_loss if trade.stop_loss is not None else float("inf")
                    if candidate < old_stop and stop_on_correct_side(
                        trade.direction, candidate, mark=current_price, entry=trade.entry_price
                    ):
                        trade.stop_loss = candidate
                        logger.info(f"  [ {symbol} ] TRAIL ↓ stop {old_stop if old_stop != float('inf') else 'None'} -> {candidate:.6f}")
                        TrailingStopManager._sync_exchange_stop(trade, candidate, broker, mark=current_price)
        except Exception as e:
            logger.warning(f"  [ {symbol} ] trailing-stop error (stop unchanged): {e}")

    @staticmethod
    def _sync_exchange_stop(trade, new_stop: float, broker, mark: float | None = None):
        """Push a trailed DB stop to Binance for binance_futures rows.

        Dual-live keeps ACTIVE_BROKER=ctrader; still sync crypto legs when
        live Binance orders are allowed. Never push a stop on the wrong side
        of mark/entry (AVAX 7.459 below mark on a short).
        """
        try:
            if is_ctrader_trade(trade):
                return
            if not live_binance_orders_allowed():
                return
            if not stop_on_correct_side(
                getattr(trade, "direction", None),
                new_stop,
                mark=mark,
                entry=getattr(trade, "entry_price", None),
            ):
                logger.warning(
                    f"  [ {trade.symbol} ] refuse exchange-stop sync: "
                    f"{new_stop} wrong-side of mark={mark} entry={getattr(trade, 'entry_price', None)} "
                    f"{getattr(trade, 'direction', None)}"
                )
                return
            res = broker.replace_stop_loss(
                symbol=trade.symbol,
                direction=trade.direction,
                new_stop_price=new_stop,
                quantity=trade.quantity,
            )
            status = res.get("status") if isinstance(res, dict) else None
            if status not in ("replaced", "simulated", "skipped"):
                logger.warning(f"  [ {trade.symbol} ] exchange-stop sync returned {res}")
        except Exception as e:
            logger.warning(f"  [ {trade.symbol} ] exchange-stop sync error: {e}")


def _ratchet_stop_to_be_fees(trade, risk_config, broker=None, mark=None, entry_override=None) -> bool:
    """After a confirmed partial TP, tighten stop to entry ± round-trip cost.

    Tightening only; never widens. Side-checked before the DB write and
    before TrailingStopManager._sync_exchange_stop.
    Returns True when the stop was moved.
    """
    entry = entry_override if entry_override is not None else getattr(trade, "entry_price", None)
    try:
        entry_px = float(entry) if entry is not None else 0.0
    except (TypeError, ValueError):
        return False
    if entry_px <= 0:
        return False

    try:
        cost_rate = float(getattr(risk_config, "roundtrip_cost_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost_rate = 0.0
    if cost_rate < 0:
        cost_rate = 0.0
    fee_offset = entry_px * cost_rate
    direction = (getattr(trade, "direction", None) or "").upper()
    old_stop = getattr(trade, "stop_loss", None)

    if direction == "BUY":
        candidate = entry_px + fee_offset
        floor = float(old_stop) if old_stop is not None else float("-inf")
        if candidate <= floor:
            return False
    elif direction == "SELL":
        candidate = entry_px - fee_offset
        ceiling = float(old_stop) if old_stop is not None else float("inf")
        if candidate >= ceiling:
            return False
    else:
        return False

    if not stop_on_correct_side(direction, candidate, mark=mark, entry=entry_px):
        logger.info(
            f"  [ {getattr(trade, 'symbol', '?')} ] PARTIAL-TP BE ratchet refused: "
            f"{candidate} wrong-side of mark={mark} entry={entry_px} {direction}"
        )
        return False

    trade.stop_loss = candidate
    logger.info(
        f"  [ {getattr(trade, 'symbol', '?')} ] PARTIAL-TP BE ratchet "
        f"{old_stop} -> {candidate:.6f} (entry ± roundtrip_cost_rate)"
    )
    TrailingStopManager._sync_exchange_stop(trade, candidate, broker, mark=mark)
    return True


class PartialTPManager:
    """Manages partial profit taking (TP) by closing a portion of the position."""

    @staticmethod
    def apply_partial_tp(
        db,
        symbol: str,
        bars: list[dict],
        risk_config,
        strategy_name: str,
    ) -> None:
        """Paper / non-live: close a fraction of each DB row independently."""
        if not getattr(risk_config, "partial_tp_enabled", False):
            return
        if not bars or len(bars) < 16:
            return
        try:
            current_price = bars[-1]["close"]

            atr = atr_from_bars(bars, current_price)
            if atr <= 0:
                atr = current_price * 0.02

            partial_dist = risk_config.partial_tp_atr_mult * atr
            close_pct = risk_config.partial_tp_close_pct

            trades = db.query(Trade).filter(Trade.symbol == symbol, Trade.status.in_(["open", "filled"])).all()
            for trade in trades:
                if not trade.entry_price or not trade.quantity:
                    continue
                notes = trade.notes or ""
                if "PARTIAL_TP_DONE" in notes:
                    continue

                if trade.direction == "BUY":
                    profit_dist = current_price - trade.entry_price
                else:
                    profit_dist = trade.entry_price - current_price

                if profit_dist < partial_dist:
                    continue

                close_qty = trade.quantity * close_pct
                if close_qty <= 0:
                    continue

                ut = UnifiedTrading()
                close_side = OrderSide.SELL if trade.direction == "BUY" else OrderSide.BUY
                res = ut.place_order(UnifiedOrder(
                    symbol=trade.symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=close_qty,
                    price=float(current_price or 0),
                    reduce_only=True,
                ), session_id=binance_order_session_id())

                if res.success:
                    filled_px = res.filled_price or current_price
                    if trade.direction == "BUY":
                        partial_pnl = (filled_px - trade.entry_price) * close_qty
                    else:
                        partial_pnl = (trade.entry_price - filled_px) * close_qty

                    trade.quantity = trade.quantity - close_qty
                    trade.notes = (notes + f" | PARTIAL_TP_DONE: closed {close_pct*100:.0f}% "
                                   f"({close_qty:.6f}) @ {filled_px:.6f}, partial PnL=${partial_pnl:+.4f}")
                    logger.info(f"  [ {symbol} ] PARTIAL TP: closed {close_pct*100:.0f}% ({close_qty:.6f}) @ {filled_px:.6f}")
                    _ratchet_stop_to_be_fees(trade, risk_config, broker=None, mark=current_price)
                    try:
                        asyncio.get_event_loop().create_task(
                            influx._write(
                                influx.BUCKET_SYSTEM, "partial_tp",
                                {"symbol": symbol, "direction": trade.direction},
                                {"pnl": partial_pnl, "close_qty": close_qty, "filled_price": filled_px},
                            )
                        )
                    except Exception:
                        pass
                else:
                    logger.warning(f"  [ {symbol} ] PARTIAL TP failed: {res.message}")
        except Exception as e:
            logger.warning(f"  [ {symbol} ] partial-TP error: {e}")

    @staticmethod
    def apply_partial_tp_live(
        db,
        symbol: str,
        bars: list[dict],
        risk_config,
        broker,
    ) -> None:
        """Live Binance: one reduce-only close sized from exchange qty (not per DB row).

        Pyramid stacks multiple DB rows against a single hedge leg. Closing
        close_pct of each row over-closes the exchange position. Exchange qty
        is the source of truth; DB rows are rescaled after a confirmed fill.
        """
        if not getattr(risk_config, "partial_tp_enabled", False):
            return
        if not bars or len(bars) < 16:
            return
        try:
            trades = (
                db.query(Trade)
                .filter(Trade.symbol == symbol, Trade.status.in_(["open", "filled"]))
                .all()
            )
            if not trades:
                return
            # Symbol-level idempotency: any row already stamped → skip.
            if any("PARTIAL_TP_DONE" in (t.notes or "") for t in trades):
                return

            direction = (trades[0].direction or "").upper()
            if direction not in ("BUY", "SELL"):
                return
            # Require a consistent side across pyramid layers.
            if any((t.direction or "").upper() != direction for t in trades):
                logger.warning(f"  [ {symbol} ] PARTIAL TP live skipped: mixed directions")
                return

            current_price = bars[-1]["close"]
            atr = atr_from_bars(bars, current_price)
            if atr <= 0:
                atr = current_price * 0.02
            partial_dist = risk_config.partial_tp_atr_mult * atr
            close_pct = float(risk_config.partial_tp_close_pct)

            # VWAP of open rows as the profit reference.
            total_qty = sum(float(t.quantity or 0) for t in trades)
            if total_qty <= 0:
                return
            vwap = sum(float(t.entry_price) * float(t.quantity or 0) for t in trades) / total_qty
            if direction == "BUY":
                profit_dist = current_price - vwap
            else:
                profit_dist = vwap - current_price
            if profit_dist < partial_dist:
                return

            futures_sym = broker._to_futures_symbol(symbol) if hasattr(broker, "_to_futures_symbol") else symbol
            if not futures_sym:
                return
            position_side = "LONG" if direction == "BUY" else "SHORT"
            try:
                live_qty = float(broker._live_position_qty(futures_sym, position_side, raise_on_error=True))
            except Exception as e:
                logger.warning(f"  [ {symbol} ] PARTIAL TP live qty check failed: {e}")
                return
            if live_qty <= 0:
                return

            raw_close = live_qty * close_pct
            if hasattr(broker, "_round_qty"):
                close_qty = broker._round_qty(futures_sym, raw_close, bump_to_min=False)
                remainder = broker._round_qty(futures_sym, live_qty - close_qty, bump_to_min=False)
            else:
                close_qty = raw_close
                remainder = live_qty - close_qty
            if close_qty <= 0 or remainder <= 0:
                logger.info(
                    f"  [ {symbol} ] PARTIAL TP live skipped: close={close_qty} remainder={remainder} "
                    f"(live={live_qty})"
                )
                return

            ut = UnifiedTrading()
            close_side = OrderSide.SELL if direction == "BUY" else OrderSide.BUY
            res = ut.place_order(UnifiedOrder(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=close_qty,
                price=float(current_price or 0),
                reduce_only=True,
            ), session_id=binance_order_session_id())
            if not res.success:
                logger.warning(f"  [ {symbol} ] PARTIAL TP live failed: {res.message}")
                return

            # Re-read exchange qty after fill; fall back to arithmetic remainder.
            try:
                remaining_live = float(
                    broker._live_position_qty(futures_sym, position_side, raise_on_error=True)
                )
            except Exception:
                remaining_live = remainder
            if remaining_live < 0:
                remaining_live = 0.0

            filled_px = res.filled_price or current_price
            if direction == "BUY":
                partial_pnl = (filled_px - vwap) * close_qty
            else:
                partial_pnl = (vwap - filled_px) * close_qty

            # Rescale DB rows so sum(qty) == remaining exchange qty.
            db_total = sum(float(t.quantity or 0) for t in trades)
            scale = (remaining_live / db_total) if db_total > 0 else 0.0
            note = (
                f" | PARTIAL_TP_DONE: live closed {close_pct*100:.0f}% "
                f"({close_qty:.6f}) @ {filled_px:.6f}, partial PnL=${partial_pnl:+.4f}"
            )
            for trade in trades:
                trade.quantity = float(trade.quantity or 0) * scale
                trade.notes = (trade.notes or "") + note
                _ratchet_stop_to_be_fees(
                    trade, risk_config, broker=broker, mark=current_price, entry_override=vwap,
                )

            logger.info(
                f"  [ {symbol} ] PARTIAL TP live: closed {close_qty:.6f} @ {filled_px:.6f} "
                f"(remaining_live={remaining_live:.6f})"
            )
            try:
                asyncio.get_event_loop().create_task(
                    influx._write(
                        influx.BUCKET_SYSTEM, "partial_tp",
                        {"symbol": symbol, "direction": direction, "mode": "live"},
                        {"pnl": partial_pnl, "close_qty": close_qty, "filled_price": filled_px},
                    )
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"  [ {symbol} ] partial-TP live error: {e}")


class ExchangeProtectionManager:
    """Validates and restores missing SL/TP orders on the exchange."""

    @staticmethod
    def restore_all_open_positions(db, broker) -> dict:
        """Re-place missing SL/TP for every symbol with open DB trades."""
        summary = {"checked": 0, "restored": 0, "errors": 0}
        try:
            if os.getenv("ACTIVE_BROKER", "ctrader") != "binance_futures":
                return summary
            from backend.services.trading_mode import get_trading_mode, TradingMode
            if get_trading_mode() != TradingMode.LIVE:
                return summary

            trades = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
            by_symbol: dict[str, Trade] = {}
            for trade in trades:
                if is_ctrader_trade(trade):
                    continue
                if not trade.stop_loss and not trade.take_profit:
                    continue
                existing = by_symbol.get(trade.symbol)
                if not existing or (trade.stop_loss and trade.take_profit):
                    by_symbol[trade.symbol] = trade

            for symbol, trade in by_symbol.items():
                summary["checked"] += 1
                res = broker.ensure_protective_orders(
                    trade.symbol,
                    trade.direction,
                    trade.stop_loss,
                    trade.take_profit,
                )
                status = res.get("status") if isinstance(res, dict) else None
                if status == "restored":
                    summary["restored"] += 1
                    logger.warning(
                        f"  [ {symbol} ] protection restored on startup/resume: "
                        f"{res.get('restored')}"
                    )
                elif status == "error":
                    summary["errors"] += 1
                    logger.error(
                        f"  [ {symbol} ] protection restore failed: {res.get('message')}"
                    )
        except Exception as e:
            logger.warning(f"Bulk protection restore failed: {e}")
            summary["errors"] += 1
        return summary

    @staticmethod
    def ensure_exchange_protection(
        db,
        symbol: str,
        broker,
    ) -> None:
        try:
            if os.getenv("ACTIVE_BROKER", "ctrader") != "binance_futures":
                return
            from backend.services.trading_mode import get_trading_mode, TradingMode
            if get_trading_mode() != TradingMode.LIVE:
                return

            trades = db.query(Trade).filter(Trade.symbol == symbol, Trade.status.in_(["open", "filled"])).all()
            for trade in trades:
                if is_ctrader_trade(trade):
                    continue
                if not trade.stop_loss and not trade.take_profit:
                    continue
                res = broker.ensure_protective_orders(
                    trade.symbol,
                    trade.direction,
                    trade.stop_loss,
                    trade.take_profit,
                )
                if res.get("status") == "restored":
                    logger.warning(f"  [ {symbol} ] Exchange protection restored: {res.get('restored')}")
        except Exception as e:
            logger.warning(f"  [ {symbol} ] exchange protection check failed: {e}")


class PerformanceMetricsWriter:
    """Computes and writes rolling win rate and drawdown metrics to InfluxDB."""

    @staticmethod
    async def write_metrics(
        db,
        cycle_equity: float,
        open_count: int,
        cycle_count: int,
    ) -> None:
        try:
            from sqlalchemy import func as _func, case

            # Use SQL aggregates instead of loading all closed trades into memory
            stats = db.query(
                _func.count(Trade.id).label("total"),
                _func.sum(case((Trade.pnl > 0, 1), else_=0)).label("wins"),
                _func.sum(case((Trade.pnl < 0, 1), else_=0)).label("losses"),
                _func.coalesce(_func.sum(Trade.pnl), 0.0).label("realized_pnl"),
            ).filter(
                Trade.status == "closed",
                Trade.pnl.isnot(None),
            ).one()

            total = int(stats.total or 0)
            wins = int(stats.wins or 0)
            losses = int(stats.losses or 0)
            decided = wins + losses
            win_rate = (wins / decided * 100.0) if decided else 0.0
            realized_pnl = round(float(stats.realized_pnl or 0.0), 4)

            # Drawdown vs peak equity seen in portfolio snapshots
            peak = db.query(_func.max(PortfolioSnapshot.total_value)).scalar() or cycle_equity
            drawdown_pct = ((cycle_equity - peak) / peak * 100.0) if peak and peak > 0 else 0.0

            await influx.write_performance(
                equity=cycle_equity,
                realized_pnl=realized_pnl,
                win_rate=round(win_rate, 2),
                wins=wins,
                losses=losses,
                total_trades=total,
                drawdown_pct=round(drawdown_pct, 3),
                open_positions=open_count,
                cycle=cycle_count,
            )
        except Exception as e:
            logger.warning(f"Failed to write performance metrics: {e}")
