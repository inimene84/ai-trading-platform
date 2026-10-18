"""
Binance Futures order status poller.
Runs as a background task, syncs open DB trades against Binance order status every 30 seconds.
Updates trade.status and trade.pnl when orders are FILLED, CANCELED, or EXPIRED on Binance.
Independent of the trading loop - always active when backend starts (only for binance_futures broker).
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

from backend.services.trading_mode import live_binance_orders_allowed
from backend.services.binance_maker import is_maker_tp_order

logger = logging.getLogger(__name__)

_task = None


async def start_order_poller():
    """Supervisor entry: run the order poll loop until cancelled."""
    global _task
    _task = asyncio.current_task()
    interval = int(os.getenv("BINANCE_ORDER_POLL_INTERVAL", "60"))
    logger.info(f"Binance order poller started (interval={interval}s)")
    await _poll_loop()


async def stop_order_poller():
    """Stop the background order polling loop."""
    global _task
    if _task:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
        logger.info("Binance order poller stopped")


async def _poll_loop():
    """Main polling loop - runs until process exits."""
    try:
        from backend.services.binance_futures_service import binance_futures_broker
        svc = binance_futures_broker
        logger.info("Order poller: using binance_futures_broker singleton")
    except Exception as e:
        logger.error(f"Order poller: failed to init BinanceFuturesService: {e}")
        return

    while True:
        try:
            # Only run when Binance Futures is the active broker
            broker_name = os.getenv("ACTIVE_BROKER", "ctrader")
            if broker_name == "binance_futures" and live_binance_orders_allowed():
                await _sync_open_orders(svc)
                await _cancel_orphaned_orders(svc)
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, svc.reconcile_maker_tps)
                except Exception as tp_e:
                    logger.debug("Order poller: maker TP reconcile skipped: %s", tp_e)
        except Exception as e:
            logger.warning(f"Order poller cycle error: {e}")
        
        interval = int(os.getenv("BINANCE_ORDER_POLL_INTERVAL", "60"))
        await asyncio.sleep(interval)


_PROTECTIVE_ORDER_TYPES = {
    "STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET", "STOP", "TAKE_PROFIT",
}


async def _cancel_orphaned_orders(svc):
    """Cancel protective orders (SL/TP/trailing) left on the exchange for symbols
    that no longer have an open position.

    When a position closes on the exchange (e.g. the trailing stop or SL fires),
    its sibling reduce-only orders are orphaned. They can't open a new position
    but they clutter the account; clean them up within one poll cycle (~30s).
    Pending entry orders (LIMIT/MARKET) are left untouched.
    """
    try:
        positions = await asyncio.get_event_loop().run_in_executor(None, lambda: svc.get_positions(raise_on_error=True))
        pos_syms = {p["symbol"] for p in positions if abs(float(p.get("quantity", 0) or 0)) > 0}
        orders = await asyncio.get_event_loop().run_in_executor(None, lambda: svc.get_open_orders(raise_on_error=True))
    except Exception as e:
        logger.warning(f"Orphan reconcile: failed to fetch state: {e}")
        return

    client = svc._get_client()
    cancelled = 0
    for o in orders:
        sym = o.get("symbol")
        if sym in pos_syms:
            logger.debug(f"Orphan reconcile: keeping protective orders for {sym} — position still open")
            continue  # live position — keep its protective orders
        if o.get("type") not in _PROTECTIVE_ORDER_TYPES and not is_maker_tp_order(o):
            continue
        try:
            if o.get("algo_id"):
                # Conditional/algo order (SL/TP/trailing on this account)
                client.futures_cancel_algo_order(algoId=o["algo_id"])
            else:
                client.futures_cancel_order(symbol=sym, orderId=int(o["order_id"]))
            cancelled += 1
        except Exception as ce:
            logger.debug(f"Orphan cancel failed for {sym} {o.get('order_id')}: {ce}")
    if cancelled:
        logger.info(f"Order poller: cancelled {cancelled} orphaned protective order(s)")


async def _sync_open_orders(svc):
    """Compare DB open trades against live Binance order status and sync."""
    from backend.database.connection import SessionLocal
    from backend.database.models import Trade
    from backend.services.influxdb_writer import influx

    db = SessionLocal()
    try:
        # 1. Query DB for open trades that have a real binance_order_id (not paper trades)
        open_trades = (
            db.query(Trade)
            .filter(
                Trade.status == "open",
                Trade.binance_order_id.isnot(None),
                Trade.binance_order_id != "",
                ~Trade.binance_order_id.like("paper_%"),  # Skip paper orders
            )
            .all()
        )

        if not open_trades:
            return

        # 2. Fetch currently open orders from Binance (set of order_ids still open)
        try:
            live_open = svc.get_open_orders()
            live_open_ids = {o["order_id"] for o in live_open}
        except Exception as e:
            logger.warning(f"Order poller: get_open_orders failed: {e}")
            return

        # 3. For each DB trade, check if its order is still open on Binance
        updated = 0
        for trade in open_trades:
            if trade.binance_order_id in live_open_ids:
                # Still open on Binance - no action needed
                continue

            # Order is no longer open → fetch precise status from Binance
            try:
                order_status = await _get_order_status(svc, trade)
                if order_status is None:
                    continue

                binance_status = order_status.get("status", "UNKNOWN")
                logger.info(
                    f"Order poller: trade.id={trade.id} {trade.symbol} "
                    f"order={trade.binance_order_id} status={binance_status}"
                )

                if binance_status == "FILLED":
                    # Parse to float first to prevent "0.00000" truthy string bug from overriding price with 0.0
                    avg_px = float(order_status.get("avgPrice") or 0.0)
                    ord_px = float(order_status.get("price") or 0.0)
                    filled_price = avg_px if avg_px > 0.0 else (ord_px if ord_px > 0.0 else float(trade.entry_price or 0.0))
                    filled_qty = float(order_status.get("executedQty") or trade.quantity or 0)

                    # Update fill details if more precise than what we stored
                    if filled_price and filled_price != trade.entry_price:
                        trade.filled_price = filled_price
                        trade.entry_price = filled_price  # update to actual fill price

                    if filled_qty and filled_qty != trade.quantity:
                        trade.quantity = filled_qty

                    # Mark as 'filled' so the poller stops re-querying this order.
                    # Position is now live; exit will close it later.
                    trade.status = "filled"
                    logger.info(
                        f"  -> Order FILLED: trade.id={trade.id} fill_price={filled_price} qty={filled_qty}"
                    )
                    # Record the confirmed fill in InfluxDB for Grafana
                    try:
                        await influx.write_trade(
                            symbol=trade.symbol, direction=trade.direction,
                            quantity=filled_qty, entry_price=filled_price,
                            status="filled", strategy=trade.strategy or "",
                            pnl=0.0,
                        )
                    except Exception as _ie:
                        logger.warning(f"  -> InfluxDB write_trade(filled) failed: {_ie}")

                elif binance_status in ("CANCELED", "EXPIRED", "REJECTED"):
                    trade.status = "failed"
                    trade.closed_at = datetime.now(timezone.utc)
                    trade.notes = (trade.notes or "") + f" | Order {binance_status} on Binance"
                    logger.warning(
                        f"  -> Order {binance_status}: trade.id={trade.id} marked as failed"
                    )
                    # Write failed status to InfluxDB
                    await influx.write_trade(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        quantity=trade.quantity,
                        entry_price=trade.entry_price or 0.0,
                        status="failed",
                        strategy=trade.strategy or "",
                        pnl=0.0,
                    )

                elif binance_status == "PARTIALLY_FILLED":
                    filled_qty = float(order_status.get("executedQty") or 0)
                    # Parse to float first to prevent "0.00000" truthy string bug
                    avg_px = float(order_status.get("avgPrice") or 0.0)
                    filled_price = avg_px if avg_px > 0.0 else float(trade.entry_price or 0.0)
                    if filled_price:
                        trade.filled_price = filled_price
                    logger.info(
                        f"  -> Order PARTIALLY_FILLED: trade.id={trade.id} "
                        f"executed={filled_qty}/{trade.quantity}"
                    )

                updated += 1

            except Exception as e:
                logger.warning(f"Order poller: error processing trade.id={trade.id}: {e}")
                continue

        if updated:
            db.commit()
            logger.info(f"Order poller: synced {updated} trade(s)")

    except Exception as e:
        db.rollback()
        logger.error(f"Order poller sync database error: {e}")
        raise
    finally:
        db.close()


async def _get_order_status(svc, trade) -> dict | None:
    """Fetch a specific order's status from Binance Futures."""
    try:
        client = await asyncio.get_event_loop().run_in_executor(
            None, svc._get_client
        )
        # Convert symbol to futures format
        futures_sym = svc._to_futures_symbol(trade.symbol)
        if not futures_sym:
            futures_sym = trade.symbol  # try raw symbol as fallback

        order = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.futures_get_order(
                symbol=futures_sym,
                orderId=int(trade.binance_order_id)
            )
        )
        return order
    except Exception as e:
        logger.warning(
            f"_get_order_status: trade.id={trade.id} order={trade.binance_order_id}: {e}"
        )
        return None
