"""
cTrader Open API Service — Direct TCP + Protobuf Protocol Adapter.

Implements the BrokerService protocol for QuantumTrade Pro.
Supports Forex, CFDs, Commodities, and Crypto with native SL/TP calculations,
safe paper mode by default, token lifecycle rotation persistence, and auto-reconnect watchdog.
"""

import logging
import os
import struct
import threading
import time
from decimal import Decimal
from typing import Optional, Dict, List, Any

from backend.brokers.base import BrokerService
from backend.services.ctrader_tokens import token_store
from backend.services.trading_mode import live_ctrader_orders_allowed

logger = logging.getLogger(__name__)


class CTraderProtocol:
    """cTrader TCP Protobuf framing protocol handler."""

    def __init__(self, service: "CTraderService", creds: dict):
        self._service = service
        self._creds = creds
        self._buf = b""
        self.transport = None
        self.last_heartbeat = time.time()
        self._subscribed_spot_ids: set[int] = set()

    def connectionMade(self, transport):
        self.transport = transport
        self.last_heartbeat = time.time()
        logger.info("cTrader TCP connected — initiating Application Authorization (2100)")
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            req = msgs.ProtoOAApplicationAuthReq()
            req.clientId = self._creds.get("client_id", "")
            req.clientSecret = self._creds.get("client_secret", "")
            self._send(req, 2100)
        except Exception as e:
            logger.error(f"Failed to send cTrader App Auth: {e}")

    def _send(self, message: Any, payload_type: int):
        if not self.transport:
            return
        try:
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
            pm = ProtoMessage()
            pm.payloadType = payload_type
            pm.payload = message.SerializeToString() if hasattr(message, "SerializeToString") else message
            data = pm.SerializeToString()
            self.transport.write(struct.pack(">I", len(data)) + data)
        except Exception as e:
            logger.error(f"cTrader protocol send error (payloadType={payload_type}): {e}")

    def _subscribe_spots(self, symbol_ids):
        """Ask for streaming quotes on symbols we hold.

        Spot events (2131) are the only price source for marking a forex
        position. Without this request the broker never sends them, so every
        position reported a 0.00 P&L.
        """
        wanted = {int(sid) for sid in symbol_ids if sid}
        new_ids = wanted - self._subscribed_spot_ids
        if not new_ids:
            return
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs

            req = msgs.ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self._creds.get("account_id", 0)
            req.symbolId.extend(sorted(new_ids))
            self._send(req, 2127)
            self._subscribed_spot_ids |= new_ids
            logger.info(f"cTrader subscribed to spot quotes for {sorted(new_ids)}")
        except Exception as e:
            logger.error(f"cTrader spot subscription failed: {e}")

    def _request_symbol_specs(self, symbol_ids):
        """Load per-symbol lotSize / minVolume. Metals are not 10M cents/lot."""
        wanted = sorted({int(sid) for sid in symbol_ids if sid})
        if not wanted:
            return
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs

            req = msgs.ProtoOASymbolByIdReq()
            req.ctidTraderAccountId = self._creds.get("account_id", 0)
            req.symbolId.extend(wanted)
            self._send(req, 2116)
            logger.info("cTrader requested symbol specs for %s ids", len(wanted))
        except Exception as e:
            logger.error(f"cTrader symbol-spec request failed: {e}")

    def dataReceived(self, data: bytes):
        self.last_heartbeat = time.time()
        self._buf += data
        while len(self._buf) >= 4:
            length = struct.unpack(">I", self._buf[:4])[0]
            if len(self._buf) < 4 + length:
                break
            raw = self._buf[4: 4 + length]
            self._buf = self._buf[4 + length:]
            try:
                from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
                pm = ProtoMessage()
                pm.ParseFromString(raw)
                self._handle(pm)
            except Exception as e:
                logger.error(f"Failed to decode cTrader message frame: {e}")

    def _handle(self, msg: Any):
        ptype = msg.payloadType
        self.last_heartbeat = time.time()
        logger.debug(f"cTrader message received type={ptype}")

        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent

            if ptype == 2101:  # ProtoOAApplicationAuthRes
                # Discover accounts for this token first. Authing a stale/wrong
                # ctidTraderAccountId is a common CANT_ROUTE_REQUEST cause.
                if hasattr(msgs, "ProtoOAGetAccountListByAccessTokenReq"):
                    logger.info("cTrader App authenticated — requesting account list (2149)")
                    req = msgs.ProtoOAGetAccountListByAccessTokenReq()
                    req.accessToken = self._creds.get("access_token", "")
                    self._send(req, 2149)
                else:
                    self._send_account_auth()

            elif ptype == 2150:  # ProtoOAGetAccountListByAccessTokenRes
                res = msgs.ProtoOAGetAccountListByAccessTokenRes()
                res.ParseFromString(msg.payload)
                rows = []
                for acc in getattr(res, "ctidTraderAccount", []) or []:
                    rows.append({
                        "id": int(getattr(acc, "ctidTraderAccountId", 0) or 0),
                        "is_live": bool(getattr(acc, "isLive", False)),
                        "login": int(getattr(acc, "traderLogin", 0) or 0),
                    })
                prefer_live = "live.ctraderapi.com" in str(self._creds.get("host") or "")
                wanted = int(self._creds.get("account_id") or 0)
                chosen = CTraderService.pick_trader_account_id(rows, wanted, prefer_live)
                if not chosen:
                    logger.error("cTrader token has no trader accounts")
                    self._service._last_protocol_error = "NO_ACCOUNTS"
                    self._service._auth_event.set()
                else:
                    if chosen != wanted:
                        logger.warning(
                            "cTrader configured account %s not in token list %s; using %s",
                            wanted,
                            [r["id"] for r in rows],
                            chosen,
                        )
                    self._creds["account_id"] = chosen
                    self._service._account_id = chosen
                    chosen_row = next((r for r in rows if r["id"] == chosen), {})
                    account_live = bool(chosen_row.get("is_live"))
                    current_live = "live.ctraderapi.com" in str(self._creds.get("host") or "")
                    if account_live != current_live:
                        live_confirm = os.getenv(
                            "CTRADER_LIVE_CONFIRM", os.getenv("CTRADE_LIVE_CONFIRM", "")
                        ).strip()
                        if account_live and live_confirm != "I_UNDERSTAND":
                            logger.error(
                                "cTrader account %s is live but CTRADER_LIVE_CONFIRM is not set — staying on demo",
                                chosen,
                            )
                            self._service._last_protocol_error = "LIVE_CONFIRM_REQUIRED"
                            self._service._auth_event.set()
                            if self.transport:
                                try:
                                    self.transport.loseConnection()
                                except Exception:
                                    pass
                            return
                        correct = "live.ctraderapi.com" if account_live else "demo.ctraderapi.com"
                        logger.warning(
                            "cTrader account %s is_live=%s but host is %s — switching to %s",
                            chosen,
                            account_live,
                            self._creds.get("host"),
                            correct,
                        )
                        self._service._host_override = correct
                        self._service._retry_correct_host = True
                        self._service._last_protocol_error = (
                            f"HOST_MISMATCH — account is {'live' if account_live else 'demo'}"
                        )
                        self._service._auth_event.set()
                        if self.transport:
                            try:
                                self.transport.loseConnection()
                            except Exception:
                                pass
                    else:
                        logger.info("cTrader account list: %s; authenticating %s", rows, chosen)
                        self._send_account_auth()

            elif ptype == 2103:  # ProtoOAAccountAuthRes
                logger.info(f"cTrader Account {self._creds.get('account_id')} authenticated! Ready to trade.")
                self._service._authenticated = True
                self._service._protocol = self
                self._service._auth_event.set()

                # 1. Fetch balance & account specs
                req_trader = msgs.ProtoOATraderReq()
                req_trader.ctidTraderAccountId = self._creds.get("account_id", 0)
                self._send(req_trader, 2121)

                # 2. Fetch symbol list for dynamic mapping
                req_syms = msgs.ProtoOASymbolsListReq()
                req_syms.ctidTraderAccountId = self._creds.get("account_id", 0)
                self._send(req_syms, 2114)

                # 3. Reconcile existing open positions
                req_reconcile = msgs.ProtoOAReconcileReq()
                req_reconcile.ctidTraderAccountId = self._creds.get("account_id", 0)
                self._send(req_reconcile, 2124)

            elif ptype == 2115:  # ProtoOASymbolsListRes (2116 is SymbolByIdReq)
                res = msgs.ProtoOASymbolsListRes()
                res.ParseFromString(msg.payload)
                sym_map = {}
                for sym in res.symbol:
                    name = CTraderService.normalize_catalog_name(getattr(sym, "symbolName", "") or "")
                    if name:
                        sym_map[name] = sym.symbolId
                self._service._symbol_ids = CTraderService.merge_symbol_catalog(
                    dict(self._service.DEFAULT_SYMBOL_IDS),
                    sym_map,
                )
                self._service.relabel_cached_symbols()
                logger.info(
                    "cTrader dynamic symbol catalog updated: %s instruments (NZDJPY id=%s)",
                    len(self._service._symbol_ids),
                    self._service._symbol_ids.get("NZDJPY"),
                )
                self._request_symbol_specs(self._service.priority_symbol_ids())
                # Reconcile often arrives before this large catalog. Refresh names
                # after IDs are unique so NZDJPY is not left labeled XAUUSD.
                req_rec = msgs.ProtoOAReconcileReq()
                req_rec.ctidTraderAccountId = self._creds.get("account_id", 0)
                self._send(req_rec, 2124)

            elif ptype == 2117:  # ProtoOASymbolByIdRes
                res = msgs.ProtoOASymbolByIdRes()
                res.ParseFromString(msg.payload)
                for sym in res.symbol:
                    name = self._service.symbol_name_for_id(sym.symbolId)
                    if not name:
                        continue
                    spec = {
                        "lot_size": int(getattr(sym, "lotSize", 0) or 0),
                        "min_volume": int(getattr(sym, "minVolume", 0) or 0),
                        "max_volume": int(getattr(sym, "maxVolume", 0) or 0),
                        "step_volume": int(getattr(sym, "stepVolume", 0) or 0),
                        "digits": int(getattr(sym, "digits", 0) or 0),
                        "pip_position": int(getattr(sym, "pipPosition", 0) or 0),
                    }
                    self._service._symbol_specs[name] = spec
                    logger.info(
                        "cTrader symbol spec %s id=%s lotSize=%s minVolume=%s stepVolume=%s",
                        name, sym.symbolId, spec["lot_size"],
                        spec["min_volume"], spec["step_volume"],
                    )
                self._service.refresh_position_lots()

            elif ptype == 2122:  # ProtoOATraderRes
                trader_res = msgs.ProtoOATraderRes()
                trader_res.ParseFromString(msg.payload)
                trader = trader_res.trader

                money_digits = trader.moneyDigits if trader.moneyDigits else 2
                divisor = 10 ** money_digits
                balance = trader.balance / divisor

                self._service.balance = balance
                self._service.equity = balance
                self._service.margin = 0.0
                self._service._leverage = trader.leverageInCents / 100 if trader.leverageInCents else 500
                self._service._broker_name = trader.brokerName if trader.brokerName else "cTrader Broker"
                self._service._trader_login = trader.traderLogin if trader.traderLogin else 0
                logger.info(f"cTrader balance: {balance:.2f} | leverage 1:{self._service._leverage:.0f}")

            elif ptype == 2125:  # ProtoOAReconcileRes
                reconcile_res = msgs.ProtoOAReconcileRes()
                reconcile_res.ParseFromString(msg.payload)
                positions = []
                for p in reconcile_res.position:
                    side = "BUY" if p.tradeData.tradeSide == 1 else "SELL"
                    sym_name = self._service.symbol_name_for_id(p.tradeData.symbolId)
                    raw_volume = int(p.tradeData.volume or 0)
                    volume_lots = self._service.protocol_volume_to_lots(
                        raw_volume,
                        sym_name,
                        lot_size=self._service.lot_size_cents(sym_name),
                    )
                    # Surface the protection the broker actually holds. A stop
                    # inside the broker's stop level is dropped silently, so
                    # "requested" and "attached" are not the same thing.
                    positions.append({
                        "symbol": sym_name,
                        "symbol_id": p.tradeData.symbolId,
                        "side": side,
                        "quantity": volume_lots,
                        "volume_cents": raw_volume,
                        "entry_price": self._service.normalize_position_price(
                            getattr(p, "price", None),
                            sym_name,
                        ),
                        "unrealized_pnl": 0.0,
                        "stop_loss": float(p.stopLoss) if getattr(p, "stopLoss", 0) else None,
                        "take_profit": float(p.takeProfit) if getattr(p, "takeProfit", 0) else None,
                        "position_id": str(p.positionId),
                        "broker": "ctrader",
                    })
                self._service._positions = positions
                logger.info(f"cTrader positions reconciled: {len(positions)} open")
                self._subscribe_spots({p["symbol_id"] for p in positions})
                missing_specs = [
                    p["symbol_id"]
                    for p in positions
                    if p.get("symbol_id")
                    and not self._service._symbol_specs.get(str(p.get("symbol") or ""))
                ]
                if missing_specs:
                    self._request_symbol_specs(missing_specs)
                self._service._notify_positions_changed(positions)

            elif ptype == 2126:  # ProtoOAExecutionEvent
                ev = msgs.ProtoOAExecutionEvent()
                ev.ParseFromString(msg.payload)
                logger.info(f"cTrader execution event received: type={ev.executionType}")
                if ev.executionType in (2, 3):  # ORDER_ACCEPTED / ORDER_FILLED
                    oid = None
                    pid = None
                    if hasattr(ev, "order") and ev.order:
                        oid = str(getattr(ev.order, "orderId", "") or "") or None
                        pid = str(getattr(ev.order, "positionId", "") or "") or None
                    if not pid and hasattr(ev, "deal") and ev.deal:
                        pid = str(getattr(ev.deal, "positionId", "") or "") or None
                    self._service.signal_place_order_ack(
                        error=None, order_id=oid, position_id=pid,
                    )

                # Record deal execution details if present
                if hasattr(ev, "deal") and ev.deal and getattr(ev.deal, "dealId", 0):
                    deal = ev.deal
                    pid = str(getattr(deal, "positionId", "") or "").strip()
                    sym_id = getattr(deal, "symbolId", 0)
                    sym_name = self._service.symbol_name_for_id(sym_id)
                    raw_price = getattr(deal, "executionPrice", None)
                    exec_price = self._service.normalize_position_price(raw_price, sym_name) if raw_price else None
                    close_detail = getattr(deal, "closePositionDetail", None)
                    gross_profit = None
                    closed_volume = None
                    if close_detail:
                        raw_gp = getattr(close_detail, "grossProfit", None)
                        money_digits = getattr(close_detail, "moneyDigits", 2) or 2
                        if raw_gp is not None:
                            gross_profit = float(raw_gp) / (10 ** money_digits)
                        raw_vol = getattr(close_detail, "closedVolume", None)
                        if raw_vol:
                            closed_volume = self._service.protocol_volume_to_lots(
                                raw_vol, sym_name
                            )

                    deal_type = getattr(deal, "closePositionType", None)
                    if close_detail:
                        close_type_name = str(deal_type) if deal_type is not None else "CLOSED"
                    else:
                        close_type_name = "OPEN"
                    if hasattr(ev, "order") and ev.order and getattr(ev.order, "orderType", None) == 4:
                        close_type_name = "SL_TP"

                    self._service.record_deal({
                        "position_id": pid,
                        "deal_id": str(getattr(deal, "dealId", "")),
                        "symbol": sym_name,
                        "execution_price": exec_price,
                        "gross_profit": gross_profit,
                        "closed_volume": closed_volume,
                        "close_type": close_type_name,
                        "timestamp": getattr(deal, "executionTimestamp", int(time.time() * 1000)),
                    })

                # Refresh positions on fill/close
                req_rec = msgs.ProtoOAReconcileReq()
                req_rec.ctidTraderAccountId = self._creds.get("account_id", 0)
                self._send(req_rec, 2124)

            elif ptype == 2132:  # ProtoOAOrderErrorEvent
                err_ev = msgs.ProtoOAOrderErrorEvent()
                err_ev.ParseFromString(msg.payload)
                desc = f"{err_ev.errorCode} — {err_ev.description}"
                logger.error(f"cTrader Order Error: {desc}")
                self._service.signal_place_order_ack(error=desc)

            elif ptype == 2131:  # ProtoOASpotEvent (2128 is SubscribeSpotsRes)
                spot_ev = msgs.ProtoOASpotEvent()
                spot_ev.ParseFromString(msg.payload)
                sym_id = spot_ev.symbolId
                sym_name = self._service.symbol_name_for_id(sym_id)
                bid = self._service.decode_spotware_price(spot_ev.bid, sym_name) if spot_ev.bid else None
                ask = self._service.decode_spotware_price(spot_ev.ask, sym_name) if spot_ev.ask else None
                self._service._last_spots[sym_name] = {
                    "symbol": sym_name,
                    "symbol_id": sym_id,
                    "bid": bid,
                    "ask": ask,
                    "timestamp": int(time.time() * 1000)
                }

            elif ptype == 2138:  # ProtoOAGetTrendbarsRes
                tb_res = msgs.ProtoOAGetTrendbarsRes()
                tb_res.ParseFromString(msg.payload)
                sym_id = tb_res.symbolId
                sym_name = self._service.symbol_name_for_id(sym_id)
                digits = self._service.digits_for(sym_name)
                scale = float(self._service.SPOTWARE_PRICE_SCALE)
                bars = []
                for tb in tb_res.trendbar:
                    low = tb.low / scale
                    open_p = (tb.low + (tb.deltaOpen or 0)) / scale
                    high = (tb.low + (tb.deltaHigh or 0)) / scale
                    close_p = (tb.low + (tb.deltaClose or 0)) / scale
                    # Timestamp in milliseconds
                    ts = (tb.utcTimestampInMinutes or 0) * 60 * 1000
                    bars.append({
                        "timestamp": ts,
                        "time": int(ts / 1000),
                        "open": round(open_p, digits),
                        "high": round(high, digits),
                        "low": round(low, digits),
                        "close": round(close_p, digits),
                        "volume": float(tb.volume or 1.0)
                    })
                self._service._trendbar_cache[f"{sym_name}_{tb_res.period}"] = bars
                logger.info(f"cTrader received {len(bars)} trendbars for {sym_name} (period={tb_res.period})")
                waiter = self._service._trendbar_events.get(f"{sym_name}_{tb_res.period}")
                if waiter is not None:
                    waiter.set()

            elif ptype == 2146:  # ProtoOAGetTickDataRes
                tick_res = msgs.ProtoOAGetTickDataRes()
                tick_res.ParseFromString(msg.payload)
                ticks = []
                for t in tick_res.tickData:
                    ticks.append({
                        "timestamp": t.timestamp,
                        "tick": t.tick / 100000.0
                    })
                logger.info(f"cTrader received {len(ticks)} ticks")

            elif ptype == 51:  # ProtoHeartbeatEvent
                hb = ProtoHeartbeatEvent()
                self._send(hb, 51)

            elif ptype == 2142:  # ProtoOAErrorRes
                err = msgs.ProtoOAErrorRes()
                err.ParseFromString(msg.payload)
                code = str(getattr(err, "errorCode", "") or "")
                desc = str(getattr(err, "description", "") or "")
                logger.error(f"cTrader Protocol Error: {code} — {desc}")
                self._service._last_protocol_error = f"{code} — {desc}".strip(" —")
                if not CTraderService.protocol_error_should_disconnect(code, desc):
                    # Spotware is throttling us: serve stale trendbar cache and
                    # stop making fresh requests for a minute. Do not drop the
                    # socket — a disconnect re-auths, reloads the symbol catalog,
                    # and reconciles, which is what piles onto the limiter.
                    self._service._rate_limited_until = time.time() + 60
                    logger.warning("cTrader rate-limited — 60s request cooldown, session kept")
                    return
                self._service._auth_event.set()
                if self.transport:
                    try:
                        self.transport.loseConnection()
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"cTrader protocol handle error (ptype={ptype}): {e}")

    def _send_account_auth(self) -> None:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs

        req = msgs.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self._creds.get("account_id", 0)
        req.accessToken = self._creds.get("access_token", "")
        logger.info("cTrader sending Account Auth (2102) for %s", req.ctidTraderAccountId)
        self._send(req, 2102)

    def connectionLost(self, reason):
        logger.warning(f"cTrader connection lost: {reason}")
        was_authenticated = self._service._authenticated
        self._service._connected = False
        self._service._authenticated = False
        self._service._protocol = None
        # Only retry after a live session drops. Retrying failed account-auth
        # stacks TCP sessions and Spotware answers CANT_ROUTE_REQUEST.
        if was_authenticated and not self._service._intentional_disconnect:
            self._service._schedule_reconnect()


class CTraderService(BrokerService):
    """
    cTrader Open API Broker implementation.
    Adheres strictly to the BrokerService interface.
    """

    # Symbol normalization map
    SYMBOL_MAP = {
        "EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD", "USDJPY=X": "USDJPY",
        "AUDUSD=X": "AUDUSD", "USDCAD=X": "USDCAD", "USDCHF=X": "USDCHF",
        "NZDUSD=X": "NZDUSD", "EURGBP=X": "EURGBP", "EURJPY=X": "EURJPY",
        "GBPJPY=X": "GBPJPY", "EURAUD=X": "EURAUD", "GBPAUD=X": "GBPAUD",
        "NZDJPY=X": "NZDJPY", "USDNOK=X": "USDNOK", "CHFJPY=X": "CHFJPY",
        "BTC-USD": "BTCUSD", "ETH-USD": "ETHUSD", "SOL-USD": "SOLUSD",
        "BNB-USD": "BNBUSD", "XRP-USD": "XRPUSD",
    }

    # Standard known symbol IDs (fallback before dynamic catalog is loaded)
    DEFAULT_SYMBOL_IDS = {
        "EURUSD": 1, "GBPUSD": 2, "USDJPY": 3, "USDCHF": 4,
        "AUDUSD": 5, "USDCAD": 6, "NZDUSD": 7, "EURGBP": 8,
        "EURJPY": 9, "GBPJPY": 10, "EURCHF": 11, "EURAUD": 12,
        "GBPCHF": 13, "GBPAUD": 14, "AUDJPY": 15, "AUDNZD": 16,
        "CADJPY": 17, "CHFJPY": 18, "EURCAD": 19, "EURNZD": 20,
        "XAUUSD": 21, "XAGUSD": 22,
        "BTCUSD": 36, "ETHUSD": 37, "SOLUSD": 38, "XRPUSD": 39, "BNBUSD": 40,
    }

    # Standard pip size for SL/TP offset calculation
    PIP_SIZE = {
        "EURUSD": Decimal("0.0001"), "GBPUSD": Decimal("0.0001"), "AUDUSD": Decimal("0.0001"),
        "NZDUSD": Decimal("0.0001"), "USDCAD": Decimal("0.0001"), "USDCHF": Decimal("0.0001"),
        "EURGBP": Decimal("0.0001"), "USDJPY": Decimal("0.01"),   "EURJPY": Decimal("0.01"),
        "GBPJPY": Decimal("0.01"),   "AUDJPY": Decimal("0.01"),   "CADJPY": Decimal("0.01"),
        "CHFJPY": Decimal("0.01"),   "NZDJPY": Decimal("0.01"),
        "USDNOK": Decimal("0.0001"),
        "XAUUSD": Decimal("0.01"),   "XAGUSD": Decimal("0.001"),
        "BTCUSD": Decimal("1.0"),    "ETHUSD": Decimal("0.1"),    "SOLUSD": Decimal("0.01"),
        "BNBUSD": Decimal("0.01"),   "XRPUSD": Decimal("0.0001"),
    }

    DIGITS = {
        "EURUSD": 5, "GBPUSD": 5, "AUDUSD": 5, "NZDUSD": 5, "USDCAD": 5, "USDCHF": 5,
        "EURGBP": 5, "USDJPY": 3, "EURJPY": 3, "GBPJPY": 3, "AUDJPY": 3, "CADJPY": 3,
        "CHFJPY": 3, "NZDJPY": 3, "USDNOK": 5,
        "XAUUSD": 2, "XAGUSD": 3, "BTCUSD": 2, "ETHUSD": 2, "SOLUSD": 2, "BNBUSD": 2, "XRPUSD": 4
    }

    PIP_POSITION = {
        "EURUSD": 4, "GBPUSD": 4, "AUDUSD": 4, "NZDUSD": 4, "USDCAD": 4, "USDCHF": 4,
        "EURGBP": 4, "USDJPY": 2, "EURJPY": 2, "GBPJPY": 2, "AUDJPY": 2, "CADJPY": 2,
        "CHFJPY": 2, "NZDJPY": 2, "USDNOK": 4,
        "XAUUSD": 2, "XAGUSD": 2, "BTCUSD": 0, "ETHUSD": 1, "SOLUSD": 2, "BNBUSD": 2, "XRPUSD": 4
    }

    BASE_PRICES = {
        "EURUSD": 1.0850, "GBPUSD": 1.2950, "USDJPY": 154.20, "AUDUSD": 0.6550,
        "USDCAD": 1.3650, "USDCHF": 0.8850, "NZDUSD": 0.5950, "EURGBP": 0.8375,
        "EURJPY": 167.30, "GBPJPY": 199.70, "NZDJPY": 94.50, "CHFJPY": 180.00,
        "USDNOK": 9.35, "XAUUSD": 2500.00, "XAGUSD": 29.50,
        "BTCUSD": 64500.00, "ETHUSD": 2750.00, "SOLUSD": 155.00, "BNBUSD": 570.00, "XRPUSD": 0.5850
    }

    PERIOD_MAP = {
        "1T": 1, "10S": 1, "30S": 1,
        "1M": 1, "M1": 1,
        "2M": 2, "M2": 2,
        "3M": 3, "M3": 3,
        "4M": 4, "M4": 4,
        "5M": 5, "M5": 5,
        "15M": 6, "M15": 6,
        "30M": 7, "M30": 7,
        "1H": 8, "H1": 8,
        "4H": 9, "H4": 9,
        "12H": 10, "H12": 10,
        "1D": 11, "D1": 11,
        "1W": 12, "W1": 12,
        "1MO": 13, "MN1": 13
    }
    # Spotware ProtoOATrendbarPeriod seconds. Unknown enums fall back to M5.
    PERIOD_SECONDS = {
        1: 60, 2: 120, 3: 180, 4: 240, 5: 300,
        6: 900, 7: 1800, 8: 3600, 9: 14400, 10: 43200,
        11: 86400, 12: 604800, 13: 2592000,
    }
    TRENDBAR_LOOKBACK_CAP_MS = 7 * 86400 * 1000

    @classmethod
    def trendbar_lookback_ms(cls, period_enum: int, count: int) -> int:
        """Bound a live trendbar request to about ``count`` bars.

        Callers only keep the tail (``cached[-count:]``). Asking Spotware for
        a flat 7 days of M5 bars (~2000 rows) on every refresh is what trips
        BLOCKED_PAYLOAD_TYPE. Daily and weekly requests stay capped at 7 days.
        """
        sec = cls.PERIOD_SECONDS.get(int(period_enum), 300)
        bars = max(int(count), 1) + 2
        return min(bars * sec * 1000, cls.TRENDBAR_LOOKBACK_CAP_MS)

    @staticmethod
    def protocol_error_should_disconnect(code: str, desc: str) -> bool:
        """Rate limits keep the socket. Other protocol errors drop it."""
        if code == "BLOCKED_PAYLOAD_TYPE" or "rate limit" in (desc or "").lower():
            return False
        return True

    def __init__(self):
        self._connected = False
        self._authenticated = False
        self._dry_run = True
        self._protocol: Optional[CTraderProtocol] = None
        self._reactor_thread: Optional[threading.Thread] = None
        self._auth_event = threading.Event()
        self._account_id: Optional[int] = None
        self._symbol_ids: Dict[str, int] = dict(self.DEFAULT_SYMBOL_IDS)
        self._symbol_specs: Dict[str, Dict[str, Any]] = {}
        self._positions: List[Dict[str, Any]] = []
        self._trendbar_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._last_spots: Dict[str, Dict[str, Any]] = {}
        self._tick_cache: Dict[str, List[Dict[str, Any]]] = {}

        self.balance: float = 0.0
        self.equity: float = 0.0
        self.margin: float = 0.0
        self._leverage: float = 500.0
        self._broker_name: str = "cTrader"
        self._trader_login: int = 0
        self._reconnect_pending = False
        self._reconnect_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._intentional_disconnect = False
        self._next_connect_ok_at: float = 0.0
        self._reconnect_delay: float = 5.0
        self._last_protocol_error: Optional[str] = None
        self._rate_limited_until: float = 0.0
        self._host_override: Optional[str] = None
        self._retry_correct_host: bool = False
        self._last_order_error: Optional[str] = None
        self._order_ack_event = threading.Event()
        self._order_ack_lock = threading.Lock()
        self._order_ack_waiters: List[Dict[str, Any]] = []
        self._trendbar_last_req: Dict[str, float] = {}
        self._trendbar_events: Dict[str, threading.Event] = {}
        self._recent_deals: Dict[str, Dict[str, Any]] = {}
        self._on_positions_changed_callbacks: List[Any] = []
        self._sl_restore_sent: Dict[str, float] = {}

    # Spotware volume is in cents of a unit (0.01 of 1 unit of the base asset).
    # FX: 1.00 lot = 100,000 units = 10,000,000 protocol volume.
    # Metals use the broker lotSize (IC Markets silver = 1,000 oz, not COMEX 5,000).
    VOLUME_CENTS_PER_LOT = 10_000_000
    MIN_VOLUME_CENTS = 100_000  # 0.01 FX lot
    STEP_VOLUME_CENTS = 1_000
    CONTRACT_UNITS_PER_LOT = 100_000
    # IC Markets cTrader commodity sheet: gold 100 oz, silver 1,000 oz per 1.00 lot.
    # Treating XAGUSD as 5,000 oz made QuantTrade show -€6 while the desk showed -€114
    # on the same 12-cent move (1,000 oz * $0.12 ≈ $120 / €114).
    UNITS_PER_LOT: Dict[str, float] = {
        "XAUUSD": 100.0,
        "XAGUSD": 1_000.0,
        "GOLD": 100.0,
        "SILVER": 1_000.0,
    }
    MIN_VOLUME_CENTS_BY_SYMBOL: Dict[str, int] = {
        "XAUUSD": 100,     # 1 oz
        "XAGUSD": 1_000,   # 10 oz
        "GOLD": 100,
        "SILVER": 1_000,
    }

    @classmethod
    def lot_size_cents_for(cls, symbol: Optional[str] = None) -> int:
        name = cls.normalize_symbol_name(symbol) if symbol else ""
        units = cls.UNITS_PER_LOT.get(name)
        if units:
            return int(round(float(units) * 100.0))
        return cls.VOLUME_CENTS_PER_LOT

    @classmethod
    def min_volume_cents_for(cls, symbol: Optional[str] = None) -> int:
        name = cls.normalize_symbol_name(symbol) if symbol else ""
        return int(cls.MIN_VOLUME_CENTS_BY_SYMBOL.get(name, cls.MIN_VOLUME_CENTS))

    def lot_size_cents(self, symbol: Optional[str] = None) -> int:
        name = self.normalize_symbol_name(symbol) if symbol else ""
        spec = self._symbol_specs.get(name) or {}
        if spec.get("lot_size"):
            return int(spec["lot_size"])
        return self.lot_size_cents_for(symbol)

    def min_volume_cents(self, symbol: Optional[str] = None) -> int:
        name = self.normalize_symbol_name(symbol) if symbol else ""
        spec = self._symbol_specs.get(name) or {}
        if spec.get("min_volume"):
            return int(spec["min_volume"])
        return self.min_volume_cents_for(symbol)

    def step_volume_cents(self, symbol: Optional[str] = None) -> int:
        name = self.normalize_symbol_name(symbol) if symbol else ""
        spec = self._symbol_specs.get(name) or {}
        if spec.get("step_volume"):
            return int(spec["step_volume"])
        return self.STEP_VOLUME_CENTS

    @classmethod
    def units_per_lot(cls, symbol: str, lot_size_cents: Optional[int] = None) -> float:
        """Base-asset units in one standard lot for mark-to-market P&L."""
        if lot_size_cents:
            return float(lot_size_cents) / 100.0
        name = cls.normalize_symbol_name(symbol) if symbol else ""
        return float(cls.UNITS_PER_LOT.get(name, cls.CONTRACT_UNITS_PER_LOT))

    # Bid/ask/trendbar integers are always 1/100000 of a price unit (not 10^digits).
    SPOTWARE_PRICE_SCALE = 100_000
    MAX_FX_STOP_PIPS = 120
    PLACE_ORDER_SEND_TIMEOUT_SEC = 10.0
    PLACE_ORDER_ACK_TIMEOUT_SEC = 3.0
    # Broker stop level. Protection closer than this is silently discarded.
    MIN_FX_STOP_PIPS = float(os.getenv("CTRADER_MIN_STOP_PIPS", "10"))
    # Exotic JPY crosses (NZDJPY/CHFJPY) have a wider broker stop level than
    # majors. 10 JPY pips (0.10) is silently discarded; TP survives because
    # it is farther out. Live NZDJPY/CHFJPY at 21:33 UTC filled with TP and no SL.
    MIN_JPY_STOP_PIPS = float(os.getenv("CTRADER_MIN_JPY_STOP_PIPS", "30"))

    @classmethod
    def lots_to_protocol_volume(
        cls,
        lots: float,
        symbol: str = "",
        lot_size: Optional[int] = None,
        min_volume: Optional[int] = None,
        step_volume: Optional[int] = None,
    ) -> int:
        size = int(lot_size or cls.lot_size_cents_for(symbol))
        min_vol = int(min_volume or cls.min_volume_cents_for(symbol))
        step = int(step_volume or cls.STEP_VOLUME_CENTS)
        raw = int(round(float(lots) * size))
        raw = max(min_vol, raw)
        if step > 0:
            raw = int(round(raw / step) * step)
        return max(min_vol, raw)

    @classmethod
    def protocol_volume_to_lots(
        cls,
        raw: int,
        symbol: str = "",
        lot_size: Optional[int] = None,
    ) -> float:
        size = int(lot_size or cls.lot_size_cents_for(symbol))
        if size <= 0:
            return 0.0
        return float(raw) / size

    @classmethod
    def relative_stop_units(cls, current_price: float, target_price: float, digits: int) -> int:
        """Convert a price distance to Spotware relativeStopLoss/TP (1/100000 of price)."""
        diff = abs(Decimal(str(current_price)) - Decimal(str(target_price)))
        units = int((diff * Decimal(cls.SPOTWARE_PRICE_SCALE)).to_integral_value())
        step = 10 ** max(0, 5 - int(digits))
        units = (units // step) * step
        return units

    @classmethod
    def normalize_symbol_name(cls, sym: str) -> str:
        if not sym:
            return ""
        mapped = cls.SYMBOL_MAP.get(sym)
        if mapped:
            return mapped
        return cls.normalize_catalog_name(sym.replace("-USD", "USD").replace("=X", ""))

    @staticmethod
    def normalize_catalog_name(name: str) -> str:
        """IC Markets light-symbol names may be 'NZD/JPY' rather than 'NZDJPY'."""
        return (name or "").replace("/", "").replace("-", "").replace(" ", "").upper()

    @classmethod
    def digits_for(cls, symbol: str) -> int:
        """Tick decimals for rounding. Not used to decode Spotware integers."""
        sym = cls.normalize_symbol_name(symbol)
        if sym in cls.DIGITS:
            return cls.DIGITS[sym]
        if sym.endswith("JPY"):
            return 3
        if sym.startswith("XAU"):
            return 2
        if sym.startswith("XAG"):
            return 3
        return 5

    @classmethod
    def pip_size_for(cls, symbol: str) -> Decimal:
        """Pip size in price units. JPY quotes are 0.01, not the 0.0001 FX default."""
        sym = cls.normalize_symbol_name(symbol)
        if sym in cls.PIP_SIZE:
            return cls.PIP_SIZE[sym]
        if sym.endswith("JPY"):
            return Decimal("0.01")
        if sym.startswith("XAU"):
            return Decimal("0.01")
        if sym.startswith("XAG"):
            return Decimal("0.001")
        return Decimal("0.0001")

    @classmethod
    def pip_position_for(cls, symbol: str) -> int:
        sym = cls.normalize_symbol_name(symbol)
        if sym in cls.PIP_POSITION:
            return cls.PIP_POSITION[sym]
        return max(0, cls.digits_for(sym) - 1)

    @classmethod
    def normalize_position_price(cls, raw: Optional[float], symbol: str = "") -> float:
        """Decode reconcile position prices whether Spotware sent ints or floats."""
        if raw is None:
            return 0.0
        val = float(raw)
        if val <= 0:
            return 0.0
        # Spotware sends prices as integers scaled by 1/100000. Already-decoded
        # floats for FX/metals sit in a narrow band (≈0.5–250).
        if val > 1000:
            decoded = cls.decode_spotware_price(val, symbol)
            return float(decoded or 0.0)
        return val

    @classmethod
    def decode_spotware_price(cls, raw: Optional[float], symbol: str = "") -> Optional[float]:
        """Convert a Spotware integer price (always 1/100000) to a float."""
        if raw is None:
            return None
        price = float(raw) / float(cls.SPOTWARE_PRICE_SCALE)
        if symbol:
            return round(price, cls.digits_for(symbol))
        return price

    @classmethod
    def max_protective_distance(cls, symbol: str, entry: float) -> float:
        pip = float(cls.pip_size_for(symbol))
        entry_abs = abs(float(entry) or 0.0)
        return max(pip * cls.MAX_FX_STOP_PIPS, entry_abs * 0.012)

    @classmethod
    def is_plausible_stop(
        cls, symbol: str, entry: float, stop: float, direction: str = "",
    ) -> bool:
        """False when a DB stop is impossibly far from entry.

        CADJPY SELL @ 112.9 with SL 95.9 is ~1700 pips — a live 30-pip stop
        must not be overwritten by that row. Stops at/through entry are
        allowed (break-even / trail lock). Callers still clamp vs mark.
        """
        try:
            entry_f = float(entry)
            stop_f = float(stop)
        except (TypeError, ValueError):
            return False
        if entry_f <= 0 or stop_f <= 0:
            return False
        # Distance rejects garbage (CADJPY ~1700-pip "stop"). After
        # break-even the ratchet may lock at/through entry, so do not
        # require the stop to stay on the loss side of entry. Callers
        # still clamp vs the current mark before sending.
        return abs(stop_f - entry_f) <= cls.max_protective_distance(symbol, entry_f) * 1.05

    @classmethod
    def min_protective_distance(cls, symbol: str, entry: float) -> float:
        """Closest a stop may sit and still be accepted by the broker.

        IC Markets rejects protection inside its stop level, and does so
        silently: the order fills and the stop is simply absent. M5 ATR on a
        quiet pair produced 2 pip stops, so live positions ran with no stop
        at all while the wider take-profit survived.
        """
        pips = cls.MIN_FX_STOP_PIPS
        if cls.normalize_symbol_name(symbol).endswith("JPY"):
            pips = max(pips, cls.MIN_JPY_STOP_PIPS)
        return float(cls.pip_size_for(symbol)) * pips

    @classmethod
    def clamp_protective_prices(
        cls,
        symbol: str,
        entry: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        direction: Optional[str] = None,
    ) -> tuple[Optional[float], Optional[float]]:
        """Force SL/TP onto the correct side of entry and into a workable band.

        Levels too far out (thousands of pips on JPY) are capped; levels too
        close are pushed out to the broker minimum so they are not dropped.
        """
        if not entry:
            return stop_loss, take_profit
        max_dist = cls.max_protective_distance(symbol, entry)
        min_dist = min(cls.min_protective_distance(symbol, entry), max_dist)
        digits = cls.digits_for(symbol)
        side = (direction or "").upper()
        entry_f = float(entry)

        def _clamp(target: Optional[float], want_above: Optional[bool]) -> Optional[float]:
            if target is None:
                return None
            dist = abs(float(target) - float(entry))
            if want_above is None:
                sign = 1.0 if float(target) > entry_f else -1.0
            else:
                sign = 1.0 if want_above else -1.0
                if (float(target) > entry_f) is not want_above:
                    logger.warning(
                        "%s protective level %s is on the wrong side of entry %s; mirroring",
                        symbol, target, entry,
                    )
            bounded = min(max(dist, min_dist), max_dist)
            if bounded != dist:
                logger.warning(
                    "Adjusting %s protective distance %s -> %s (entry=%s min=%s max=%s)",
                    symbol, dist, bounded, entry, min_dist, max_dist,
                )
            return round(entry_f + sign * bounded, digits)

        if side in ("BUY", "LONG"):
            return _clamp(stop_loss, False), _clamp(take_profit, True)
        if side in ("SELL", "SHORT"):
            return _clamp(stop_loss, True), _clamp(take_profit, False)
        return _clamp(stop_loss, None), _clamp(take_profit, None)

    # ── Post-clamp geometry guards ──────────────────────────────────────────
    # The broker min-distance clamp can silently rewrite a trade into a
    # symmetric 1:1 coin-flip: on 2026-09-08 the live FX book measured a
    # median planned R:R of exactly 1.00, 20% win rate, -$1.13 expectancy
    # per trade. These guards REJECT deformed geometry instead of executing
    # it — not trading beats negative-expectancy trading.
    MIN_EFFECTIVE_RR = float(os.getenv("CTRADER_MIN_EFFECTIVE_RR", "1.5"))
    REJECT_SUBMIN_STOP = os.getenv("CTRADER_REJECT_SUBMIN_STOP", "true").lower() == "true"

    @classmethod
    def validate_protective_geometry(
        cls,
        symbol: str,
        entry: float,
        natural_stop_loss: Optional[float],
        natural_take_profit: Optional[float],
        clamped_stop_loss: Optional[float],
        clamped_take_profit: Optional[float],
        direction: Optional[str] = None,
    ) -> Optional[tuple[Optional[float], Optional[float]]]:
        """Validate clamped SL/TP against strategy-designed geometry.

        Returns the (sl, tp) pair to trade, or None to veto the entry.

        Rule A (sub-minimum stop): the strategy's natural stop sits inside the
        broker minimum distance, so the clamp had to invent a wider stop the
        strategy never designed for. The setup needs precision the broker
        cannot express — skip it.

        Rule B (effective R:R): after clamping, reward:risk must still meet
        MIN_EFFECTIVE_RR. The clamp widens tight stops AND tight targets to
        the same minimum, which produced the 1:1 churn cohort. We never widen
        the TP to compensate — the TP encodes the strategy thesis (mean,
        measured move), moving it breaks the thesis. Reject instead.
        """
        if not entry or clamped_stop_loss is None or clamped_take_profit is None:
            return clamped_stop_loss, clamped_take_profit

        entry_f = float(entry)
        pip = max(float(cls.pip_size_for(symbol)), 1e-9)
        min_dist = cls.min_protective_distance(symbol, entry_f)

        # Rule A — natural stop inside broker minimum
        if cls.REJECT_SUBMIN_STOP and natural_stop_loss is not None:
            natural_sl_dist = abs(float(natural_stop_loss) - entry_f)
            # tolerance: half a pip of rounding slack
            if 0 < natural_sl_dist < (min_dist - 0.5 * pip):
                logger.info(
                    "%s geometry veto: natural stop %.5f is %.1f pips from entry, "
                    "inside broker minimum %.1f pips — setup too tight to express",
                    symbol, float(natural_stop_loss),
                    natural_sl_dist / pip, min_dist / pip,
                )
                return None

        # Rule B — effective R:R after clamp
        sl_dist = abs(float(clamped_stop_loss) - entry_f)
        tp_dist = abs(float(clamped_take_profit) - entry_f)
        if sl_dist > 0:
            eff_rr = tp_dist / sl_dist
            if eff_rr < cls.MIN_EFFECTIVE_RR:
                logger.info(
                    "%s geometry veto: post-clamp R:R %.2f < %.2f "
                    "(SL %.1f pips, TP %.1f pips) — refusing 1:1 coin-flip",
                    symbol, eff_rr, cls.MIN_EFFECTIVE_RR,
                    sl_dist / pip, tp_dist / pip,
                )
                return None

        return clamped_stop_loss, clamped_take_profit

    @staticmethod
    def merge_symbol_catalog(
        defaults: Dict[str, int],
        catalog: Dict[str, int],
    ) -> Dict[str, int]:
        """Broker catalog wins. Drop default names whose IDs now belong to another symbol.

        IC Markets NZDJPY can reuse id 21 (our default XAUUSD). Keeping both names
        for the same id made reconcile report NZDJPY 94.54 as XAUUSD.
        """
        merged = dict(catalog)
        used_ids = set(merged.values())
        for name, sid in defaults.items():
            if name in merged:
                continue
            if sid in used_ids:
                continue
            merged[name] = sid
        return merged

    def priority_symbol_ids(self) -> List[int]:
        """Metals plus any open position — enough to size volume correctly."""
        wanted: set[int] = set()
        for name in ("XAUUSD", "XAGUSD", "GOLD", "SILVER"):
            sid = self._symbol_ids.get(name)
            if sid:
                wanted.add(int(sid))
        for pos in self._positions:
            sid = pos.get("symbol_id")
            if sid:
                wanted.add(int(sid))
        return sorted(wanted)

    def relabel_cached_symbols(self) -> None:
        """Re-apply catalog names after merge (reconcile can beat ProtoOASymbolsListRes)."""
        for pos in self._positions:
            sid = pos.get("symbol_id")
            if sid is None:
                continue
            pos["symbol"] = self.symbol_name_for_id(sid)
        relabeled_spots: Dict[str, Dict[str, Any]] = {}
        for spot in self._last_spots.values():
            sid = spot.get("symbol_id")
            if sid is None:
                continue
            name = self.symbol_name_for_id(sid)
            spot["symbol"] = name
            relabeled_spots[name] = spot
        if relabeled_spots:
            self._last_spots = relabeled_spots
        self.refresh_position_lots()

    def refresh_position_lots(self) -> None:
        """Recompute displayed lots from protocol cents after lotSize is known."""
        for pos in self._positions:
            raw = pos.get("volume_cents")
            symbol = pos.get("symbol") or ""
            if raw in (None, 0):
                continue
            lots = self.protocol_volume_to_lots(
                int(raw), symbol, lot_size=self.lot_size_cents(symbol)
            )
            pos["quantity"] = lots

    def symbol_name_for_id(self, symbol_id: int) -> str:
        """Resolve broker symbolId → name. Catalog-first merge keeps IDs unique."""
        try:
            sid = int(symbol_id)
        except (TypeError, ValueError):
            return str(symbol_id)
        for name, mapped in self._symbol_ids.items():
            if mapped == sid:
                return name
        return str(sid)

    @property
    def is_connected(self) -> bool:
        return self._connected and self._authenticated

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def _orders_live_blocked(self) -> bool:
        """True when TRADING_MODE is not live — reads stay up, orders must not."""
        return not live_ctrader_orders_allowed()

    def has_credentials(self) -> bool:
        """True when OAuth tokens or env vars are present for a live session."""
        tokens = token_store.get_tokens()
        client_id = tokens.get("client_id") or os.getenv("CTRADER_CLIENT_ID", "")
        access_token = (
            token_store.refresh_if_needed()
            or tokens.get("access_token")
            or os.getenv("CTRADER_ACCESS_TOKEN", "")
        )
        return bool(client_id and access_token)

    @staticmethod
    def pick_trader_account_id(
        accounts: List[Dict[str, Any]],
        wanted: int,
        prefer_live: bool,
    ) -> Optional[int]:
        """Choose a ctidTraderAccountId from ProtoOAGetAccountListByAccessTokenRes."""
        valid = [a for a in accounts if int(a.get("id") or 0) > 0]
        if not valid:
            return None
        ids = {int(a["id"]) for a in valid}
        if wanted in ids:
            return int(wanted)
        for acc in valid:
            if bool(acc.get("is_live")) == prefer_live:
                return int(acc["id"])
        return int(valid[0]["id"])

    def ensure_connected(self) -> bool:
        """Connect to cTrader when credentials exist.

        Without this, backend restarts leave the service in paper mode and
        n8n execute-candidate calls silently return status=simulated.
        """
        if self.is_connected and not self._dry_run:
            return True
        if not self.has_credentials():
            return False
        if time.time() < self._next_connect_ok_at:
            return False
        self._intentional_disconnect = False
        return self.connect()

    def _normalize_symbol(self, sym: str) -> str:
        """Map generic/yfinance symbols to cTrader symbol string."""
        return self.normalize_symbol_name(sym)

    def connect(self) -> bool:
        """Authenticate and connect to cTrader Open API."""
        with self._connect_lock:
            if self.is_connected and not self._dry_run:
                return True
            if time.time() < self._next_connect_ok_at:
                remaining = self._next_connect_ok_at - time.time()
                logger.warning("cTrader connect cooling down for %.0fs", remaining)
                return False
            return self._connect_unlocked()

    def _connect_unlocked(self) -> bool:
        tokens = token_store.get_tokens()
        client_id = tokens.get("client_id") or os.getenv("CTRADER_CLIENT_ID", "")
        force_refresh = "CANT_ROUTE" in str(self._last_protocol_error or "")
        access_token = (
            token_store.refresh_if_needed(force=force_refresh)
            or tokens.get("access_token")
            or os.getenv("CTRADER_ACCESS_TOKEN", "")
        )
        account_id = tokens.get("account_id") or int(os.getenv("CTRADER_ACCOUNT_ID", "0") or 0)

        if not client_id or not access_token:
            logger.warning("cTrader credentials missing. Operating in paper/dry-run mode.")
            return False

        # Host security gate: paper mode defaults to demo host.
        # TRADING_MODE=paper must never select live.ctraderapi.com even when
        # CTRADER_PAPER_MODE=false and CTRADER_LIVE_CONFIRM is set.
        paper_mode = (
            os.getenv("CTRADER_PAPER_MODE", os.getenv("CTRADE_PAPER_MODE", "true")).lower() == "true"
        )
        live_confirm = os.getenv("CTRADER_LIVE_CONFIRM", os.getenv("CTRADE_LIVE_CONFIRM", "")).strip()
        is_live = (
            (not paper_mode)
            and (live_confirm == "I_UNDERSTAND")
            and live_ctrader_orders_allowed()
        )
        if self._host_override == "live.ctraderapi.com" and not is_live:
            logger.error("Ignoring cTrader live host override without live confirmation")
            self._host_override = None
        host = self._host_override or ("live.ctraderapi.com" if is_live else "demo.ctraderapi.com")
        port = 5035
        if self._host_override:
            logger.info("cTrader using host override %s", host)

        self._auth_event.clear()
        self._authenticated = False
        self._connected = False
        self._dry_run = False
        self._intentional_disconnect = False
        self._account_id = account_id

        creds = {
            "client_id": client_id,
            "client_secret": tokens.get("client_secret") or os.getenv("CTRADER_CLIENT_SECRET", ""),
            "access_token": access_token,
            "account_id": account_id,
            "host": host,
            "port": port,
        }

        try:
            from twisted.internet import reactor, ssl as tssl

            class _TwistedFactory:
                noisy = False
                numPorts = 0
                def doStart(self): self.numPorts += 1
                def doStop(self): self.numPorts -= 1
                def startedConnecting(self, connector): pass

                def __init__(self, svc, c):
                    self._svc = svc
                    self._c = c

                def buildProtocol(self, addr):
                    from twisted.protocols.basic import Int32StringReceiver
                    svc = self._svc
                    proto = CTraderProtocol(svc, self._c)

                    class _Int32Proto(Int32StringReceiver):
                        MAX_LENGTH = 15_000_000
                        def connectionMade(self):
                            svc._connected = True
                            proto.connectionMade(self.transport)
                        def dataReceived(self, data):
                            proto.dataReceived(data)
                        def connectionLost(self, reason):
                            proto.connectionLost(reason.getErrorMessage())

                    return _Int32Proto()

                def clientConnectionFailed(self, connector, reason):
                    logger.error(f"cTrader connection failed: {reason.getErrorMessage()}")
                    self._svc._auth_event.set()

                def clientConnectionLost(self, connector, reason):
                    logger.warning(f"cTrader connection dropped: {reason.getErrorMessage()}")
                    self._svc._connected = False
                    self._svc._authenticated = False

            factory = _TwistedFactory(self, creds)
            ctx = tssl.CertificateOptions(verify=False)

            if reactor.running:
                reactor.callFromThread(lambda: reactor.connectSSL(host, port, factory, ctx))
            else:
                def run():
                    reactor.callLater(0, lambda: reactor.connectSSL(host, port, factory, ctx))
                    reactor.run(installSignalHandlers=False)

                self._reactor_thread = threading.Thread(target=run, daemon=True, name="ctrader-reactor")
                self._reactor_thread.start()

            logger.info("Awaiting cTrader authentication (15s timeout)...")
            self._auth_event.wait(timeout=15.0)

            if self._authenticated:
                self._reconnect_pending = False
                self._reconnect_delay = 5.0
                self._next_connect_ok_at = 0.0
                self._last_protocol_error = None
                self._retry_correct_host = False
                logger.info("cTrader connected and authenticated successfully!")
                return True
            if self._retry_correct_host and self._host_override:
                self._retry_correct_host = False
                self._next_connect_ok_at = 0.0
                logger.info("cTrader retrying on %s after account-list host mismatch", self._host_override)
                return self._connect_unlocked()
            self._mark_connect_failure("authentication timed out")
            return False

        except ImportError:
            logger.info("Twisted networking stack not installed locally. Operating in simulated paper mode.")
            self._dry_run = True
            return True
        except Exception as e:
            logger.error(f"cTrader connect error: {e}")
            self._mark_connect_failure(str(e))
            return False

    def _mark_connect_failure(self, reason: str) -> None:
        self._dry_run = True
        delay = 60.0 if "CANT_ROUTE" in str(self._last_protocol_error or reason) else self._reconnect_delay
        self._next_connect_ok_at = time.time() + delay
        self._reconnect_delay = min(max(delay, self._reconnect_delay) * 2.0, 180.0)
        logger.warning(
            "cTrader authentication failed (%s) — paper/dry-run; retry in %.0fs",
            reason or self._last_protocol_error or "unknown",
            delay,
        )

    def _schedule_reconnect(self):
        """Auto-reconnect with exponential backoff on unexpected disconnect."""
        if self._intentional_disconnect:
            return
        with self._reconnect_lock:
            if self._reconnect_pending:
                return
            self._reconnect_pending = True

        def _worker():
            delay = 5
            time.sleep(delay)
            if not self.is_connected and not self._intentional_disconnect:
                logger.info("cTrader auto-reconnect attempting recovery...")
                self._dry_run = False
                self.connect()
            self._reconnect_pending = False

        threading.Thread(target=_worker, daemon=True, name="ctrader-reconnect").start()

    def disconnect(self) -> None:
        """Gracefully disconnect from cTrader and return to paper mode."""
        self._intentional_disconnect = True
        self._connected = False
        self._authenticated = False
        self._protocol = None
        self._dry_run = True
        logger.info("cTrader disconnected — paper mode enabled")

    def get_balance(self) -> Dict[str, Any]:
        """Fetch current balance, equity, and margin."""
        return {
            "balance": self.balance,
            "available": self.balance - self.margin,
            "equity": self.equity,
            "margin_used": self.margin,
            "unrealized_pnl": 0.0,
            "broker": "ctrader",
        }

    def ensure_spot_quotes(self, symbols: List[str]) -> None:
        """Subscribe to spot streams for symbols that need a live mark."""
        if self._dry_run or not self.is_connected or self._protocol is None:
            return
        wanted: set[int] = set()
        for raw_sym in symbols:
            sym = self._normalize_symbol(raw_sym)
            sid = self._symbol_ids.get(sym)
            if sid:
                wanted.add(int(sid))
        if wanted:
            self._protocol._subscribe_spots(wanted)

    def get_spread_pips(self, symbol: str) -> Optional[float]:
        """Live bid/ask spread in pips; None when no spot has streamed yet."""
        spot = self._last_spots.get(self._normalize_symbol(symbol)) or self._last_spots.get(
            self.normalize_symbol_name(symbol)
        )
        if not spot:
            return None
        bid, ask = spot.get("bid"), spot.get("ask")
        if not bid or not ask:
            return None
        pip = max(float(self.pip_size_for(symbol)), 1e-9)
        return abs(float(ask) - float(bid)) / pip

    def get_mark_price(self, symbol: str, side: Optional[str] = None) -> Optional[float]:
        """Last streamed price for a symbol.

        A position is closed against the opposite side of the book, so a BUY
        exits on the bid and a SELL exits on the ask. Without a side, return the
        mid. Returns None when no spot has been streamed yet, so callers can
        record "unknown" instead of inventing a price.
        """
        spot = self._last_spots.get(self._normalize_symbol(symbol)) or self._last_spots.get(
            self.normalize_symbol_name(symbol)
        )
        if not spot:
            return None
        bid = spot.get("bid")
        ask = spot.get("ask")
        wanted = (side or "").upper()
        if wanted in ("BUY", "LONG"):
            price = bid or ask
        elif wanted in ("SELL", "SHORT"):
            price = ask or bid
        elif bid and ask:
            price = (float(bid) + float(ask)) / 2
        else:
            price = bid or ask
        return float(price) if price else None

    def quote_to_usd_rate(self, symbol: str, mark: float) -> Optional[float]:
        """Rate that converts this pair's quote currency into USD.

        Raw P&L lands in the quote currency, so a USDJPY move reads in yen
        while EURUSD reads in dollars. Showing both in one column made a
        -490 JPY position look ~150x worse than a -3 USD one.
        """
        sym = self.normalize_symbol_name(symbol)
        if len(sym) != 6:
            return None
        base, quote = sym[:3], sym[3:]
        if quote == "USD":
            return 1.0
        if base == "USD":
            # Quote per USD is the pair itself (USDJPY -> 159.7 JPY per USD).
            return 1.0 / mark if mark else None
        # Cross pair: price the quote currency against USD directly.
        direct = self.get_mark_price(f"{quote}USD")
        if direct:
            return float(direct)
        inverse = self.get_mark_price(f"USD{quote}")
        if inverse:
            return 1.0 / float(inverse)
        return None

    def register_positions_callback(self, callback: Any) -> None:
        """Register a callback to be invoked when positions are reconciled or updated."""
        if callback not in self._on_positions_changed_callbacks:
            self._on_positions_changed_callbacks.append(callback)

    def unregister_positions_callback(self, callback: Any) -> None:
        """Unregister a positions callback."""
        if callback in self._on_positions_changed_callbacks:
            self._on_positions_changed_callbacks.remove(callback)

    def _notify_positions_changed(self, positions: List[Dict[str, Any]]) -> None:
        """Notify registered callbacks of positions change."""
        for cb in list(self._on_positions_changed_callbacks):
            try:
                cb(positions)
            except Exception as exc:
                logger.warning(f"Error in cTrader positions callback: {exc}")

    def record_deal(self, deal_info: Dict[str, Any]) -> None:
        """Cache recent deal execution info for position reconciliation."""
        pid = str(deal_info.get("position_id") or "").strip()
        sym = str(deal_info.get("symbol") or "").strip().upper()
        if pid:
            self._recent_deals[pid] = deal_info
        if sym:
            self._recent_deals[sym] = deal_info
        # Limit cache size to prevent memory leak
        if len(self._recent_deals) > 200:
            keys = list(self._recent_deals.keys())[:50]
            for k in keys:
                self._recent_deals.pop(k, None)

    def get_recent_deal(
        self,
        position_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return cached deal execution details if available."""
        if position_id and str(position_id) in self._recent_deals:
            return self._recent_deals[str(position_id)]
        if symbol:
            sym = self._normalize_symbol(symbol)
            if sym in self._recent_deals:
                return self._recent_deals[sym]
        return None

    def get_exit_price(
        self,
        symbol: str,
        direction: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> Optional[float]:
        """Best-effort exit price for a just-closed position.

        1. Checks recent deal execution price for this position_id / symbol.
        2. Falls back to the latest live mark / spot quote.
        """
        sym = self._normalize_symbol(symbol)
        if position_id:
            deal = self.get_recent_deal(position_id=position_id)
            if deal and deal.get("execution_price"):
                return float(deal["execution_price"])
        deal = self.get_recent_deal(symbol=sym)
        if deal and deal.get("execution_price"):
            return float(deal["execution_price"])
        mark = self.get_mark_price(sym, direction)
        if mark and mark > 0:
            return float(mark)
        return None

    def get_positions(self, raise_on_error: bool = False) -> List[Dict[str, Any]]:
        """Return all open positions across cTrader, marked to the latest spot.

        `raise_on_error` exists for parity with the Binance broker so callers
        can treat the two interchangeably.
        """
        positions = []
        for pos in self._positions:
            enriched = dict(pos)
            symbol = str(enriched.get("symbol") or "")
            mark = self.get_mark_price(symbol, enriched.get("side"))
            entry = float(enriched.get("entry_price") or 0)
            volume_cents = enriched.get("volume_cents")
            if volume_cents:
                lots = self.protocol_volume_to_lots(
                    int(volume_cents), symbol, lot_size=self.lot_size_cents(symbol)
                )
                enriched["quantity"] = lots
                units = float(volume_cents) / 100.0
            else:
                lots = float(enriched.get("quantity") or 0)
                units = lots * self.units_per_lot(
                    symbol, lot_size_cents=self.lot_size_cents(symbol)
                )
            if mark and entry and units:
                direction = 1 if str(enriched.get("side", "")).upper() == "BUY" else -1
                quote_pnl = (mark - entry) * units * direction
                enriched["current_price"] = mark
                rate = self.quote_to_usd_rate(symbol, mark)
                if rate:
                    enriched["unrealized_pnl"] = round(quote_pnl * rate, 2)
                    enriched["pnl_currency"] = "USD"
                else:
                    enriched["unrealized_pnl"] = round(quote_pnl, 2)
                    enriched["pnl_currency"] = symbol[3:] if len(symbol) == 6 else ""
            else:
                # Always expose a mark for the dashboard — entry until spot arrives.
                enriched["current_price"] = mark if mark and mark > 0 else entry
            positions.append(enriched)
        return positions

    def signal_place_order_ack(
        self,
        error: Optional[str] = None,
        order_id: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> None:
        """Complete the oldest in-flight place_order waiter (per-request ack)."""
        self._last_order_error = error
        self._order_ack_event.set()
        with self._order_ack_lock:
            for waiter in self._order_ack_waiters:
                if not waiter["event"].is_set():
                    waiter["error"] = error
                    if order_id:
                        waiter["order_id"] = str(order_id)
                    if position_id:
                        waiter["position_id"] = str(position_id)
                    waiter["event"].set()
                    return

    def place_order(
        self,
        symbol: str = "",
        direction: str = "",
        action: str = "open",
        quantity: Optional[float] = None,
        volume: Optional[float] = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Place order with exact volume conversion and pip-offset calculations.
        Supports standard lots (1.0 lot = 100,000 units).
        """
        # Resolve argument aliases
        raw_sym = symbol or kwargs.get("yfinance_symbol", "")
        ct_symbol = self._normalize_symbol(raw_sym)
        from backend.services.multi_asset_bars import classify_symbol
        if classify_symbol(raw_sym or ct_symbol) == "crypto":
            logger.error(
                "cTrader refusing crypto symbol %s — crypto orders must use "
                "binance_futures_live, not the default cTrader session",
                raw_sym or ct_symbol,
            )
            return {
                "status": "error",
                "error": (
                    f"crypto symbol {raw_sym or ct_symbol} is not a cTrader market"
                ),
            }
        side = "BUY" if direction.upper() in ("BUY", "LONG") else "SELL"
        lots = volume if volume is not None else (quantity if quantity is not None else 1.0)

        # A close is a different protocol message. This method only ever sent
        # ProtoOANewOrderReq, so a caller asking to close silently doubled the
        # position instead of flattening it.
        if str(action).lower() == "close":
            held = next(
                (p for p in self._positions if str(p.get("symbol")) == ct_symbol),
                None,
            )
            if not held:
                return {
                    "status": "already_flat",
                    "broker": "ctrader",
                    "symbol": ct_symbol,
                }
            return self.close_position(
                position_id=held.get("position_id"),
                symbol=ct_symbol,
                volume=lots if quantity is not None or volume is not None else held.get("quantity"),
            )
        current_price = price or kwargs.get("current_price")
        stop_loss_price = stop_loss or kwargs.get("stop_loss_price")
        take_profit_price = take_profit or kwargs.get("take_profit_price")
        if current_price:
            stop_loss_price, take_profit_price = self.clamp_protective_prices(
                ct_symbol, float(current_price), stop_loss_price, take_profit_price,
                direction=side,
            )

        # Simulated fallback in dry-run or when TRADING_MODE is not live
        if self._orders_live_blocked() or self._dry_run or not self.is_connected or self._protocol is None:
            logger.info(f"[PAPER] cTrader: {side} {ct_symbol} lots={lots:.2f}")
            return {
                "status": "simulated",
                "broker": "ctrader:paper",
                "symbol": ct_symbol,
                "direction": side,
                "quantity": lots,
                "price": float(current_price or 1.0),
                "order_id": f"sim_ct_{int(time.time() * 1000)}",
            }

        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            from twisted.internet import reactor

            symbol_id = self._symbol_ids.get(ct_symbol)
            if not symbol_id:
                logger.error(f"cTrader symbol ID not found for {ct_symbol}")
                return {"status": "error", "error": f"Unknown symbol {ct_symbol}"}

            # Spotware volume is cents of a unit. Metals use broker lotSize,
            # not the FX 10_000_000 cents/lot scale.
            raw_volume = self.lots_to_protocol_volume(
                lots,
                ct_symbol,
                lot_size=self.lot_size_cents(ct_symbol),
                min_volume=self.min_volume_cents(ct_symbol),
                step_volume=self.step_volume_cents(ct_symbol),
            )
            digits = self.digits_for(ct_symbol)

            order_req = msgs.ProtoOANewOrderReq()
            order_req.ctidTraderAccountId = self._account_id or 0
            order_req.symbolId = symbol_id
            order_req.orderType = 1  # MARKET
            order_req.tradeSide = 1 if side == "BUY" else 2
            order_req.volume = raw_volume

            # relativeStopLoss/TP are 1/100000 of a price unit, aligned to symbol digits
            if current_price and stop_loss_price:
                sl_units = self.relative_stop_units(float(current_price), float(stop_loss_price), digits)
                step = 10 ** max(0, 5 - int(digits))
                min_units = int(round(
                    self.min_protective_distance(ct_symbol, float(current_price))
                    * self.SPOTWARE_PRICE_SCALE
                ))
                min_units = max(step, (min_units // step) * step)
                if 0 < sl_units < min_units:
                    sl_units = min_units
                if sl_units > 0:
                    order_req.relativeStopLoss = sl_units
                else:
                    # Sending the entry anyway would open an unprotected
                    # position, which is how live forex ended up running naked.
                    logger.critical(
                        f"Refusing {ct_symbol} entry: stop loss {stop_loss_price} "
                        f"rounds to zero distance from {current_price}"
                    )
                    return {
                        "status": "error",
                        "error": "stop loss too close to entry to encode",
                        "symbol": ct_symbol,
                    }

            if current_price and take_profit_price:
                tp_units = self.relative_stop_units(float(current_price), float(take_profit_price), digits)
                if tp_units > 0:
                    order_req.relativeTakeProfit = tp_units

            # Final geometry gate at execution price: entry drift between scan
            # and fill can shrink one leg and break the designed R:R even
            # though both levels passed validation at scan time.
            if current_price and stop_loss_price and take_profit_price:
                sl_d = abs(float(stop_loss_price) - float(current_price))
                tp_d = abs(float(take_profit_price) - float(current_price))
                if sl_d > 0 and (tp_d / sl_d) < self.MIN_EFFECTIVE_RR:
                    logger.critical(
                        f"Refusing {ct_symbol} entry: effective R:R {tp_d / sl_d:.2f} < "
                        f"{self.MIN_EFFECTIVE_RR:.2f} at execution price {current_price} "
                        f"(entry drift broke geometry)"
                    )
                    return {
                        "status": "error",
                        "error": "effective R:R below minimum at execution price",
                        "symbol": ct_symbol,
                    }

            send_done = threading.Event()
            result = {"status": "pending"}
            waiter = {
                "event": threading.Event(),
                "error": None,
                "order_id": None,
                "position_id": None,
            }
            self._last_order_error = None
            self._order_ack_event.clear()
            with self._order_ack_lock:
                self._order_ack_waiters.append(waiter)

            def _send():
                try:
                    result["dispatched"] = True
                    self._protocol._send(order_req, 2106)
                    result["status"] = "sent"
                    result["broker"] = "ctrader:live"
                    result["symbol"] = ct_symbol
                    result["direction"] = side
                    result["quantity"] = lots
                    logger.info(f"cTrader live order dispatched: {side} {ct_symbol} (volume={raw_volume})")
                except Exception as e:
                    result["status"] = "error"
                    result["error"] = str(e)
                finally:
                    send_done.set()

            try:
                reactor.callFromThread(_send)
                send_done.wait(timeout=self.PLACE_ORDER_SEND_TIMEOUT_SEC)
                if result.get("status") == "pending":
                    # Reactor may still send. Never classify this as error —
                    # callers treat failure as retryable and would double size.
                    result["status"] = "sent"
                    result["broker"] = "ctrader:live"
                    result["symbol"] = ct_symbol
                    result["direction"] = side
                    result["quantity"] = lots
                    result["ack"] = "dispatch_timeout_do_not_retry"
                    logger.warning(
                        "cTrader place_order dispatch timed out for %s; "
                        "treating as sent/unconfirmed (do not retry)",
                        ct_symbol,
                    )
                if result.get("status") == "sent":
                    waiter["event"].wait(timeout=self.PLACE_ORDER_ACK_TIMEOUT_SEC)
                    if waiter["error"]:
                        result["status"] = "error"
                        result["error"] = waiter["error"]
                        logger.error(f"cTrader order rejected: {waiter['error']}")
                    else:
                        if waiter["order_id"]:
                            result["order_id"] = waiter["order_id"]
                        if waiter["position_id"]:
                            result["position_id"] = waiter["position_id"]
                        if not waiter["event"].is_set():
                            result["ack"] = result.get("ack") or "ack_timeout"
                            logger.warning(
                                "cTrader place_order ack timed out for %s; "
                                "keeping sent (do not retry)",
                                ct_symbol,
                            )
                return result
            finally:
                with self._order_ack_lock:
                    if waiter in self._order_ack_waiters:
                        self._order_ack_waiters.remove(waiter)

        except Exception as e:
            logger.error(f"cTrader place_order error: {e}")
            return {"status": "error", "error": str(e)}

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel pending order by ID."""
        if self._orders_live_blocked() or self._dry_run or not self.is_connected or self._protocol is None:
            return {
                "success": True,
                "simulated": True,
                "message": "Cancel skipped: TRADING_MODE is not live" if self._orders_live_blocked() else "Order cancelled in paper mode",
                "order_id": order_id,
            }

        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            from twisted.internet import reactor

            req = msgs.ProtoOACancelOrderReq()
            req.ctidTraderAccountId = self._account_id or 0
            req.orderId = int(order_id) if str(order_id).isdigit() else 0

            reactor.callFromThread(lambda: self._protocol._send(req, 2108))
            return {"success": True, "message": "Cancel request sent", "order_id": order_id}
        except Exception as e:
            return {"success": False, "message": str(e), "order_id": order_id}

    @classmethod
    def missing_stop_price(
        cls,
        symbol: str,
        side: str,
        entry: float,
        mark: Optional[float] = None,
    ) -> float:
        """Absolute stop for a live position that the broker dropped."""
        entry_f = float(entry)
        mark_f = float(mark) if mark and float(mark) > 0 else entry_f
        min_dist = cls.min_protective_distance(symbol, entry_f)
        digits = cls.digits_for(symbol)
        if str(side).upper() in ("BUY", "LONG"):
            sl = min(entry_f - min_dist, mark_f - min_dist)
        else:
            sl = max(entry_f + min_dist, mark_f + min_dist)
        sl, _ = cls.clamp_protective_prices(symbol, entry_f, sl, None, direction=side)
        return float(round(sl, digits))

    def amend_position_sltp(
        self,
        position_id: str | int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Set absolute SL/TP on an open cTrader position (ProtoOAAmendPositionSLTPReq)."""
        if self._orders_live_blocked() or self._dry_run or not self.is_connected or self._protocol is None:
            return {
                "status": "simulated",
                "position_id": str(position_id),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        matched = next(
            (p for p in self._positions if str(p.get("position_id")) == str(position_id)),
            None,
        )
        if stop_loss:
            if not matched:
                return {
                    "status": "error",
                    "error": "cannot validate stop without cached position",
                    "position_id": str(position_id),
                    "stop_loss": stop_loss,
                }
            entry = float(matched.get("entry_price") or 0)
            symbol = str(matched.get("symbol") or "")
            side = str(matched.get("side") or matched.get("direction") or "")
            if entry <= 0 or not symbol:
                return {
                    "status": "error",
                    "error": "cannot validate stop without entry/symbol",
                    "position_id": str(position_id),
                    "stop_loss": stop_loss,
                }
            if not self.is_plausible_stop(symbol, entry, stop_loss, side):
                logger.error(
                    "Refusing implausible cTrader amend SL %s for %s %s @ %s",
                    stop_loss, side, symbol, entry,
                )
                return {
                    "status": "error",
                    "error": "implausible stop versus live entry",
                    "position_id": str(position_id),
                    "stop_loss": stop_loss,
                    "entry_price": entry,
                }
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            from twisted.internet import reactor

            req = msgs.ProtoOAAmendPositionSLTPReq()
            req.ctidTraderAccountId = self._account_id or 0
            req.positionId = int(position_id)
            if stop_loss:
                req.stopLoss = float(stop_loss)
            if take_profit:
                req.takeProfit = float(take_profit)
            payload_type = int(getattr(req, "payloadType", 0) or 2113)
            reactor.callFromThread(lambda: self._protocol._send(req, payload_type))
            logger.info(
                "cTrader amend SL/TP sent for %s sl=%s tp=%s",
                position_id,
                stop_loss,
                take_profit,
            )
            return {
                "status": "sent",
                "position_id": str(position_id),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        except Exception as e:
            logger.error("cTrader amend SL/TP failed for %s: %s", position_id, e)
            return {"status": "error", "error": str(e), "position_id": str(position_id)}

    def restore_missing_stops(self) -> List[Dict[str, Any]]:
        """Attach a broker-legal stop to any live position that has none."""
        restored: List[Dict[str, Any]] = []
        now = time.time()
        for pos in self.get_positions():
            if pos.get("stop_loss"):
                continue
            pid = str(pos.get("position_id") or "")
            if pid and now - self._sl_restore_sent.get(pid, 0) < 60:
                continue
            symbol = str(pos.get("symbol") or "")
            side = str(pos.get("side") or "BUY")
            entry = float(pos.get("entry_price") or 0)
            if not symbol or entry <= 0:
                continue
            sl = self.missing_stop_price(
                symbol, side, entry, pos.get("current_price")
            )
            res = self.amend_position_sltp(
                pos.get("position_id"),
                stop_loss=sl,
                take_profit=pos.get("take_profit"),
            )
            if pid:
                self._sl_restore_sent[pid] = now
            restored.append({
                "symbol": symbol,
                "position_id": pos.get("position_id"),
                "stop_loss": sl,
                "result": res.get("status"),
            })
        return restored

    def close_position(
        self,
        position_id: str | int,
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Close open position by cTrader position ID. Volume is required on the live protocol."""
        lots = volume if volume is not None else kwargs.get("quantity")
        matched = next(
            (p for p in self._positions if str(p.get("position_id")) == str(position_id)),
            None,
        )
        if lots is None and matched is not None:
            lots = matched.get("quantity")
        if not symbol and matched is not None:
            symbol = matched.get("symbol")

        if self._orders_live_blocked():
            # Paper + a live read socket must not rewrite the connected book
            # or report closed (the dashboard would then persist a SQL close).
            if self.is_connected and self._protocol is not None and not self._dry_run:
                return {
                    "status": "error",
                    "error": "cTrader live closes are disabled unless TRADING_MODE=live",
                    "position_id": position_id,
                    "symbol": symbol,
                }
            self._positions = [p for p in self._positions if str(p.get("position_id")) != str(position_id)]
            return {
                "status": "closed",
                "position_id": position_id,
                "symbol": symbol,
                "quantity": lots,
                "broker": "ctrader:paper",
            }

        if self._dry_run or not self.is_connected or self._protocol is None:
            self._positions = [p for p in self._positions if str(p.get("position_id")) != str(position_id)]
            return {
                "status": "closed",
                "position_id": position_id,
                "symbol": symbol,
                "quantity": lots,
                "broker": "ctrader:paper",
            }

        if lots is None:
            return {
                "status": "error",
                "error": "volume is required to close a live cTrader position",
                "position_id": position_id,
            }

        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            from twisted.internet import reactor

            req = msgs.ProtoOAClosePositionReq()
            req.ctidTraderAccountId = self._account_id or 0
            req.positionId = int(position_id) if str(position_id).isdigit() else 0
            if (
                matched
                and matched.get("volume_cents")
                and volume is None
                and kwargs.get("quantity") is None
            ):
                req.volume = int(matched["volume_cents"])
            else:
                req.volume = self.lots_to_protocol_volume(
                    float(lots),
                    symbol,
                    lot_size=self.lot_size_cents(symbol),
                    min_volume=self.min_volume_cents(symbol),
                    step_volume=self.step_volume_cents(symbol),
                )

            reactor.callFromThread(lambda: self._protocol._send(req, 2111))
            return {
                "status": "sent",
                "position_id": position_id,
                "symbol": symbol,
                "quantity": lots,
                "broker": "ctrader:live",
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "position_id": position_id}

    def get_symbol_specification(self, symbol: str) -> Dict[str, Any]:
        """Returns standard specification parameters for a symbol (digits, pip position, lot size, etc.)."""
        ct_symbol = self._normalize_symbol(symbol)
        digits = self.digits_for(ct_symbol)
        pip_pos = self.pip_position_for(ct_symbol)
        pip_size = float(self.pip_size_for(ct_symbol))
        tick_size = 1.0 / (10 ** digits)
        base_price = self.BASE_PRICES.get(ct_symbol)
        if base_price is None:
            base_price = 154.20 if ct_symbol.endswith("JPY") else 1.0

        # Determine base and quote assets
        if len(ct_symbol) == 6 and not any(ct_symbol.startswith(k) for k in ("BTC", "ETH", "SOL", "BNB", "XRP", "XAU", "XAG")):
            base_asset = ct_symbol[:3]
            quote_asset = ct_symbol[3:]
        elif any(ct_symbol.startswith(k) for k in ("BTC", "ETH", "SOL", "BNB", "XRP")):
            base_asset = ct_symbol.replace("USD", "")
            quote_asset = "USD"
        elif ct_symbol in ("XAUUSD", "XAGUSD"):
            base_asset = ct_symbol[:3]
            quote_asset = "USD"
        else:
            base_asset = ct_symbol
            quote_asset = "USD"

        return {
            "symbol": ct_symbol,
            "raw_symbol": symbol,
            "symbol_id": self._symbol_ids.get(ct_symbol, 0),
            "digits": digits,
            "pip_position": pip_pos,
            "pip_size": pip_size,
            "tick_size": tick_size,
            "lot_size": self.units_per_lot(
                ct_symbol, lot_size_cents=self.lot_size_cents(ct_symbol)
            ),
            "min_volume": self.min_volume_cents(ct_symbol),
            "max_volume": int(
                (self._symbol_specs.get(ct_symbol) or {}).get("max_volume")
                or 100_000_000
            ),
            "step_volume": self.step_volume_cents(ct_symbol),
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "base_price": base_price,
        }

    def calculate_pip_margin(
        self,
        symbol: str,
        lots: float,
        price: Optional[float] = None,
        leverage: float = 100.0,
        deposit_asset: str = "USD"
    ) -> Dict[str, Any]:
        """
        Calculates pip value, tick value, required margin, and lot volume conversion
        following OpenAPI.Net SymbolExtensions specifications.
        """
        spec = self.get_symbol_specification(symbol)
        curr_price = float(price) if price else float(spec["base_price"])
        pip_size = float(spec["pip_size"])
        tick_size = float(spec["tick_size"])
        lot_size = float(spec["lot_size"])
        # Floor is 0.01 lot in that instrument's units (FX=1000, gold=1, silver=50).
        # The old max(1000, ...) treated 0.01 gold as 10 lots and inflated metal margin.
        min_units = max(1, int(round(0.01 * lot_size)))
        volume_units = max(min_units, int(round(float(lots) * lot_size)))

        # Pip Value calculation based on quote asset vs deposit asset
        if spec["quote_asset"] == deposit_asset:
            pip_value = pip_size * volume_units
            tick_value = tick_size * volume_units
        elif spec["base_asset"] == deposit_asset and curr_price > 0:
            pip_value = (pip_size / curr_price) * volume_units
            tick_value = (tick_size / curr_price) * volume_units
        else:
            pip_value = pip_size * volume_units
            tick_value = tick_size * volume_units

        # Required Margin = (volume_units * curr_price) / leverage
        notional_value = volume_units * (curr_price if spec["base_asset"] != deposit_asset else 1.0)
        required_margin = notional_value / max(1.0, float(leverage))

        return {
            "symbol": spec["symbol"],
            "lots": float(lots),
            "volume_units": volume_units,
            "price": curr_price,
            "leverage": float(leverage),
            "pip_size": pip_size,
            "tick_size": tick_size,
            "pip_value": round(pip_value, 4),
            "tick_value": round(tick_value, 4),
            "notional_value": round(notional_value, 2),
            "required_margin": round(required_margin, 2),
            "deposit_asset": deposit_asset,
        }

    def _generate_synthetic_trendbars(
        self,
        ct_symbol: str,
        period: str,
        count: int,
    ) -> List[Dict[str, Any]]:
        """Paper-mode OHLCV generator. Never used on a live connected session."""
        import math
        import random

        digits = self.digits_for(ct_symbol)
        pip_size = float(self.pip_size_for(ct_symbol))
        base_p = self.BASE_PRICES.get(ct_symbol)
        if base_p is None:
            if ct_symbol.endswith("JPY"):
                base_p = 154.20 if ct_symbol.startswith("USD") else 94.50
            else:
                base_p = 1.0850

        step_seconds = 300  # default 5m
        p_up = period.upper()
        if "1M" in p_up or p_up == "M1":
            step_seconds = 60
        elif "15M" in p_up or p_up == "M15":
            step_seconds = 900
        elif "30M" in p_up or p_up == "M30":
            step_seconds = 1800
        elif "1H" in p_up or p_up == "H1":
            step_seconds = 3600
        elif "4H" in p_up or p_up == "H4":
            step_seconds = 14400
        elif "1D" in p_up or p_up == "D1":
            step_seconds = 86400
        elif "1W" in p_up or p_up == "W1":
            step_seconds = 604800

        now_sec = int(time.time())
        aligned_end = (now_sec // step_seconds) * step_seconds
        rnd = random.Random(hash(ct_symbol) + step_seconds)

        bars = []
        p = base_p
        volatility = pip_size * 8.0

        for i in range(count, 0, -1):
            bar_time = aligned_end - (i * step_seconds)
            drift = math.sin(i / 15.0) * (pip_size * 4.0)
            noise = (rnd.random() - 0.49) * volatility
            open_price = p
            close_price = max(pip_size * 10, open_price + drift + noise)
            high_price = max(open_price, close_price) + rnd.random() * (volatility * 0.6)
            low_price = min(open_price, close_price) - rnd.random() * (volatility * 0.6)
            volume = round(rnd.uniform(50, 500), 2)

            p = close_price
            bars.append({
                "timestamp": bar_time * 1000,
                "time": bar_time,
                "open": round(open_price, digits),
                "high": round(high_price, digits),
                "low": round(low_price, digits),
                "close": round(close_price, digits),
                "volume": volume,
                "synthetic": True,
            })

        return bars

    def get_trendbars(
        self,
        symbol: str,
        period: str = "M5",
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
        count: int = 120
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV trendbars for a symbol and timeframe period.

        Live sessions use ProtoOAGetTrendbarsReq and return cached broker data only.
        When live and no real bars are available (timeout, missing symbol id, empty
        cache), returns an empty list so callers skip the symbol instead of trading
        on fabricated prices.

        Paper/dry-run mode may synthesize bars; those rows include ``synthetic: true``.
        """
        ct_symbol = self._normalize_symbol(symbol)
        period_enum = self.PERIOD_MAP.get(period.upper(), 5)
        cache_key = f"{ct_symbol}_{period_enum}"
        live_mode = not self._dry_run and self.is_connected and self._protocol is not None

        if live_mode:
            if time.time() < self._rate_limited_until:
                cached = self._trendbar_cache.get(cache_key) or []
                return cached[-count:] if cached else []
            last = self._trendbar_last_req.get(cache_key, 0)
            if time.time() - last < 45:
                cached = self._trendbar_cache.get(cache_key) or []
                if cached:
                    return cached[-count:]
            else:
                try:
                    from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
                    from twisted.internet import reactor

                    sym_id = self._symbol_ids.get(ct_symbol)
                    if not sym_id:
                        logger.warning(
                            "No cTrader symbol id for %s; skipping live trendbar request",
                            ct_symbol,
                        )
                    else:
                        now_ms = int(time.time() * 1000)
                        f_ts = from_ts or (now_ms - self.trendbar_lookback_ms(period_enum, count))
                        t_ts = to_ts or now_ms

                        req = msgs.ProtoOAGetTrendbarsReq()
                        req.ctidTraderAccountId = self._account_id or 0
                        req.symbolId = sym_id
                        req.period = period_enum
                        req.fromTimestamp = f_ts
                        req.toTimestamp = t_ts

                        # Register the waiter before sending so a fast response cannot
                        # arrive (and be missed) before we start waiting.
                        waiter = threading.Event()
                        self._trendbar_events[cache_key] = waiter
                        reactor.callFromThread(lambda: self._protocol._send(req, 2137))
                        self._trendbar_last_req[cache_key] = time.time()
                        waiter.wait(timeout=2.0)
                except Exception as e:
                    logger.warning(f"Live trendbar request failed: {e}")

            cached = self._trendbar_cache.get(cache_key) or []
            if cached:
                return cached[-count:]

            logger.warning(
                "Live trendbar fetch returned no data for %s %s; refusing synthetic fallback",
                ct_symbol,
                period,
            )
            return []

        cached = self._trendbar_cache.get(cache_key) or []
        if cached:
            return cached[-count:]

        return self._generate_synthetic_trendbars(ct_symbol, period, count)

    def get_tick_data(
        self,
        symbol: str,
        quote_type: str = "BID",
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
        hours: int = 4
    ) -> List[Dict[str, Any]]:
        """Get historical tick stream for a symbol."""
        import random
        ct_symbol = self._normalize_symbol(symbol)
        digits = self.digits_for(ct_symbol)
        base_p = self.BASE_PRICES.get(ct_symbol, 1.0850)
        pip_size = float(self.pip_size_for(ct_symbol))

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (hours * 3600 * 1000)

        rnd = random.Random(hash(ct_symbol) + 42)
        ticks = []
        curr = base_p

        for i in range(150):
            t_offset = int((i / 150.0) * (hours * 3600 * 1000))
            delta = (rnd.random() - 0.495) * (pip_size * 0.8)
            curr = round(curr + delta, digits)
            ticks.append({
                "timestamp": start_ms + t_offset,
                "symbol": ct_symbol,
                "type": quote_type.upper(),
                "price": curr,
                "volume": rnd.randint(1000, 50000)
            })

        return ticks

    def status(self) -> Dict[str, Any]:
        tokens = token_store.get_tokens()
        return {
            "connected": self.is_connected,
            "authenticated": self._authenticated,
            "env": "live" if not self._dry_run else "paper",
            "dry_run": self._dry_run,
            "account_id": tokens.get("account_id"),
            "login": self._trader_login,
            "broker": self._broker_name,
            "open_positions": len(self._positions),
            "symbols_supported": list(self.SYMBOL_MAP.keys()),
            "symbol_map": self.SYMBOL_MAP,
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "last_error": self._last_protocol_error,
            "connect_cooldown_s": max(0, int(self._next_connect_ok_at - time.time())),
            "host": self._host_override,
            "account_id_live": self._account_id,
        }


# Global service singleton
ctrader_service = CTraderService()
ctrader_broker = ctrader_service
YFINANCE_TO_CTRADER = CTraderService.SYMBOL_MAP
