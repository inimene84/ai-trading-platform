"""
Binance USDT-M Futures Service
Handles live order execution on Binance Futures (perpetuals).
Supports LONG/SHORT positions with configurable leverage (default 10x).

Interface compatible with ctrader_service for drop-in integration with trading_loop.py.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from backend.services.trading_mode import live_binance_orders_allowed
from backend.services.multi_asset_bars import classify_symbol

load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)

logger = logging.getLogger(__name__)

# ── Symbol Mapping ────────────────────────────────────────────────────────────
SYMBOL_MAP = {
    # Legacy yfinance-style → Binance Futures
    'BTC-USD': 'BTCUSDT',
    'ETH-USD': 'ETHUSDT',
    'SOL-USD': 'SOLUSDT',
    'BNB-USD': 'BNBUSDT',
    'XRP-USD': 'XRPUSDT',
    'ADA-USD': 'ADAUSDT',
    'DOGE-USD': 'DOGEUSDT',
    'AVAX-USD': 'AVAXUSDT',
    'DOT-USD': 'DOTUSDT',
    'LINK-USD': 'LINKUSDT',
    'MATIC-USD': 'POLUSDT',
    'LTC-USD': 'LTCUSDT',
    'UNI-USD': 'UNIUSDT',
    'ATOM-USD': 'ATOMUSDT',
    'NEAR-USD': 'NEARUSDT',
    'OP-USD': 'OPUSDT',
    'ARB-USD': 'ARBUSDT',
    'APT-USD': 'APTUSDT',
    'INJ-USD': 'INJUSDT',
    'SUI-USD': 'SUIUSDT',
    # Native Binance format passthrough
    'BTCUSDT': 'BTCUSDT', 'ETHUSDT': 'ETHUSDT', 'SOLUSDT': 'SOLUSDT',
    'BNBUSDT': 'BNBUSDT', 'XRPUSDT': 'XRPUSDT', 'ADAUSDT': 'ADAUSDT',
    'DOGEUSDT': 'DOGEUSDT', 'AVAXUSDT': 'AVAXUSDT', 'DOTUSDT': 'DOTUSDT',
    'LINKUSDT': 'LINKUSDT', 'POLUSDT': 'POLUSDT', 'LTCUSDT': 'LTCUSDT',
    'UNIUSDT': 'UNIUSDT', 'ATOMUSDT': 'ATOMUSDT', 'NEARUSDT': 'NEARUSDT',
    'OPUSDT': 'OPUSDT', 'ARBUSDT': 'ARBUSDT', 'APTUSDT': 'APTUSDT',
    'INJUSDT': 'INJUSDT', 'SUIUSDT': 'SUIUSDT',
    # USDC-margined perpetuals (MiCA / EEA — USDT is restricted for EU users).
    # Native passthrough + yfinance-style alias so either form resolves to USDC.
    'BTCUSDC': 'BTCUSDC', 'ETHUSDC': 'ETHUSDC', 'SOLUSDC': 'SOLUSDC',
    'BNBUSDC': 'BNBUSDC', 'XRPUSDC': 'XRPUSDC', 'ADAUSDC': 'ADAUSDC',
    'AVAXUSDC': 'AVAXUSDC', 'LINKUSDC': 'LINKUSDC', 'UNIUSDC': 'UNIUSDC',
}

# Forex / stock symbols not tradeable on Binance Futures — skip gracefully
UNSUPPORTED_SYMBOLS = {'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURJPY=X', 'EURGBP=X'}

# Minimum order quantity per symbol
MIN_QTY = {
    'BTCUSDT': 0.001, 'ETHUSDT': 0.001, 'SOLUSDT': 0.1,
    'BNBUSDT': 0.01,  'XRPUSDT': 1.0,   'ADAUSDT': 1.0,
    'DOGEUSDT': 1.0,  'AVAXUSDT': 1.0,  'DOTUSDT': 0.1,
    'LINKUSDT': 0.01, 'POLUSDT': 1.0, 'LTCUSDT': 0.001,
    'UNIUSDT': 1.0,   'ATOMUSDT': 0.01, 'NEARUSDT': 0.1,
    'OPUSDT': 0.1,    'ARBUSDT': 1.0,   'APTUSDT': 0.1,
    'INJUSDT': 0.1,   'SUIUSDT': 1.0,
    # USDC perpetuals (real LOT_SIZE.minQty from /fapi/v1/exchangeInfo)
    'BTCUSDC': 0.001, 'ETHUSDC': 0.001, 'SOLUSDC': 0.01,
    'BNBUSDC': 0.01,  'XRPUSDC': 0.1,   'ADAUSDC': 0.1,
    'AVAXUSDC': 0.01, 'LINKUSDC': 0.01, 'UNIUSDC': 1.0,
}

# Precision (decimal places) for quantity
QTY_PRECISION = {
    'BTCUSDT': 3, 'ETHUSDT': 3, 'SOLUSDT': 1,
    'BNBUSDT': 2, 'XRPUSDT': 0, 'ADAUSDT': 0,
    'DOGEUSDT': 0, 'AVAXUSDT': 0, 'DOTUSDT': 1,
    'LINKUSDT': 2, 'POLUSDT': 0, 'LTCUSDT': 3,
    'UNIUSDT': 0, 'ATOMUSDT': 2, 'NEARUSDT': 1,
    'OPUSDT': 1, 'ARBUSDT': 0, 'APTUSDT': 1,
    'INJUSDT': 1,   'SUIUSDT': 0,
    # USDC perpetuals (real quantityPrecision from /fapi/v1/exchangeInfo)
    'BTCUSDC': 3, 'ETHUSDC': 3, 'SOLUSDC': 2,
    'BNBUSDC': 2, 'XRPUSDC': 1, 'ADAUSDC': 1,
    'AVAXUSDC': 2, 'LINKUSDC': 2, 'UNIUSDC': 0,
}

# Price precision per symbol (loaded from futures_exchange_info at runtime)
PRICE_PRECISION: Dict[str, int] = {}

# Price tick sizes per symbol (min increment — from /fapi/v1/exchangeInfo PRICE_FILTER)
# Used by _round_price() to avoid Binance -1111 precision errors on SL/TP orders.
TICK_SIZES = {
    'BTCUSDT':  0.1,       'ETHUSDT':  0.01,     'SOLUSDT':  0.001,
    'BNBUSDT':  0.01,      'XRPUSDT':  0.0001,   'ADAUSDT':  0.00001,
    'DOGEUSDT': 0.00001,   'AVAXUSDT': 0.001,    'DOTUSDT':  0.001,
    'LINKUSDT': 0.001,     'POLUSDT':  0.0001,   'LTCUSDT':  0.01,
    'UNIUSDT':  0.001,     'ATOMUSDT': 0.001,    'NEARUSDT': 0.0001,
    'OPUSDT':   0.0001,    'ARBUSDT':  0.0001,   'APTUSDT':  0.001,
    'INJUSDT':  0.001,     'SUIUSDT':  0.00001,
    # USDC perpetuals (real PRICE_FILTER.tickSize from /fapi/v1/exchangeInfo)
    'BTCUSDC':  0.1,       'ETHUSDC':  0.01,     'SOLUSDC':  0.01,
    'BNBUSDC':  0.01,      'XRPUSDC':  0.0001,   'ADAUSDC':  0.0001,
    'AVAXUSDC': 0.001,     'LINKUSDC': 0.001,    'UNIUSDC':  0.001,
}

# Minimum order notional per symbol (from /fapi/v1/exchangeInfo MIN_NOTIONAL filter)
MIN_NOTIONAL = {
    'BTCUSDT': 100.0, 'ETHUSDT': 20.0, 'SOLUSDT': 20.0,
    'BNBUSDT': 20.0,  'XRPUSDT': 20.0, 'ADAUSDT': 20.0,
    'DOGEUSDT': 20.0, 'AVAXUSDT': 20.0, 'DOTUSDT': 20.0,
    'LINKUSDT': 20.0, 'LTCUSDT': 20.0,
    'BTCUSDC': 100.0, 'ETHUSDC': 20.0, 'SOLUSDC': 20.0,
    'BNBUSDC': 20.0,  'XRPUSDC': 20.0, 'ADAUSDC': 20.0,
    'AVAXUSDC': 20.0, 'LINKUSDC': 20.0,
}



class BinanceClientProxy:
    """Wrapper around binance.client.Client that intercepts all method calls
    and runs _handle_api_exception if any call raises an exception.
    """
    def __init__(self, client, service_instance):
        self._client = client
        self._service = service_instance

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                try:
                    self._service._check_ban_status()
                    return attr(*args, **kwargs)
                except Exception as e:
                    self._service._handle_api_exception(e)
                    raise
            return wrapper
        return attr


def entry_client_order_id(
    futures_sym: str,
    side: str,
    is_pyramid: bool,
    quantity: float,
    now: Optional[float] = None,
) -> str:
    """Stable Binance newClientOrderId for one entry intent in the current minute.

    Same symbol+side+purpose+qty in the same minute → same id (retry/restart
    of that intent is rejected as a duplicate). A different size or
    entry-vs-pyramid purpose in the same minute stays distinct. Per-second
    buckets were a live-behavior accident: a restart 1s later placed a
    second order.
    """
    minute = int(now if now is not None else time.time()) // 60
    purpose = "p" if is_pyramid else "e"
    qty_tag = format(float(quantity), ".8f").rstrip("0").rstrip(".").replace(".", "d")
    if not qty_tag or qty_tag in ("-0", "-"):
        qty_tag = "0"
    raw = f"x{futures_sym[:6]}{side[0]}{purpose}{qty_tag}{minute}"
    if len(raw) <= 36:
        return raw
    # Idempotency digest, not a security MAC.
    digest = hashlib.sha1(
        f"{futures_sym}|{side}|{purpose}|{quantity}|{minute}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"x{futures_sym[:6]}{side[0]}{purpose}{digest}"[:36]


class BinanceFuturesService:
    """Binance USDT-M Futures broker — compatible with trading_loop.py interface."""
    _banned_until = None

    def __init__(self):
        self.api_key    = os.getenv('BINANCE_API_KEY', '')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY', '') or os.getenv('BINANCE_API_SECRET', '')
        self.testnet    = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        self.leverage   = int(os.getenv('BINANCE_LEVERAGE', '10'))
        margin_env = os.getenv('BINANCE_MARGIN_TYPE', 'ISOLATED').upper()
        self.margin_type = 'CROSSED' if margin_env in ('CROSS', 'CROSSED') else 'ISOLATED'
        self.dry_run    = os.getenv('BINANCE_DRY_RUN', 'false').lower() == 'true'
        self._client    = None
        self._leverage_set: set = set()
        # Runtime LOT_SIZE from /fapi/v1/exchangeInfo (authoritative step/min qty).
        self._lot_step: Dict[str, float] = {}
        self._lot_min: Dict[str, float] = {}
        self._qty_precision: Dict[str, int] = {}
        # TRADING USDT/USDC perps from the last exchangeInfo ingest. Empty
        # means "unknown" (do not treat every symbol as unlisted).
        self._tradable_symbols: set = set()
        logger.info(
            f"BinanceFuturesService: testnet={self.testnet} "
            f"leverage={self.leverage}x margin={self.margin_type} dry_run={self.dry_run}"
        )

        # Load price + lot precision per symbol from exchange info
        try:
            client = self._get_client()
            exchange_info = client.futures_exchange_info()
            for sym_info in exchange_info.get('symbols', []):
                sym = sym_info['symbol']
                PRICE_PRECISION[sym] = sym_info.get('pricePrecision', 2)
                self._qty_precision[sym] = sym_info.get(
                    'quantityPrecision', QTY_PRECISION.get(sym, 3)
                )
                if (
                    sym_info.get('status') == 'TRADING'
                    and str(sym_info.get('contractType', '')).upper() == 'PERPETUAL'
                    and str(sym_info.get('quoteAsset', '')).upper() in ('USDT', 'USDC')
                ):
                    self._tradable_symbols.add(sym)
                for filt in sym_info.get('filters', []):
                    if filt.get('filterType') == 'LOT_SIZE':
                        self._lot_step[sym] = float(filt['stepSize'])
                        self._lot_min[sym] = float(filt['minQty'])
                    elif filt.get('filterType') == 'PRICE_FILTER':
                        try:
                            TICK_SIZES[sym] = float(filt['tickSize'])
                        except (TypeError, ValueError, KeyError):
                            pass
            logger.info(
                f"Exchange filters loaded for {len(PRICE_PRECISION)} symbols "
                f"({len(self._lot_step)} lot steps, "
                f"{len(self._tradable_symbols)} TRADING USDT/USDC perps)"
            )
        except Exception as e:
            self._handle_api_exception(e)
            logger.warning(f"Could not load exchange filters: {e} — using static defaults")

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _handle_api_exception(cls, e: Exception) -> None:
        """Inspect exception for a real Binance IP ban (HTTP 418/429 / code -1003).

        Match status/code attributes, not substrings — a price like 64185.2 or
        our own "Cooldown remaining: 418.3s" message used to self-inflict a ban.
        """
        err_msg = str(e)
        if "IP ban active" in err_msg or "Suppressing API request" in err_msg:
            return
        code = getattr(e, "code", None)
        status_code = getattr(e, "status_code", None)
        banned = (
            code in (-1003, "-1003")
            or status_code in (418, 429, "418", "429")
            or "banned until" in err_msg.lower()
        )
        if not banned:
            return
        import re
        match = re.search(r"banned until (\d+)", err_msg)
        if match:
            try:
                ts_ms = int(match.group(1))
                cls._banned_until = datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc)
                logger.error(f"!!! BINANCE IP BAN DETECTED !!! Local cooldown set until {cls._banned_until} UTC")
                return
            except Exception:
                pass
        cls._banned_until = datetime.now(timezone.utc) + timedelta(minutes=10)
        logger.error(f"!!! BINANCE IP BAN DETECTED !!! Local fallback cooldown set for 10 minutes (until {cls._banned_until} UTC)")

    def _check_ban_status(self) -> None:
        """Raise an exception if the local IP ban cooldown is active."""
        if BinanceFuturesService._banned_until:
            now = datetime.now(timezone.utc)
            if now < BinanceFuturesService._banned_until:
                remaining = (BinanceFuturesService._banned_until - now).total_seconds()
                raise Exception(
                    f"Binance IP ban active. Suppressing API request to avoid extending ban. "
                    f"Cooldown remaining: {remaining:.1f}s (until {BinanceFuturesService._banned_until} UTC)"
                )

    def _get_client(self):
        self._check_ban_status()
        if not self._client:
            from binance.client import Client
            self._client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )
            logger.info(f"Binance Futures client connected (testnet={self.testnet})")
        return BinanceClientProxy(self._client, self)

    def _to_futures_symbol(self, symbol: str) -> Optional[str]:
        """Convert internal symbol → Binance Futures format. Returns None if unsupported."""
        if not symbol:
            return None
        if symbol in UNSUPPORTED_SYMBOLS:
            return None
        if symbol in SYMBOL_MAP:
            return SYMBOL_MAP[symbol]
        # Generic: ETH-USD → ETHUSDT, BTC/USDT → BTCUSDT, BTC/USDC → BTCUSDC
        cleaned = symbol.replace('=X', '').replace('/', '').upper()
        # Preserve an explicit USDC/USDT quote; only the legacy yfinance '-USD'
        # form (no explicit stablecoin) defaults to USDT.
        if cleaned.endswith('USDC') or cleaned.endswith('USDT'):
            return cleaned
        # Crypto yfinance aliases (BTC-USD) must map before FX classify —
        # ETHUSD is 6 letters and would otherwise look like a forex pair.
        if cleaned.endswith('-USD'):
            return cleaned.replace('-USD', 'USDT')
        # EURUSD / XAUUSD used to become EURUSDUSDT and hit Binance.
        asset = classify_symbol(cleaned.replace('-', ''))
        if asset in ('forex', 'metal', 'oil'):
            logger.info(f"[Binance Futures] Skipping non-crypto symbol: {symbol} ({asset})")
            return None
        cleaned = cleaned.replace('-USD', 'USDT')
        if not (cleaned.endswith('USDT') or cleaned.endswith('USDC')):
            cleaned += 'USDT'
        return cleaned

    def futures_listing_status(self, symbol: str) -> Optional[bool]:
        """Whether *symbol* is a TRADING USDT/USDC perp on this venue.

        True = listed and TRADING. False = resolved but not in the
        exchangeInfo TRADING set (delisted/halted/wrong quote). None =
        cache empty so listing is unknown — callers must not treat that
        as a hard reject (tests / exchangeInfo outage).
        """
        resolved = self._to_futures_symbol(symbol)
        if not resolved:
            return False
        tradable = getattr(self, "_tradable_symbols", None)
        if not tradable:
            return None
        return resolved in tradable

    def _setup_symbol(self, client, sym: str) -> None:
        """Configure leverage + margin type once per symbol per session."""
        if sym in self._leverage_set:
            return
        # Margin type
        try:
            client.futures_change_margin_type(symbol=sym, marginType=self.margin_type)
            logger.info(f"[{sym}] Margin → {self.margin_type}")
        except Exception as e:
            err = str(e)
            if '-4046' in err or '-4175' in err or '-4067' in err:  # -4046=already set, -4175=credit status (BNFCR), -4067=open orders
                pass  # safe to ignore
            else:
                logger.warning(f"[{sym}] margin_type: {e}")
        # Leverage
        try:
            client.futures_change_leverage(symbol=sym, leverage=self.leverage)
            logger.info(f"[{sym}] Leverage → {self.leverage}x")
        except Exception as e:
            logger.warning(f"[{sym}] leverage: {e}")
        self._leverage_set.add(sym)

    def _round_qty(self, sym: str, qty: float, round_up: bool = False, bump_to_min: bool = True) -> float:
        """Round quantity to Binance LOT_SIZE step (avoids -1111 precision errors).

        Defaults to flooring, which is the safe direction for a caller-supplied
        quantity (never order more than requested). Pass `round_up=True` when
        `qty` is itself a *minimum* required quantity (e.g. derived from
        Binance's MIN_NOTIONAL filter) — flooring a minimum can round it back
        below the threshold it was meant to satisfy and trigger -4164.

        `bump_to_min=False` (use on reduce-only remainders) returns 0 instead of
        bumping a sub-minQty value up to minQty, which would over-close.
        """
        import math
        step = self._lot_step.get(sym) or MIN_QTY.get(sym, 0.001)
        min_q = self._lot_min.get(sym) or MIN_QTY.get(sym, step)
        if step <= 0:
            step = 0.001
        if round_up:
            rounded = math.ceil(qty / step - 1e-9) * step
        else:
            rounded = math.floor(qty / step + 1e-9) * step
        if rounded < min_q:
            if not bump_to_min:
                return 0.0
            rounded = math.ceil(min_q / step - 1e-9) * step
        prec = self._qty_precision.get(sym)
        if prec is None:
            prec = QTY_PRECISION.get(sym, 3)
        if prec == 0:
            return float(int(round(rounded)))
        return round(rounded, prec)

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to the exchange tick size (not pricePrecision)."""
        import math
        tick = TICK_SIZES.get(symbol)
        if tick is None or tick <= 0:
            precision = PRICE_PRECISION.get(symbol)
            if precision is not None:
                tick = 10 ** (-int(precision))
            else:
                tick = 0.001
        rounded = round(price / tick) * tick
        if tick >= 1:
            decimals = 0
        else:
            decimals = max(0, -int(math.floor(math.log10(tick))))
        return round(rounded, decimals)

    def _min_quantity(self, sym: str, price: float) -> float:
        """Return tradeable quantity based on TRADE_USDT_AMOUNT env var (default 10 USDT)."""
        min_qty = MIN_QTY.get(sym, 0.001)
        trade_usdt = float(os.getenv("TRADE_USDT_AMOUNT", "10.0"))
        # Per-symbol min notional from Binance /fapi/v1/exchangeInfo MIN_NOTIONAL filter
        min_notional = MIN_NOTIONAL.get(sym, 20.0)  # safe default: $20
        if price > 0:
            notional_qty = trade_usdt / price
            min_notional_qty = min_notional / price
            qty = max(min_qty, notional_qty, min_notional_qty)
        else:
            qty = min_qty
        # Round up: `qty` is a *minimum* viable size here, so flooring it to the
        # LOT_SIZE step could push the resulting notional back under Binance's
        # MIN_NOTIONAL filter and cause a -4164 rejection at order time.
        rounded = self._round_qty(sym, qty, round_up=True)
        # Final safety: ensure rounded qty still meets min_qty and min_notional.
        if rounded < min_qty:
            rounded = self._round_qty(sym, min_qty, round_up=True)
        if price > 0 and rounded * price < min_notional:
            rounded = self._round_qty(sym, min_notional / price, round_up=True)
        logger.info(f"[Binance] Quantity for {sym}: {rounded:.6f} @ ${price:.2f} (${trade_usdt} USDT, minNotional=${min_notional})")
        return rounded

    # ── Public interface ──────────────────────────────────────────────────────

    def get_balance(self, raise_on_error: bool = False) -> dict:
        """Fetch Binance Futures wallet balance (sync).
        Uses futures_account() totals which aggregate ALL margin assets
        (USDT, USDC, BNFCR, BFUSD, etc.) into a single USD-equivalent value.
        The old USDT-only filter returned 0 on USDC/BNFCR accounts.
        """
        try:
            client = self._get_client()
            acct = client.futures_account()
            total_wallet = float(acct.get('totalWalletBalance', 0))
            available    = float(acct.get('availableBalance', 0))
            margin_bal   = float(acct.get('totalMarginBalance', total_wallet))
            return {
                'balance':         total_wallet,
                'available':       available,
                'equity':          margin_bal,
                'unrealized_pnl':  float(acct.get('totalUnrealizedProfit', 0)),
                'margin_used':     float(acct.get('totalInitialMargin', 0)),
                'broker':          'binance_futures',
            }
        except Exception as e:
            logger.error(f"get_balance error: {e}")
            if raise_on_error:
                raise
            return {'balance': 0.0, 'equity': 0.0, 'available': 0.0,
                    'broker': 'binance_futures', 'error': str(e)}

    def get_positions(self, raise_on_error: bool = False) -> list:
        """Return all non-zero open futures positions."""
        try:
            client = self._get_client()
            all_pos = client.futures_position_information()
            return [
                {
                    'symbol':          p['symbol'],
                    'side':            'BUY' if float(p['positionAmt']) > 0 else 'SELL',
                    'quantity':        abs(float(p['positionAmt'])),
                    'entry_price':     float(p['entryPrice']),
                    'unrealized_pnl':  float(p['unRealizedProfit']),
                    'leverage':        int(p.get('leverage', self.leverage)),
                    'margin_type':     p.get('marginType', self.margin_type),
                    'mark_price':      float(p.get('markPrice', 0)),
                    'liquidation_price': float(p.get('liquidationPrice', 0)),
                }
                for p in all_pos
                if abs(float(p.get('positionAmt', 0))) > 0
            ]
        except Exception as e:
            logger.error(f"get_positions error: {e}")
            if raise_on_error:
                raise
            return []

    def get_exit_price(self, symbol: str) -> Optional[float]:
        """Best-effort exit price for a just-closed position.

        Prefers the most recent actual fill price (accurate when SL/TP filled
        on the exchange), and falls back to the current mark price.
        """
        sym = self._to_futures_symbol(symbol)
        if not sym:
            return None
        client = self._get_client()
        try:
            trades = client.futures_account_trades(symbol=sym, limit=1)
            if trades:
                # futures_account_trades returns the most recent fill EVER for
                # the symbol — it may be days old and unrelated to the close we
                # are reconciling. Only trust it when it's recent; otherwise
                # fall through to the live mark price.
                fill = trades[-1]
                px = float(fill.get('price', 0) or 0)
                fill_time_ms = float(fill.get('time', 0) or 0)
                age_sec = (datetime.now(timezone.utc).timestamp() - fill_time_ms / 1000.0) if fill_time_ms else None
                if px > 0 and age_sec is not None and age_sec < 24 * 3600:
                    return px
                if px > 0:
                    logger.warning(
                        f"[{sym}] get_exit_price: last fill is stale "
                        f"(age={age_sec if age_sec is not None else 'unknown'}s) — using mark price instead"
                    )
        except Exception as e:
            logger.warning(f"[{sym}] get_exit_price (fills) failed: {e}")
        try:
            data = client.futures_mark_price(symbol=sym)
            px = float(data.get('markPrice', 0) or 0)
            if px > 0:
                return px
        except Exception as e:
            logger.warning(f"[{sym}] get_exit_price (mark) failed: {e}")
        return None

    def get_open_orders(self, raise_on_error: bool = False) -> list:
        """Return all open futures orders, INCLUDING conditional/algo orders.

        On this account SL/TP/trailing orders placed via futures_create_order are
        routed as CONDITIONAL (algo) orders, which do NOT appear in the regular
        /fapi/v1/openOrders feed. We merge both so callers see every open order.
        Conditional orders are tagged with `algo_id` for cancellation.
        """
        out = []
        client = self._get_client()
        try:
            for o in client.futures_get_open_orders():
                out.append({
                    'order_id': str(o['orderId']),
                    'symbol':   o['symbol'],
                    'side':     o['side'],
                    'type':     o['type'],
                    'quantity': float(o['origQty']),
                    'price':    float(o.get('price', 0)),
                    'status':   o['status'],
                    'algo_id':  None,
                })
        except Exception as e:
            logger.error(f"get_open_orders error: {e}")
            if raise_on_error:
                raise
        # Conditional / algo orders (STOP/TP/TRAILING on this account)
        try:
            algo = client.futures_get_open_algo_orders()
            algo_list = algo if isinstance(algo, list) else algo.get('orders', [])
            for o in algo_list:
                out.append({
                    'order_id': str(o.get('algoId', '')),
                    'symbol':   o.get('symbol'),
                    'side':     o.get('side'),
                    'type':     o.get('orderType') or o.get('algoType'),
                    'quantity': float(o.get('quantity', 0) or 0),
                    'price':    float(o.get('triggerPrice', 0) or 0),
                    'status':   o.get('algoStatus', 'NEW'),
                    'algo_id':  o.get('algoId'),
                })
        except Exception as e:
            logger.debug(f"get_open_algo_orders error: {e}")
            if raise_on_error:
                raise
        return out

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel ALL open orders for a symbol — regular AND conditional/algo."""
        if not live_binance_orders_allowed():
            logger.info("[SAFETY] skip cancel_all_orders: TRADING_MODE is not live")
            return
        client = self._get_client()
        fsym = self._to_futures_symbol(symbol)
        if not fsym:
            return
        try:
            client.futures_cancel_all_open_orders(symbol=fsym)
        except Exception as e:
            logger.debug(f"[{fsym}] cancel regular orders: {e}")
        try:
            algo = client.futures_get_open_algo_orders()
            algo_list = algo if isinstance(algo, list) else algo.get('orders', [])
            for o in algo_list:
                if o.get('symbol') == fsym and o.get('algoId'):
                    try:
                        client.futures_cancel_algo_order(algoId=o['algoId'])
                    except Exception as ce:
                        logger.debug(f"[{fsym}] cancel algo {o.get('algoId')}: {ce}")
        except Exception as e:
            logger.debug(f"[{fsym}] cancel algo orders: {e}")

    def _is_protective_order_type(self, order_type: str) -> bool:
        upper = (order_type or "").upper()
        return any(t in upper for t in self._PROTECTIVE_ORDER_TYPES)

    def cancel_non_protective_orders(self, symbol: str) -> int:
        """Cancel entry/limit orders only — preserve live SL/TP/trailing protection."""
        if not live_binance_orders_allowed():
            logger.info("[SAFETY] skip cancel_non_protective_orders: TRADING_MODE is not live")
            return 0
        fsym = self._to_futures_symbol(symbol)
        if not fsym:
            return 0
        cancelled = 0
        client = self._get_client()
        for o in self.get_open_orders(raise_on_error=True):
            if o.get("symbol") != fsym:
                continue
            if self._is_protective_order_type(o.get("type") or ""):
                continue
            try:
                if o.get("algo_id"):
                    client.futures_cancel_algo_order(algoId=o["algo_id"])
                elif o.get("order_id"):
                    client.futures_cancel_order(symbol=fsym, orderId=int(o["order_id"]))
                cancelled += 1
            except Exception as ce:
                logger.debug(
                    f"[{fsym}] cancel non-protective order "
                    f"{o.get('order_id') or o.get('algo_id')}: {ce}"
                )
        return cancelled

    _PROTECTIVE_ORDER_TYPES = frozenset({
        "STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET", "STOP", "TAKE_PROFIT",
    })

    def _collect_protective_orders(self, futures_sym: str, position_side: str, raise_on_error: bool = False) -> list:
        """Return open SL/TP/trailing orders for a symbol+positionSide (regular + algo)."""
        sl_side = 'SELL' if position_side == 'LONG' else 'BUY'
        return [
            o for o in self.get_open_orders(raise_on_error)
            if o.get('symbol') == futures_sym
            and o.get('type') in self._PROTECTIVE_ORDER_TYPES
            and o.get('side') == sl_side
        ]

    def _cancel_listed_orders(self, client, futures_sym: str, orders: list) -> int:
        cancelled = 0
        for o in orders:
            try:
                if o.get('algo_id'):
                    client.futures_cancel_algo_order(algoId=o['algo_id'])
                elif o.get('order_id'):
                    client.futures_cancel_order(symbol=futures_sym, orderId=int(o['order_id']))
                cancelled += 1
            except Exception as ce:
                logger.warning(
                    f"  [!] Could not cancel protective order {o.get('order_id') or o.get('algo_id')} "
                    f"on {futures_sym}: {ce}"
                )
        return cancelled

    def _live_position_qty(self, futures_sym: str, position_side: str, raise_on_error: bool = False) -> float:
        """Exchange position size for emergency closes (full leg, not pyramid add qty)."""
        for p in self.get_positions(raise_on_error):
            if p.get('symbol') != futures_sym:
                continue
            side = (p.get('side') or '').upper()
            if position_side == 'LONG' and side in ('BUY', 'LONG'):
                return float(p.get('quantity') or 0)
            if position_side == 'SHORT' and side in ('SELL', 'SHORT'):
                return float(p.get('quantity') or 0)
        return 0.0

    def _has_exchange_stop(self, futures_sym: str, position_side: str, raise_on_error: bool = False) -> bool:
        return any(
            'STOP' in (o.get('type') or '')
            for o in self._collect_protective_orders(futures_sym, position_side, raise_on_error)
        )

    def _has_exchange_take_profit(self, futures_sym: str, position_side: str, raise_on_error: bool = False) -> bool:
        return any(
            'TAKE_PROFIT' in (o.get('type') or '')
            for o in self._collect_protective_orders(futures_sym, position_side, raise_on_error)
        )

    @staticmethod
    def _is_existing_close_position_error(err: Exception) -> bool:
        """Binance -4130: only one closePosition SL/TP allowed per position side."""
        return '-4130' in str(err)

    def ensure_protective_orders(
        self,
        symbol: str,
        direction: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """Re-place missing exchange SL/TP from DB levels — never cancels existing protection first."""
        futures_sym = self._to_futures_symbol(symbol)
        if not futures_sym:
            return {'status': 'skipped', 'reason': f'{symbol} unsupported'}
        if self.dry_run:
            return {'status': 'simulated', 'symbol': futures_sym}
        if not live_binance_orders_allowed():
            logger.info("[SAFETY] skip ensure_protective_orders: TRADING_MODE is not live")
            return {'status': 'skipped', 'reason': 'live Binance orders disabled unless TRADING_MODE=live'}

        position_side = 'LONG' if direction.upper() == 'BUY' else 'SHORT'
        try:
            qty = self._live_position_qty(futures_sym, position_side, raise_on_error=True)
        except Exception as e:
            logger.error(f"  [PROTECT-RESTORE] {futures_sym} failed to check position qty: {e}")
            return {'status': 'error', 'message': str(e), 'symbol': futures_sym}

        if qty <= 0:
            return {'status': 'skipped', 'reason': 'no open position'}

        live_entry = None
        implausible_db_stop = False
        try:
            for p in self.get_positions():
                if p.get("symbol") == futures_sym and float(p.get("quantity") or 0) > 0:
                    live_entry = float(p.get("entry_price") or p.get("entryPrice") or 0)
                    break
        except Exception:
            live_entry = None
        if stop_loss and live_entry and live_entry > 0:
            deviation = abs(float(stop_loss) - live_entry) / live_entry
            if deviation > 0.15:
                logger.error(
                    f"  [PROTECT-RESTORE] {futures_sym} refusing implausible DB stop "
                    f"{stop_loss} vs entry {live_entry} ({deviation:.1%})"
                )
                implausible_db_stop = True
                stop_loss = None

        try:
            has_sl = self._has_exchange_stop(futures_sym, position_side, raise_on_error=True)
            needs_tp = bool(take_profit and take_profit > 0)
            has_tp = self._has_exchange_take_profit(futures_sym, position_side, raise_on_error=True) if needs_tp else True
        except Exception as e:
            logger.error(f"  [PROTECT-RESTORE] {futures_sym} failed to check exchange orders: {e}")
            return {'status': 'error', 'message': str(e), 'symbol': futures_sym}

        if has_sl and has_tp:
            return {'status': 'ok', 'symbol': futures_sym}
        if implausible_db_stop and not has_sl:
            return {
                'status': 'error',
                'message': (
                    f'implausible DB stop vs entry {live_entry}; '
                    'exchange has no stop'
                ),
                'symbol': futures_sym,
            }

        client = self._get_client()
        sl_side = 'SELL' if direction.upper() == 'BUY' else 'BUY'
        restored = []

        try:
            ticker = client.futures_symbol_ticker(symbol=futures_sym)
            ref_price = float(ticker.get('price') or 0)
        except Exception:
            ref_price = 0.0

        if stop_loss and stop_loss > 0 and not has_sl:
            sl_valid = (
                (direction.upper() == 'BUY' and stop_loss < ref_price)
                or (direction.upper() == 'SELL' and stop_loss > ref_price)
                or ref_price <= 0
            )
            if sl_valid:
                try:
                    sl_params = {
                        "symbol": futures_sym,
                        "side": sl_side,
                        "type": 'STOP_MARKET',
                        "stopPrice": self._round_price(futures_sym, stop_loss),
                        "closePosition": "true",
                        "positionSide": position_side,
                    }
                    sl_order = self._safe_create_order(client, sl_params)
                    restored.append(f"SL@{stop_loss} id={sl_order.get('algoId') or sl_order.get('orderId')}")
                    logger.warning(
                        f"  [PROTECT-RESTORE] {futures_sym} replaced missing SL @ {stop_loss}"
                    )
                except Exception as sl_e:
                    logger.error(
                        f"  [PROTECT-RESTORE] {futures_sym} failed to restore SL @ {stop_loss}: {sl_e}"
                    )
                    return {'status': 'error', 'message': str(sl_e), 'symbol': futures_sym}

        if take_profit and take_profit > 0 and needs_tp and not has_tp:
            tp_valid = (
                (direction.upper() == 'BUY' and take_profit > ref_price)
                or (direction.upper() == 'SELL' and take_profit < ref_price)
                or ref_price <= 0
            )
            if tp_valid:
                try:
                    tp_params = {
                        "symbol": futures_sym,
                        "side": sl_side,
                        "type": 'TAKE_PROFIT_MARKET',
                        "stopPrice": self._round_price(futures_sym, take_profit),
                        "closePosition": "true",
                        "positionSide": position_side,
                    }
                    tp_order = self._safe_create_order(client, tp_params)
                    restored.append(f"TP@{take_profit} id={tp_order.get('algoId') or tp_order.get('orderId')}")
                    logger.warning(
                        f"  [PROTECT-RESTORE] {futures_sym} replaced missing TP @ {take_profit}"
                    )
                except Exception as tp_e:
                    logger.warning(
                        f"  [PROTECT-RESTORE] {futures_sym} failed to restore TP @ {take_profit}: {tp_e}"
                    )

        if restored:
            return {'status': 'restored', 'symbol': futures_sym, 'restored': restored}
        return {'status': 'skipped', 'reason': 'no valid SL/TP to restore', 'symbol': futures_sym}

    def place_order(
        self,
        symbol: str,
        direction: str,
        action: str = 'open',           # 'open' | 'close'
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = '',
        **kwargs,          # accepts ctrader-style kwarg aliases
    ) -> dict:
        """
        Place a Binance Futures market order.
        Compatible with ctrader_broker.place_order() return format.

        Returns: {'status': 'sent'|'simulated'|'skipped'|'error',
                  'broker': str, 'order_id': str, ...}
        """
        # ── ctrader-style kwarg aliases (backward-compat) ─────────────────────
        symbol      = kwargs.get('yfinance_symbol', symbol)
        quantity    = kwargs.get('volume', quantity)
        price       = kwargs.get('current_price', price)
        stop_loss   = kwargs.get('stop_loss_price', stop_loss)
        take_profit = kwargs.get('take_profit_price', take_profit)
        futures_sym = self._to_futures_symbol(symbol)
        if not futures_sym:
            logger.info(f"[Binance Futures] Skipping unsupported symbol: {symbol}")
            return {'status': 'skipped', 'broker': 'binance_futures',
                    'reason': f'{symbol} not supported on Binance Futures'}
        listed = self.futures_listing_status(symbol)
        if listed is False:
            logger.warning(
                "[Binance Futures] Skipping unlisted/halted symbol: %s (resolved %s)",
                symbol, futures_sym,
            )
            return {
                'status': 'skipped',
                'broker': 'binance_futures',
                'reason': f'{futures_sym} not TRADING on Binance futures',
                'symbol': futures_sym,
            }

        if self.dry_run:
            logger.info(f"[Binance Futures DRY-RUN] {action.upper()} {direction} {futures_sym}")
            return {'status': 'simulated', 'broker': 'binance_futures_dry',
                    'symbol': futures_sym, 'direction': direction, 'action': action}

        if not live_binance_orders_allowed():
            logger.error(
                "[SAFETY] Refusing Binance futures order: TRADING_MODE is not live "
                "(paper/backtest uses the local PaperTradingEngine, not the exchange)"
            )
            return {
                'status': 'error',
                'broker': 'binance_futures',
                'reason': 'live Binance orders disabled unless TRADING_MODE=live',
                'order_id': '',
            }

        try:
            client = self._get_client()
            self._setup_symbol(client, futures_sym)

            # Resolve current price
            if not price:
                ticker = client.futures_symbol_ticker(symbol=futures_sym)
                price  = float(ticker['price'])

            passed_reduce_only = bool(kwargs.get('reduce_only', False))
            is_close = action == 'close' or passed_reduce_only

            # Resolve quantity
            if not quantity:
                if is_close:
                    close_side = 'LONG' if direction.upper() in ('BUY', 'LONG') else 'SHORT'
                    quantity = self._live_position_qty(futures_sym, close_side)
                    if quantity <= 0:
                        return {
                            'status': 'already_flat',
                            'broker': 'binance_futures',
                            'already_closed': True,
                            'order_id': '',
                            'symbol': futures_sym,
                            'message': 'no live position to close',
                        }
                    quantity = self._round_qty(futures_sym, quantity, bump_to_min=False)
                    if quantity <= 0:
                        return {
                            'status': 'already_flat',
                            'broker': 'binance_futures',
                            'already_closed': True,
                            'order_id': '',
                            'symbol': futures_sym,
                            'message': 'live qty below LOT_SIZE min',
                        }
                else:
                    quantity = self._min_quantity(futures_sym, price)
            else:
                quantity = self._round_qty(
                    futures_sym, quantity, bump_to_min=not is_close
                )

            if price > 0 and not is_close:
                _min_not = MIN_NOTIONAL.get(futures_sym, 20.0)
                _actual_notional = quantity * price
                if _actual_notional < _min_not:
                    _old_qty = quantity
                    # round_up=True: this is a minimum required quantity, so
                    # flooring it to the LOT_SIZE step (the default direction)
                    # can undo the bump and land back under `_min_not` -> -4164.
                    quantity = self._round_qty(futures_sym, _min_not / price, round_up=True)
                    logger.info(
                        f"[Binance] Notional floor: {futures_sym} bumped qty {_old_qty:.6f} -> {quantity:.6f} "
                        f"(${_actual_notional:.2f} < min ${_min_not:.0f})"
                    )

            target_side = direction.upper()

            # ── Hedge-mode guard + pyramiding ─────────────────────────────────
            # Block accidental opposite-side legs. Allow same-direction adds
            # (pyramid/DCA) when is_pyramid=True from the decision engine OR when
            # the exchange already has a position in the same direction.
            is_pyramid = bool(kwargs.get('is_pyramid', False))
            if not is_close:
                live_on_symbol = [
                    p for p in self.get_positions()
                    if p.get('symbol') == futures_sym and float(p.get('quantity') or 0) > 0
                ]
                if live_on_symbol:
                    sides = [p.get('side') for p in live_on_symbol]
                    same_side_only = len(set(sides)) == 1 and sides[0] == target_side
                    if same_side_only:
                        is_pyramid = True
                        logger.info(
                            f"[Binance Futures] PYRAMID {target_side} add on {futures_sym} "
                            f"(existing qty={[p.get('quantity') for p in live_on_symbol]}, "
                            f"add={quantity})"
                        )
                    else:
                        logger.warning(
                            f"[Binance Futures] SKIP {futures_sym}: open position(s) already "
                            f"on exchange ({sides}) — refusing {target_side} entry"
                        )
                        return {
                            'status': 'skipped',
                            'broker': 'binance_futures',
                            'reason': f'position_already_open:{sides}',
                            'symbol': futures_sym,
                        }

            # ── Pre-trade margin check ──────────────────────────────────────────
            acct = client.futures_account()
            available = float(acct.get('availableBalance', 0))
            notional = quantity * price
            # Isolated margin requirement ≈ notional / leverage
            required_margin = notional / self.leverage
            # A reduce-only close releases margin; it must never be blocked by
            # the entry affordability gate when the account is fully deployed.
            if not is_close and available < required_margin:
                logger.warning(
                    f"[Binance Futures] SKIP {futures_sym}: available={available:.4f} < "
                    f"required_margin={required_margin:.4f} (notional={notional:.2f} / "
                    f"leverage={self.leverage}x)"
                )
                return {
                    'status': 'skipped',
                    'broker': 'binance_futures',
                    'reason': f'Insufficient margin: available={available:.4f}, '
                              f'required={required_margin:.4f}',
                    'available': available,
                    'required_margin': required_margin,
                }

            # Determine side and reduceOnly flag
            if is_close:
                # Closing a position: flip side vs original direction
                side        = 'SELL' if direction.upper() == 'BUY' else 'BUY'
                reduce_only = True
            else:
                side        = direction.upper()
                reduce_only = False

            # Hedge Mode positionSide: always the POSITION side (LONG/SHORT)
            position_side = 'LONG' if direction.upper() == 'BUY' else 'SHORT'

            logger.info(
                f"[Binance Futures] {action.upper()} {side} ({position_side}) {quantity} {futures_sym} "
                f"@ ~{price:.4f} leverage={self.leverage}x comment={comment}"
            )

            # ── Main order ────────────────────────────────────────────────────
            # FEE-CHURN FIX: 100% of historical fills were TAKER (0.05%), draining
            # ~38%/mo of the account in commission alone. For ENTRIES (not closes),
            # try a POST-ONLY (GTX) maker LIMIT at the near touch first; if it does
            # not fill within MAKER_WAIT_SEC, cancel and FALL BACK to MARKET.
            # Fail-OPEN: any error or timeout reverts to exactly today's MARKET path.
            # Closes / SL / TP / emergency stay MARKET — they must always execute.
            result = None
            order_id = ''
            filled_price = 0.0
            maker_enabled = os.getenv("MAKER_ENTRY_ENABLED", "true").lower() in ("1", "true", "yes")

            if maker_enabled and not reduce_only:
                try:
                    result = self._try_maker_entry(
                        client, futures_sym, side, quantity, position_side,
                    )
                except RuntimeError as re:
                    # Uncertain maker fill status: abort to avoid duplicate fill / doubling position size
                    logger.critical(f"  [MAKER] uncertain maker fill: {re} — aborting entry to prevent double-fill")
                    return {
                        'status': 'error',
                        'broker': 'binance_futures',
                        'reason': 'uncertain_maker_fill',
                        'message': str(re)
                    }
                except Exception as mk_e:
                    logger.warning(f"  [MAKER] post-only entry path errored ({mk_e}) — falling back to MARKET")
                    result = None

            if result is None:
                # Default / fallback: original MARKET (taker) order.
                order_params = {
                    "symbol": futures_sym,
                    "side": side,
                    "type": 'MARKET',
                    "quantity": quantity,
                    "positionSide": position_side,
                }
                # Idempotency: a deterministic client order id bucketed to the
                # current minute makes Binance reject an accidental duplicate
                # (e.g. order filled but DB commit failed → restart re-enters).
                if not reduce_only:
                    order_params["newClientOrderId"] = entry_client_order_id(
                        futures_sym, side, is_pyramid, quantity,
                    )
                # Note: In Hedge Mode, we DO NOT send reduceOnly as positionSide handles it.
                # Sending it causes APIError -1106.
                result = self._safe_create_order(client, order_params)

            order_id = str(result.get('orderId', ''))
            
            # Binance fills for MARKET orders sometimes return "0.00000" for avgPrice/price.
            # Since non-empty strings are truthy, we must parse them to float first to check if they are non-zero.
            avg_px = float(result.get('avgPrice') or 0.0)
            ord_px = float(result.get('price') or 0.0)
            filled_price = avg_px if avg_px > 0.0 else (ord_px if ord_px > 0.0 else (price or 0.0))
            
            if filled_price > 0:
                price = filled_price
            logger.info(f"  ✓ Order {order_id} filled @ {filled_price}")

            # Capture exchange-authoritative realized P&L and commission for
            # this fill. Callers otherwise compute from a stale bar close and
            # omit fees, corrupting expectancy and learning metrics.
            commission = 0.0
            exchange_realized_pnl = None
            if order_id:
                try:
                    fills = client.futures_account_trades(
                        symbol=futures_sym, orderId=int(order_id),
                    )
                    commission = sum(
                        abs(float(f.get("commission", 0) or 0))
                        for f in fills
                    )
                    exchange_realized_pnl = sum(
                        float(f.get("realizedPnl", 0) or 0)
                        for f in fills
                    )
                except Exception as fill_error:
                    logger.warning(
                        f"  [ACCOUNTING] {futures_sym} could not load fill costs "
                        f"for order {order_id}: {fill_error}"
                    )

            # Treat 0 / negative as missing — manual API calls often send stop_loss=0
            if stop_loss is not None and stop_loss <= 0:
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                take_profit = None

            logger.info(f"  [DEBUG] SL/TP Check -> reduce_only={reduce_only}, stop_loss={stop_loss}, take_profit={take_profit}")

            # ── Cleanup Orphaned Orders on Close ──────────────────────────────
            if reduce_only:
                try:
                    # A close may cover only one DB pyramid layer. Keep the
                    # closePosition SL/TP while any quantity remains; those
                    # orders automatically protect the whole residual leg.
                    remaining = self._live_position_qty(futures_sym, position_side, raise_on_error=True)
                    if remaining <= 0:
                        # Fully flat: remove orphan regular + conditional orders.
                        client.futures_cancel_all_open_orders(symbol=futures_sym)
                        try:
                            _algo = client.futures_get_open_algo_orders()
                            _algo_list = _algo if isinstance(_algo, list) else _algo.get('orders', [])
                            for _o in _algo_list:
                                if _o.get('symbol') == futures_sym and _o.get('algoId'):
                                    client.futures_cancel_algo_order(algoId=_o['algoId'])
                        except Exception as _ae:
                            logger.warning(f"  [!] Failed to cancel conditional orders for {futures_sym}: {_ae}")
                        logger.info(f"  ✓ Fully flat; cancelled remaining orders for {futures_sym}")
                    else:
                        logger.info(
                            f"  ✓ Partial close left {remaining} {futures_sym}; "
                            "keeping exchange SL/TP protection"
                        )
                except Exception as e:
                    # Unknown remaining state: fail safe by preserving orders.
                    logger.warning(f"  [!] Could not verify flat state for {futures_sym}; keeping protective orders: {e}")

            # ── Stop-loss order (STOP_MARKET, reduceOnly) ─────────────────────
            if not reduce_only and stop_loss:
                sl_valid = (side == 'BUY' and stop_loss < price) or (side == 'SELL' and stop_loss > price)
                if not sl_valid:
                    logger.warning(f"  [SL] Invalid stop_loss={stop_loss} for {side} @ {price} — skipping")
                elif is_pyramid and self._has_exchange_stop(futures_sym, position_side):
                    # closePosition SL already covers the full pyramided size; Binance
                    # rejects a second one (-4130). Tighten via replace_stop_loss only.
                    rep = self.replace_stop_loss(
                        symbol=symbol,
                        direction=direction,
                        new_stop_price=stop_loss,
                    )
                    logger.info(
                        f"  [SL] Pyramid add on {futures_sym}: existing closePosition stop kept "
                        f"(replace={rep.get('status')})"
                    )
                else:
                    try:
                        sl_side = 'SELL' if side == 'BUY' else 'BUY'
                        sl_params = {
                            "symbol": futures_sym,
                            "side": sl_side,
                            "type": 'STOP_MARKET',
                            "stopPrice": self._round_price(futures_sym, stop_loss),
                            "closePosition": "true",
                            "positionSide": position_side,
                        }
                        sl_order = self._safe_create_order(client, sl_params)
                        sl_algo_id = sl_order.get('algoId') or sl_order.get('orderId')
                        logger.info(f"  [SL] Placed for {futures_sym} @ {stop_loss} (id={sl_algo_id})")
                    except Exception as sl_e:
                        try:
                            has_sl = self._has_exchange_stop(
                                futures_sym, position_side, raise_on_error=True
                            )
                        except Exception as verify_err:
                            logger.error(
                                f"  [SL] FAILED to place stop-loss for {futures_sym}: {sl_e} "
                                f"— protection unverified ({verify_err}); leaving position open"
                            )
                            return {
                                'status': 'error',
                                'broker': 'binance_futures',
                                'message': (
                                    f'SL placement failed ({sl_e}); '
                                    f'protection unverified ({verify_err}) — no emergency close'
                                ),
                                'order_id': order_id,
                                'symbol': futures_sym,
                                'sl_error': str(sl_e),
                            }
                        if self._is_existing_close_position_error(sl_e) and has_sl:
                            logger.info(
                                f"  [SL] {futures_sym}: -4130 with live exchange stop — "
                                f"position protected, skipping emergency close"
                            )
                        elif has_sl:
                            logger.info(
                                f"  [SL] {futures_sym}: place failed ({sl_e}) but exchange "
                                f"already has a stop — skipping emergency close"
                            )
                        else:
                            # SL failed and exchange confirmed no protection → EMERGENCY CLOSE
                            logger.error(
                                f"  [SL] FAILED to place stop-loss for {futures_sym}: {sl_e}"
                                f" — initiating emergency close to prevent naked position"
                            )
                            try:
                                emg_side = 'SELL' if side == 'BUY' else 'BUY'
                                close_qty = self._live_position_qty(futures_sym, position_side) or quantity
                                close_qty = self._round_qty(futures_sym, close_qty)
                                client.futures_create_order(
                                    symbol=futures_sym,
                                    side=emg_side,
                                    type='MARKET',
                                    quantity=close_qty,
                                    positionSide=position_side,
                                )
                                logger.critical(
                                    f"  [SL-FAILSAFE] {futures_sym} closed at market due to SL failure"
                                )
                            except Exception as close_e:
                                logger.critical(
                                    f"  [SL-FAILSAFE] MARKET CLOSE ALSO FAILED for {futures_sym}: {close_e}"
                                    f" — MANUAL INTERVENTION REQUIRED"
                                )
                            return {
                                'status': 'error',
                                'broker': 'binance_futures',
                                'message': f'SL placement failed ({sl_e}) — emergency close attempted',
                                'order_id': order_id,
                                'symbol': futures_sym,
                                'sl_error': str(sl_e),
                            }

            # ── Take-profit order (TAKE_PROFIT_MARKET, reduceOnly) ────────────
            if not reduce_only and take_profit:
                tp_valid = (side == 'BUY' and take_profit > price) or (side == 'SELL' and take_profit < price)
                if not tp_valid:
                    logger.warning(f"  [TP] Invalid take_profit={take_profit} for {side} @ {price} — skipping")
                elif is_pyramid and self._has_exchange_take_profit(futures_sym, position_side):
                    logger.info(
                        f"  [TP] Pyramid add on {futures_sym}: existing closePosition TP covers "
                        f"full size — skipping duplicate"
                    )
                else:
                    try:
                        tp_side = 'SELL' if side == 'BUY' else 'BUY'
                        tp_params = {
                            "symbol": futures_sym,
                            "side": tp_side,
                            "type": 'TAKE_PROFIT_MARKET',
                            "stopPrice": self._round_price(futures_sym, take_profit),
                            "closePosition": "true",
                            "positionSide": position_side,
                        }
                        tp_order = self._safe_create_order(client, tp_params)
                        tp_algo_id = tp_order.get('algoId') or tp_order.get('orderId')
                        logger.info(f"  [TP] Placed for {futures_sym} @ {take_profit} (id={tp_algo_id})")
                    except Exception as tp_e:
                        if self._is_existing_close_position_error(tp_e) and self._has_exchange_take_profit(
                            futures_sym, position_side
                        ):
                            logger.info(
                                f"  [TP] {futures_sym}: -4130 with live exchange TP — skipping duplicate"
                            )
                        else:
                            logger.warning(f"  [TP] Failed: {tp_e}")

            # ── Native trailing stop (TRAILING_STOP_MARKET, reduceOnly) ───────
            # Trails the market continuously on the exchange. The hard STOP_MARKET
            # above remains as a catastrophe floor; this captures profit as price
            # moves favorably. Best-effort: never fails the trade.
            if not reduce_only and self._native_trailing_enabled():
                try:
                    self._place_native_trailing_stop(
                        client, futures_sym, side, quantity, price, position_side
                    )
                except Exception as tr_e:
                    logger.warning(f"  [TRAIL] native trailing stop skipped: {tr_e}")

            return {
                'status':       'sent',
                'message':      f'Order {order_id} filled @ {filled_price}',
                'broker':       'binance_futures',
                'order_id':     order_id,
                'symbol':       futures_sym,
                'side':         side,
                'quantity':     quantity,
                'filled_price': filled_price,
                'commission': commission,
                'realized_pnl': exchange_realized_pnl,
                'action':       action,
                'raw':          result,
            }

        except Exception as e:
            # -2022 is ANY reduce-only rejection, including over-size closes.
            # Only treat as already_flat when the exchange book is actually empty.
            if '-2022' in str(e) or 'ReduceOnly Order is rejected' in str(e):
                try:
                    verify_side = 'LONG' if direction.upper() in ('BUY', 'LONG') else 'SHORT'
                    live_qty = self._live_position_qty(futures_sym, verify_side, raise_on_error=True)
                except Exception as verify_err:
                    logger.error(
                        f"[Binance Futures] {symbol} -2022 but live qty unverified: {verify_err}"
                    )
                    return {
                        'status': 'error',
                        'broker': 'binance_futures',
                        'message': f'-2022 unverified ({verify_err})',
                        'error': str(e),
                    }
                if live_qty <= 0:
                    logger.info(f'[Binance Futures] {symbol} already flat (-2022) -- marking closed')
                    return {
                        'status': 'already_flat',
                        'broker': 'binance_futures',
                        'already_closed': True,
                        'order_id': '',
                        'symbol': symbol,
                        'message': '-2022 position already closed on exchange',
                    }
                logger.error(
                    f"[Binance Futures] {symbol} reduce-only rejected but {live_qty} still open"
                )
                return {
                    'status': 'error',
                    'broker': 'binance_futures',
                    'message': f'-2022 with live qty {live_qty} remaining',
                    'error': str(e),
                }
            logger.error(f"[Binance Futures] place_order error: {e}")
            return {'status': 'error', 'broker': 'binance_futures', 'message': str(e), 'error': str(e)}

    def _try_maker_entry(self, client, futures_sym, side, quantity, position_side):
        """POST-ONLY (GTX) maker entry with MARKET fallback.

        Places a post-only LIMIT at the near touch (BUY→best bid, SELL→best ask)
        so the order rests as a maker (0.02% / rebate) instead of crossing the
        spread as a taker (0.05%). Polls up to MAKER_WAIT_SEC for a fill.

        Returns the Binance order dict if FULLY filled as maker, else returns
        None so the caller falls back to a MARKET order. FAIL-OPEN: any error
        returns None (→ MARKET), so worst case == today's behavior.
        """
        import time as _t
        wait_sec = float(os.getenv("MAKER_WAIT_SEC", "8"))
        poll = 0.75

        # Best bid/ask from the book
        book = client.futures_orderbook_ticker(symbol=futures_sym)
        best_bid = float(book["bidPrice"])
        best_ask = float(book["askPrice"])
        limit_px = best_bid if side == "BUY" else best_ask
        limit_px = self._round_price(futures_sym, limit_px)

        params = {
            "symbol": futures_sym,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTX",      # GTX = post-only: rejected/expired if it would take
            "price": limit_px,
            "quantity": quantity,
            "positionSide": position_side,
        }
        try:
            order = client.futures_create_order(**params)
        except Exception as e:
            # -2021/-5022: would immediately match (not a maker) → bail to MARKET
            logger.debug(f"  [MAKER] post-only rejected for {futures_sym} ({e}) — MARKET fallback")
            return None

        oid = order.get("orderId")
        deadline = _t.time() + wait_sec
        while _t.time() < deadline:
            _t.sleep(poll)
            try:
                st = client.futures_get_order(symbol=futures_sym, orderId=oid)
            except Exception:
                continue
            status = st.get("status")
            if status == "FILLED":
                logger.info(f"  [MAKER] ✓ post-only FILLED {futures_sym} @ {st.get('avgPrice')} (saved taker fee)")
                return st
            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                logger.info(f"  [MAKER] post-only {status} for {futures_sym} — MARKET fallback")
                return None

        # Timed out unfilled (or partial) → cancel remainder and fall back to MARKET.
        try:
            client.futures_cancel_order(symbol=futures_sym, orderId=oid)
        except Exception:
            pass
        try:
            final = client.futures_get_order(symbol=futures_sym, orderId=oid)
            if final.get("status") == "FILLED":
                # Filled in the race between timeout and cancel.
                return final
            executed = float(final.get("executedQty", 0) or 0)
            if executed > 0:
                # Partial maker fill: top up the remainder at MARKET so we reach
                # target size. Return None lets caller MARKET the FULL qty, which
                # would double-fill — so handle the partial here explicitly.
                remaining = self._round_qty(futures_sym, quantity - executed)
                if remaining > 0:
                    bucket = int(time.time())
                    self._safe_create_order(client, {
                        "symbol": futures_sym,
                        "side": side,
                        "type": "MARKET",
                        "quantity": remaining,
                        "positionSide": position_side,
                        "newClientOrderId": f"x{futures_sym[:6]}{side[0]}m{bucket}"[:36],
                    })
                logger.info(f"  [MAKER] partial maker {executed}, topped up {remaining} at MARKET")
                return final
        except Exception as e:
            # Fill quantity is unknown. Falling back for the FULL quantity can
            # double the position when the maker order partially filled but
            # status lookup failed. Abort; broker-sync orphan adoption will
            # reconcile/protect any actual fill on the next cycle.
            raise RuntimeError(
                f"maker fill reconciliation uncertain for {futures_sym}: {e}"
            ) from e
        return None

    def _native_trailing_enabled(self) -> bool:
        try:
            from backend.services.risk_config import get_risk_config
            return bool(get_risk_config().native_trailing_enabled)
        except Exception:
            return False

    def _place_native_trailing_stop(self, client, futures_sym, side, quantity, entry_price, position_side):
        """Place a Binance TRAILING_STOP_MARKET to trail profit continuously."""
        from backend.services.risk_config import get_risk_config
        cfg = get_risk_config()
        # Clamp callbackRate to Binance's allowed 0.1–5.0 range
        callback = max(0.1, min(5.0, float(cfg.trailing_callback_rate)))
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        params = {
            "symbol": futures_sym,
            "side": close_side,
            "type": 'TRAILING_STOP_MARKET',
            "quantity": quantity,
            "callbackRate": callback,
            "positionSide": position_side,
        }
        # Activate only after price moves in favor by trailing_activation_pct.
        act_pct = float(getattr(cfg, "trailing_activation_pct", 0.0) or 0.0)
        if act_pct > 0 and entry_price > 0:
            if side == 'BUY':
                activation = entry_price * (1.0 + act_pct)
            else:
                activation = entry_price * (1.0 - act_pct)
            params["activationPrice"] = self._round_price(futures_sym, activation)
        try:
            order = self._safe_create_order(client, params)
            logger.info(
                f"  [TRAIL] native trailing stop placed for {futures_sym} "
                f"callbackRate={callback}% activation={params.get('activationPrice', 'now')} "
                f"(id={order.get('orderId')})"
            )
        except Exception as e:
            # If the activation price is rejected (already past market), retry
            # with immediate activation (no activationPrice).
            if "activationPrice" in params and ("-2021" in str(e) or "-1102" in str(e) or "would immediately" in str(e).lower()):
                params.pop("activationPrice", None)
                order = self._safe_create_order(client, params)
                logger.info(f"  [TRAIL] native trailing stop placed (immediate) for {futures_sym} (id={order.get('orderId')})")
            else:
                raise

    def _safe_create_order(self, client, order_params):
        """Place an order, auto-reducing precision on -1111 and retrying
        transient network / rate-limit (429) / server (5xx) errors."""
        import copy
        import time as _t
        params = copy.deepcopy(order_params)

        transient_retries = 3
        precision_retries = 0
        backoff = 0.5
        while True:
            try:
                return client.futures_create_order(**params)
            except Exception as e:
                err_str = str(e)
                # Precision error: trim decimals and retry only if params change
                if "-1111" in err_str and precision_retries < 4:
                    before = copy.deepcopy(params)
                    if "stopPrice" in params:
                        sp_str = str(params["stopPrice"])
                        if "." in sp_str:
                            decimals = len(sp_str.split(".")[1])
                            if decimals > 0:
                                params["stopPrice"] = float(f"{params['stopPrice']:.{decimals - 1}f}")
                    if "price" in params:
                        sym = params.get("symbol", "")
                        params["price"] = self._round_price(sym, float(params["price"]))
                    if "quantity" in params:
                        sym = params.get("symbol", "")
                        params["quantity"] = self._round_qty(sym, float(params["quantity"]))
                    if params != before:
                        precision_retries += 1
                        continue
                # Duplicate clientOrderId (-4015/-2010 "Duplicate") → idempotent no-op
                if "duplicate" in err_str.lower() or "-4015" in err_str:
                    client_id = params.get("newClientOrderId")
                    if client_id and params.get("symbol"):
                        try:
                            existing = client.futures_get_order(
                                symbol=params["symbol"],
                                origClientOrderId=client_id,
                            )
                            if existing and existing.get("status") in {
                                "NEW", "PARTIALLY_FILLED", "FILLED",
                            }:
                                logger.warning(
                                    f"  [idempotency] recovered duplicate order "
                                    f"{client_id} status={existing.get('status')}"
                                )
                                return existing
                        except Exception as lookup_error:
                            logger.error(
                                f"  [idempotency] duplicate {client_id} could not "
                                f"be reconciled: {lookup_error}"
                            )
                    logger.warning(
                        f"  [idempotency] duplicate order unresolved: {err_str[:120]}"
                    )
                    raise e
                # MIN_NOTIONAL violation — order is too small, no point retrying
                if "-4164" in err_str or "MIN_NOTIONAL" in err_str:
                    logger.error(
                        f"  [MIN_NOTIONAL] Order rejected: notional too small. "
                        f"Params: {params.get('symbol')} qty={params.get('quantity')} — {err_str[:200]}"
                    )
                    raise e
                # Transient errors: retry with backoff
                is_transient = any(t in err_str for t in (
                    "HTTP 429", "HTTP 418", "status_code=429", "status_code=418",
                    "-1003", " 500 ", "HTTP 500", "502", "503", "504",
                    "Timeout", "timed out", "Connection",
                ))
                if is_transient and transient_retries > 0:
                    transient_retries -= 1
                    logger.warning(f"  [retry] transient order error, retrying in {backoff}s: {err_str[:120]}")
                    _t.sleep(backoff)
                    backoff *= 2
                    continue
                raise e


    def replace_stop_loss(
        self,
        symbol: str,
        direction: str,
        new_stop_price: float,
        quantity: Optional[float] = None,
    ) -> dict:
        """Move the exchange-native reduce-only STOP_MARKET to a tighter level.

        Used by the in-loop trailing stop so the *exchange* backstop tracks the
        ratcheted DB stop. Without this, the native STOP_MARKET placed at entry
        stays frozen at the original (worst) stop while the in-memory trail
        advances — giving back all locked profit during the inter-cycle sleep or
        if the container dies.

        SAFETY CONTRACT:
          * Resolve the existing reduce-only STOP_MARKET for this symbol+side.
          * Only act if `new_stop_price` is strictly TIGHTER than the live
            exchange stop (closer to price) — never loosen, never churn.
          * Reject a stop that sits on the wrong side of current price (would
            trigger instantly, Binance -2021) — skip safely.
          * Binance permits only one closePosition STOP per position side
            (-4130), so replacement must cancel the old stop before creating
            the new one. If the new placement fails, immediately restore the
            previous stop. If restoration also fails, emergency-close the leg
            at market rather than leave it naked.

        Returns a dict with 'status': 'replaced'|'skipped'|'simulated'|'error'.
        """
        futures_sym = self._to_futures_symbol(symbol)
        if not futures_sym:
            return {'status': 'skipped', 'reason': f'{symbol} unsupported'}
        if self.dry_run:
            logger.info(f"[Binance Futures DRY-RUN] would move SL {futures_sym} -> {new_stop_price}")
            return {'status': 'simulated', 'symbol': futures_sym, 'new_stop': new_stop_price}
        if not live_binance_orders_allowed():
            logger.info("[SAFETY] skip replace_stop_loss: TRADING_MODE is not live")
            return {'status': 'skipped', 'reason': 'live Binance orders disabled unless TRADING_MODE=live'}
        if not new_stop_price or new_stop_price <= 0:
            return {'status': 'skipped', 'reason': 'invalid new_stop_price'}

        try:
            client = self._get_client()
            position_side = 'LONG' if direction.upper() == 'BUY' else 'SHORT'
            # Reduce-only stop side is opposite the position direction
            sl_side = 'SELL' if direction.upper() == 'BUY' else 'BUY'
            rounded_new = self._round_price(futures_sym, new_stop_price)

            # Current mark/last price — validate trigger side (avoid -2021)
            try:
                ticker = client.futures_symbol_ticker(symbol=futures_sym)
                current_price = float(ticker['price'])
            except Exception:
                current_price = 0.0
            if current_price > 0:
                # For a LONG the stop must sit BELOW price; for a SHORT, ABOVE.
                if direction.upper() == 'BUY' and rounded_new >= current_price:
                    return {'status': 'skipped',
                            'reason': f'new stop {rounded_new} >= price {current_price} (would trigger now)'}
                if direction.upper() == 'SELL' and rounded_new <= current_price:
                    return {'status': 'skipped',
                            'reason': f'new stop {rounded_new} <= price {current_price} (would trigger now)'}

            # Find the existing reduce-only STOP_MARKET for this symbol+side (regular + algo)
            existing = [
                o for o in self._collect_protective_orders(futures_sym, position_side, raise_on_error=True)
                if 'STOP' in (o.get('type') or '') and 'TAKE_PROFIT' not in (o.get('type') or '')
            ]

            # Only tighten: compare against the current exchange stop level
            if existing:
                cur_levels = [float(o.get('price') or 0) for o in existing if float(o.get('price') or 0) > 0]
                if not cur_levels:
                    logger.warning(
                        f"  [TRAIL-SL] {futures_sym} existing stop has no readable trigger price; "
                        "replacing without tightness comparison"
                    )
                elif direction.upper() == 'BUY':
                    cur_stop = max(cur_levels)  # highest = least loose for a long
                    if rounded_new <= cur_stop:
                        return {'status': 'skipped',
                                'reason': f'not tighter ({rounded_new} <= exchange {cur_stop})'}
                else:
                    cur_stop = min(cur_levels)  # lowest = least loose for a short
                    if rounded_new >= cur_stop:
                        return {'status': 'skipped',
                                'reason': f'not tighter ({rounded_new} >= exchange {cur_stop})'}

            # Resolve quantity to protect (position size). Fall back to existing
            # order qty, then the model-supplied quantity.
            qty = quantity
            if not qty:
                pos = next(
                    (p for p in self.get_positions(raise_on_error=True)
                     if p['symbol'] == futures_sym
                     and ((direction.upper() == 'BUY' and p['side'] == 'BUY')
                          or (direction.upper() == 'SELL' and p['side'] == 'SELL'))),
                    None,
                )
                qty = pos['quantity'] if pos else (
                    float(existing[0].get('origQty', 0)) if existing else 0.0)
            qty = self._round_qty(futures_sym, qty) if qty else 0.0
            if qty <= 0:
                return {'status': 'skipped', 'reason': 'no open position quantity to protect'}

            # Binance rejects a second closePosition stop with -4130. Cancel
            # old STOP orders first, then place the tighter stop immediately.
            cancelled = []
            old_stop_levels = [
                float(o.get('price') or 0)
                for o in existing
                if float(o.get('price') or 0) > 0
            ]
            for o in existing:
                oid = o.get('order_id') or o.get('algo_id')
                try:
                    if o.get('algo_id'):
                        client.futures_cancel_algo_order(algoId=o['algo_id'])
                    else:
                        client.futures_cancel_order(symbol=futures_sym, orderId=int(o['order_id']))
                    cancelled.append(str(oid))
                except Exception as ce:
                    logger.error(
                        f"  [TRAIL-SL] {futures_sym} refused to replace stop: "
                        f"could not cancel old stop {oid}: {ce}"
                    )
                    return {
                        'status': 'error',
                        'reason': 'old_stop_cancel_failed',
                        'message': str(ce),
                    }

            # Use closePosition=true so protection automatically tracks the
            # entire pyramided leg.
            new_params = {
                "symbol": futures_sym,
                "side": sl_side,
                "type": 'STOP_MARKET',
                "stopPrice": rounded_new,
                "closePosition": "true",
                "positionSide": position_side,
            }
            try:
                new_order = self._safe_create_order(client, new_params)
                new_id = str(new_order.get('algoId') or new_order.get('orderId', ''))
            except Exception as new_error:
                # Restore the previous exchange stop before returning. Use the
                # tightest known previous level if duplicate stale stops existed.
                restore_level = None
                if old_stop_levels:
                    restore_level = (
                        max(old_stop_levels)
                        if direction.upper() == 'BUY'
                        else min(old_stop_levels)
                    )
                try:
                    if restore_level is None:
                        raise RuntimeError("previous stop trigger price unavailable")
                    restore_params = dict(new_params)
                    restore_params["stopPrice"] = restore_level
                    restored = self._safe_create_order(client, restore_params)
                    logger.error(
                        f"  [TRAIL-SL] {futures_sym} new stop failed ({new_error}); "
                        f"restored previous stop @ {restore_level} "
                        f"(id={restored.get('algoId') or restored.get('orderId')})"
                    )
                    return {
                        'status': 'error',
                        'reason': 'new_stop_failed_old_restored',
                        'message': str(new_error),
                        'restored_stop': restore_level,
                    }
                except Exception as restore_error:
                    logger.critical(
                        f"  [TRAIL-SL] {futures_sym} STOP REPLACEMENT AND RESTORE FAILED: "
                        f"new={new_error}; restore={restore_error} — emergency closing position"
                    )
                    # Instead of a dangerous, standalone market order, trigger the global emergency_halt
                    logger.critical(
                        f"  [TRAIL-SL] {futures_sym} STOP REPLACEMENT AND RESTORE FAILED. "
                        "Triggering Sentry Emergency Halt to protect the account."
                    )
                    # Circular: sentry_emergency imports binance_futures_broker.
                    from backend.services.sentry_emergency import trigger_emergency_halt_sync
                    try:
                        trigger_emergency_halt_sync(
                            reason=f"STOP REPLACEMENT AND RESTORE FAILED for {futures_sym}",
                            source="replace_stop_loss",
                        )
                    except Exception as eh_err:
                        logger.critical(f"Failed to trigger emergency halt: {eh_err}")
                    return {
                        'status': 'critical',
                        'reason': 'stop_replace_and_restore_failed_sentry_triggered',
                        'message': str(restore_error),
                    }

            logger.info(
                f"  [TRAIL-SL] {futures_sym} exchange stop -> {rounded_new} "
                f"(new_id={new_id}, cancelled={cancelled or 'none'})"
            )
            return {'status': 'replaced', 'symbol': futures_sym,
                    'new_stop': rounded_new, 'new_order_id': new_id,
                    'cancelled': cancelled, 'quantity': qty}

        except Exception as e:
            logger.warning(
                f"  [TRAIL-SL] {symbol} exchange-stop move FAILED before replacement completed: {e}"
            )
            return {'status': 'error', 'message': str(e)}

    def replace_take_profit(
        self,
        symbol: str,
        direction: str,
        new_take_profit: float,
    ) -> dict:
        """Replace the exchange closePosition TAKE_PROFIT_MARKET order.

        Unlike SL replacement, a failed TP restore does not leave downside
        unprotected (the SL remains live), but SQL must not claim the new TP
        succeeded unless Binance accepted it.
        """
        futures_sym = self._to_futures_symbol(symbol)
        if not futures_sym:
            return {'status': 'skipped', 'reason': f'{symbol} unsupported'}
        if self.dry_run:
            return {'status': 'simulated', 'symbol': futures_sym, 'new_take_profit': new_take_profit}
        if not live_binance_orders_allowed():
            logger.info("[SAFETY] skip replace_take_profit: TRADING_MODE is not live")
            return {'status': 'skipped', 'reason': 'live Binance orders disabled unless TRADING_MODE=live'}
        if not new_take_profit or new_take_profit <= 0:
            return {'status': 'skipped', 'reason': 'invalid new_take_profit'}

        client = self._get_client()
        position_side = 'LONG' if direction.upper() == 'BUY' else 'SHORT'
        close_side = 'SELL' if direction.upper() == 'BUY' else 'BUY'
        rounded_new = self._round_price(futures_sym, new_take_profit)
        ticker = client.futures_symbol_ticker(symbol=futures_sym)
        current_price = float(ticker.get('price') or 0)
        if current_price > 0:
            if direction.upper() == 'BUY' and rounded_new <= current_price:
                return {'status': 'skipped', 'reason': 'long TP must be above current price'}
            if direction.upper() == 'SELL' and rounded_new >= current_price:
                return {'status': 'skipped', 'reason': 'short TP must be below current price'}

        existing = [
            o for o in self._collect_protective_orders(
                futures_sym, position_side, raise_on_error=True,
            )
            if 'TAKE_PROFIT' in (o.get('type') or '')
        ]
        old_levels = [
            float(o.get('price') or 0)
            for o in existing
            if float(o.get('price') or 0) > 0
        ]
        for o in existing:
            if o.get('algo_id'):
                client.futures_cancel_algo_order(algoId=o['algo_id'])
            else:
                client.futures_cancel_order(
                    symbol=futures_sym, orderId=int(o['order_id']),
                )

        params = {
            "symbol": futures_sym,
            "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": rounded_new,
            "closePosition": "true",
            "positionSide": position_side,
        }
        try:
            order = self._safe_create_order(client, params)
            return {
                'status': 'replaced',
                'symbol': futures_sym,
                'new_take_profit': rounded_new,
                'order_id': str(order.get('algoId') or order.get('orderId', '')),
            }
        except Exception as new_error:
            if old_levels:
                restore = dict(params)
                restore["stopPrice"] = old_levels[0]
                try:
                    self._safe_create_order(client, restore)
                    return {
                        'status': 'error',
                        'reason': 'new_tp_failed_old_restored',
                        'message': str(new_error),
                    }
                except Exception as restore_error:
                    logger.error(
                        f"  [TP-REPLACE] {futures_sym} new and restore both failed: "
                        f"{new_error}; {restore_error}"
                    )
            return {
                'status': 'error',
                'reason': 'take_profit_replace_failed',
                'message': str(new_error),
            }

    @property
    def is_connected(self) -> bool:
        """Check if Binance client is initialized and accessible."""
        return self._client is not None or self.dry_run

    def connect(self) -> bool:
        """Initialize client connection to Binance Futures."""
        try:
            client = self._get_client()
            return client is not None
        except Exception as e:
            logger.error(f"BinanceFutures connect error: {e}")
            return False

    def disconnect(self) -> None:
        """Clear client connection."""
        self._client = None

    def status(self) -> dict:
        """Return status dictionary for Binance Futures."""
        balance = self.get_balance()
        return {
            "connected": self.is_connected,
            "dry_run": self.dry_run,
            "broker": "binance_futures",
            "balance": balance.get("balance", 0.0),
            "equity": balance.get("equity", 0.0),
            "margin": balance.get("margin_used", 0.0),
            "open_positions": len(self.get_positions()),
        }

    def close_position(
        self,
        position_id: str | int = "",
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> dict:
        """Close an open position on Binance Futures by placing a market close order.

        `direction` is the side of the position being closed, which in hedge
        mode selects positionSide. Hardcoding SELL here aimed every close at
        the SHORT leg, so closing a long either errored or hit the wrong leg.
        """
        if not symbol:
            return {"status": "error", "message": "Symbol required to close Binance position"}

        held = (direction or "").upper()
        if held in ("LONG", "BUY"):
            held = "BUY"
        elif held in ("SHORT", "SELL"):
            held = "SELL"
        else:
            futures_sym = self._to_futures_symbol(symbol)
            held = next(
                (
                    "BUY" if (p.get("side") or "").upper() in ("BUY", "LONG") else "SELL"
                    for p in self.get_positions()
                    if p.get("symbol") == futures_sym
                    and abs(float(p.get("quantity") or 0)) > 0
                ),
                "",
            )
            if not held:
                return {"status": "already_flat", "symbol": symbol}

        futures_sym = self._to_futures_symbol(symbol)
        side = "LONG" if held == "BUY" else "SHORT"
        live_qty = self._live_position_qty(futures_sym, side) if futures_sym else 0.0
        if live_qty <= 0:
            return {"status": "already_flat", "symbol": symbol}
        return self.place_order(
            symbol=symbol, direction=held, action="close", quantity=live_qty
        )

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        if not live_binance_orders_allowed():
            logger.info("[SAFETY] skip cancel_order: TRADING_MODE is not live")
            return {'success': False, 'message': 'live Binance orders disabled unless TRADING_MODE=live'}
        try:
            client = self._get_client()
            if symbol:
                futures_sym = self._to_futures_symbol(symbol)
                result = client.futures_cancel_order(symbol=futures_sym, orderId=order_id)
            else:
                # Binance requires a symbol to cancel an order by ID. 
                # This is a limitation we must work around by trying active symbols or returning an error.
                return {'success': False, 'message': 'Binance Futures requires a symbol to cancel an order.'}
            return {'success': True, 'message': 'Order cancelled', 'raw': result}
        except Exception as e:
            return {'success': False, 'message': str(e)}


# ── Module-level singleton ────────────────────────────────────────────────────
binance_futures_broker = BinanceFuturesService()
