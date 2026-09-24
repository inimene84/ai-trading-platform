"""
Layered Multi-Asset Signal Candidate & Timing Engine
Coordinates market scanning (cTrader & Binance), news/macro correlation,
4 strategy evaluators (Momentum, Fade, Straddle, Slingshot), precision pip-margin
sizing, and timing execution windows (Pre-event, At-release, Post-reaction, Bar-close).
"""

import asyncio
import time
import logging
import os
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

from backend.services.ctrader_service import CTraderService, ctrader_service
from backend.services.ctrader_trade_sync import (
    persist_ctrader_execution,
    count_open_ctrader_db_trades,
    open_ctrader_db_symbols,
    open_ctrader_db_positions,
    try_open_ctrader_db_positions,
)
from backend.services.unified_trading import UnifiedTrading, UnifiedOrder, OrderSide, OrderType
from backend.services.binance_futures_service import binance_futures_broker
from backend.services.binance_market_data import binance_market_data
from backend.services.multi_asset_bars import classify_symbol, tf_to_binance_interval
from backend.services.trading_mode import live_ctrader_orders_allowed
from backend.services.symbol_aliases import same_crypto_perp_leg
from backend.services.market_hours import is_venue_open

logger = logging.getLogger(__name__)


class TimingMode:
    PRE_EVENT = "PRE_EVENT"
    AT_RELEASE = "AT_RELEASE"
    POST_REACTION = "POST_REACTION"
    BAR_CLOSE = "BAR_CLOSE"


class CandidateStatus:
    PENDING = "PENDING"
    READY = "READY"
    EXECUTED = "EXECUTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


TERMINAL_CANDIDATE_STATUSES = frozenset({
    CandidateStatus.EXECUTED,
    CandidateStatus.EXPIRED,
    CandidateStatus.CANCELLED,
})
CANDIDATE_TTL_SECONDS = 24 * 3600
# News must not re-arm a setup that already filled while the EXECUTED row
# is still in memory (same 24h TTL). Missing timestamps fail closed.
RECENT_EXECUTED_REARM_SECONDS = CANDIDATE_TTL_SECONDS
EQUITY_FALLBACK_USD = 10000.0
LIVE_CANDIDATE_STATUSES = frozenset({
    CandidateStatus.PENDING,
    CandidateStatus.READY,
})


class SignalCandidateEngine:
    def __init__(self):
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self._equity_cache: Dict[str, float] = {}
        self.timing_config: Dict[str, Any] = {
            "pre_event_window_min": 15,       # Minutes before release to evaluate pre-event
            "at_release_window_sec": 45,      # Max seconds for at-release before moving to post
            "post_reaction_delay_min": 2,     # Wait X mins after release to enter post-reaction
            "post_reaction_window_min": 10,   # Window duration for post-reaction entries
            "max_spread_pips": 2.0,           # Max allowable spread in pips for FX
            "max_slippage_pips": 1.5,         # Max slippage tolerance
            "default_risk_pct": 0.5,          # Default risk % per trade
            # Logged fallback only — live sizing uses broker equity via _resolve_equity().
            "account_equity_override": EQUITY_FALLBACK_USD,
            "strategies_enabled": {
                "momentum": True,
                "fade": True,
                "straddle": True,
                "slingshot": True,
            },
        }
        self.execution_config: Dict[str, Any] = {
            "forex_only": os.getenv("CTRADER_FOREX_ONLY", "false").lower() == "true",
            "max_open_ctrader_positions": int(
                os.getenv("CTRADER_MAX_OPEN_POSITIONS")
                or os.getenv("MAX_CTRADER_POSITIONS")
                or "10"
            ),
            "max_ready_per_poll": int(os.getenv("CTRADER_MAX_READY_PER_POLL", "10")),
            "max_ctrader_lots": float(os.getenv("CTRADER_MAX_LOTS", "0.10")),
            "max_metal_lots": float(os.getenv("CTRADER_MAX_METAL_LOTS", "0.01")),
            "one_position_per_symbol": os.getenv("CTRADER_ONE_POSITION_PER_SYMBOL", "true").lower() == "true",
            "include_metals": os.getenv("CTRADER_INCLUDE_METALS", "true").lower() == "true",
            "max_same_base": int(os.getenv("CTRADER_MAX_SAME_BASE", "2")),
            "max_currency_exposure": int(os.getenv("CTRADER_MAX_CURRENCY_EXPOSURE", "2")),
            "max_candidates": int(os.getenv("CTRADER_MAX_CANDIDATES", "200")),
            "max_binance_positions": int(os.getenv("MAX_BINANCE_POSITIONS", "10")),
            # Portfolio risk budget per broker book: sum of risk_usd across live
            # (PENDING/READY) candidates plus the incoming trade may not exceed
            # this % of account equity. 12 READY x 0.5% used to stack to ~6%.
            "max_portfolio_risk_pct": float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "3.0")),
        }
        self._symbol_cooldowns: Dict[str, float] = {}
        self._venue_closed_until: Dict[str, float] = {}
        self.cooldown_duration_sec: int = int(os.getenv("SYMBOL_COOLDOWN_MINUTES", "45")) * 60
        self.momentum_min_adx: float = float(os.getenv("MOMENTUM_MIN_ADX", "22.0"))

        # High-impact calendar currencies map to the liquid FX pair, never gold.
        # NZD OCR / AUD GDP used to become XAUUSD SELL and n8n retried forever.
        self.MACRO_CURRENCY_PAIRS: Dict[str, str] = {
            "USD": "EURUSD",
            "EUR": "EURUSD",
            "GBP": "GBPUSD",
            "JPY": "USDJPY",
            "AUD": "AUDUSD",
            "NZD": "NZDUSD",
            "CAD": "USDCAD",
            "CHF": "USDCHF",
        }

    def set_symbol_cooldown(self, symbol: str, duration_sec: Optional[int] = None) -> None:
        """Lock out a symbol from opening new positions after a stop-out or adverse exit."""
        duration = duration_sec if duration_sec is not None else self.cooldown_duration_sec
        expiry = time.time() + max(0, duration)
        sym = str(symbol or "").upper()
        self._symbol_cooldowns[sym] = expiry
        logger.info(
            f"[{sym}] Cooldown activated until "
            f"{datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat()} ({duration/60:.1f}m)"
        )

    def is_symbol_cooling_down(self, symbol: str) -> bool:
        sym = str(symbol or "").upper()
        expiry = self._symbol_cooldowns.get(sym)
        if not expiry:
            return False
        if time.time() < expiry:
            return True
        self._symbol_cooldowns.pop(sym, None)
        return False

    # ── P1: FX pre-execution timing gate ────────────────────────────────────
    # FX entries historically bypassed every quality gate the DecisionEngine
    # path enforces (Kronos veto, heuristic timing) — measured live win rate
    # on 2026-09-08 was 20%. FX_KRONOS_GATE=enforce (default) blocks trades
    # when Kronos predicts opposite/neutral; =shadow logs vetoes without blocking; =off disables.
    @staticmethod
    def _fx_gate_mode() -> str:
        return os.getenv("FX_KRONOS_GATE", "enforce").strip().lower()

    @staticmethod
    def _fade_h1_filter_enabled() -> bool:
        return os.getenv("FADE_H1_FILTER_ENABLED", "true").strip().lower() == "true"

    async def _fx_timing_gate(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        bars: List[Dict[str, Any]],
    ):
        from backend.services.kronos_gate import apply_pre_execution_gate
        from backend.services import kronos_service

        gate_bars = bars
        try:
            # cTrader trendbars carry epoch `time`; give kronos an ISO date so
            # its (symbol, interval, last_bar_date) dedupe key rotates per bar.
            gate_bars = [
                ({**b, "date": datetime.fromtimestamp(float(b["time"]), tz=timezone.utc).isoformat()}
                 if isinstance(b, dict) and b.get("time") and not b.get("date") else b)
                for b in bars
            ]
        except Exception:
            gate_bars = bars

        kronos_result: Dict[str, Any] = {}
        try:
            kronos_result = await kronos_service.predict(gate_bars, symbol, interval="m5")
        except Exception as err:
            logger.debug(f"FX gate: kronos predict failed for {symbol}: {err}")

        return apply_pre_execution_gate(
            strategy_signal=direction,
            strategy_confidence=confidence,
            kronos_result=kronos_result,
            bars=gate_bars,
            shadow_mode=(self._fx_gate_mode() != "enforce"),
            symbol=symbol,
        )

    def _is_ctrader_forex_candidate(self, cand: Dict[str, Any]) -> bool:
        if cand.get("broker") != "ctrader":
            return False
        asset = classify_symbol(str(cand.get("symbol", "")))
        if asset == "forex":
            return True
        if asset == "metal" and self.execution_config.get("include_metals"):
            return True
        return False

    def _open_ctrader_position_count(self) -> int:
        """Max of broker opens and DB opens so paper/sim fills cannot bypass the cap."""
        broker_n = 0
        try:
            st = ctrader_service.status()
            if st.get("connected"):
                broker_n = int(st.get("open_positions") or 0)
            else:
                broker_n = len(ctrader_service.get_positions() or [])
        except Exception as err:
            logger.warning(f"Could not read cTrader open positions: {err}")
            broker_n = 0
        db_n = 0
        try:
            db_n = int(count_open_ctrader_db_trades() or 0)
        except Exception as err:
            logger.warning(f"Could not read cTrader DB open positions: {err}")
        return max(broker_n, db_n)

    def _open_ctrader_symbols(self) -> Set[str]:
        """Union of broker + DB open symbols so one-per-symbol works in paper/sim."""
        syms: Set[str] = set()
        try:
            for p in (ctrader_service.get_positions() or []):
                if p.get("symbol"):
                    syms.add(str(p.get("symbol")).upper())
        except Exception as err:
            logger.warning(f"Could not read cTrader open symbols: {err}")
        try:
            syms |= {s.upper() for s in (open_ctrader_db_symbols() or set()) if s}
        except Exception as err:
            logger.warning(f"Could not read cTrader DB open symbols: {err}")
        return syms

    @staticmethod
    def _symbol_base(symbol: str) -> str:
        sym = str(symbol or "").upper()
        return sym[:3] if len(sym) >= 6 else sym

    def _try_open_binance_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Open Binance futures positions. None means the book could not be read."""
        try:
            return list(binance_futures_broker.get_positions(raise_on_error=True) or [])
        except Exception as err:
            logger.warning(f"Could not read Binance open positions: {err}")
            return None

    def _open_binance_positions(self) -> List[Dict[str, Any]]:
        """Open Binance futures positions; empty list on API errors (legacy)."""
        found = self._try_open_binance_positions()
        return found if found is not None else []

    def _open_binance_symbols(self) -> Set[str]:
        return {
            str(p.get("symbol") or "").upper()
            for p in self._open_binance_positions()
            if p.get("symbol")
        }

    @staticmethod
    def _normalize_direction(direction: Optional[str]) -> str:
        side = str(direction or "").upper()
        if side in ("BUY", "LONG"):
            return "BUY"
        if side in ("SELL", "SHORT"):
            return "SELL"
        return side

    # ── cTrader FX side invert ──────────────────────────────────────────────
    # IC demo book (Sep 11–24, n=55 closed): ~18% win rate / −$121 vs ~73% /
    # +$121 if every side were flipped. MOMENTUM_TREND_PULSE dominated.
    # BUY=1 / SELL=2 on Spotware is conventional — this is strategy polarity
    # vs subsequent FX moves (trend pulse in a mean-reverting tape), not a
    # protocol mapping bug. Default ON for cTrader FX only; Binance untouched.
    # Disable with CTRADER_INVERT_SIDE=false after the edge is re-measured.
    @staticmethod
    def _ctrader_invert_side_enabled() -> bool:
        return os.getenv("CTRADER_INVERT_SIDE", "true").strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def invert_trade_side(direction: Optional[str]) -> str:
        side = SignalCandidateEngine._normalize_direction(direction)
        if side == "BUY":
            return "SELL"
        if side == "SELL":
            return "BUY"
        return side

    @staticmethod
    def mirror_price_around_entry(
        entry: Optional[float],
        level: Optional[float],
        digits: int = 5,
    ) -> Optional[float]:
        """Reflect a protective level through entry so SL/TP swap sides with the trade."""
        if entry is None or level is None:
            return level
        return round((2.0 * float(entry)) - float(level), digits)

    @classmethod
    def invert_side_and_protection(
        cls,
        direction: Optional[str],
        entry: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        *,
        digits: int = 5,
    ) -> tuple[str, Optional[float], Optional[float]]:
        """Flip BUY↔SELL and mirror SL/TP around entry (stop stays on the loss side)."""
        return (
            cls.invert_trade_side(direction),
            cls.mirror_price_around_entry(entry, stop_loss, digits),
            cls.mirror_price_around_entry(entry, take_profit, digits),
        )

    def _should_invert_ctrader_fx_side(self, broker: str, symbol: str) -> bool:
        if not self._ctrader_invert_side_enabled():
            return False
        if str(broker or "").lower() != "ctrader":
            return False
        return classify_symbol(str(symbol or "")) == "forex"

    def apply_ctrader_fx_side_invert(
        self,
        symbol: str,
        signal: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Invert a cTrader FX signal in place and re-clamp SL/TP for the new side.

        Returns the mutated signal, or None when post-invert geometry is rejected.
        Idempotent when ``side_inverted`` is already set.
        """
        if signal.get("side_inverted"):
            return signal
        original = self._normalize_direction(signal.get("direction"))
        entry = signal.get("entry_price")
        if entry is None:
            logger.info(
                "[%s] cTrader FX invert skipped: missing entry, refusing unmirrored flip",
                symbol,
            )
            return None
        digits = CTraderService.digits_for(symbol)
        flipped, sl, tp = self.invert_side_and_protection(
            original,
            entry,
            signal.get("stop_loss"),
            signal.get("take_profit"),
            digits=digits,
        )
        if flipped == original:
            return signal
        natural_sl, natural_tp = sl, tp
        if entry is not None:
            sl, tp = CTraderService.clamp_protective_prices(
                symbol, float(entry), sl, tp, direction=flipped,
            )
            validated = CTraderService.validate_protective_geometry(
                symbol,
                float(entry),
                natural_sl,
                natural_tp,
                sl,
                tp,
                direction=flipped,
            )
            if validated is None:
                logger.info(
                    "[%s] cTrader FX invert veto: post-invert geometry rejected "
                    "(%s → %s)",
                    symbol, original, flipped,
                )
                return None
            sl, tp = validated
        signal["direction"] = flipped
        signal["stop_loss"] = sl
        signal["take_profit"] = tp
        signal["original_direction"] = original
        signal["side_inverted"] = True
        reason = str(signal.get("reason") or "").rstrip()
        tag = f"[cTrader FX side inverted {original}→{flipped}; CTRADER_INVERT_SIDE=true]"
        signal["reason"] = f"{reason} {tag}".strip() if reason else tag
        logger.info(
            "[%s] cTrader FX side invert %s → %s sl=%s tp=%s (entry=%s)",
            symbol, original, flipped, sl, tp, entry,
        )
        return signal

    def _ensure_ctrader_fx_side_invert(self, cand: Dict[str, Any]) -> bool:
        """Apply invert to a live candidate once. False = skip (geometry veto)."""
        if not self._should_invert_ctrader_fx_side(
            str(cand.get("broker") or ""), str(cand.get("symbol") or "")
        ):
            return True
        if cand.get("side_inverted"):
            return True
        updated = self.apply_ctrader_fx_side_invert(str(cand.get("symbol") or ""), cand)
        if updated is None:
            return False
        entry = cand.get("entry_price")
        sl = cand.get("stop_loss")
        if entry and sl:
            cand["sizing"] = self._calculate_size(
                str(cand.get("symbol") or ""),
                float(entry),
                float(sl),
                "ctrader",
            )
        return True

    def _open_position_keys(self, broker: str) -> Optional[Set[tuple]]:
        """(symbol, BUY/SELL) pairs currently open, or None if the book is unreadable."""
        keys: Set[tuple] = set()
        if broker == "binance_futures":
            positions = self._try_open_binance_positions()
            if positions is None:
                return None
            for pos in positions:
                sym = str(pos.get("symbol") or "").upper()
                side = self._normalize_direction(pos.get("direction") or pos.get("side"))
                if sym and side:
                    keys.add((sym, side))
            return keys

        try:
            status: Dict[str, Any] = {}
            try:
                status = ctrader_service.status() or {}
            except Exception:
                status = {}
            positions = list(ctrader_service.get_positions() or [])
            connected = bool(status.get("connected"))
            reported_n = int(status.get("open_positions") or 0)
            if connected and reported_n > 0 and not positions:
                logger.warning(
                    "cTrader reports %s open positions but the cache is empty — "
                    "treating the book as unknown",
                    reported_n,
                )
                return None
            for pos in positions:
                sym = str(pos.get("symbol") or "").upper()
                side = self._normalize_direction(pos.get("direction") or pos.get("side"))
                if sym and side:
                    keys.add((sym, side))
        except Exception as err:
            logger.warning(f"Could not read cTrader open position sides: {err}")
            return None
        try:
            db_rows = try_open_ctrader_db_positions()
        except Exception as err:
            logger.warning(f"Could not read cTrader DB open position sides: {err}")
            db_rows = None
        if db_rows is None:
            # Empty broker cache plus an unread DB is not a flat book.
            if not keys:
                return None
            return keys
        for row in db_rows:
            sym = str(row.get("symbol") or "").upper()
            side = self._normalize_direction(row.get("direction") or row.get("side"))
            if sym and side:
                keys.add((sym, side))
        return keys

    def _has_open_position(self, broker: str, symbol: str, direction: str) -> bool:
        """True when the side is open, or when the book cannot be verified.

        USDT/USDC perps are one leg: BTCUSDC BUY matches a live BTCUSDT BUY.
        """
        key = (str(symbol or "").upper(), self._normalize_direction(direction))
        if not key[0] or not key[1]:
            return False
        keys = self._open_position_keys(broker)
        if keys is None:
            return True
        if key in keys:
            return True
        return any(
            kside == key[1] and same_crypto_perp_leg(ksym, key[0])
            for ksym, kside in keys
        )

    def _has_live_strategy_twin(self, symbol: str, strategy: str, direction: str) -> bool:
        sym = str(symbol or "").upper()
        side = self._normalize_direction(direction)
        return any(
            same_crypto_perp_leg(str(c.get("symbol") or ""), sym)
            and c.get("strategy") == strategy
            and self._normalize_direction(c.get("direction")) == side
            and c.get("status") in LIVE_CANDIDATE_STATUSES
            for c in self.candidates.values()
        )

    def _has_live_same_direction_candidate(self, symbol: str, direction: str) -> bool:
        """PENDING/READY on the same symbol+direction, any strategy."""
        sym = str(symbol or "").upper()
        side = self._normalize_direction(direction)
        return any(
            same_crypto_perp_leg(str(c.get("symbol") or ""), sym)
            and self._normalize_direction(c.get("direction")) == side
            and c.get("status") in LIVE_CANDIDATE_STATUSES
            for c in self.candidates.values()
        )

    def _has_recent_executed_twin(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        now_ts: Optional[int] = None,
        *,
        match_strategy: bool = True,
    ) -> bool:
        """True when a recent EXECUTED twin still sits in memory.

        Missing timestamps fail closed — a filled twin with no clock must not
        be treated as stale enough to re-arm. When match_strategy is False,
        any strategy on the same symbol+direction counts (stops cross-strategy
        re-arm after a fill).
        """
        now = int(now_ts if now_ts is not None else time.time())
        sym = str(symbol or "").upper()
        side = self._normalize_direction(direction)
        for cand in self.candidates.values():
            if cand.get("status") != CandidateStatus.EXECUTED:
                continue
            if not same_crypto_perp_leg(str(cand.get("symbol") or ""), sym):
                continue
            if match_strategy and cand.get("strategy") != strategy:
                continue
            if self._normalize_direction(cand.get("direction")) != side:
                continue
            epoch = self._candidate_epoch(cand)
            if epoch is None:
                return True
            if now - epoch < RECENT_EXECUTED_REARM_SECONDS:
                return True
        return False

    def _live_committed_risk_usd(
        self,
        broker: str,
        *,
        exclude_id: Optional[str] = None,
    ) -> float:
        return sum(
            float((c.get("sizing") or {}).get("risk_usd") or 0.0)
            for c in self.candidates.values()
            if c.get("id") != exclude_id
            and c.get("status") in LIVE_CANDIDATE_STATUSES
            and c.get("broker") == broker
        )

    def _portfolio_risk_breach(
        self,
        broker: str,
        new_risk_usd: float,
        *,
        exclude_id: Optional[str] = None,
    ) -> Optional[str]:
        """Reason string when live book + incoming risk exceeds the budget."""
        max_risk_pct = float(self.execution_config.get("max_portfolio_risk_pct") or 0)
        if max_risk_pct <= 0:
            return None
        equity = self._resolve_equity(str(broker or ""))
        if equity <= 0:
            return None
        committed = self._live_committed_risk_usd(broker, exclude_id=exclude_id)
        budget = equity * max_risk_pct / 100.0
        incoming = float(new_risk_usd or 0.0)
        if committed + incoming > budget:
            return (
                f"Portfolio risk budget exceeded for {broker}: "
                f"committed ${committed:.2f} + new ${incoming:.2f} > "
                f"{max_risk_pct}% of ${equity:.2f} equity (${budget:.2f})."
            )
        return None

    def _binance_position_cap_breach(self, symbol: str) -> Optional[str]:
        """One-per-symbol and max-positions caps for the Binance book."""
        open_binance = self._try_open_binance_positions()
        if open_binance is None:
            return "Could not read Binance positions — refusing."
        cand_sym = str(symbol or "").upper()
        if self.execution_config.get("one_position_per_symbol"):
            open_syms = {
                str(p.get("symbol") or "").upper()
                for p in open_binance
                if p.get("symbol")
            }
            hit = next((s for s in open_syms if same_crypto_perp_leg(cand_sym, s)), None)
            if cand_sym and hit:
                return (
                    f"Already have an open Binance position in {hit} "
                    f"(same perp leg as {cand_sym})."
                )
        max_binance = int(self.execution_config.get("max_binance_positions") or 0)
        if max_binance > 0 and len(open_binance) >= max_binance:
            return f"Max open Binance positions reached ({max_binance})."
        return None

    def _refuse_new_candidate_reason(
        self,
        *,
        symbol: str,
        broker: str,
        strategy: str,
        direction: str,
        risk_usd: float = 0.0,
        check_recent_executed: bool = False,
    ) -> Optional[str]:
        """Scan-time admission control. Returns a skip reason or None.

        Does not loosen any execute-time cap — those still re-check.
        """
        if self._has_live_strategy_twin(symbol, strategy, direction):
            return (
                f"live {strategy} {self._normalize_direction(direction)} "
                f"candidate already armed"
            )
        if self._has_live_same_direction_candidate(symbol, direction):
            return (
                f"live {self._normalize_direction(direction)} candidate already "
                f"armed for {str(symbol or '').upper()} (any strategy)"
            )
        if check_recent_executed and self._has_recent_executed_twin(
            symbol, strategy, direction, match_strategy=False
        ):
            return (
                f"recently executed {self._normalize_direction(direction)} "
                f"twin — refusing re-arm"
            )
        if self._has_open_position(broker, symbol, direction):
            return (
                f"open {broker} position already exists for "
                f"{str(symbol or '').upper()} {self._normalize_direction(direction)}"
            )
        risk_block = self._portfolio_risk_breach(broker, risk_usd)
        if risk_block:
            return risk_block
        if broker == "binance_futures":
            return self._binance_position_cap_breach(symbol)
        return None

    def cancel_candidate(self, candidate_id: str, reason: str = "admin cancel") -> bool:
        """Mark a live candidate CANCELLED. Does not touch broker positions."""
        cand = self.candidates.get(candidate_id)
        if not cand:
            return False
        if cand.get("status") not in LIVE_CANDIDATE_STATUSES:
            return False
        cand["status"] = CandidateStatus.CANCELLED
        cand["cancel_reason"] = reason
        cand["cancelled_at"] = datetime.now(timezone.utc).isoformat()
        return True

    def cancel_live_same_direction(
        self,
        symbol: str,
        direction: str,
        *,
        reason: str = "duplicate of open position",
    ) -> List[str]:
        """Cancel PENDING/READY rows that stack on an open symbol+direction."""
        cancelled: List[str] = []
        sym = str(symbol or "").upper()
        side = self._normalize_direction(direction)
        for cid, cand in list(self.candidates.items()):
            if cand.get("status") not in LIVE_CANDIDATE_STATUSES:
                continue
            if not same_crypto_perp_leg(str(cand.get("symbol") or ""), sym):
                continue
            if self._normalize_direction(cand.get("direction")) != side:
                continue
            cand["status"] = CandidateStatus.CANCELLED
            cand["cancel_reason"] = reason
            cand["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            cancelled.append(cid)
        return cancelled

    def _open_ctrader_base_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for sym in self._open_ctrader_symbols():
            base = self._symbol_base(sym)
            if base:
                counts[base] = counts.get(base, 0) + 1
        return counts

    def _open_ctrader_currency_exposure(self) -> Dict[str, float]:
        """Net signed currency exposure across open cTrader positions.

        BUY EURUSD = long EUR / short USD. Broker positions win on symbol
        conflicts; DB rows cover paper/sim fills the broker never saw.
        The 2026-09-08 book held 4 JPY-linked positions at once — one
        JPY headline would have hit all four.
        """
        exposure: Dict[str, float] = {}
        seen_syms: Set[str] = set()

        def _apply(sym: str, direction: str) -> None:
            s = str(sym or "").upper()
            if len(s) < 6:
                return
            sign = 1.0 if str(direction or "").upper() == "BUY" else -1.0
            exposure[s[:3]] = exposure.get(s[:3], 0.0) + sign
            exposure[s[3:6]] = exposure.get(s[3:6], 0.0) - sign

        try:
            for p in (ctrader_service.get_positions() or []):
                sym = str(p.get("symbol") or "").upper()
                if not sym:
                    continue
                seen_syms.add(sym)
                _apply(sym, p.get("direction") or p.get("side"))
        except Exception as err:
            logger.warning(f"Could not read broker positions for exposure: {err}")
        try:
            for row in open_ctrader_db_positions():
                sym = str(row.get("symbol") or "").upper()
                if sym and sym not in seen_syms:
                    _apply(sym, row.get("direction"))
        except Exception as err:
            logger.warning(f"Could not read DB positions for exposure: {err}")
        return exposure

    def _currency_exposure_slots_available(self, symbol: str, direction: str) -> bool:
        """True unless the candidate piles onto an at-cap currency cluster.

        Only blocks when the candidate INCREASES |net exposure| beyond the
        cap — trades that reduce or hedge an existing cluster stay allowed.
        """
        cap = float(self.execution_config.get("max_currency_exposure") or 0)
        if cap <= 0:
            return True
        sym = str(symbol or "").upper()
        if len(sym) < 6:
            return True
        exposure = self._open_ctrader_currency_exposure()
        sign = 1.0 if str(direction or "").upper() == "BUY" else -1.0
        for curr, delta in ((sym[:3], sign), (sym[3:6], -sign)):
            current = exposure.get(curr, 0.0)
            if abs(current) >= cap and abs(current + delta) > abs(current):
                return False
        return True

    def _same_base_slots_available(self, symbol: str) -> bool:
        cap = int(self.execution_config.get("max_same_base") or 0)
        if cap <= 0:
            return True
        base = self._symbol_base(symbol)
        return self._open_ctrader_base_counts().get(base, 0) < cap

    def _venue_closed_backoff_seconds(self) -> float:
        try:
            value = float(os.getenv("VENUE_CLOSED_BACKOFF_SECONDS", "900"))
        except (TypeError, ValueError):
            value = 900.0
        return value if value > 0 else 900.0

    def _note_venue_closed(self, venue: str, symbol: str) -> None:
        """Log once per backoff window so weekend polls do not spam MARKET_CLOSED."""
        now = time.time()
        until = float(self._venue_closed_until.get(venue) or 0.0)
        if now < until:
            return
        backoff = self._venue_closed_backoff_seconds()
        self._venue_closed_until[venue] = now + backoff
        logger.info(
            "[%s] %s venue closed — skipping IC orders/scans for %.0fs (weekend/session closed)",
            symbol,
            venue,
            backoff,
        )

    def _ctrader_venue_closed(self, symbol: str) -> bool:
        if is_venue_open(symbol):
            return False
        self._note_venue_closed("ctrader", symbol)
        return True

    def _account_equity(self) -> float:
        """Live cTrader equity when connected; otherwise a conservative fallback.

        A live account with no snapshot must not inherit the $10k paper override —
        that sized 0.10 lots on a ~$159 IC Markets book.
        """
        try:
            equity = float(getattr(ctrader_service, "equity", 0) or 0)
            if equity > 0:
                return equity
        except Exception:
            pass
        if ctrader_service.has_credentials():
            return float(os.getenv("CTRADER_FALLBACK_EQUITY", "150"))
        return float(self.timing_config.get("account_equity_override") or 10_000.0)

    def prune_candidates(self, now_ts: Optional[int] = None) -> int:
        """Drop expired/cancelled rows so the in-memory queue cannot grow without bound."""
        now_ts = now_ts or int(time.time())
        kept: Dict[str, Dict[str, Any]] = {}
        removed = 0
        for cid, cand in self.candidates.items():
            status = cand.get("status")
            latest = int(cand.get("latest_exec_at") or 0)
            if status in (CandidateStatus.EXPIRED, CandidateStatus.CANCELLED):
                removed += 1
                continue
            if latest and now_ts > latest and status != CandidateStatus.EXECUTED:
                removed += 1
                continue
            kept[cid] = cand
        max_keep = int(self.execution_config.get("max_candidates") or 200)
        if len(kept) > max_keep:
            executed = [
                c for c in kept.values()
                if c.get("status") == CandidateStatus.EXECUTED
            ]
            others = [
                c for c in kept.values()
                if c.get("status") != CandidateStatus.EXECUTED
            ]
            others.sort(key=lambda c: str(c.get("created_at") or ""), reverse=True)
            # Never evict in-TTL EXECUTED rows — they are the re-arm brake.
            room = max(0, max_keep - len(executed))
            kept_rows = list(executed) + others[:room]
            kept = {c["id"]: c for c in kept_rows if c.get("id")}
            removed += len(self.candidates) - len(kept)
        self.candidates = kept
        return removed
    def _ctrader_execution_slots_remaining(self) -> int:
        cap = int(self.execution_config.get("max_open_ctrader_positions", 10))
        return max(0, cap - self._open_ctrader_position_count())

    def _ctrader_execution_slot_available(self) -> bool:
        return self._ctrader_execution_slots_remaining() > 0

    def clear_candidates(self, statuses: Optional[List[str]] = None) -> int:
        """Drop in-memory candidates (default: everything except EXECUTED)."""
        if statuses is None:
            statuses = [
                CandidateStatus.PENDING,
                CandidateStatus.READY,
                CandidateStatus.EXPIRED,
                CandidateStatus.CANCELLED,
            ]
        removed = 0
        for cid, cand in list(self.candidates.items()):
            if cand.get("status") in statuses:
                del self.candidates[cid]
                removed += 1
        return removed
    def _candidate_epoch(self, cand: Dict[str, Any]) -> Optional[float]:
        """Unix timestamp for TTL age. Prefer executed/created ISO, else latest_exec_at."""
        for key in ("executed_at", "created_at"):
            raw = cand.get(key)
            if not raw:
                continue
            try:
                if isinstance(raw, datetime):
                    dt = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
                else:
                    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                continue
        latest = cand.get("latest_exec_at")
        if isinstance(latest, (int, float)):
            return float(latest)
        return None

    def _prune_terminal_candidates(self, now_ts: Optional[int] = None) -> int:
        """Drop EXECUTED/EXPIRED/CANCELLED candidates older than 24h.

        Called on every scan/queue so the in-memory store cannot grow without bound.
        Live PENDING/READY candidates are never pruned here.
        """
        now = int(now_ts if now_ts is not None else time.time())
        removed = 0
        for cid, cand in list(self.candidates.items()):
            if cand.get("status") not in TERMINAL_CANDIDATE_STATUSES:
                continue
            epoch = self._candidate_epoch(cand)
            if epoch is None:
                continue
            if now - epoch >= CANDIDATE_TTL_SECONDS:
                del self.candidates[cid]
                removed += 1
        if removed:
            logger.info("Pruned %s terminal signal candidates older than 24h", removed)
        return removed
    MIN_FEATURE_BARS = 20
    # Entries trigger on the signal timeframe, but stops are sized from a
    # slower one. M5 ATR on a quiet pair produced 2 pip stops that the broker
    # discarded outright, leaving positions with no protection at all.
    STOP_TIMEFRAME = os.getenv("SIGNAL_STOP_TIMEFRAME", "H1")

    @staticmethod
    def stop_atr(features: Dict[str, Any]) -> float:
        """ATR to size protective levels from, falling back to the signal ATR."""
        return float(features.get("stop_atr") or features.get("atr") or 0.0)

    async def _attach_stop_atr(self, symbol: str, broker: str, features: Dict[str, Any]) -> None:
        """Add a slower-timeframe ATR for stop sizing.

        Leaves `atr` untouched so entry triggers keep their signal-timeframe
        sensitivity. On any failure the signal ATR is used and the broker
        minimum-distance clamp remains the backstop.
        """
        tf = self.STOP_TIMEFRAME
        if not tf or tf == "":
            return
        try:
            if broker == "ctrader":
                bars = await asyncio.to_thread(
                    ctrader_service.get_trendbars, symbol, tf, count=40
                )
            else:
                bars = await binance_market_data.get_klines(
                    symbol, interval=tf_to_binance_interval(tf), limit=40
                )
            slow = self._compute_features(bars)
            if slow and slow.get("atr"):
                features["stop_atr"] = slow["atr"]
                features["stop_timeframe"] = tf
            if slow and slow.get("trend"):
                # Higher-timeframe trend context for counter-trend gating
                # (FADE shorts into H1 rallies were the biggest losing cohort).
                features["h1_trend"] = slow["trend"]
        except Exception as err:
            logger.warning(f"{symbol}: {tf} stop ATR unavailable ({err}); using signal ATR")

    @staticmethod
    def _compute_features(bars: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Compute standard technical indicators from candlestick bars.

        Returns None when there is not enough history. The previous placeholder
        (price 1.0, RSI 50, volatility 0.1%) looked exactly like a compressed
        range, so an empty API response would fire a straddle candidate on
        fabricated data. Callers must skip the symbol instead.
        """
        if not bars or len(bars) < SignalCandidateEngine.MIN_FEATURE_BARS:
            return None

        closes = [b["close"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]

        last_close = closes[-1]

        # ATR calculation (simple true range over last min(14, len))
        n = min(14, len(bars) - 1)
        trs = []
        for i in range(len(bars) - n, len(bars)):
            h_l = highs[i] - lows[i]
            h_cp = abs(highs[i] - closes[i - 1])
            l_cp = abs(lows[i] - closes[i - 1])
            trs.append(max(h_l, h_cp, l_cp))
        atr = sum(trs) / len(trs) if trs else (highs[-1] - lows[-1])
        if last_close > 0 and atr > last_close * 0.05:
            logger.warning(
                "ATR %s exceeds 5%% of price %s — clamping (possible JPY scale mismatch)",
                atr,
                last_close,
            )
            atr = last_close * 0.012

        # Simple RSI (14 period)
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff >= 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))

        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0.0001
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.0001
        rs = avg_gain / (avg_loss if avg_loss != 0 else 0.00001)
        rsi = 100 - (100 / (1 + rs))

        # EMAs
        def _ema(series, period):
            k = 2 / (period + 1)
            ema = series[0]
            for price in series[1:]:
                ema = (price * k) + (ema * (1 - k))
            return ema

        ema_fast = _ema(closes, min(9, len(closes)))
        ema_slow = _ema(closes, min(21, len(closes)))

        trend = "BULLISH" if ema_fast > ema_slow else "BEARISH" if ema_fast < ema_slow else "NEUTRAL"
        volatility_pct = (atr / last_close) * 100 if last_close > 0 else 0.1

        # ADX calculation (14 period)
        adx = 0.0
        if len(bars) >= 15:
            try:
                import pandas as pd
                s_high = pd.Series(highs, dtype=float)
                s_low = pd.Series(lows, dtype=float)
                s_close = pd.Series(closes, dtype=float)
                up_move = s_high.diff()
                down_move = -s_low.diff()
                plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
                minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
                tr = pd.concat([
                    s_high - s_low,
                    (s_high - s_close.shift()).abs(),
                    (s_low - s_close.shift()).abs(),
                ], axis=1).max(axis=1)
                period = min(14, len(bars) - 1)
                atr_s = tr.ewm(span=period, adjust=False).mean()
                plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
                minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
                dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
                adx_val = dx.ewm(span=period, adjust=False).mean()
                adx = float(adx_val.iloc[-1])
            except Exception as err:
                logger.debug(f"ADX computation fallback: {err}")
                adx = 25.0

        return {
            "last_close": last_close,
            "atr": round(atr, 5),
            "rsi": round(rsi, 2),
            "adx": round(adx, 2),
            "ema_fast": round(ema_fast, 5),
            "ema_slow": round(ema_slow, 5),
            "trend": trend,
            "volatility_pct": round(volatility_pct, 3),
            "recent_high": max(highs[-10:]),
            "recent_low": min(lows[-10:]),
        }

    # ── Strategy 1: Momentum (Trend + Breakout Confirmation) ─────────────────
    def _evaluate_momentum(self, symbol: str, features: Dict[str, Any], broker: str) -> Optional[Dict[str, Any]]:
        if not self.timing_config["strategies_enabled"].get("momentum", True):
            return None

        # Check ADX / Chop filter: do not trade momentum breakouts during low-ADX range/consolidation
        adx_val = float(features.get("adx", 0.0))
        if adx_val < self.momentum_min_adx:
            logger.info(
                f"[{symbol}] MOMENTUM veto: ADX {adx_val:.1f} < {self.momentum_min_adx} indicates choppy/ranging market"
            )
            return None

        # Bullish momentum: EMA trend is BULLISH + RSI between 52 and 72 (healthy impulse)
        # Bearish momentum: EMA trend is BEARISH + RSI between 28 and 48
        direction = None
        if features["trend"] == "BULLISH" and 52 <= features["rsi"] <= 72:
            direction = "BUY"
        elif features["trend"] == "BEARISH" and 28 <= features["rsi"] <= 48:
            direction = "SELL"

        if not direction:
            return None

        # Higher-timeframe trend alignment: never trade M5 momentum against H1 trend
        h1_trend = features.get("h1_trend")
        if h1_trend == "BEARISH" and direction == "BUY":
            logger.info(
                f"[{symbol}] MOMENTUM veto: BUY impulse against H1 BEARISH trend"
            )
            return None
        if h1_trend == "BULLISH" and direction == "SELL":
            logger.info(
                f"[{symbol}] MOMENTUM veto: SELL impulse against H1 BULLISH trend"
            )
            return None

        price = features["last_close"]
        atr = self.stop_atr(features)

        if direction == "BUY":
            sl = round(price - (1.5 * atr), 5)
            tp = round(price + (2.5 * atr), 5)
        else:
            sl = round(price + (1.5 * atr), 5)
            tp = round(price - (2.5 * atr), 5)

        return {
            "strategy": "MOMENTUM_TREND_PULSE",
            "direction": direction,
            "entry_price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "timing_mode": TimingMode.POST_REACTION,
            "confidence": 0.82,
            "reason": f"M5 {direction} momentum confirmed. RSI {features['rsi']} aligned with {features['trend']} trend (ADX {adx_val:.1f}).",
        }

    # ── Strategy 2: Fade (Contrarian Mean-Reversion) ─────────────────────────
    def _evaluate_fade(self, symbol: str, features: Dict[str, Any], broker: str) -> Optional[Dict[str, Any]]:
        if not self.timing_config["strategies_enabled"].get("fade", True):
            return None

        # Overextended spike: RSI > 75 (overbought -> Fade Short) or RSI < 25 (oversold -> Fade Long)
        direction = None
        if features["rsi"] >= 75:
            direction = "SELL"
        elif features["rsi"] <= 25:
            direction = "BUY"

        if not direction:
            return None

        # H1 trend alignment: fading an H1 rally/selloff is shorting strength.
        # 2026-09-08 live: 15 FADE trades, 19/21 SELLs, net negative — the
        # strategy was systematically counter-trend. Block fades that fight
        # the H1 EMA cross; allow when H1 is unavailable or NEUTRAL.
        if self._fade_h1_filter_enabled():
            h1_trend = features.get("h1_trend")
            if h1_trend == "BULLISH" and direction == "SELL":
                logger.info(
                    f"[{symbol}] FADE veto: SELL overbought spike against H1 BULLISH trend"
                )
                return None
            if h1_trend == "BEARISH" and direction == "BUY":
                logger.info(
                    f"[{symbol}] FADE veto: BUY oversold dip against H1 BEARISH trend"
                )
                return None

        price = features["last_close"]
        atr = self.stop_atr(features)

        if direction == "BUY":
            sl = round(features["recent_low"] - (0.5 * atr), 5)
            tp = round(features["ema_slow"], 5)
        else:
            sl = round(features["recent_high"] + (0.5 * atr), 5)
            tp = round(features["ema_slow"], 5)

        return {
            "strategy": "FADE_OVEREXTENSION",
            "direction": direction,
            "entry_price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "timing_mode": TimingMode.POST_REACTION,
            "confidence": 0.78,
            "reason": f"Extreme RSI {features['rsi']} reached. Fading overbought/oversold surge back to mean {features['ema_slow']}.",
        }

    # ── Strategy 3: Straddle (Pre-Event Compression Bracket) ─────────────────
    def _evaluate_straddle(self, symbol: str, features: Dict[str, Any], broker: str) -> Optional[Dict[str, Any]]:
        if not self.timing_config["strategies_enabled"].get("straddle", True):
            return None

        # Low volatility compression before catalyst (volatility_pct < 0.15% or RSI near 50)
        if 46 <= features["rsi"] <= 54 and features["volatility_pct"] < 0.25:
            price = features["last_close"]
            atr = self.stop_atr(features)
            upper_bracket = round(features["recent_high"] + (0.4 * atr), 5)
            lower_bracket = round(features["recent_low"] - (0.4 * atr), 5)

            return {
                "strategy": "STRADDLE_VOLATILITY_BRACKET",
                "direction": "BUY",  # Primary breakout bias, with lower bracket hedge
                "entry_price": upper_bracket,
                "stop_loss": round(price - (0.8 * atr), 5),
                "take_profit": round(upper_bracket + (2.0 * atr), 5),
                "timing_mode": TimingMode.PRE_EVENT,
                "confidence": 0.75,
                "reason": f"Tight range compression. Volatility {features['volatility_pct']}%. Pre-event bracket around {lower_bracket} - {upper_bracket}.",
            }
        return None

    # ── Strategy 4: Slingshot (Rejection Reversal) ───────────────────────────
    def _evaluate_slingshot(self, symbol: str, features: Dict[str, Any], broker: str) -> Optional[Dict[str, Any]]:
        if not self.timing_config["strategies_enabled"].get("slingshot", True):
            return None

        # Rejection of extreme wick against prevailing higher timeframe trend
        price = features["last_close"]
        atr = self.stop_atr(features)
        if features["trend"] == "BULLISH" and features["rsi"] < 42:
            # Bullish trend pulling back deeply, potential spring / slingshot long
            return {
                "strategy": "SLINGSHOT_PULLBACK_REENTRY",
                "direction": "BUY",
                "entry_price": price,
                "stop_loss": round(features["recent_low"] - (0.3 * atr), 5),
                "take_profit": round(features["recent_high"] + (1.0 * atr), 5),
                "timing_mode": TimingMode.BAR_CLOSE,
                "confidence": 0.80,
                "reason": f"Slingshot pullback in bullish trend. Discounted entry near {price}.",
            }
        elif features["trend"] == "BEARISH" and features["rsi"] > 58:
            # Bearish trend rallying into resistance, slingshot short
            return {
                "strategy": "SLINGSHOT_PULLBACK_REENTRY",
                "direction": "SELL",
                "entry_price": price,
                "stop_loss": round(features["recent_high"] + (0.3 * atr), 5),
                "take_profit": round(features["recent_low"] - (1.0 * atr), 5),
                "timing_mode": TimingMode.BAR_CLOSE,
                "confidence": 0.80,
                "reason": f"Slingshot bounce into resistance in bearish trend. Entry near {price}.",
            }
        return None

    def _resolve_equity(self, broker: str) -> float:
        """Live account equity for risk sizing. Conservative fallback, never $10k on a live book."""
        if broker == "ctrader":
            return float(self._account_equity())
        cached = self._equity_cache.get(broker)
        if cached is not None:
            return cached
        fallback = float(os.getenv("BINANCE_FALLBACK_EQUITY", "150"))
        equity = 0.0
        try:
            bal = binance_futures_broker.get_balance() or {}
            equity = float(bal.get("equity") or bal.get("balance") or 0.0)
        except Exception as err:
            logger.warning(
                "Live equity lookup failed for %s (%s); falling back to $%.0f",
                broker, err, fallback,
            )
            self._equity_cache[broker] = fallback
            return fallback
        if equity > 0:
            self._equity_cache[broker] = equity
            return equity
        logger.warning(
            "Live equity unavailable for %s (got %s); falling back to $%.0f",
            broker, equity, fallback,
        )
        self._equity_cache[broker] = fallback
        return fallback

    # ── Sizing Calculator ───────────────────────────────────────────────────
    def _calculate_size(self, symbol: str, entry_price: float, stop_loss: float, broker: str) -> Dict[str, Any]:
        """Compute exact lot / quantity sizing from live equity and the candidate stop."""
        equity = self._resolve_equity(broker)
        risk_pct = self.timing_config.get("default_risk_pct", 0.5)
        risk_amount = (equity * risk_pct) / 100.0
        stop_distance = abs(float(entry_price) - float(stop_loss or 0.0))

        if broker == "ctrader":
            spec = ctrader_service.get_symbol_specification(symbol)
            pip_size = spec.get("pip_size", 0.0001)
            stop_pips = max(stop_distance / pip_size, 5.0)

            # pip_value for 1.0 standard lot
            calc_1lot = ctrader_service.calculate_pip_margin(symbol, 1.0, entry_price, 100.0, "USD")
            pip_val_1lot = calc_1lot.get("pip_value", 10.0)

            # lots = risk_amount / (stop_pips * pip_val_1lot)
            lots = round(risk_amount / max(stop_pips * pip_val_1lot, 0.1), 2)
            if classify_symbol(symbol) == "metal":
                metal_cap = float(self.execution_config.get("max_metal_lots") or 0.01)
                lots = min(lots, metal_cap)
            lots = max(0.01, min(lots, 10.0))

            calc_final = ctrader_service.calculate_pip_margin(symbol, lots, entry_price, 100.0, "USD")
            return {
                "lots": lots,
                "quantity": lots,
                "stop_pips": round(stop_pips, 1),
                "risk_usd": round(risk_amount, 2),
                "margin_required": calc_final.get("required_margin", 0.0),
                "pip_value": calc_final.get("pip_value", 0.0),
            }
        else:
            # Size from the candidate's own stop. Tiny epsilon only to avoid divide-by-zero.
            price_dist = stop_distance if stop_distance > 0 else max(abs(entry_price) * 1e-6, 1e-9)
            qty = round(risk_amount / price_dist, 4)
            qty = max(0.001, min(qty, 100.0))
            return {
                "lots": qty,
                "quantity": qty,
                "stop_pips": 0,
                "risk_usd": round(risk_amount, 2),
                "margin_required": round((qty * entry_price) / 20.0, 2),  # assume 20x leverage
                "pip_value": 0.0,
            }

    # ── Market Scanning Execution ───────────────────────────────────────────
    async def scan_markets(self, universe: Optional[List[str]] = None, timeframe: str = "M5") -> List[Dict[str, Any]]:
        """Scan specified market universe and produce trade candidate signals."""
        from backend.services.sentry_state import is_trading_allowed
        if not is_trading_allowed():
            logger.warning("Market scan skipped: trading is halted by sentry")
            return []

        if not universe:
            universe = [
                # Forex Majors
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                # Forex Crosses
                "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "GBPAUD", "NZDJPY", "CHFJPY", "AUDJPY", "CADJPY",
                # Metals
                "XAUUSD", "XAGUSD",
                # Crypto
                "BTCUSDT", "ETHUSDT", "SOLUSDT",
            ]

        self._equity_cache.clear()
        self._prune_terminal_candidates()

        candidates_generated = []
        now_ts = int(time.time())
        self.prune_candidates(now_ts)

        for sym in universe:
            try:
                if self.is_symbol_cooling_down(sym):
                    logger.info(
                        f"[{sym}] Skipping candidate generation — symbol is in post-loss cooldown"
                    )
                    continue
                asset_class = classify_symbol(sym)
                if asset_class == "metal" and not self.execution_config.get("include_metals"):
                    continue
                if self.execution_config.get("forex_only") and asset_class == "crypto":
                    continue
                broker = "binance_futures" if asset_class == "crypto" else "ctrader"
                if broker == "ctrader" and self._ctrader_venue_closed(sym):
                    continue

                # Ingest bars
                if broker == "ctrader":
                    bars = ctrader_service.get_trendbars(sym, timeframe, count=40)
                else:
                    bars = await binance_market_data.get_klines(
                        sym, interval=tf_to_binance_interval(timeframe), limit=40
                    )

                features = self._compute_features(bars)
                if not features:
                    logger.warning(
                        f"Skipping {sym}: only {len(bars or [])} bars returned, "
                        f"need {self.MIN_FEATURE_BARS}"
                    )
                    continue
                await self._attach_stop_atr(sym, broker, features)

                # Evaluate strategies in order of priority
                evaluators = [
                    self._evaluate_momentum,
                    self._evaluate_fade,
                    self._evaluate_straddle,
                    self._evaluate_slingshot,
                ]

                for ev in evaluators:
                    raw_signal = ev(sym, features, broker)
                    if raw_signal:
                        if broker == "ctrader":
                            sl, tp = CTraderService.clamp_protective_prices(
                                sym,
                                raw_signal["entry_price"],
                                raw_signal.get("stop_loss"),
                                raw_signal.get("take_profit"),
                                direction=raw_signal.get("direction"),
                            )
                            # P0: veto geometry the broker clamp deformed into a
                            # 1:1 (or worse) coin-flip instead of executing it.
                            validated = CTraderService.validate_protective_geometry(
                                sym,
                                raw_signal["entry_price"],
                                raw_signal.get("stop_loss"),
                                raw_signal.get("take_profit"),
                                sl,
                                tp,
                                direction=raw_signal.get("direction"),
                            )
                            if validated is None:
                                continue  # next strategy may produce valid geometry
                            raw_signal["stop_loss"], raw_signal["take_profit"] = validated

                        # P1: Kronos + heuristic timing gate. Shadow mode logs
                        # and annotates the candidate; enforce mode blocks.
                        # Gate the *strategy* side first; invert after approval
                        # so Kronos cannot veto the empirically-winning flip.
                        gate_result = None
                        if broker == "ctrader" and self._fx_gate_mode() != "off":
                            try:
                                gate_result = await self._fx_timing_gate(
                                    sym,
                                    raw_signal["direction"],
                                    raw_signal["confidence"],
                                    bars,
                                )
                            except Exception as err:
                                logger.debug(f"FX gate error for {sym}: {err}")
                            if (
                                gate_result is not None
                                and gate_result.action == "veto"
                                and self._fx_gate_mode() == "enforce"
                            ):
                                logger.info(f"[{sym}] FX gate ACTIVE VETO: {gate_result.reasoning}")
                                continue

                        if self._should_invert_ctrader_fx_side(broker, sym):
                            inverted = self.apply_ctrader_fx_side_invert(sym, raw_signal)
                            if inverted is None:
                                continue
                            raw_signal = inverted

                        size_data = self._calculate_size(sym, raw_signal["entry_price"], raw_signal["stop_loss"], broker)

                        # Timing window parameters
                        timing_mode = raw_signal["timing_mode"]
                        if timing_mode == TimingMode.PRE_EVENT:
                            earliest = now_ts
                            latest = now_ts + (self.timing_config["pre_event_window_min"] * 60)
                        elif timing_mode == TimingMode.POST_REACTION:
                            earliest = now_ts + (self.timing_config["post_reaction_delay_min"] * 60)
                            latest = earliest + (self.timing_config["post_reaction_window_min"] * 60)
                        else:
                            earliest = now_ts
                            latest = now_ts + 900  # 15 minutes default validity

                        candidate = {
                            "id": f"sig-{uuid.uuid4().hex[:8]}",
                            "symbol": sym,
                            "broker": broker,
                            "strategy": raw_signal["strategy"],
                            "direction": raw_signal["direction"],
                            "entry_price": raw_signal["entry_price"],
                            "stop_loss": raw_signal["stop_loss"],
                            "take_profit": raw_signal["take_profit"],
                            "timing_mode": timing_mode,
                            "status": CandidateStatus.PENDING if earliest > now_ts else CandidateStatus.READY,
                            "confidence": raw_signal["confidence"],
                            "reason": raw_signal["reason"],
                            "original_direction": raw_signal.get("original_direction"),
                            "side_inverted": bool(raw_signal.get("side_inverted")),
                            "features": features,
                            "sizing": size_data,
                            "fx_gate": (
                                {
                                    "action": gate_result.action,
                                    "shadow_veto": gate_result.is_shadow_veto,
                                    "reasoning": gate_result.reasoning[:300],
                                }
                                if gate_result is not None
                                else None
                            ),
                            "earliest_exec_at": earliest,
                            "latest_exec_at": latest,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }

                        # Cross-scan admission: refuse live twins (same
                        # strategy or any strategy on this symbol+direction),
                        # an open book position, portfolio-risk overflow, and
                        # Binance one-per-symbol / max-positions. No await
                        # between check and insert so concurrent scans cannot
                        # interleave here.
                        refuse = self._refuse_new_candidate_reason(
                            symbol=sym,
                            broker=broker,
                            strategy=raw_signal["strategy"],
                            direction=raw_signal["direction"],
                            risk_usd=float((size_data or {}).get("risk_usd") or 0.0),
                            check_recent_executed=True,
                        )
                        if refuse:
                            logger.info(f"[{sym}] scan skip: {refuse}")
                            continue

                        self.candidates[candidate["id"]] = candidate
                        candidates_generated.append(candidate)
                        break  # 1 candidate per symbol per scan run

            except Exception as err:
                logger.warning(f"Error scanning {sym}: {err}")

        return candidates_generated

    # ── News & Macro Scanning Execution ─────────────────────────────────────
    async def scan_news_and_events(self, lookahead_minutes: int = 60) -> List[Dict[str, Any]]:
        """Correlate economic events and news sentiment to propose news-triggered trade setups."""
        from backend.services.sentry_state import is_trading_allowed
        if not is_trading_allowed():
            logger.warning("News scan skipped: trading is halted by sentry")
            return []

        from backend.routes.news import get_economic_calendar, get_news_feed, get_market_sentiment

        self._equity_cache.clear()
        self._prune_terminal_candidates()

        news_candidates = []
        try:
            cal_res = await get_economic_calendar()
            await get_news_feed()
            await get_market_sentiment()

            events = cal_res.get("events", [])
            high_impact = [e for e in events if e.get("impact") == "high"]

            # Check for major-currency catalysts. Unmapped currencies are skipped
            # instead of becoming a gold trade.
            self.prune_candidates()
            for ev in high_impact[:3]:
                curr = str(ev.get("currency") or "USD").upper()
                matched_sym = self.MACRO_CURRENCY_PAIRS.get(curr)
                if not matched_sym:
                    logger.info(
                        "Skipping macro event %s (%s): no FX pair mapping",
                        ev.get("event"),
                        curr,
                    )
                    continue
                # One live macro candidate per symbol: keying on the event name
                # let 3 same-hour USD events arm 3 identical EURUSD candidates.
                already = any(
                    c.get("strategy") == "MACRO_EVENT_POST_REACTION"
                    and c.get("symbol") == matched_sym
                    and c.get("status") in LIVE_CANDIDATE_STATUSES
                    for c in self.candidates.values()
                )
                if already:
                    continue

                bars = ctrader_service.get_trendbars(matched_sym, "M5", count=30)
                features = self._compute_features(bars)
                if not features:
                    logger.warning(
                        f"Skipping macro candidate for {matched_sym}: insufficient bars"
                    )
                    continue
                await self._attach_stop_atr(matched_sym, "ctrader", features)
                direction = "BUY" if features["trend"] == "BULLISH" else "SELL"
                entry = features["last_close"]
                atr = self.stop_atr(features)
                # Protection must sit on the correct side of entry. A short with
                # a stop below and a target above is an inverted bracket that
                # never caps the loss.
                if direction == "BUY":
                    sl = round(entry - (1.2 * atr), 5)
                    tp = round(entry + (2.4 * atr), 5)
                else:
                    sl = round(entry + (1.2 * atr), 5)
                    tp = round(entry - (2.4 * atr), 5)
                natural_sl, natural_tp = sl, tp
                sl, tp = CTraderService.clamp_protective_prices(
                    matched_sym, entry, sl, tp, direction=direction
                )
                validated = CTraderService.validate_protective_geometry(
                    matched_sym, entry, natural_sl, natural_tp, sl, tp,
                    direction=direction,
                )
                if validated is None:
                    continue
                sl, tp = validated
                raw_macro = {
                    "direction": direction,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": f"High Impact Macro Event: '{ev.get('event')}' ({ev.get('currency')}). Post-reaction momentum window armed.",
                }
                if self._should_invert_ctrader_fx_side("ctrader", matched_sym):
                    inverted = self.apply_ctrader_fx_side_invert(matched_sym, raw_macro)
                    if inverted is None:
                        continue
                    raw_macro = inverted
                    direction = raw_macro["direction"]
                    sl = raw_macro["stop_loss"]
                    tp = raw_macro["take_profit"]
                size_data = self._calculate_size(matched_sym, entry, sl, "ctrader")

                now_ts = int(time.time())
                candidate = {
                    "id": f"sig-news-{uuid.uuid4().hex[:8]}",
                    "symbol": matched_sym,
                    "broker": "ctrader",
                    "strategy": "MACRO_EVENT_POST_REACTION",
                    "direction": direction,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "timing_mode": TimingMode.POST_REACTION,
                    "status": CandidateStatus.PENDING,
                    "confidence": 0.85,
                    "reason": raw_macro["reason"],
                    "original_direction": raw_macro.get("original_direction"),
                    "side_inverted": bool(raw_macro.get("side_inverted")),
                    "features": features,
                    "sizing": size_data,
                    "earliest_exec_at": now_ts + (self.timing_config["post_reaction_delay_min"] * 60),
                    "latest_exec_at": now_ts + (self.timing_config["post_reaction_window_min"] * 60) + 300,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                refuse = self._refuse_new_candidate_reason(
                    symbol=matched_sym,
                    broker="ctrader",
                    strategy="MACRO_EVENT_POST_REACTION",
                    direction=direction,
                    risk_usd=float((size_data or {}).get("risk_usd") or 0.0),
                    check_recent_executed=True,
                )
                if refuse:
                    logger.info(f"[{matched_sym}] news scan skip: {refuse}")
                    continue
                self.candidates[candidate["id"]] = candidate
                news_candidates.append(candidate)

        except Exception as e:
            logger.error(f"Error during scan_news_and_events: {e}")

        return news_candidates

    # ── Ready for Execution Queue ───────────────────────────────────────────
    def get_ready_signals(
        self,
        current_ts: Optional[int] = None,
        *,
        broker: Optional[str] = None,
        forex_only: bool = False,
        limit: Optional[int] = None,
        enforce_ctrader_position_cap: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return signals whose timing window is currently open and valid."""
        if not current_ts:
            current_ts = int(time.time())
        self._prune_terminal_candidates(current_ts)

        ready = []
        for cid, cand in list(self.candidates.items()):
            # Check expiration
            if current_ts > cand["latest_exec_at"]:
                if cand["status"] in [CandidateStatus.PENDING, CandidateStatus.READY]:
                    cand["status"] = CandidateStatus.EXPIRED
                continue

            # Check timing window
            if cand["earliest_exec_at"] <= current_ts <= cand["latest_exec_at"]:
                if cand["status"] == CandidateStatus.PENDING:
                    cand["status"] = CandidateStatus.READY
                if cand["status"] == CandidateStatus.READY:
                    ready.append(cand)

        self.prune_candidates(current_ts)
        if broker:
            ready = [c for c in ready if c.get("broker") == broker.lower()]
        if forex_only or self.execution_config.get("forex_only"):
            ready = [c for c in ready if self._is_ctrader_forex_candidate(c)]
        cap = int(self.execution_config.get("max_same_base") or 0)
        if cap > 0:
            base_counts = self._open_ctrader_base_counts()
            limited: List[Dict[str, Any]] = []
            for cand in ready:
                base = self._symbol_base(str(cand.get("symbol") or ""))
                used = base_counts.get(base, 0)
                if used >= cap:
                    continue
                limited.append(cand)
                base_counts[base] = used + 1
            ready = limited

        # Net currency exposure cap (base AND quote side), simulated
        # incrementally so a ready batch cannot itself form a cluster.
        fx_cap = float(self.execution_config.get("max_currency_exposure") or 0)
        if fx_cap > 0:
            exposure = self._open_ctrader_currency_exposure()
            fx_limited: List[Dict[str, Any]] = []
            for cand in ready:
                sym = str(cand.get("symbol") or "").upper()
                side = str(cand.get("direction") or "").upper()
                if len(sym) < 6:
                    fx_limited.append(cand)
                    continue
                sign = 1.0 if side == "BUY" else -1.0
                base, quote = sym[:3], sym[3:6]
                blocked = False
                for curr, delta in ((base, sign), (quote, -sign)):
                    current = exposure.get(curr, 0.0)
                    if abs(current) >= fx_cap and abs(current + delta) > abs(current):
                        blocked = True
                        break
                if blocked:
                    logger.info(
                        f"[{sym}] currency exposure cap: {side} would push "
                        f"{base}/{quote} past net {fx_cap}"
                    )
                    continue
                exposure[base] = exposure.get(base, 0.0) + sign
                exposure[quote] = exposure.get(quote, 0.0) - sign
                fx_limited.append(cand)
            ready = fx_limited

        ready.sort(key=lambda c: float(c.get("confidence") or 0), reverse=True)

        if self.execution_config.get("one_position_per_symbol"):
            open_syms = self._open_ctrader_symbols()
            ready = [c for c in ready if str(c.get("symbol", "")).upper() not in open_syms]
            seen: Set[str] = set()
            unique: List[Dict[str, Any]] = []
            for cand in ready:
                sym = str(cand.get("symbol", "")).upper()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                unique.append(cand)
            ready = unique

        # Direction-aware book filter: a READY that duplicates an open
        # symbol+direction on its broker must not reach the executor even
        # when one_position_per_symbol is off. An unreadable book drops
        # that broker's READY rows rather than failing open.
        keys_by_broker: Dict[str, Optional[Set[tuple]]] = {}
        filtered_ready: List[Dict[str, Any]] = []
        for cand in ready:
            broker_name = str(cand.get("broker") or "")
            if broker_name not in keys_by_broker:
                keys_by_broker[broker_name] = (
                    self._open_position_keys(broker_name) if broker_name else set()
                )
            keys = keys_by_broker[broker_name]
            if keys is None:
                logger.info(
                    f"[{cand.get('symbol')}] ready skip: {broker_name} book unreadable"
                )
                continue
            side_key = (
                str(cand.get("symbol") or "").upper(),
                self._normalize_direction(cand.get("direction")),
            )
            if side_key in keys or any(
                kside == side_key[1] and same_crypto_perp_leg(ksym, side_key[0])
                for ksym, kside in keys
            ):
                continue
            filtered_ready.append(cand)
        ready = filtered_ready

        if enforce_ctrader_position_cap:
            slots = self._ctrader_execution_slots_remaining()
            if slots <= 0:
                ready = []
            else:
                ready = ready[:slots]

        poll_limit = limit
        if poll_limit is None:
            poll_limit = self.execution_config.get("max_ready_per_poll")
        if poll_limit is not None:
            ready = ready[: max(0, int(poll_limit))]

        return ready

    async def _resolve_mark_price(self, symbol: str, broker: str, fallback: float = 0.0) -> float:
        """Fetch a live quote for paper market fills when no explicit price is set."""
        if fallback and fallback > 0:
            return float(fallback)
        sym = symbol.upper()
        try:
            if broker == "binance_futures":
                tick = await binance_market_data.get_ticker_24h(sym)
                if tick and tick.get("lastPrice"):
                    return float(tick["lastPrice"])
            tick = await binance_market_data.get_ticker_24h(sym)
            if tick and tick.get("lastPrice"):
                return float(tick["lastPrice"])
        except Exception as err:
            logger.warning(f"Could not resolve mark price for {sym}: {err}")
        return 0.0

    @staticmethod
    def _resolve_broker_session(ut: UnifiedTrading, broker: str) -> Optional[str]:
        """Session id that routes to the candidate's own broker.

        place_order() without a session id follows the global default session —
        whatever startup or the trading loop last selected — so a binance_futures
        candidate could fill on an unrelated (e.g. paper) book. Pick explicitly.
        """
        from backend.services.trading_mode import (
            BINANCE_PAPER_SESSION_ID,
            TradingMode,
            binance_paper_parallel_enabled,
            get_trading_mode,
        )

        sessions = [s for s in ut.list_sessions() if s.get("broker") == broker]
        if not sessions:
            return None
        if broker == "binance_futures" and binance_paper_parallel_enabled():
            for s in sessions:
                if s.get("id") == BINANCE_PAPER_SESSION_ID:
                    return s["id"]
        want_live = get_trading_mode() == TradingMode.LIVE
        for s in sessions:
            if (s.get("mode") == "live") == want_live:
                return s.get("id")
        return sessions[0].get("id")

    # ── Execution Dispatcher ────────────────────────────────────────────────
    async def execute_candidate(self, candidate_id: str, force: bool = False) -> Dict[str, Any]:
        """Dispatch candidate trade signal through smart unified broker router."""
        cand = self.candidates.get(candidate_id)
        if not cand:
            return {"success": False, "error": f"Candidate {candidate_id} not found."}

        from backend.services.sentry_state import is_trading_allowed
        if not is_trading_allowed():
            return {"success": False, "error": "Trading halted by sentry."}

        now_ts = int(time.time())
        if not force:
            if now_ts < cand["earliest_exec_at"]:
                return {
                    "success": False,
                    "error": f"Timing window not yet open. Opens in {cand['earliest_exec_at'] - now_ts}s.",
                }
            if now_ts > cand["latest_exec_at"]:
                cand["status"] = CandidateStatus.EXPIRED
                return {"success": False, "error": "Timing window expired."}

        cand_sym = str(cand.get("symbol", "")).upper()
        if not force and cand.get("broker") == "ctrader" and self._ctrader_venue_closed(cand_sym):
            return {
                "success": False,
                "skipped": True,
                "error": "VENUE_CLOSED — market is closed; not sending IC order.",
            }
        if not force and self.is_symbol_cooling_down(cand_sym):
            return {
                "success": False,
                "skipped": True,
                "error": f"Symbol {cand_sym} is in post-loss cooldown.",
            }

        if self.execution_config.get("forex_only") and cand.get("broker") == "binance_futures":
            return {
                "success": False,
                "error": "Forex-only execution mode: crypto candidates are blocked.",
            }
        if cand.get("broker") == "ctrader":
            if self.execution_config.get("forex_only") and not self._is_ctrader_forex_candidate(cand):
                return {
                    "success": False,
                    "skipped": True,
                    "error": "Forex-only execution mode: candidate is not a cTrader FX pair.",
                }
            if not self._ctrader_execution_slot_available():
                return {
                    "success": False,
                    "error": (
                        f"Max open cTrader positions reached "
                        f"({self.execution_config.get('max_open_ctrader_positions', 10)})."
                    ),
                }
            if self.execution_config.get("one_position_per_symbol"):
                sym = str(cand.get("symbol", "")).upper()
                if sym in self._open_ctrader_symbols():
                    return {
                        "success": False,
                        "skipped": True,
                        "error": f"Already have an open cTrader position in {sym}.",
                    }
            if not self._same_base_slots_available(str(cand.get("symbol") or "")):
                base = self._symbol_base(str(cand.get("symbol") or ""))
                return {
                    "success": False,
                    "skipped": True,
                    "error": (
                        f"Same-base cap reached for {base} "
                        f"({self.execution_config.get('max_same_base', 2)} open)."
                    ),
                }

        if cand.get("broker") == "binance_futures":
            binance_block = self._binance_position_cap_breach(cand_sym)
            if binance_block:
                return {
                    "success": False,
                    "skipped": True,
                    "error": binance_block,
                }

        # After skip gates, invert cTrader FX so same-side / exposure checks
        # see the side we will actually send. In-flight candidates created
        # before this flag still flip once (side_inverted is the lock).
        if cand.get("broker") == "ctrader" and not self._ensure_ctrader_fx_side_invert(cand):
            return {
                "success": False,
                "skipped": True,
                "error": (
                    f"cTrader FX side invert rejected {cand.get('symbol')} "
                    "geometry; not sending IC order."
                ),
            }

        # Same-direction open book: refuse even when one_position_per_symbol
        # is disabled so scans cannot stack a second EURUSD SELL after fill.
        cand_side = self._normalize_direction(cand.get("direction"))
        if cand_side and self._has_open_position(str(cand.get("broker") or ""), cand_sym, cand_side):
            return {
                "success": False,
                "skipped": True,
                "error": (
                    f"Already have an open {cand.get('broker')} "
                    f"{cand_side} position in {cand_sym}."
                ),
            }

        # Portfolio risk budget: live candidate risk (this broker's book) plus
        # the incoming trade must stay under max_portfolio_risk_pct of equity.
        risk_block = self._portfolio_risk_breach(
            str(cand.get("broker") or ""),
            float((cand.get("sizing") or {}).get("risk_usd") or 0.0),
            exclude_id=candidate_id,
        )
        if risk_block:
            return {
                "success": False,
                "skipped": True,
                "error": risk_block,
            }

        # Net currency exposure re-check at execution (the ready queue check
        # ran earlier; positions may have changed since).
        if cand.get("broker") == "ctrader" and not self._currency_exposure_slots_available(
            str(cand.get("symbol") or ""), str(cand.get("direction") or "")
        ):
            return {
                "success": False,
                "skipped": True,
                "error": (
                    f"Currency exposure cap reached for {cand.get('symbol')} "
                    f"(net {self.execution_config.get('max_currency_exposure', 2)} per currency)."
                ),
            }

        # Spread gate: with 10-30 pip stops, a 2+ pip spread is 7-20% of risk
        # paid upfront. max_spread_pips existed in config but was never enforced.
        if cand.get("broker") == "ctrader":
            max_spread = float(self.timing_config.get("max_spread_pips") or 0)
            if max_spread > 0:
                try:
                    spread = ctrader_service.get_spread_pips(str(cand.get("symbol") or ""))
                except Exception:
                    spread = None
                if spread is not None and spread > max_spread:
                    logger.info(
                        f"[{cand.get('symbol')}] spread gate: {spread:.1f} pips > "
                        f"max {max_spread} — skipping execution"
                    )
                    return {
                        "success": False,
                        "skipped": True,
                        "error": (
                            f"Spread {spread:.1f} pips exceeds max {max_spread} "
                            f"for {cand.get('symbol')}"
                        ),
                    }

        # Route through appropriate broker engine
        try:
            qty = cand["sizing"]["lots"] if cand["broker"] == "ctrader" else cand["sizing"]["quantity"]
            if cand["broker"] == "ctrader":
                live_size = self._calculate_size(
                    str(cand.get("symbol") or ""),
                    float(cand.get("entry_price") or 0),
                    float(cand.get("stop_loss") or 0),
                    "ctrader",
                )
                cand["sizing"] = live_size
                qty = live_size["lots"]
                max_lots = float(self.execution_config.get("max_ctrader_lots") or 0)
                if classify_symbol(str(cand.get("symbol") or "")) == "metal":
                    metal_cap = float(self.execution_config.get("max_metal_lots") or 0.01)
                    max_lots = min(max_lots or metal_cap, metal_cap)
                if max_lots > 0:
                    qty = min(float(qty), max_lots)
            side = cand["direction"].upper()

            if cand["broker"] == "ctrader":
                if live_ctrader_orders_allowed():
                    connected = await asyncio.to_thread(ctrader_service.ensure_connected)
                    if not connected and ctrader_service.has_credentials():
                        return {
                            "success": False,
                            "error": "cTrader is not connected; refused to simulate a live forex fill.",
                            "candidate_id": candidate_id,
                            "symbol": cand["symbol"],
                        }
                order_res = ctrader_service.place_order(
                    symbol=cand["symbol"],
                    direction=side,
                    quantity=qty,
                    price=cand.get("entry_price"),
                    stop_loss=cand.get("stop_loss"),
                    take_profit=cand.get("take_profit"),
                )
                status = (order_res or {}).get("status", "")
                err_text = str((order_res or {}).get("error") or "")
                success = status in ("ok", "filled", "sent")
                if status == "simulated" and (
                    not live_ctrader_orders_allowed()
                    or not ctrader_service.has_credentials()
                ):
                    success = True
                order_id = order_res.get("order_id") if order_res else None
                msg = f"cTrader order {order_id or 'pending'} placed ({status or 'unknown'})"
                if not success and "MARKET_CLOSED" in err_text.upper():
                    self._note_venue_closed("ctrader", cand_sym)
                    cand["status"] = CandidateStatus.READY
                    cand["execution_result"] = {
                        "success": False,
                        "skipped": True,
                        "order_id": None,
                        "message": err_text,
                        "broker": "ctrader",
                    }
                    return {
                        "success": False,
                        "skipped": True,
                        "candidate_id": candidate_id,
                        "symbol": cand["symbol"],
                        "error": err_text,
                    }
            else:
                order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
                fill_price = await self._resolve_mark_price(
                    cand["symbol"], cand["broker"], float(cand.get("entry_price") or 0)
                )
                ut = UnifiedTrading()
                session_id = self._resolve_broker_session(ut, cand["broker"])
                if not session_id:
                    return {
                        "success": False,
                        "candidate_id": candidate_id,
                        "symbol": cand["symbol"],
                        "error": (
                            f"No trading session registered for broker "
                            f"{cand.get('broker')}; refusing to route via the default session."
                        ),
                    }
                order_req = UnifiedOrder(
                    symbol=cand["symbol"],
                    side=order_side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    price=fill_price,
                    stop_loss=cand.get("stop_loss"),
                    take_profit=cand.get("take_profit"),
                )
                res = ut.place_order(order_req, session_id=session_id)
                success = res.success
                order_id = res.order_id
                msg = res.message

            cand["execution_result"] = {
                "success": success,
                "order_id": order_id,
                "message": msg,
                "broker": cand["broker"],
            }
            if success:
                cand["status"] = CandidateStatus.EXECUTED
                cand["executed_at"] = datetime.now(timezone.utc).isoformat()
                # Defuse live twins: the position is on, so any remaining
                # same symbol+direction candidate is pure overlapping risk.
                for other_id, other in list(self.candidates.items()):
                    if other_id == candidate_id:
                        continue
                    if (
                        other.get("status") in (CandidateStatus.PENDING, CandidateStatus.READY)
                        and same_crypto_perp_leg(str(other.get("symbol") or ""), cand_sym)
                        and str(other.get("direction") or "").upper() == side
                    ):
                        other["status"] = CandidateStatus.CANCELLED
                        other["cancel_reason"] = f"duplicate of executed {candidate_id}"
                        logger.info(
                            f"[{cand_sym}] Cancelled duplicate candidate {other_id} "
                            f"(superseded by executed {candidate_id})"
                        )
                if cand["broker"] == "ctrader":
                    persist_ctrader_execution(
                        symbol=cand["symbol"],
                        direction=side,
                        quantity=qty,
                        entry_price=float(
                            (order_res or {}).get("price")
                            or (order_res or {}).get("entry_price")
                            or cand.get("entry_price")
                            or 0
                        ),
                        stop_loss=cand.get("stop_loss"),
                        take_profit=cand.get("take_profit"),
                        strategy=cand.get("strategy"),
                        order_id=order_id,
                        position_id=(order_res or {}).get("position_id"),
                        notes=msg,
                    )
            else:
                logger.warning(
                    f"Candidate {candidate_id} execution failed ({cand['broker']}): {msg}"
                )

            return {
                "success": success,
                "candidate_id": candidate_id,
                "symbol": cand["symbol"],
                "broker": cand["broker"],
                "order_result": {
                    "success": success,
                    "order_id": order_id,
                    "message": msg,
                },
            }

        except Exception as err:
            logger.error(f"Failed to execute candidate {candidate_id}: {err}")
            return {"success": False, "error": str(err)}


signal_candidate_engine = SignalCandidateEngine()
