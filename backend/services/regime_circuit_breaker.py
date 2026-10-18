"""Regime-Shift Circuit Breaker.

Monitors per-symbol and market-wide regime transitions produced by
:class:`backend.strategies.market_regime.MarketRegimeDetector` and trips a
circuit breaker when structural regime instability is detected.

State machine per symbol (and globally):
  - ``CLOSED``    : Normal operation (size_factor = 1.0).
  - ``OPEN``      : Tripped — blocks NEW entries only (size_factor = 0.0 in
                    enforce mode). Exits, stops, and emergency flattens are
                    NEVER blocked.
  - ``HALF_OPEN`` : Cooldown expired — allows new entries at reduced size
                    (``size_factor = 0.5``) until ``recovery_cycles`` consecutive
                    stable observations restore ``CLOSED``.

Trip triggers (all configurable via constructor or env vars for GPU grid-search):
  1. **Vol-shock transition**: regime flips into ``VOLATILE`` (or flips between
     ``TRENDING`` and ``RANGING``) accompanied by an ATR ratio jump of at least
     ``atr_jump_ratio`` (default ``1.50``, i.e. +50% ATR expansion vs prior bar)
     or absolute ``atr_ratio >= max_atr_ratio`` (default ``0.045``).
  2. **Flip-flop churn**: ``>= flip_threshold`` (default ``3``) regime changes
     within the last ``window_size`` (default ``6``) observations for a symbol.
  3. **Market-wide systemic shift**: within a single scan cycle, at least
     ``universe_shift_pct`` (default ``0.40``, min ``universe_min_symbols=4``)
     of evaluated symbols experience a regime change simultaneously.

Modes (``REGIME_CB_MODE``):
  - ``off``     : No-op.
  - ``shadow``  : Tracks state, logs transitions, records telemetry, never blocks.
  - ``enforce`` : Blocks new entries when ``OPEN`` and scales size by ``0.5`` in
                  ``HALF_OPEN``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Literal, Optional

logger = logging.getLogger(__name__)

CBState = Literal["CLOSED", "OPEN", "HALF_OPEN"]


@dataclass
class RegimeCheckResult:
    symbol: str
    state: CBState
    mode: str
    allow_entry: bool
    would_block: bool
    size_factor: float
    reason: str
    regime: str
    previous_regime: Optional[str] = None
    flip_count: int = 0
    atr_jump: float = 1.0


@dataclass
class _SymbolTracker:
    regimes: Deque[str] = field(default_factory=lambda: deque(maxlen=6))
    atr_ratios: Deque[float] = field(default_factory=lambda: deque(maxlen=6))
    state: CBState = "CLOSED"
    tripped_at: float = 0.0
    trip_reason: str = ""
    stable_cycles_in_half_open: int = 0


class RegimeCircuitBreaker:
    """Per-symbol + market-wide regime-shift circuit breaker."""

    def __init__(
        self,
        mode: Optional[str] = None,
        window_size: Optional[int] = None,
        flip_threshold: Optional[int] = None,
        atr_jump_ratio: Optional[float] = None,
        max_atr_ratio: Optional[float] = None,
        cooldown_seconds: Optional[int] = None,
        recovery_cycles: Optional[int] = None,
        universe_shift_pct: Optional[float] = None,
        universe_min_symbols: Optional[int] = None,
        half_open_size_factor: float = 0.50,
    ) -> None:
        self.mode = (mode or os.getenv("REGIME_CB_MODE", "shadow")).strip().lower()
        self.window_size = int(window_size if window_size is not None else os.getenv("REGIME_CB_WINDOW", "6"))
        self.flip_threshold = int(
            flip_threshold if flip_threshold is not None else os.getenv("REGIME_CB_FLIP_THRESHOLD", "3")
        )
        self.atr_jump_ratio = float(
            atr_jump_ratio if atr_jump_ratio is not None else os.getenv("REGIME_CB_ATR_JUMP_RATIO", "1.50")
        )
        self.max_atr_ratio = float(
            max_atr_ratio if max_atr_ratio is not None else os.getenv("REGIME_CB_MAX_ATR_RATIO", "0.045")
        )
        self.cooldown_seconds = int(
            cooldown_seconds if cooldown_seconds is not None else os.getenv("REGIME_CB_COOLDOWN_SEC", "900")
        )
        self.recovery_cycles = int(
            recovery_cycles if recovery_cycles is not None else os.getenv("REGIME_CB_RECOVERY_CYCLES", "2")
        )
        self.universe_shift_pct = float(
            universe_shift_pct
            if universe_shift_pct is not None
            else os.getenv("REGIME_CB_UNIVERSE_PCT", "0.40")
        )
        self.universe_min_symbols = int(
            universe_min_symbols
            if universe_min_symbols is not None
            else os.getenv("REGIME_CB_UNIVERSE_MIN_SYMBOLS", "4")
        )
        self.half_open_size_factor = float(half_open_size_factor)

        self._symbols: Dict[str, _SymbolTracker] = {}
        self._global_state: CBState = "CLOSED"
        self._global_tripped_at: float = 0.0
        self._global_reason: str = ""
        self._cycle_total: int = 0
        self._cycle_shifted: int = 0

    def _get_tracker(self, symbol: str) -> _SymbolTracker:
        key = symbol.upper()
        tracker = self._symbols.get(key)
        if tracker is None:
            tracker = _SymbolTracker(
                regimes=deque(maxlen=self.window_size),
                atr_ratios=deque(maxlen=self.window_size),
            )
            self._symbols[key] = tracker
        return tracker

    @staticmethod
    def _count_flips(seq: Deque[str]) -> int:
        if len(seq) < 2:
            return 0
        items = list(seq)
        return sum(1 for i in range(1, len(items)) if items[i] != items[i - 1])

    def _notify_telegram(self, text: str) -> None:
        try:
            from backend.utils.telegram import send_telegram_message
            loop = asyncio.get_running_loop()
            loop.create_task(send_telegram_message(text))
        except Exception:
            pass

    def _refresh_cooldowns(self, tracker: _SymbolTracker, now: float) -> None:
        if tracker.state == "OPEN" and (now - tracker.tripped_at) >= self.cooldown_seconds:
            tracker.state = "HALF_OPEN"
            tracker.stable_cycles_in_half_open = 0
            logger.info("[REGIME CB] Symbol cooldown expired -> HALF_OPEN")

        if self._global_state == "OPEN" and (now - self._global_tripped_at) >= self.cooldown_seconds:
            self._global_state = "HALF_OPEN"
            logger.info("[REGIME CB] Market-wide cooldown expired -> HALF_OPEN")

    def observe(
        self,
        symbol: str,
        regime_result: Any,
        now: Optional[float] = None,
    ) -> RegimeCheckResult:
        """Record a new ``RegimeResult`` (or dict) for ``symbol`` and return gate decision."""
        current_mode = (self.mode or os.getenv("REGIME_CB_MODE", "shadow")).strip().lower()
        regime = str(
            getattr(regime_result, "regime", None)
            or (regime_result.get("regime") if isinstance(regime_result, dict) else "UNKNOWN")
        ).upper()
        atr_ratio = float(
            getattr(regime_result, "atr_ratio", 0.0)
            if not isinstance(regime_result, dict)
            else regime_result.get("atr_ratio", 0.0) or 0.0
        )

        if current_mode == "off":
            return RegimeCheckResult(
                symbol=symbol.upper(),
                state="CLOSED",
                mode="off",
                allow_entry=True,
                would_block=False,
                size_factor=1.0,
                reason="regime_cb_off",
                regime=regime,
            )

        ts = now if now is not None else time.monotonic()
        tracker = self._get_tracker(symbol)
        self._refresh_cooldowns(tracker, ts)

        prev_regime = tracker.regimes[-1] if tracker.regimes else None
        prev_atr = tracker.atr_ratios[-1] if tracker.atr_ratios else atr_ratio
        atr_jump = (atr_ratio / prev_atr) if prev_atr > 1e-9 else 1.0

        shifted = prev_regime is not None and prev_regime != regime
        self._cycle_total += 1
        if shifted:
            self._cycle_shifted += 1

        tracker.regimes.append(regime)
        tracker.atr_ratios.append(atr_ratio)
        flips = self._count_flips(tracker.regimes)

        trip_reason: Optional[str] = None
        if shifted and regime == "VOLATILE" and (atr_jump >= self.atr_jump_ratio or atr_ratio >= self.max_atr_ratio):
            trip_reason = (
                f"vol_shock_shift ({prev_regime}->{regime}, atr_jump={atr_jump:.2f}x, atr_ratio={atr_ratio:.4f})"
            )
        elif shifted and {prev_regime, regime} == {"TRENDING", "RANGING"} and atr_jump >= self.atr_jump_ratio:
            trip_reason = f"regime_whipsaw ({prev_regime}->{regime}, atr_jump={atr_jump:.2f}x)"
        elif flips >= self.flip_threshold:
            trip_reason = f"regime_flip_churn ({flips} flips in last {len(tracker.regimes)} bars)"

        if trip_reason:
            was_open = tracker.state == "OPEN"
            tracker.state = "OPEN"
            tracker.tripped_at = ts
            tracker.trip_reason = trip_reason
            tracker.stable_cycles_in_half_open = 0
            if not was_open:
                logger.warning(
                    "[REGIME CB] TRIPPED for %s (mode=%s): %s",
                    symbol.upper(),
                    current_mode,
                    trip_reason,
                )
                self._notify_telegram(
                    f"⚡ *Regime Circuit Breaker TRIPPED* (`{symbol.upper()}`, mode=`{current_mode}`): {trip_reason}"
                )
        elif tracker.state == "HALF_OPEN":
            if shifted:
                tracker.state = "OPEN"
                tracker.tripped_at = ts
                tracker.trip_reason = f"half_open_relapse ({prev_regime}->{regime})"
                tracker.stable_cycles_in_half_open = 0
                logger.warning("[REGIME CB] %s relapsed in HALF_OPEN -> OPEN", symbol.upper())
            else:
                tracker.stable_cycles_in_half_open += 1
                if tracker.stable_cycles_in_half_open >= self.recovery_cycles:
                    tracker.state = "CLOSED"
                    tracker.trip_reason = ""
                    tracker.stable_cycles_in_half_open = 0
                    logger.info("[REGIME CB] %s recovered HALF_OPEN -> CLOSED", symbol.upper())

        effective_state: CBState = tracker.state
        reason = tracker.trip_reason or "ok"
        if self._global_state == "OPEN":
            effective_state = "OPEN"
            reason = self._global_reason or reason
        elif self._global_state == "HALF_OPEN" and effective_state == "CLOSED":
            effective_state = "HALF_OPEN"
            reason = self._global_reason or "market_wide_half_open"

        would_block = effective_state == "OPEN"
        allow_entry = not (would_block and current_mode == "enforce")
        if effective_state == "OPEN":
            size_factor = 0.0 if current_mode == "enforce" else 1.0
        elif effective_state == "HALF_OPEN":
            size_factor = self.half_open_size_factor if current_mode == "enforce" else 1.0
        else:
            size_factor = 1.0

        return RegimeCheckResult(
            symbol=symbol.upper(),
            state=effective_state,
            mode=current_mode,
            allow_entry=allow_entry,
            would_block=would_block,
            size_factor=size_factor,
            reason=reason,
            regime=regime,
            previous_regime=prev_regime,
            flip_count=flips,
            atr_jump=round(atr_jump, 3),
        )

    def end_cycle(self, now: Optional[float] = None) -> dict[str, Any]:
        """Evaluate market-wide shift ratio at the end of a symbol scan cycle."""
        ts = now if now is not None else time.monotonic()
        total = self._cycle_total
        shifted = self._cycle_shifted
        self._cycle_total = 0
        self._cycle_shifted = 0

        ratio = (shifted / total) if total > 0 else 0.0
        tripped_now = False
        if total >= self.universe_min_symbols and ratio >= self.universe_shift_pct:
            self._global_state = "OPEN"
            self._global_tripped_at = ts
            self._global_reason = (
                f"market_wide_regime_shift ({shifted}/{total} symbols = {ratio * 100:.1f}% shifted in 1 cycle)"
            )
            tripped_now = True
            logger.warning("[REGIME CB] MARKET-WIDE TRIP: %s", self._global_reason)
            self._notify_telegram(f"🚨 *Market-Wide Regime Circuit Breaker*: {self._global_reason}")
        elif self._global_state == "HALF_OPEN" and total >= self.universe_min_symbols and ratio < (self.universe_shift_pct * 0.5):
            self._global_state = "CLOSED"
            self._global_reason = ""
            logger.info("[REGIME CB] Market-wide breaker recovered HALF_OPEN -> CLOSED")

        return {
            "total_symbols": total,
            "shifted_symbols": shifted,
            "shift_ratio": round(ratio, 3),
            "global_state": self._global_state,
            "tripped_now": tripped_now,
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._symbols.clear()
            self._global_state = "CLOSED"
            self._global_tripped_at = 0.0
            self._global_reason = ""
            self._cycle_total = 0
            self._cycle_shifted = 0
        else:
            self._symbols.pop(symbol.upper(), None)

    def get_status(self) -> dict[str, Any]:
        now = time.monotonic()
        symbols_out: dict[str, Any] = {}
        for sym, tr in self._symbols.items():
            self._refresh_cooldowns(tr, now)
            rem = max(0, int(self.cooldown_seconds - (now - tr.tripped_at))) if tr.state == "OPEN" else 0
            symbols_out[sym] = {
                "state": tr.state,
                "last_regime": tr.regimes[-1] if tr.regimes else None,
                "flips_in_window": self._count_flips(tr.regimes),
                "cooldown_remaining_sec": rem,
                "reason": tr.trip_reason,
            }
        return {
            "mode": self.mode,
            "global_state": self._global_state,
            "global_reason": self._global_reason,
            "symbols": symbols_out,
        }


regime_circuit_breaker = RegimeCircuitBreaker()
