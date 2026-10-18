"""Detect a stuck Jesse ML conformal-uncertainty veto.

BNBUSDC was vetoed 144 times with the identical reason
``uncertainty margin 0.064 < 0.08``. That is not a changing forecast —
the feature vector (or the cached prediction) is frozen, so the gate can
never clear and the symbol never trades.

Safety intent stays: a *moving* HIGH-uncertainty or thin-margin forecast
still vetoes. Only an identical (margin, close) pair repeated
``JESSE_ML_STUCK_VETO_HITS`` times (default 5) is treated as stale and
the veto is skipped for that evaluation.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional


def _stuck_hits() -> int:
    raw = os.getenv("JESSE_ML_STUCK_VETO_HITS", "5").strip()
    try:
        return max(3, int(raw))
    except ValueError:
        return 5


class StuckUncertaintyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[str, int]] = {}

    def _fingerprint(self, *, margin: Any, latest_close: Any, reason: str) -> str:
        try:
            margin_s = f"{float(margin):.3f}"
        except (TypeError, ValueError):
            margin_s = str(margin or "")
        try:
            close_s = f"{float(latest_close):.6f}" if latest_close not in (None, "") else ""
        except (TypeError, ValueError):
            close_s = str(latest_close or "")
        return f"{margin_s}|{close_s}|{(reason or '')[:80]}"

    def observe(
        self,
        symbol: str,
        *,
        margin: Any = None,
        latest_close: Any = None,
        reason: str = "",
    ) -> bool:
        """Record one identical uncertainty veto. True = fail-open (stuck)."""
        key = str(symbol or "").upper()
        if not key:
            return False
        fp = self._fingerprint(margin=margin, latest_close=latest_close, reason=reason)
        with self._lock:
            prev_fp, count = self._hits.get(key, ("", 0))
            if prev_fp == fp:
                count += 1
            else:
                count = 1
            self._hits[key] = (fp, count)
            hits = _stuck_hits()
            # Fail-open only every Nth identical hit so a frozen model
            # cannot trade every cycle, but also cannot lock a symbol forever.
            return count >= hits and count % hits == 0

    def reset(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol is None:
                self._hits.clear()
                return
            self._hits.pop(str(symbol).upper(), None)

    def hits(self, symbol: str) -> int:
        with self._lock:
            _fp, count = self._hits.get(str(symbol).upper(), ("", 0))
            return count


stuck_uncertainty = StuckUncertaintyTracker()
