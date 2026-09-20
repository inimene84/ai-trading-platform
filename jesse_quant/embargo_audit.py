"""Purged-CV + embargo audit for Triple-Barrier labels.

The label at t uses the path t … t+max_holding_bars. If a training observation
whose label window overlaps the test fold survives, the fold leaks. Embargo
after the test window must be at least max(max_holding_bars, longest feature
lookback). On the live 1h book that is max(48, 200) = 200 bars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from barrier_config import MAX_HOLDING_BARS

# Longest causal rolling window in ml_features.py (EMA-200). FracDiff kernel
# width is added on top when a d is supplied.
EMA200_LOOKBACK_BARS = 200
DEFAULT_FEATURE_LOOKBACK_BARS = EMA200_LOOKBACK_BARS


@dataclass(frozen=True)
class EmbargoAudit:
    ok: bool
    embargo_bars: int
    purge_horizon_bars: int
    required_embargo_bars: int
    max_holding_bars: int
    longest_lookback_bars: int
    timeframe: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def longest_feature_lookback_bars(fracdiff_width: int = 0) -> int:
    return int(max(DEFAULT_FEATURE_LOOKBACK_BARS, int(fracdiff_width)))


def required_embargo_bars(
    max_holding_bars: int = MAX_HOLDING_BARS,
    longest_lookback_bars: int = DEFAULT_FEATURE_LOOKBACK_BARS,
) -> int:
    return int(max(int(max_holding_bars), int(longest_lookback_bars)))


def audit_purged_embargo(
    *,
    embargo_bars: int,
    purge_horizon_bars: int,
    max_holding_bars: int = MAX_HOLDING_BARS,
    longest_lookback_bars: int = DEFAULT_FEATURE_LOOKBACK_BARS,
    timeframe: str = "1h",
) -> EmbargoAudit:
    """Fail closed when embargo or purge is shorter than the label/feature horizon."""
    required = required_embargo_bars(max_holding_bars, longest_lookback_bars)
    reasons: list[str] = []
    if int(embargo_bars) < required:
        reasons.append(
            f"embargo {int(embargo_bars)} < required {required} "
            f"(max(max_holding={int(max_holding_bars)}, lookback={int(longest_lookback_bars)}))"
        )
    if int(purge_horizon_bars) < int(max_holding_bars):
        reasons.append(
            f"purge horizon {int(purge_horizon_bars)} < max_holding_bars {int(max_holding_bars)}"
        )
    if timeframe == "1h" and int(max_holding_bars) < MAX_HOLDING_BARS:
        reasons.append(
            f"1h max_holding_bars {int(max_holding_bars)} < live geometry {MAX_HOLDING_BARS}"
        )
    return EmbargoAudit(
        ok=len(reasons) == 0,
        embargo_bars=int(embargo_bars),
        purge_horizon_bars=int(purge_horizon_bars),
        required_embargo_bars=required,
        max_holding_bars=int(max_holding_bars),
        longest_lookback_bars=int(longest_lookback_bars),
        timeframe=str(timeframe),
        reasons=reasons,
    )


def embargo_bars_for_n(
    n_samples: int,
    *,
    max_holding_bars: int = MAX_HOLDING_BARS,
    longest_lookback_bars: int = DEFAULT_FEATURE_LOOKBACK_BARS,
    embargo_pct: Optional[float] = None,
) -> int:
    """Integer embargo used by PurgedKFold: bars win over a percent of n."""
    required = required_embargo_bars(max_holding_bars, longest_lookback_bars)
    if embargo_pct is not None:
        return max(required, int(n_samples * float(embargo_pct)))
    return required


def audit_from_validation_block(
    validation: Mapping[str, Any],
    *,
    max_holding_bars: int = MAX_HOLDING_BARS,
    longest_lookback_bars: int = DEFAULT_FEATURE_LOOKBACK_BARS,
    n_samples: int = 0,
    timeframe: str = "1h",
) -> EmbargoAudit:
    """Read purge/embargo from geometry.json validation{} or a trainer dict."""
    raw_bars = validation.get("embargo_bars")
    if raw_bars is None and n_samples > 0:
        raw_bars = int(n_samples * float(validation.get("embargo_fraction") or 0.0))
    embargo = int(raw_bars or 0)
    purge = int(validation.get("purge_horizon") or validation.get("purge_horizon_bars") or 0)
    return audit_purged_embargo(
        embargo_bars=embargo,
        purge_horizon_bars=purge,
        max_holding_bars=max_holding_bars,
        longest_lookback_bars=longest_lookback_bars,
        timeframe=timeframe,
    )
