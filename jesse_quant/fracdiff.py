"""Causal fractional differentiation (Lopez de Prado FFD).

Finds the smallest d in [d_min, d_max] such that the transformed series is
ADF-stationary (p < adf_p_max) while keeping correlation with the raw level.
The chosen d is stored on the model artifact so live inference uses the same
kernel — never re-searched at serve time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
except ImportError:
    _adfuller = None

D_SEARCH_MIN = 0.20
D_SEARCH_MAX = 0.70
D_MIN = 0.35
D_MAX = 0.55
D_STEP = 0.05
DEFAULT_D = 0.45
ADF_P_MAX = 0.05
WEIGHT_THRESHOLD = 1e-4
MAX_WEIGHTS = 80


def fracdiff_weights(d: float, threshold: float = WEIGHT_THRESHOLD, max_weights: int = MAX_WEIGHTS) -> np.ndarray:
    """Fixed-window FFD kernel. w[0] is the oldest weight (causal convolution)."""
    if d < 0:
        raise ValueError(f"fracdiff d must be >= 0, got {d}")
    weights = [1.0]
    k = 1
    while k < max_weights:
        nxt = -weights[-1] / k * (d - k + 1.0)
        if abs(nxt) < threshold:
            break
        weights.append(float(nxt))
        k += 1
    return np.asarray(weights[::-1], dtype=float)


def fracdiff_series(
    series: pd.Series | np.ndarray | Sequence[float],
    d: float,
    *,
    threshold: float = WEIGHT_THRESHOLD,
) -> pd.Series:
    """Apply a causal FFD transform. Value at t uses observations <= t only."""
    if isinstance(series, pd.Series):
        values = series.to_numpy(dtype=float)
        index = series.index
    else:
        values = np.asarray(series, dtype=float)
        index = pd.RangeIndex(len(values))
    weights = fracdiff_weights(float(d), threshold=threshold)
    width = len(weights)
    out = np.full(len(values), np.nan, dtype=float)
    if width == 0 or len(values) < width:
        return pd.Series(out, index=index)
    # Sliding-window dot product: window [t-width+1, t] inclusive.
    for i in range(width - 1, len(values)):
        window = values[i - width + 1 : i + 1]
        if not np.isfinite(window).all():
            continue
        out[i] = float(np.dot(weights, window))
    return pd.Series(out, index=index)


def _adf_pvalue(values: np.ndarray) -> float:
    """ADF p-value. Prefers statsmodels; falls back to a MacKinnon-style t-map."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return 1.0
    if _adfuller is not None:
        try:
            return float(_adfuller(arr, maxlag=1, regression="c", autolag=None)[1])
        except Exception:
            pass
    y = np.diff(arr)
    lag = arr[:-1]
    design = np.column_stack([np.ones(len(lag)), lag])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        dof = max(len(y) - design.shape[1], 1)
        sigma2 = float(np.sum(resid**2) / dof)
        xtx_inv = np.linalg.inv(design.T @ design)
        se = float(np.sqrt(max(sigma2 * xtx_inv[1, 1], 1e-18)))
        tstat = float(beta[1] / se)
    except (np.linalg.LinAlgError, ValueError):
        return 1.0
    # Constant-only ADF 5% critical value is about -2.86. Map t to a coarse p.
    if tstat <= -3.43:
        return 0.01
    if tstat <= -2.86:
        return 0.04
    if tstat <= -2.57:
        return 0.10
    return 0.50


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 10:
        return 0.0
    a = left[mask]
    b = right[mask]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def select_fracdiff_d(
    series: pd.Series | np.ndarray | Sequence[float],
    *,
    d_min: float = D_MIN,
    d_max: float = D_MAX,
    d_step: float = D_STEP,
    adf_p_max: float = ADF_P_MAX,
    default_d: float = DEFAULT_D,
) -> Dict[str, Any]:
    """Grid-search the smallest stationary d.

    Full search is [0.20, 0.70]; the research band [0.35, 0.55] wins when any
    candidate there is ADF-stationary (p < adf_p_max).
    """
    if isinstance(series, pd.Series):
        raw = series.to_numpy(dtype=float)
    else:
        raw = np.asarray(series, dtype=float)
    search_min = min(float(d_min), D_SEARCH_MIN)
    search_max = max(float(d_max), D_SEARCH_MAX)
    grid = [round(float(d), 4) for d in np.arange(search_min, search_max + 1e-9, d_step)]
    candidates: list[dict[str, Any]] = []
    for d in grid:
        transformed = fracdiff_series(raw, d).to_numpy()
        p_value = _adf_pvalue(transformed)
        corr = _correlation(transformed, raw)
        row = {"d": d, "adf_pvalue": p_value, "level_corr": corr, "stationary": p_value < adf_p_max}
        candidates.append(row)
    stationary = [row for row in candidates if row["stationary"]]
    preferred = [row for row in stationary if d_min - 1e-9 <= row["d"] <= d_max + 1e-9]
    if preferred:
        chosen = min(preferred, key=lambda row: (row["d"], -row["level_corr"]))
        source = "prefer_band_stationary"
    elif stationary:
        chosen = min(stationary, key=lambda row: (row["d"], -row["level_corr"]))
        source = "extended_grid_stationary"
    elif candidates:
        prefer_only = [row for row in candidates if d_min - 1e-9 <= row["d"] <= d_max + 1e-9]
        pool = prefer_only or candidates
        chosen = min(pool, key=lambda row: (row["adf_pvalue"], -row["level_corr"]))
        source = "default_or_best_effort"
    else:
        chosen = {"d": default_d, "adf_pvalue": 1.0, "level_corr": 0.0, "stationary": False}
        source = "default_or_best_effort"
    return {
        "d": float(chosen["d"]),
        "adf_pvalue": float(chosen["adf_pvalue"]),
        "level_corr": float(chosen["level_corr"]),
        "stationary": bool(chosen["stationary"]),
        "grid": candidates,
        "d_min": d_min,
        "d_max": d_max,
        "d_search_min": search_min,
        "d_search_max": search_max,
        "source": source,
    }


def resolve_artifact_d(
    artifact: Optional[Dict[str, Any]],
    *,
    symbol: Optional[str] = None,
    default_d: float = DEFAULT_D,
) -> float:
    """Read per-symbol d from a saved joblib/meta payload. Never re-grid at serve."""
    payload = artifact or {}
    per_symbol = payload.get("fracdiff_d_by_symbol") or {}
    if symbol and symbol in per_symbol:
        return float(per_symbol[symbol])
    if payload.get("fracdiff_d") is not None:
        return float(payload["fracdiff_d"])
    metrics = payload.get("metrics") or {}
    if metrics.get("fracdiff_d") is not None:
        return float(metrics["fracdiff_d"])
    return float(default_d)


def fracdiff_feature_frame(
    close: pd.Series,
    d: float,
) -> pd.DataFrame:
    """Return fd_close (FFD of price) and fd_logret (FFD of log price)."""
    close = close.astype(float)
    log_close = np.log(close.clip(lower=1e-12))
    return pd.DataFrame(
        {
            "fd_close": fracdiff_series(close, d),
            "fd_logret": fracdiff_series(log_close, d),
        },
        index=close.index,
    )


def kernel_width(d: float, threshold: float = WEIGHT_THRESHOLD) -> int:
    return int(len(fracdiff_weights(d, threshold=threshold)))
