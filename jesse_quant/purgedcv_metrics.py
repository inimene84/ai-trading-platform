"""GPU-side DSR / PBO / CPCV via the pinned purgedcv library.

The Decision Engine never imports this module to recompute DSR/PBO from
prices. Training jobs call these helpers, write metrics.json, and stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# Optional GPU extra — Decision Engine must not hard-require purgedcv.
try:
    import purgedcv as _PURGEDCV
except ImportError:
    _PURGEDCV = None

try:
    from purgedcv.optuna_integration import TrialSharpeRecorder as _TrialSharpeRecorder
except ImportError:
    _TrialSharpeRecorder = None

PURGEDCV_IMPORTS = (
    "CombinatorialPurgedCV",
    "PurgedKFold",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_full",
    "DSRDiagnostics",
    "probability_of_backtest_overfitting",
    "PBOResult",
    "effective_n_trials",
    "path_metrics",
    "reconstruct_paths",
    "probabilistic_sharpe_ratio",
    "minimum_backtest_length",
)

EFFECTIVE_SOURCE = "effective_n_trials"
RECORDER_SOURCE = "TrialSharpeRecorder.n_effective"


class PurgedcvUnavailable(RuntimeError):
    """Raised when the GPU job cannot import the pinned purgedcv library."""


class PromotionMetricError(ValueError):
    """Fail-closed error while assembling promotion metrics."""


def import_purgedcv() -> Any:
    """Return the pinned library. Do not substitute a homegrown DSR."""
    if _PURGEDCV is None:
        raise PurgedcvUnavailable(
            "purgedcv is required on the GPU training job "
            "(pip install purgedcv — eslazarev/purged-cross-validation)"
        )
    return _PURGEDCV


def import_trial_sharpe_recorder() -> Any:
    if _TrialSharpeRecorder is not None:
        return _TrialSharpeRecorder
    purgedcv = import_purgedcv()
    recorder = getattr(purgedcv, "TrialSharpeRecorder", None)
    if recorder is None:
        raise PurgedcvUnavailable(
            "purgedcv.optuna_integration.TrialSharpeRecorder is required "
            "(pip install purgedcv[optuna])"
        )
    return recorder


@dataclass
class EffectiveTrials:
    n_effective: float
    n_raw: int
    source: str
    var_sharpe: Optional[float] = None


@dataclass
class DSRComputation:
    dsr: float
    dsr_is_probability: bool
    n_trials_effective: float
    n_trials_raw: int
    n_trials_source: str
    var_sharpe: Optional[float] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PBOComputation:
    pbo: float
    n_configs: int
    n_obs: int
    n_splits: int
    feasible: bool
    matrix_complete: bool
    reject_reason: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


def resolve_effective_n_trials(
    *,
    trial_sharpes: Sequence[float] | None = None,
    recorder: Any = None,
    raw_n_trials: int | None = None,
) -> EffectiveTrials:
    """n_trials MUST be effective_n_trials or TrialSharpeRecorder.n_effective().

    Passing study.n_trials is N_TRIALS_NOT_EFFECTIVE — this helper refuses it.
    """
    if recorder is not None:
        n_eff = float(recorder.n_effective())
        n_raw = int(recorder.n_trials()) if hasattr(recorder, "n_trials") else int(raw_n_trials or n_eff)
        var_sharpe = float(recorder.var_sharpe(ddof=1)) if hasattr(recorder, "var_sharpe") else None
        return EffectiveTrials(
            n_effective=n_eff,
            n_raw=n_raw,
            source=RECORDER_SOURCE,
            var_sharpe=var_sharpe,
        )

    if trial_sharpes is None:
        raise PromotionMetricError("effective n_trials requires TrialSharpeRecorder or a trial-Sharpe series")

    sharpes = np.asarray(list(trial_sharpes), dtype=float)
    if sharpes.size < 1:
        raise PromotionMetricError("trial Sharpe series is empty")
    purgedcv = import_purgedcv()
    n_eff = float(purgedcv.effective_n_trials(sharpes))
    n_raw = int(raw_n_trials if raw_n_trials is not None else sharpes.size)
    var_sharpe = float(np.var(sharpes, ddof=1)) if sharpes.size > 1 else None
    return EffectiveTrials(
        n_effective=n_eff,
        n_raw=n_raw,
        source=EFFECTIVE_SOURCE,
        var_sharpe=var_sharpe,
    )


def _extract_dsr(result: Any) -> tuple[float, dict[str, Any]]:
    if hasattr(result, "dsr"):
        extras = {}
        for name in ("observed_sr", "sr_star", "expected_max_z", "var_sharpe"):
            if hasattr(result, name):
                extras[name] = getattr(result, name)
        return float(result.dsr), extras
    return float(result), {}


def compute_dsr(
    costed_returns: Sequence[float] | np.ndarray,
    *,
    effective: EffectiveTrials,
    bars_per_year: float | None = None,
    use_full: bool = True,
) -> DSRComputation:
    """DSR as a probability in [0, 1] on **costed** selected-config returns."""
    if effective.source not in {EFFECTIVE_SOURCE, RECORDER_SOURCE}:
        raise PromotionMetricError("N_TRIALS_NOT_EFFECTIVE: refuse raw Optuna n_trials")

    returns = np.asarray(list(costed_returns), dtype=float)
    if returns.size < 2:
        raise PromotionMetricError("DSR_MISSING: costed returns too short")

    purgedcv = import_purgedcv()
    kwargs: dict[str, Any] = {
        "n_trials": effective.n_effective,
        "var_sharpe": effective.var_sharpe,
    }
    if bars_per_year is not None:
        kwargs["bars_per_year"] = bars_per_year

    if use_full:
        raw = purgedcv.deflated_sharpe_ratio_full(returns, **kwargs)
    else:
        raw = purgedcv.deflated_sharpe_ratio(returns, **kwargs)
    dsr, extras = _extract_dsr(raw)
    if not (0.0 <= dsr <= 1.0):
        raise PromotionMetricError(
            f"DSR_MISSING: purgedcv returned {dsr!r} which is not a probability in [0, 1]"
        )
    return DSRComputation(
        dsr=dsr,
        dsr_is_probability=True,
        n_trials_effective=effective.n_effective,
        n_trials_raw=effective.n_raw,
        n_trials_source=effective.source,
        var_sharpe=effective.var_sharpe,
        diagnostics=extras,
    )


def _normalize_pbo_matrix(returns: np.ndarray) -> np.ndarray:
    """Accept (n_obs, n_configs) parquet layout or (n_configs, n_obs) library layout."""
    if returns.ndim != 2:
        raise PromotionMetricError("PBO_MATRIX_INCOMPLETE: returns must be 2-D")
    return returns


def compute_pbo(
    trial_returns: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_splits: int = 16,
    completed_trial_count: int | None = None,
    orientation: str = "configs_by_obs",
) -> PBOComputation:
    """PBO over the full completed-trial returns matrix. Never winner-only.

    ``orientation``:
      - ``configs_by_obs``: shape (n_configs, n_obs) as purgedcv expects
      - ``obs_by_configs``: shape (n_obs, n_configs) as trials_returns.parquet
    """
    matrix = np.asarray(trial_returns, dtype=float)
    matrix = _normalize_pbo_matrix(matrix)
    if orientation == "obs_by_configs":
        matrix = matrix.T
    elif orientation != "configs_by_obs":
        raise PromotionMetricError(f"unknown PBO matrix orientation {orientation!r}")

    n_configs, n_obs = int(matrix.shape[0]), int(matrix.shape[1])
    if n_configs < 2 or n_obs < n_splits:
        return PBOComputation(
            pbo=1.0,
            n_configs=n_configs,
            n_obs=n_obs,
            n_splits=n_splits,
            feasible=False,
            matrix_complete=False,
            reject_reason="PBO_INFEASIBLE",
        )

    if completed_trial_count is not None and n_configs < int(completed_trial_count):
        return PBOComputation(
            pbo=1.0,
            n_configs=n_configs,
            n_obs=n_obs,
            n_splits=n_splits,
            feasible=True,
            matrix_complete=False,
            reject_reason="PBO_MATRIX_INCOMPLETE",
        )

    purgedcv = import_purgedcv()
    result = purgedcv.probability_of_backtest_overfitting(matrix, n_splits=n_splits)
    pbo = float(getattr(result, "pbo", result))
    extras: dict[str, Any] = {}
    if hasattr(result, "n_combos"):
        extras["n_combos"] = int(result.n_combos)
    if hasattr(result, "slope"):
        extras["slope"] = float(result.slope)
    return PBOComputation(
        pbo=pbo,
        n_configs=n_configs,
        n_obs=n_obs,
        n_splits=n_splits,
        feasible=True,
        matrix_complete=True,
        extras=extras,
    )


def compute_path_metrics(
    *,
    prediction_times: Any,
    evaluation_times: Any,
    returns: Sequence[float] | np.ndarray,
    n_splits: int = 16,
    n_test_groups: int = 2,
    n_test_folds: Optional[int] = None,
    purge_horizon: Any = None,
    embargo: Any = None,
    embargo_fraction: Optional[float] = 0.01,
) -> dict[str, Any]:
    """Outer CombinatorialPurgedCV + reconstruct_paths + path_metrics."""
    purgedcv = import_purgedcv()
    k = n_test_folds if n_test_folds is not None else n_test_groups
    kwargs: dict[str, Any] = {
        "prediction_times": prediction_times,
        "evaluation_times": evaluation_times,
    }
    if purge_horizon is not None:
        kwargs["purge_horizon"] = purge_horizon
    if embargo_fraction is not None:
        kwargs["embargo_fraction"] = embargo_fraction
    elif embargo is not None:
        kwargs["embargo"] = embargo

    splitter = purgedcv.CombinatorialPurgedCV(
        n_splits,
        k,
        **kwargs,
    )
    rets_arr = np.asarray(returns, dtype=float)
    n_samples = len(rets_arr)
    fold_preds = [rets_arr[te] for _, te in splitter.split(np.zeros((n_samples, 1)))]
    paths = splitter.reconstruct_paths(fold_preds)
    table = purgedcv.path_metrics(paths)
    sharpes: list[float] = []
    if hasattr(table, "get"):
        col = table.get("sharpe") if hasattr(table, "get") else None
        if col is not None:
            sharpes = [float(x) for x in col if np.isfinite(x)]
    elif hasattr(table, "__iter__"):
        for row in table:
            if isinstance(row, dict) and "sharpe" in row and np.isfinite(row["sharpe"]):
                sharpes.append(float(row["sharpe"]))
    p10 = float(np.percentile(sharpes, 10)) if sharpes else None
    return {
        "n_paths": len(sharpes),
        "path_sharpe_p10": p10,
        "path_sharpes": sharpes,
    }

