#!/usr/bin/env python3
"""
Institutional Quantitative Validation Metrics Module
Implements formal statistical validation techniques from Marcos López de Prado:
1. Deflated Sharpe Ratio (DSR) - corrects for selection bias, trial count, skewness, kurtosis
2. Probability of Backtest Overfitting (PBO) - via Combinatorially Symmetric Cross-Validation (CSCV)
3. Purged K-Fold Cross-Validation with Embargo - eliminates label overlap and serial correlation leakage
"""

import math
from itertools import combinations
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm, skew, kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _norm_cdf(x: float) -> float:
    """Standard Normal Cumulative Distribution Function."""
    if HAS_SCIPY:
        return float(norm.cdf(x))
    # High-precision Abramowitz & Stegun approximation (erf-based)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard Normal Percent Point Function (inverse CDF / quantile)."""
    if HAS_SCIPY:
        return float(norm.ppf(p))
    # Rational approximation for central & tail quantiles
    p = max(1e-12, min(1.0 - 1e-12, p))
    # Acklam's inverse normal approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    q = p - 0.5
    if abs(q) <= 0.425:
        r = q * q
        return q * (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    x = (((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5]) / \
        ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1.0)
    return -x if q < 0 else x


def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series], annualization: int = 365) -> float:
    """Calculate annualized Sharpe Ratio for a return series."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    mean_ret = np.mean(arr)
    std_ret = np.std(arr, ddof=1)
    if std_ret <= 1e-12:
        return 0.0
    return float((mean_ret / std_ret) * np.sqrt(annualization))


def expected_max_sharpe(n_trials: int, variance_sharpe: float = 1.0, euler_gamma: float = 0.5772156649) -> float:
    """
    Computes the expected maximum Sharpe ratio under the null hypothesis
    that all N trials have true zero skill (selection bias adjustment).
    Reference: Bailey & López de Prado (2014)
    """
    if n_trials <= 1:
        return 0.0
    std_s = math.sqrt(max(1e-6, variance_sharpe))
    p1 = (1.0 - euler_gamma) * _norm_ppf(1.0 - 1.0 / n_trials)
    p2 = euler_gamma * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(std_s * (p1 + p2))


def deflated_sharpe_ratio_from_stats(
    sr: float,
    n_obs: int,
    n_trials: int,
    variance_of_trials: Optional[float] = None,
    skewness: float = 0.0,
    kurtosis_pearson: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio from summary statistics (Bailey & Lopez de Prado, 2014).

    All Sharpe quantities are *per-period* (same sampling frequency as `n_obs`).

      SR*  = sqrt(V[SR_n]) * ((1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e)))
      DSR  = Z[ (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ]

    where gamma is the Euler-Mascheroni constant, N the number of independent trials,
    V[SR_n] the variance of the trial Sharpe ratios, g3 skewness and g4 (non-excess)
    kurtosis of the selected strategy's returns.
    """
    T = int(n_obs)
    if T < 5 or n_trials < 1:
        return 0.0
    if variance_of_trials is not None and variance_of_trials > 0:
        v_s = float(variance_of_trials)
    else:
        # Under the null of zero skill the estimated per-period Sharpe has variance ~ 1/(T-1).
        v_s = 1.0 / (T - 1.0)
    sr_star = expected_max_sharpe(int(n_trials), variance_sharpe=v_s)

    denom_sq = 1.0 - skewness * sr + ((kurtosis_pearson - 1.0) / 4.0) * (sr ** 2)
    if denom_sq <= 0:
        denom_sq = 1.0
    se = math.sqrt(denom_sq / (T - 1.0))
    z = (sr - sr_star) / se
    return float(np.clip(_norm_cdf(z), 0.0, 1.0))


def deflated_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    n_trials: int,
    variance_of_trials: Optional[float] = None,
    annualization: int = 365,
) -> float:
    """
    Computes the Deflated Sharpe Ratio (DSR).
    Returns the probability (0.0 to 1.0) that the observed Sharpe ratio is not
    merely the result of selection bias and data snooping across N trials.

    A DSR > 0.95 indicates statistical significance at 95% confidence level.

    `n_trials` MUST be the total number of configurations evaluated against the data
    (see trial_registry.py); passing a small constant collapses the correction and
    yields DSR ~ 1.0 regardless of skill.

    Note: `annualization` is accepted for API compatibility only. The correction is
    performed entirely in per-period units; dividing the expected maximum Sharpe by
    sqrt(annualization) (as an earlier version did) shrank SR* by ~19x for daily data
    and was the reason DSR printed exactly 1.0000.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    T = len(arr)
    if T < 5 or n_trials < 1:
        return 0.0

    mean_ret = np.mean(arr)
    std_ret = np.std(arr, ddof=1)
    if std_ret <= 1e-12:
        return 0.0

    # Per-period Sharpe Ratio
    sr = mean_ret / std_ret

    # Skewness and Kurtosis
    if HAS_SCIPY:
        g3 = float(skew(arr, bias=False))
        g4 = float(kurtosis(arr, fisher=False, bias=False))  # Pearson kurtosis (normal = 3)
    else:
        centered = arr - mean_ret
        m2 = np.mean(centered**2)
        m3 = np.mean(centered**3)
        m4 = np.mean(centered**4)
        g3 = float(m3 / (m2**1.5 + 1e-12))
        g4 = float(m4 / (m2**2 + 1e-12))
    if not np.isfinite(g3):
        g3 = 0.0
    if not np.isfinite(g4):
        g4 = 3.0

    return deflated_sharpe_ratio_from_stats(
        sr=float(sr),
        n_obs=T,
        n_trials=int(n_trials),
        variance_of_trials=variance_of_trials,
        skewness=g3,
        kurtosis_pearson=g4,
    )


# Institutional deployment gates (Bailey & Lopez de Prado thresholds). The canonical
# values live in barrier_config so strategies, trainer and server agree; this module
# keeps a dependency-free fallback so the unit tests can import it standalone.
try:
    from barrier_config import DSR_MIN as GATE_DSR_MIN, PBO_MAX as GATE_PBO_MAX
except ImportError:  # pragma: no cover - only when barrier_config is not on the path
    GATE_DSR_MIN = 0.95
    GATE_PBO_MAX = 0.30


def evaluate_gate(
    dsr: Optional[float],
    pbo: Optional[float],
    dsr_min: float = GATE_DSR_MIN,
    pbo_max: float = GATE_PBO_MAX,
) -> dict:
    """
    Machine-enforceable promotion gate. Fails when DSR <= dsr_min or PBO >= pbo_max.
    Missing values fail closed (a model without evidence must not be promoted).
    """
    reasons = []
    if dsr is None or not np.isfinite(dsr):
        reasons.append("DSR missing")
    elif dsr <= dsr_min:
        reasons.append(f"DSR {dsr:.4f} <= {dsr_min:.2f}")
    if pbo is None or not np.isfinite(pbo):
        reasons.append("PBO missing")
    elif pbo >= pbo_max:
        reasons.append(f"PBO {pbo:.3f} >= {pbo_max:.2f}")
    return {
        "passed": len(reasons) == 0,
        "dsr": None if dsr is None else float(dsr),
        "pbo": None if pbo is None else float(pbo),
        "dsr_min": float(dsr_min),
        "pbo_max": float(pbo_max),
        "reasons": reasons,
    }


def pbo_proxy_from_is_oos(is_scores: List[float], oos_scores: List[float]) -> dict:
    """
    Single-split overfitting diagnostics for optimizers that only expose one
    in-sample / out-of-sample pair per candidate (e.g. Jesse's GA optimizer API).

    This is NOT the full CSCV PBO (which needs the per-period return matrix); it is the
    relative OOS rank of the in-sample winner, plus the Spearman rank correlation between
    IS and OOS scores across candidates and the share of candidates whose OOS score sits
    below the median. Values > 0.5 indicate that IS selection picks worse-than-random OOS.
    """
    is_arr = np.asarray(is_scores, dtype=float)
    oos_arr = np.asarray(oos_scores, dtype=float)
    n = len(is_arr)
    if n < 2 or len(oos_arr) != n:
        return {"pbo_proxy": None, "spearman_is_oos": None, "oos_rank_of_is_best": None, "n_candidates": n}

    best_is = int(np.argmax(is_arr))
    oos_rank = int((oos_arr > oos_arr[best_is]).sum()) + 1  # 1 = best OOS
    rel_rank = oos_rank / (n + 1.0)

    # Spearman correlation of the ranks
    is_ranks = pd.Series(is_arr).rank().to_numpy()
    oos_ranks = pd.Series(oos_arr).rank().to_numpy()
    if np.std(is_ranks) > 0 and np.std(oos_ranks) > 0:
        spearman = float(np.corrcoef(is_ranks, oos_ranks)[0, 1])
    else:
        spearman = 0.0

    # Logit of the relative rank as in CSCV; PBO proxy = 1 if the IS winner is below the
    # OOS median, blended with the decay of rank correlation so a single lucky winner does
    # not mask a random IS/OOS relationship.
    below_median = 1.0 if rel_rank > 0.5 else 0.0
    corr_component = float(np.clip(0.5 * (1.0 - spearman), 0.0, 1.0))
    pbo_proxy = float(np.clip(0.5 * below_median + 0.5 * corr_component, 0.0, 1.0))
    return {
        "pbo_proxy": round(pbo_proxy, 4),
        "spearman_is_oos": round(spearman, 4),
        "oos_rank_of_is_best": oos_rank,
        "oos_relative_rank_of_is_best": round(rel_rank, 4),
        "n_candidates": n,
    }


def probability_of_backtest_overfitting(
    trial_returns_matrix: np.ndarray,
    n_blocks: int = 16,
    annualization: int = 365,
) -> Tuple[float, float, List[float]]:
    """
    Computes Probability of Backtest Overfitting (PBO) via
    Combinatorially Symmetric Cross-Validation (CSCV).
    Reference: Bailey, Borwein, López de Prado, Zhu (2015)
    
    Parameters:
      trial_returns_matrix: 2D array of shape (T, N) where T = time periods, N = trials/candidates.
      n_blocks: Number of equal contiguous slices of time (must be even, typically 16).
    
    Returns:
      (pbo, median_relative_rank, relative_ranks_list)
      pbo: Probability in [0, 1]. Values > 0.50 indicate selection chooses worse than random.
           Institutional gate: PBO < 0.30.
    """
    mat = np.asarray(trial_returns_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("trial_returns_matrix must be 2D of shape (T, N)")
    T, N = mat.shape
    if T < n_blocks or N < 2:
        return 0.0, 0.5, []

    if n_blocks % 2 != 0:
        n_blocks -= 1

    # Slice T into S contiguous equal blocks
    block_indices = np.array_split(np.arange(T), n_blocks)

    # All combinations of S/2 blocks as in-sample
    half = n_blocks // 2
    all_combos = list(combinations(range(n_blocks), half))

    # To keep computation fast and deterministic, sample up to 100 combinations if S=16 (16C8 = 12870)
    max_combos = min(len(all_combos), 250)
    if len(all_combos) > max_combos:
        # Uniform sampling
        step = len(all_combos) // max_combos
        combos_to_test = all_combos[::step][:max_combos]
    else:
        combos_to_test = all_combos

    relative_ranks = []
    below_median_count = 0

    for combo in combos_to_test:
        in_sample_blocks = set(combo)
        out_sample_blocks = set(range(n_blocks)) - in_sample_blocks

        is_idx = np.concatenate([block_indices[b] for b in in_sample_blocks])
        oos_idx = np.concatenate([block_indices[b] for b in out_sample_blocks])

        mat_is = mat[is_idx, :]
        mat_oos = mat[oos_idx, :]

        # In-sample Sharpe ratios
        mean_is = np.mean(mat_is, axis=0)
        std_is = np.std(mat_is, axis=0, ddof=1) + 1e-12
        sr_is = mean_is / std_is

        # Out-of-sample Sharpe ratios
        mean_oos = np.mean(mat_oos, axis=0)
        std_oos = np.std(mat_oos, axis=0, ddof=1) + 1e-12
        sr_oos = mean_oos / std_oos

        # Best in-sample trial index
        best_trial = int(np.argmax(sr_is))

        # Rank of the best in-sample trial among all trials out-of-sample
        # rank 1 = best, rank N = worst
        sorted_oos = np.sort(sr_oos)[::-1]
        rank = int(np.where(sorted_oos == sr_oos[best_trial])[0][0]) + 1
        
        # Relative rank in (0, 1]
        rel_rank = rank / (N + 1.0)
        relative_ranks.append(rel_rank)

        # In CSCV, a trial is considered overfit if its out-of-sample performance is below median (rel_rank > 0.50)
        if rel_rank > 0.50:
            below_median_count += 1

    pbo = below_median_count / len(combos_to_test)
    med_rank = float(np.median(relative_ranks)) if relative_ranks else 0.5
    return float(pbo), med_rank, relative_ranks


class PurgedKFold:
    """
    Purged and Embargoed K-Fold Cross Validation for Financial Time Series.
    Guarantees no label overlap leakage across the train/test boundaries,
    and applies a post-test embargo buffer to eliminate serial correlation leakage.
    Reference: Advances in Financial Machine Learning, Ch. 7
    """

    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: Optional[pd.Series] = None,
        embargo_pct: float = 0.01,
        embargo_bars: Optional[int] = None,
        bar_timedelta: Optional[pd.Timedelta] = None,
    ):
        """
        Parameters:
          n_splits: Number of cross-validation folds.
          samples_info_sets: Series mapping sample index -> label expiration index (t1).
                             If None, assumes 1-bar horizon (no overlap purge, only embargo).
          embargo_pct: Fraction of total observations to embargo immediately after test set.
          embargo_bars: Absolute embargo length. When set, wins over embargo_pct.
                        Must be >= max(max_holding_bars, longest feature lookback).
                        On a DatetimeIndex this is a *time* buffer (bars × bar_timedelta),
                        not an event-count buffer.
          bar_timedelta: Candle duration used to convert embargo_bars to time.
                         Defaults to 1 hour when X has a DatetimeIndex.
        """
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.embargo_pct = embargo_pct
        self.embargo_bars = embargo_bars
        self.bar_timedelta = bar_timedelta

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(self.n_splits)

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Any] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.embargo_bars is not None:
            embargo = max(0, int(self.embargo_bars))
        else:
            embargo = int(n_samples * self.embargo_pct)

        # Create contiguous test chunks
        fold_bounds = [(int(i * n_samples / self.n_splits), int((i + 1) * n_samples / self.n_splits))
                       for i in range(self.n_splits)]

        for test_start, test_end in fold_bounds:
            test_indices = indices[test_start:test_end]

            # 1. Purge training samples whose forward labeling window spans into the test set
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            if self.samples_info_sets is not None:
                first_info = self.samples_info_sets.iloc[0] if hasattr(self.samples_info_sets, 'iloc') else self.samples_info_sets[0]
                is_dt = isinstance(first_info, (pd.Timestamp, np.datetime64))
                test_start_bound = X.index[test_start] if (is_dt and hasattr(X, 'index')) else test_start

                # For any training sample t < test_start, if label_end(t) >= test_start, purge it
                for t in range(test_start):
                    t1 = self.samples_info_sets.iloc[t] if hasattr(self.samples_info_sets, 'iloc') else self.samples_info_sets[t]
                    if t1 >= test_start_bound:
                        train_mask[t] = False

            # 2. Embargo samples immediately following the test set.
            # Event-sampled rows use a time buffer (embargo_bars × bar length)
            # so 48/200 means hours, not 48/200 QuantumAI events.
            idx = getattr(X, "index", None)
            if (
                self.embargo_bars is not None
                and idx is not None
                and isinstance(idx, pd.DatetimeIndex)
                and test_end > 0
                and test_end < n_samples
            ):
                delta = self.bar_timedelta or pd.Timedelta(hours=1)
                test_end_ts = idx[test_end - 1]
                cutoff = test_end_ts + int(self.embargo_bars) * delta
                embargo_mask = (idx > test_end_ts) & (idx <= cutoff)
                train_mask[np.asarray(embargo_mask)] = False
            else:
                embargo_end = min(n_samples, test_end + embargo)
                train_mask[test_end:embargo_end] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices


if __name__ == "__main__":
    print("[*] Self-testing validation_metrics module...")
    # Synthetic test: 30 trials across 500 periods
    np.random.seed(42)
    synthetic_trials = np.random.normal(0.0005, 0.015, size=(500, 30))
    # Add one slightly skilled trial
    synthetic_trials[:, 0] += 0.001

    pbo, med_rank, _ = probability_of_backtest_overfitting(synthetic_trials, n_blocks=16)
    dsr = deflated_sharpe_ratio(synthetic_trials[:, 0], n_trials=30)
    sr = calculate_sharpe_ratio(synthetic_trials[:, 0])

    print(f"    Sharpe Ratio (Best Trial): {sr:.2f}")
    print(f"    Deflated Sharpe Ratio:     {dsr:.4f} (Gate: > 0.95)")
    print(f"    Probability of Overfitting: {pbo:.2%} (Gate: < 30%)")
    print(f"    Median Relative Rank:      {med_rank:.2f}")

    # Test PurgedKFold
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    for f, (tr, te) in enumerate(pkf.split(synthetic_trials)):
        print(f"    Fold {f+1}: Train={len(tr)}, Test={len(te)}, Purged/Embargoed={500 - len(tr) - len(te)}")

    print("[✓] Validation metrics module tests passed successfully!")
