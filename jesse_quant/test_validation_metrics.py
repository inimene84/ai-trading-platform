#!/usr/bin/env python3
"""
Unit Tests for Validation Metrics (DSR, PBO, PurgedKFold)
"""

import unittest
import numpy as np
import pandas as pd

from validation_metrics import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_stats,
    probability_of_backtest_overfitting,
    calculate_sharpe_ratio,
    expected_max_sharpe,
    evaluate_gate,
    pbo_proxy_from_is_oos,
    PurgedKFold,
)


class TestValidationMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # 500 periods, 20 trials
        self.T = 500
        self.N = 20
        self.trials = np.random.normal(0.0002, 0.01, size=(self.T, self.N))
        # Add strong skill to trial 0
        self.trials[:, 0] += 0.002

    def test_sharpe_ratio_calculation(self):
        returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.025])
        sr = calculate_sharpe_ratio(returns, annualization=365)
        self.assertIsInstance(sr, float)
        self.assertGreater(sr, 0)

    def test_expected_max_sharpe_increases_with_trials(self):
        e10 = expected_max_sharpe(10, variance_sharpe=1.0)
        e100 = expected_max_sharpe(100, variance_sharpe=1.0)
        e1000 = expected_max_sharpe(1000, variance_sharpe=1.0)
        self.assertGreater(e100, e10)
        self.assertGreater(e1000, e100)

    def test_deflated_sharpe_ratio_behavior(self):
        # High skill trial with large sample should have high DSR
        high_skill_returns = np.random.normal(0.003, 0.01, size=1000)
        dsr_high = deflated_sharpe_ratio(high_skill_returns, n_trials=5)
        self.assertGreaterEqual(dsr_high, 0.90)

        # Pure noise across 100 trials should have low DSR
        noise_returns = np.random.normal(0.0001, 0.01, size=200)
        dsr_low = deflated_sharpe_ratio(noise_returns, n_trials=100)
        self.assertLess(dsr_low, 0.95)

    def test_dsr_decreases_with_number_of_trials(self):
        """The whole point of the DSR: the same track record is worth less after more trials."""
        returns = np.random.normal(0.0008, 0.01, size=600)
        dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
        dsr_10 = deflated_sharpe_ratio(returns, n_trials=10)
        dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
        dsr_1000 = deflated_sharpe_ratio(returns, n_trials=1000)
        self.assertGreater(dsr_1, dsr_10)
        self.assertGreater(dsr_10, dsr_100)
        self.assertGreater(dsr_100, dsr_1000)
        # A hardcoded n_trials must never produce a saturated 1.0000 for a modest edge.
        self.assertLess(dsr_10, 1.0)

    def test_dsr_from_stats_matches_series_version(self):
        returns = np.random.normal(0.0008, 0.01, size=600)
        sr = float(np.mean(returns) / np.std(returns, ddof=1))
        dsr_series = deflated_sharpe_ratio(returns, n_trials=25, variance_of_trials=0.001)
        dsr_stats = deflated_sharpe_ratio_from_stats(sr, n_obs=len(returns), n_trials=25, variance_of_trials=0.001, skewness=0.0, kurtosis_pearson=3.0)
        # Same inputs up to the sample skew/kurtosis of the simulated series.
        self.assertAlmostEqual(dsr_series, dsr_stats, delta=0.05)

    def test_dsr_is_not_annualization_dependent(self):
        """Regression: the old implementation divided by sqrt(annualization) and saturated at 1.0."""
        returns = np.random.normal(0.0008, 0.01, size=600)
        d365 = deflated_sharpe_ratio(returns, n_trials=20, annualization=365)
        d8760 = deflated_sharpe_ratio(returns, n_trials=20, annualization=8760)
        self.assertAlmostEqual(d365, d8760, places=6)

    def test_gate_rejects_failing_metrics(self):
        self.assertTrue(evaluate_gate(0.97, 0.10, dsr_min=0.95, pbo_max=0.30)["passed"])
        self.assertFalse(evaluate_gate(0.95, 0.10, dsr_min=0.95, pbo_max=0.30)["passed"])   # DSR must exceed
        self.assertFalse(evaluate_gate(0.99, 0.30, dsr_min=0.95, pbo_max=0.30)["passed"])   # PBO must be below
        self.assertFalse(evaluate_gate(0.90, 0.45, dsr_min=0.95, pbo_max=0.30)["passed"])
        self.assertFalse(evaluate_gate(None, 0.10)["passed"])                               # missing -> fail closed
        self.assertFalse(evaluate_gate(0.99, float("nan"))["passed"])
        reasons = evaluate_gate(0.90, 0.45)["reasons"]
        self.assertEqual(len(reasons), 2)

    def test_pbo_proxy_detects_is_oos_reversal(self):
        is_scores = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        good = pbo_proxy_from_is_oos(is_scores, [2.8, 2.4, 1.9, 1.4, 0.9, 0.4])
        bad = pbo_proxy_from_is_oos(is_scores, [0.4, 0.9, 1.4, 1.9, 2.4, 2.8])
        self.assertLess(good["pbo_proxy"], bad["pbo_proxy"])
        self.assertGreater(good["spearman_is_oos"], 0.9)
        self.assertLess(bad["spearman_is_oos"], -0.9)

    def test_pbo_calculation(self):
        pbo, med_rank, ranks = probability_of_backtest_overfitting(self.trials, n_blocks=16)
        self.assertGreaterEqual(pbo, 0.0)
        self.assertLessEqual(pbo, 1.0)
        self.assertGreaterEqual(med_rank, 0.0)
        self.assertLessEqual(med_rank, 1.0)
        self.assertEqual(len(ranks), min(12870, 250))

    def test_purged_kfold_no_leakage(self):
        n_samples = 200
        data = np.arange(n_samples)
        pkf = PurgedKFold(n_splits=4, embargo_pct=0.02)
        
        splits = list(pkf.split(data))
        self.assertEqual(len(splits), 4)

        for train_idx, test_idx in splits:
            # 1. No overlap between train and test
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0)

            # 2. Embargo buffer immediately after test set is excluded
            embargo_samples = int(n_samples * 0.02) # 4 samples
            test_max = max(test_idx)
            embargo_range = set(range(test_max + 1, min(n_samples, test_max + 1 + embargo_samples)))
            embargo_leak = set(train_idx).intersection(embargo_range)
            self.assertEqual(len(embargo_leak), 0)

    def test_datetime_embargo_uses_bars_not_event_count(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="6h")
        frame = pd.DataFrame({"x": np.arange(20)}, index=idx)
        pkf = PurgedKFold(n_splits=2, embargo_bars=48, bar_timedelta=pd.Timedelta(hours=1))
        train_idx, test_idx = next(pkf.split(frame))
        test_end_ts = idx[int(max(test_idx))]
        cutoff = test_end_ts + pd.Timedelta(hours=48)
        embargoed = {i for i, ts in enumerate(idx) if test_end_ts < ts <= cutoff}
        self.assertLess(len(embargoed), 48)
        self.assertEqual(len(set(train_idx).intersection(embargoed)), 0)
        later = {i for i, ts in enumerate(idx) if ts > cutoff}
        self.assertTrue(later)
        self.assertTrue(later.issubset(set(train_idx)))


if __name__ == "__main__":
    unittest.main()
