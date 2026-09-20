#!/usr/bin/env python3
"""
Unit tests for the machine-enforced promotion gate, the trial registry and the geometry /
payoff plumbing shared by train_ml.py, predict_server.py and the strategies.
"""

import json
import os
import tempfile
import unittest

import numpy as np

import barrier_config
import trial_registry
from promotion_gates import evaluate_promotion
from signal_policy import decide_signal, expected_value_r
from triple_barrier import realized_payoff_stats
from validation_metrics import evaluate_gate


class TestGeometrySingleSourceOfTruth(unittest.TestCase):
    def test_defaults_mirror_deployed_geometry(self):
        self.assertAlmostEqual(barrier_config.SL_ATR_MULT, 1.75)
        self.assertAlmostEqual(barrier_config.TP_ATR_MULT, 5.5)
        self.assertAlmostEqual(barrier_config.theoretical_payoff_ratio(), 5.5 / 1.75)
        g = barrier_config.geometry_dict()
        self.assertEqual(g["sl_atr_mult"], 1.75)
        self.assertEqual(g["tp_atr_mult"], 5.5)
        self.assertEqual(g["max_holding_bars"], 48)
        self.assertAlmostEqual(g["trail_activation_atr"], 2.2)
        self.assertAlmostEqual(g["trail_atr_mult"], 1.6)
        self.assertAlmostEqual(g["breakeven_win_probability"], 1.0 / (1.0 + 5.5 / 1.75), places=4)

    def test_promotion_gates_use_barrier_config(self):
        from promotion_gates import STRATEGY_PT_ATR, STRATEGY_SL_ATR, DSR_GATE, PBO_GATE

        self.assertEqual(STRATEGY_PT_ATR, barrier_config.TP_ATR_MULT)
        self.assertEqual(STRATEGY_SL_ATR, barrier_config.SL_ATR_MULT)
        self.assertEqual(DSR_GATE, barrier_config.DSR_MIN)
        self.assertEqual(PBO_GATE, barrier_config.PBO_MAX)

    def test_round_trip_cost_zero_toggle(self):
        self.assertGreater(barrier_config.round_trip_cost_pct(48, 1.0, zero_cost=False), 0.0)
        self.assertEqual(barrier_config.round_trip_cost_pct(48, 1.0, zero_cost=True), 0.0)
        # 48h holding -> 6 funding intervals at 0.01% plus 2 fees and 2 slippage legs
        expected = (2 * 0.0006 + 2 * 0.0003 + 6 * 0.0001) * 100
        self.assertAlmostEqual(barrier_config.round_trip_cost_pct(48, 1.0, zero_cost=False), expected, places=6)


class TestPromotionGate(unittest.TestCase):
    def test_gate_rejects_wrong_geometry(self):
        d = evaluate_promotion({"deflated_sharpe_ratio": 0.99, "prob_backtest_overfitting": 0.05}, pt_mult=4.0, sl_mult=2.0)
        self.assertFalse(d.ok)
        self.assertIn("geometry", d.reason)

    def test_gate_rejects_bad_dsr_or_pbo(self):
        self.assertFalse(evaluate_promotion({"deflated_sharpe_ratio": 0.90, "prob_backtest_overfitting": 0.05}, pt_mult=5.5, sl_mult=1.75, allow_overfit=False).ok)
        self.assertFalse(evaluate_promotion({"deflated_sharpe_ratio": 0.99, "prob_backtest_overfitting": 0.35}, pt_mult=5.5, sl_mult=1.75, allow_overfit=False).ok)
        self.assertFalse(evaluate_promotion({"deflated_sharpe_ratio": 0.949, "prob_backtest_overfitting": 0.05}, pt_mult=5.5, sl_mult=1.75, allow_overfit=False).ok)
        self.assertTrue(evaluate_promotion({"deflated_sharpe_ratio": 0.99, "prob_backtest_overfitting": 0.05}, pt_mult=5.5, sl_mult=1.75, allow_overfit=False).ok)

    def test_gate_fails_closed_without_evidence(self):
        self.assertFalse(evaluate_promotion({}, pt_mult=5.5, sl_mult=1.75, allow_overfit=False).ok)
        self.assertFalse(evaluate_gate(None, None)["passed"])

    def test_evaluate_gate_boundary(self):
        self.assertFalse(evaluate_gate(0.95, 0.10)["passed"])
        self.assertFalse(evaluate_gate(0.96, 0.30)["passed"])
        self.assertTrue(evaluate_gate(0.9501, 0.2999)["passed"])


class TestTrialRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "trial_registry.json")
        self._orig = trial_registry.REGISTRY_PATH
        trial_registry.REGISTRY_PATH = self.path

    def tearDown(self):
        trial_registry.REGISTRY_PATH = self._orig
        self.tmp.cleanup()

    def test_registry_accumulates_and_persists(self):
        scope = "ml:TEST-USDT:1h"
        self.assertEqual(trial_registry.total_trials(scope), 0)
        trial_registry.record_trials(scope, 8, note="run 1")
        trial_registry.record_trials(scope, 8, note="run 2")
        self.assertEqual(trial_registry.total_trials(scope), 16)
        self.assertEqual(trial_registry.effective_trials(scope, 8), 24)
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["scopes"][scope]["total_trials"], 16)

    def test_historical_seed_present(self):
        self.assertGreater(trial_registry.total_trials("ml:BTC-USDT:1h"), 1)


class TestPayoffAndDecision(unittest.TestCase):
    def test_realized_payoff_stats(self):
        rets = np.array([3.0, 3.0, -1.0, -1.0, -1.0, 0.0])
        s = realized_payoff_stats(rets)
        self.assertEqual(s["n_wins"], 2)
        self.assertEqual(s["n_losses"], 3)
        self.assertAlmostEqual(s["payoff_ratio"], 3.0)
        self.assertAlmostEqual(s["win_rate"], 2 / 6)  # win rate over all trades incl. flat

    def test_decision_rule_uses_breakeven_probability(self):
        b = barrier_config.theoretical_payoff_ratio()  # 3.14 -> break-even p = 0.2414
        self.assertAlmostEqual(expected_value_r(1.0 / (1.0 + b), b), 0.0, places=9)
        # p slightly above break-even but below the min EV -> NEUTRAL
        self.assertEqual(decide_signal(0.5, 0.26, 0.24, b, min_ev_r=0.10)["signal"], "NEUTRAL")
        # p comfortably above the minimum expectancy -> BUY
        self.assertEqual(decide_signal(0.4, 0.40, 0.20, b, min_ev_r=0.10)["signal"], "BUY")
        # symmetric for the short side
        self.assertEqual(decide_signal(0.4, 0.20, 0.40, b, min_ev_r=0.10)["signal"], "SELL")


if __name__ == "__main__":
    unittest.main()
