"""Hashes, costed returns, geometry lock, purgedcv wiring, SHADOW sizing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.ml.artifacts import PromotionBundle
from backend.ml.costs import apply_costs, costed_edge_bps, per_bar_roundtrip_cost
from backend.ml.geometry import (
    HOUSE_ATR_PERIOD,
    HOUSE_PT_ATR_MULT,
    HOUSE_SL_ATR_MULT,
    training_barrier_multipliers,
)
from backend.ml.gpu_job import TrainingPathViolation, assert_training_path_isolated
from backend.ml.hashes import canonical_json, geometry_hash, holdout_id_hash, sha256_hex
from backend.ml.holdout_registry import HoldoutRegistry
from backend.ml.live_signal import evaluate_live_four_numbers
from backend.ml.promotion_gates import GateResult
from backend.ml.promotion_service import (
    PromotionState,
    evaluate_bundle_for_engine,
    live_geometry_from_risk_config,
)
from backend.ml.purgedcv_metrics import (
    EffectiveTrials,
    PromotionMetricError,
    compute_dsr,
    compute_pbo,
    resolve_effective_n_trials,
)
from backend.services.decision_engine import DecisionEngine
from backend.services.risk_config import RiskConfig
from backend.strategies.base import StrategySignal


def test_canonical_json_hash_is_stable():
    payload = {"b": 2, "a": 1}
    expected = sha256_hex(canonical_json({"a": 1, "b": 2}))
    assert geometry_hash(payload) == expected
    assert geometry_hash({"hashes": {"x": 1}, "a": 1, "b": 2}) == expected


def test_holdout_id_is_tuple_hash():
    left = holdout_id_hash(start="2025-01-01", end="2025-06-01", instrument="BTC-USDT", timeframe="1h")
    right = holdout_id_hash(start="2025-01-01", end="2025-06-01", instrument="BTC-USDT", timeframe="1h")
    other = holdout_id_hash(start="2025-01-01", end="2025-06-02", instrument="BTC-USDT", timeframe="1h")
    assert left == right
    assert left != other
    assert len(left) == 64


def test_training_barriers_match_live_house_defaults():
    sl, pt = training_barrier_multipliers()
    assert sl == HOUSE_SL_ATR_MULT == 1.75
    assert pt == HOUSE_PT_ATR_MULT == 5.5
    cfg = RiskConfig.model_construct()
    assert cfg.sl_atr_mult == 1.75
    assert cfg.tp_atr_mult == 5.5
    assert cfg.atr_period == HOUSE_ATR_PERIOD
    live = live_geometry_from_risk_config(cfg)
    assert live["triple_barrier"]["sl_atr_mult"] == 1.75
    assert live["triple_barrier"]["pt_atr_mult"] == 5.5


def test_costed_returns_subtract_fees_slippage_and_funding():
    raw = np.array([0.01, -0.005, 0.002])
    costed = apply_costs(
        raw,
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
        funding_rate_per_8h=0.0008,
        timeframe="1h",
    )
    drag = per_bar_roundtrip_cost(
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
        funding_rate_per_8h=0.0008,
        timeframe="1h",
    )
    assert drag == pytest.approx(2.0 * (0.0004 + 0.0002) + 0.0008 / 8.0)
    assert np.allclose(costed, raw - drag)
    assert costed_edge_bps(costed, bars_per_year=1.0) == pytest.approx(float(np.mean(costed)) * 10_000.0)


def test_effective_n_trials_refuses_empty_and_uses_recorder():
    with pytest.raises(PromotionMetricError):
        resolve_effective_n_trials()

    class _Recorder:
        def n_effective(self):
            return 18.5

        def n_trials(self):
            return 120

        def var_sharpe(self, ddof=1):
            return 0.04

    resolved = resolve_effective_n_trials(recorder=_Recorder())
    assert resolved.source == "TrialSharpeRecorder.n_effective"
    assert resolved.n_effective == 18.5
    assert resolved.n_raw == 120
    assert resolved.var_sharpe == 0.04


def test_compute_dsr_requires_probability_and_effective_trials():
    fake = MagicMock()
    fake.deflated_sharpe_ratio_full.return_value = MagicMock(dsr=0.97)

    with patch("backend.ml.purgedcv_metrics.import_purgedcv", return_value=fake):
        out = compute_dsr(
            [0.001, -0.0005, 0.002, 0.0],
            effective=EffectiveTrials(
                n_effective=20.0, n_raw=80, source="effective_n_trials", var_sharpe=0.01
            ),
            bars_per_year=8760.0,
        )
    assert out.dsr == 0.97
    assert out.dsr_is_probability is True
    kwargs = fake.deflated_sharpe_ratio_full.call_args.kwargs
    assert kwargs["n_trials"] == 20.0

    with pytest.raises(PromotionMetricError, match="N_TRIALS_NOT_EFFECTIVE"):
        compute_dsr(
            [0.001, 0.002],
            effective=EffectiveTrials(n_effective=80, n_raw=80, source="study.n_trials"),
        )


def test_compute_pbo_infeasible_and_incomplete():
    tiny = np.ones((1, 10))
    infeas = compute_pbo(tiny, n_splits=16, orientation="configs_by_obs")
    assert infeas.reject_reason == "PBO_INFEASIBLE"
    assert infeas.feasible is False

    matrix = np.ones((3, 40))
    incomplete = compute_pbo(
        matrix, n_splits=16, completed_trial_count=10, orientation="configs_by_obs"
    )
    assert incomplete.reject_reason == "PBO_MATRIX_INCOMPLETE"
    assert incomplete.matrix_complete is False


def test_compute_pbo_calls_purgedcv_on_full_matrix():
    fake = MagicMock()
    fake.probability_of_backtest_overfitting.return_value = MagicMock(pbo=0.22, n_combos=8)
    obs_by_cfg = np.random.default_rng(0).normal(0, 0.01, size=(40, 5))
    with patch("backend.ml.purgedcv_metrics.import_purgedcv", return_value=fake):
        out = compute_pbo(obs_by_cfg, n_splits=16, completed_trial_count=5, orientation="obs_by_configs")
    assert out.pbo == 0.22
    assert out.matrix_complete is True
    called = fake.probability_of_backtest_overfitting.call_args
    assert called.args[0].shape == (5, 40)


def test_gpu_job_refuses_broker_credentials_and_trading_imports(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "dummy-not-a-real-key")
    with pytest.raises(TrainingPathViolation, match="BINANCE_API_KEY"):
        assert_training_path_isolated(modules={})
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    with pytest.raises(TrainingPathViolation, match="/trading"):
        assert_training_path_isolated(modules={"backend.routes.trading": object()})


def test_live_four_number_stub_blocks_and_skips():
    skipped = evaluate_live_four_numbers({"side": "BUY"})
    assert skipped.allowed is True
    assert skipped.applied is False

    blocked = evaluate_live_four_numbers({"side": "BUY", "p_win": 0.40, "conformal_width": 0.1, "costed_edge_bps": 2.0})
    assert blocked.allowed is False
    assert "p_win" in blocked.reason

    ok = evaluate_live_four_numbers({"side": "BUY", "p_win": 0.70, "conformal_width": 0.1, "costed_edge_bps": 2.0})
    assert ok.allowed is True
    assert ok.applied is True


def _make_bars(n: int = 50, base: float = 100.0) -> list:
    bars = []
    for i in range(n):
        c = base + i * 0.1
        bars.append({"open": c - 0.1, "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1000.0})
    return bars


@pytest.mark.asyncio
async def test_shadow_logs_prediction_and_does_not_apply_kelly(monkeypatch):
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    cfg = RiskConfig(
        min_signal_strength=0.45,
        ai_analysis_threshold=0.30,
        trade_usdt_amount=10.0,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        sl_atr_mult=1.75,
        tp_atr_mult=5.5,
        enable_jesse_ml=True,
    )
    engine = DecisionEngine(cfg)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    engine.promotion_state = PromotionState(result=GateResult(verdict="SHADOW", reason="fixture"))
    bars = _make_bars()
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    mock_ml = {
        "status": "success",
        "signal": "SELL",
        "confidence": 0.80,
        "probabilities": {"bullish": 0.10, "bearish": 0.80, "neutral": 0.10},
        "kelly": {"size_multiplier": 1.8},
    }
    with patch("backend.services.decision_engine.jesse_bridge.get_ml_prediction", AsyncMock(return_value=mock_ml)):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)

    assert decision is not None
    assert decision.action == "BUY"
    assert getattr(mock_signal, "kelly_multiplier", None) == 0.0
    assert engine.promotion_state.logged_predictions
    assert engine.last_evaluation.get("shadow_ml", {}).get("signal") == "SELL"


@pytest.mark.asyncio
async def test_reject_promotion_fail_closed_in_live(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    cfg = RiskConfig(
        min_signal_strength=0.45,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        sl_atr_mult=1.75,
        tp_atr_mult=5.5,
        enable_jesse_ml=True,
    )
    engine = DecisionEngine(cfg)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    engine.promotion_state = PromotionState(
        result=GateResult(verdict="REJECT", reason="DSR_FLOOR", failed_gate="DSR_FLOOR")
    )
    bars = _make_bars()
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))
    decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
    assert decision is None
    assert "promotion contract" in engine.last_evaluation.get("reason", "")


@pytest.mark.asyncio
async def test_promoted_live_missing_four_number_telemetry_vetoes(monkeypatch):
    """Promoted model in live without p_win/width/edge telemetry must fail closed."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    cfg = RiskConfig(
        min_signal_strength=0.45,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        sl_atr_mult=1.75,
        tp_atr_mult=5.5,
        enable_jesse_ml=True,
    )
    engine = DecisionEngine(cfg)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    engine.promotion_state = PromotionState(result=GateResult(verdict="PROMOTE", reason="fixture"))
    bars = _make_bars()
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    mock_ml = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.80,
        "probabilities": {"bullish": 0.80, "bearish": 0.05, "neutral": 0.15},
        "p_win": None,
        "conformal_width": None,
        "costed_edge_bps": None,
    }
    with patch("backend.services.decision_engine.jesse_bridge.get_ml_prediction", AsyncMock(return_value=mock_ml)):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)

    assert decision is None
    assert "telemetry missing" in engine.last_evaluation.get("reason", "")


@pytest.mark.asyncio
async def test_promoted_paper_missing_four_number_telemetry_skips(monkeypatch):
    """Paper mode keeps the fail-open skip when the four numbers are absent."""
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    cfg = RiskConfig(
        min_signal_strength=0.45,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        sl_atr_mult=1.75,
        tp_atr_mult=5.5,
        enable_jesse_ml=True,
    )
    engine = DecisionEngine(cfg)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    engine.promotion_state = PromotionState(result=GateResult(verdict="PROMOTE", reason="fixture"))
    bars = _make_bars()
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    mock_ml = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.80,
        "probabilities": {"bullish": 0.80, "bearish": 0.05, "neutral": 0.15},
        "p_win": None,
        "conformal_width": None,
        "costed_edge_bps": None,
    }
    with patch("backend.services.decision_engine.jesse_bridge.get_ml_prediction", AsyncMock(return_value=mock_ml)):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)

    assert decision is not None
    assert decision.action == "BUY"


def test_promotion_required_without_artifacts(monkeypatch):
    monkeypatch.setenv("QTP_PROMOTION_REQUIRED", "true")
    monkeypatch.delenv("QTP_PROMOTION_ARTIFACT_DIR", raising=False)
    from backend.ml.promotion_service import resolve_promotion

    state = resolve_promotion()
    assert state is not None
    assert state.verdict == "REJECT"
    assert state.result.failed_gate == "SCHEMA_HASH"


def test_holdout_registry_rejects_second_peek(tmp_path):
    registry = HoldoutRegistry(tmp_path / "holdout_registry.json")
    hid = "a" * 64
    registry.mark_spent(hid, meta={"geometry_hash": "1" * 64, "feature_schema_hash": "2" * 64})
    assert registry.is_second_peek(hid, geometry_hash="3" * 64, feature_schema_hash="2" * 64) is True
    assert registry.is_second_peek(hid, geometry_hash="1" * 64, feature_schema_hash="2" * 64) is False


def test_mark_spent_does_not_overwrite_existing_hashes(tmp_path):
    registry = HoldoutRegistry(tmp_path / "holdout_registry.json")
    hid = "a" * 64
    original = {"geometry_hash": "1" * 64, "feature_schema_hash": "2" * 64}
    registry.mark_spent(hid, meta=original)
    registry.mark_spent(
        hid,
        meta={"geometry_hash": "9" * 64, "feature_schema_hash": "8" * 64},
    )
    recorded = registry.record(hid)
    assert recorded["geometry_hash"] == original["geometry_hash"]
    assert recorded["feature_schema_hash"] == original["feature_schema_hash"]
    assert registry.is_second_peek(hid, geometry_hash="9" * 64, feature_schema_hash="8" * 64) is True


def test_second_peek_reject_does_not_replace_holdout_hashes(tmp_path):
    examples = Path(__file__).resolve().parents[3] / "docs/ml/qtp-promotion-contract/examples"
    geometry = json.loads((examples / "geometry.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((examples / "feature_schema.json").read_text(encoding="utf-8"))
    metrics = json.loads((examples / "metrics.pass.json").read_text(encoding="utf-8"))
    hid = str(metrics["hashes"]["holdout_id"])
    sealed_geo = "1" * 64
    sealed_feat = "2" * 64

    registry = HoldoutRegistry(tmp_path / "holdout_registry.json")
    registry.mark_spent(hid, meta={"geometry_hash": sealed_geo, "feature_schema_hash": sealed_feat})

    bundle = PromotionBundle(
        directory=tmp_path,
        geometry=geometry,
        metrics=metrics,
        feature_schema=feature_schema,
    )
    state = evaluate_bundle_for_engine(bundle, holdout_registry=registry)
    assert state.verdict == "REJECT"
    assert state.result.failed_gate == "HOLDOUT_SPENT"
    recorded = registry.record(hid)
    assert recorded["geometry_hash"] == sealed_geo
    assert recorded["feature_schema_hash"] == sealed_feat

    replay = evaluate_bundle_for_engine(bundle, holdout_registry=registry)
    assert replay.verdict == "REJECT"
    assert replay.result.failed_gate == "HOLDOUT_SPENT"


def test_jesse_sync_defaults_use_house_geometry():
    import inspect

    from backend.services.jesse_bridge import JesseBridgeService

    params = inspect.signature(JesseBridgeService.sync_strategy_to_risk_config).parameters
    assert params["sl_atr_mult"].default == 1.75
    assert params["tp_atr_mult"].default == 5.5


def test_example_metrics_are_valid_json():
    examples = Path(__file__).resolve().parents[3] / "docs/ml/qtp-promotion-contract/examples"
    for path in examples.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
