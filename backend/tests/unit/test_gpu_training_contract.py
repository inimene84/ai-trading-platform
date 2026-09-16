"""Unit tests for GPU device detection and the QTP promotion contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JESSE = ROOT / "jesse_quant"
if str(JESSE) not in sys.path:
    sys.path.insert(0, str(JESSE))

from gpu_device import TrainingDevice, lightgbm_device_kwargs
from promotion_contract import (
    SPEC_VERSION,
    evaluate_contract,
    geometry_hash,
)
from promotion_gates import MIN_CLASS_RECALL, evaluate_promotion


def _device(**overrides):
    base = dict(
        kind="cpu",
        name="cpu",
        cuda_available=False,
        vram_mb=None,
        driver=None,
        cuda_version=None,
        torch_version=None,
        torch_cuda=None,
        lightgbm_device="cpu",
        source="cpu",
    )
    base.update(overrides)
    return TrainingDevice(**base)


def test_lightgbm_kwargs_empty_on_cpu():
    assert lightgbm_device_kwargs(_device()) == {}


def test_lightgbm_kwargs_cuda_device():
    d = _device(kind="cuda", name="NVIDIA B200", cuda_available=True, lightgbm_device="cuda", vram_mb=182631)
    kw = lightgbm_device_kwargs(d)
    assert kw["device"] == "cuda"
    assert kw["gpu_device_id"] == 0


def _geo():
    return {
        "spec_version": SPEC_VERSION,
        "triple_barrier": {"sl_atr_mult": 1.75, "pt_atr_mult": 5.5, "atr_period": 14, "vertical_timeout_bars": 48},
        "bar": {"timeframe": "1h", "primary_type": "time", "bars_per_year": 8760},
        "gates": {"dsr_min": 0.95, "pbo_max": 0.30},
    }


def _metrics(**overrides):
    data = {
        "spec_version": SPEC_VERSION,
        "deflated_sharpe_ratio": 0.99,
        "prob_backtest_overfitting": 0.12,
        "pt_mult": 5.5,
        "sl_mult": 1.75,
        "n_grid": 6,
        "n_trials": 40,
        "used_raw_n_as_effective": False,
        "bullish_recall": 0.22,
        "bearish_recall": 0.71,
        "n_events": 400,
        "holdout_sharpe": 1.2,
        "feature_schema_hash": "cd15d2380809b247",
    }
    data.update(overrides)
    return data


def test_contract_shadow_when_gates_pass_holdout_unspent():
    geo = _geo()
    decision = evaluate_contract(geo, _metrics(), live_geometry=geo, promote_requested=False)
    assert decision.verdict == "SHADOW"
    assert decision.code == "SHADOW"


def test_contract_rejects_wrong_geometry():
    geo = _geo()
    metrics = _metrics(pt_mult=4.0, sl_mult=2.0)
    # live lock reads geometry first; mutate geometry too
    geo["triple_barrier"]["pt_atr_mult"] = 4.0
    geo["triple_barrier"]["sl_atr_mult"] = 2.0
    decision = evaluate_contract(geo, metrics, live_geometry=_geo())
    assert decision.verdict == "REJECT"
    assert decision.code in ("GEOMETRY_HASH", "GEOMETRY_LIVE_LOCK")


def test_contract_rejects_high_pbo():
    geo = _geo()
    decision = evaluate_contract(geo, _metrics(prob_backtest_overfitting=0.46), live_geometry=geo)
    assert decision.verdict == "REJECT"
    assert decision.code == "PBO_CEILING"


def test_contract_rejects_collapsed_recall():
    geo = _geo()
    decision = evaluate_contract(
        geo,
        _metrics(bullish_recall=0.033, bearish_recall=0.97),
        live_geometry=geo,
    )
    assert decision.verdict == "REJECT"
    assert decision.code == "COLLAPSED_CLASSIFIER"


def test_contract_rejects_raw_n_trials_flag():
    geo = _geo()
    decision = evaluate_contract(geo, _metrics(used_raw_n_as_effective=True), live_geometry=geo)
    assert decision.verdict == "REJECT"
    assert decision.code == "N_TRIALS_NOT_EFFECTIVE"


def test_geometry_hash_stable():
    a = geometry_hash(_geo())
    b = geometry_hash(_geo())
    assert a == b
    assert len(a) == 64


def test_promotion_gate_min_class_recall_floor():
    assert MIN_CLASS_RECALL == 0.10
    d = evaluate_promotion(
        {
            "deflated_sharpe_ratio": 0.99,
            "prob_backtest_overfitting": 0.10,
            "bullish_recall": 0.08,
            "bearish_recall": 0.70,
        },
        pt_mult=5.5,
        sl_mult=1.75,
    )
    assert d.ok is False
    assert "bullish recall" in d.reason


def test_artifact_gate_refuses_collapsed_btc_even_if_old_pass_stamp():
    from promotion_gates import artifact_gate

    payload = {
        "gate": {"passed": True, "reasons": []},
        "dsr": 1.0,
        "pbo": 0.02,
        "pt_mult": 5.5,
        "sl_mult": 1.75,
        "metrics": {
            "deflated_sharpe_ratio": 1.0,
            "prob_backtest_overfitting": 0.02,
            "bullish_recall": 0.0,
            "bearish_recall": 0.999,
            "pt_mult": 5.5,
            "sl_mult": 1.75,
            "n_trials": 6,
        },
    }
    gate = artifact_gate(payload)
    assert gate["passed"] is False
    assert gate["status"] == "FAIL"
    assert any("collapsed" in r.lower() or "bullish recall" in r.lower() for r in gate["reasons"])


def test_deploy_ref_wired_in_ssh_wrapper():
    text = (ROOT / "scripts" / "ssh_vps_remote.sh").read_text(encoding="utf-8")
    assert "DEPLOY_REF" in text
    # Exec via bash so a missing +x bit cannot yield "Is a directory"/126.
    assert "bash scripts/hostinger_vps_apply.sh" in text
    apply = (ROOT / "scripts" / "hostinger_vps_apply.sh").read_text(encoding="utf-8")
    assert 'DEPLOY_REF="${DEPLOY_REF:-main}"' in apply
