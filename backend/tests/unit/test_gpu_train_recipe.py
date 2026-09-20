"""GPU trainer recipe must match the 2026-09-20 instruction pack / live book."""

from pathlib import Path

from barrier_config import MAX_HOLDING_BARS, TRAIL_ACTIVATION_ATR, TRAIL_ATR_MULT
from promotion_gates import evaluate_promotion
from train_cli import build_parser


ROOT = Path(__file__).resolve().parents[3]


def test_gpu_cli_holding_defaults_to_live_48():
    args = build_parser().parse_args([])
    assert args.holding == 48
    assert args.holding == MAX_HOLDING_BARS
    assert args.pt_mult == 5.5
    assert args.sl_mult == 1.75
    assert args.model == "lightgbm"


def test_gpu_remote_script_pins_holding_48():
    text = (ROOT / "scripts" / "gpu_train_remote.sh").read_text(encoding="utf-8")
    assert "--holding 48" in text
    assert "JESSE_SYNC_TO_LIVE" not in text


def test_static_geometry_matches_live_trail_and_embargo():
    import json

    geo = json.loads((ROOT / "jesse_quant" / "geometry.json").read_text(encoding="utf-8"))
    tb = geo["triple_barrier"]
    assert tb["sl_atr_mult"] == 1.75
    assert tb["pt_atr_mult"] == 5.5
    assert tb["vertical_timeout_bars"] == 48
    assert tb["trail_activation_atr"] == TRAIL_ACTIVATION_ATR
    assert tb["trail_atr_mult"] == TRAIL_ATR_MULT
    assert geo["validation"]["embargo_bars"] >= max(48, 200)
    assert geo["validation"]["purge_horizon"] == 48


def test_promotion_blocks_failed_monte_carlo_even_if_dsr_pbo_pass():
    decision = evaluate_promotion(
        {
            "deflated_sharpe_ratio": 0.99,
            "prob_backtest_overfitting": 0.10,
            "monte_carlo": {"gate_ok": False, "reason": "ruin 40% / worst-5% ruin 100%"},
        },
        pt_mult=5.5,
        sl_mult=1.75,
        allow_overfit=False,
    )
    assert decision.ok is False
    assert "Monte Carlo" in decision.reason


def test_promotion_blocks_miscalibrated_probabilities():
    decision = evaluate_promotion(
        {
            "deflated_sharpe_ratio": 0.99,
            "prob_backtest_overfitting": 0.10,
            "calibration": {"calibration_ok": False, "reason": "|slope-1|=0.8"},
        },
        pt_mult=5.5,
        sl_mult=1.75,
        allow_overfit=False,
    )
    assert decision.ok is False
    assert "calibration" in decision.reason
