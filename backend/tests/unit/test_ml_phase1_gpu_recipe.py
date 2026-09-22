"""GPU campaign CLI must default to live Triple-Barrier geometry."""

from train_cli import build_parser
from barrier_config import MAX_HOLDING_BARS, SL_ATR_MULT, TP_ATR_MULT


def test_gpu_cli_defaults_match_live_geometry():
    args = build_parser().parse_args([])
    assert MAX_HOLDING_BARS == 48
    assert args.holding == 48
    assert args.pt_mult == TP_ATR_MULT == 5.5
    assert args.sl_mult == SL_ATR_MULT == 1.75
    assert args.model == "lightgbm"
    assert args.events == "quantum_ai"
    assert args.fracdiff is False
    flagged = build_parser().parse_args(["--fracdiff"])
    assert flagged.fracdiff is True
