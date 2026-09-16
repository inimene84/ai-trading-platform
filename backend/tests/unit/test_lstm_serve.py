"""Unit tests for LSTM artifact serving (no GPU, torch optional)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
JESSE = ROOT / "jesse_quant"
if str(JESSE) not in sys.path:
    sys.path.insert(0, str(JESSE))

from lstm_infer import (  # noqa: E402
    infer_lstm_arch,
    is_rejected_artifact,
    parse_artifact_stem,
    sidecar_meta_from_payload,
)

try:
    from ml_features import FEATURE_NAMES, compute_latest_sequence
except ImportError:  # pandas/numpy missing in slim unit-test env
    FEATURE_NAMES = []
    compute_latest_sequence = None  # type: ignore[assignment]


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def test_infer_lstm_arch_from_b200_shapes():
    hidden, n_features, layers = 768, 26, 3
    state = {"lstm.weight_ih_l0": _FakeTensor((4 * hidden, n_features))}
    for layer in range(1, layers):
        state[f"lstm.weight_ih_l{layer}"] = _FakeTensor((4 * hidden, hidden))
    arch = infer_lstm_arch(state)
    assert arch == {"hidden": hidden, "n_features": n_features, "layers": layers}


def test_rejected_artifacts_never_parse_as_production():
    assert is_rejected_artifact("ETH-USDT_1h_lstm.rejected.pt")
    assert is_rejected_artifact("/tmp/BTC-USDT_1h_lightgbm.rejected.joblib")
    assert parse_artifact_stem("ETH-USDT_1h_lstm.rejected.pt") is None
    assert parse_artifact_stem("ETH-USDT_1h_lstm.pt") == {
        "symbol": "ETH-USDT",
        "timeframe": "1h",
        "model_type": "lstm",
    }
    assert parse_artifact_stem("BTC-USDT_15m_lstm.pt")["timeframe"] == "15m"


def test_sidecar_meta_copies_gate_fields():
    payload = {
        "trained_at": "2026-09-16T03:41:34+00:00",
        "feature_hash": "cd15d2380809b247",
        "pt_mult": 5.5,
        "sl_mult": 1.75,
        "promotion_ok": True,
        "metrics": {
            "deflated_sharpe_ratio": 1.0,
            "prob_backtest_overfitting": 0.028,
            "bullish_recall": 0.30,
            "bearish_recall": 0.75,
            "seq_len": 32,
        },
    }
    meta = sidecar_meta_from_payload(payload, symbol="ETH-USDT", timeframe="1h")
    assert meta["promotion_ok"] is True
    assert meta["metrics"]["deflated_sharpe_ratio"] == 1.0
    assert meta["pt_mult"] == 5.5
    assert meta["model_type"] == "lstm"


def test_latest_sequence_shape_and_short_history():
    if compute_latest_sequence is None:
        return
    short = np.ones((10, 6), dtype=float)
    short[:, 0] = np.arange(10)
    nan_seq = compute_latest_sequence(short, seq_len=32)
    assert nan_seq.shape == (32, len(FEATURE_NAMES))
    assert np.isnan(nan_seq).all()

    n = 300
    rng = np.random.default_rng(0)
    candles = np.column_stack(
        [
            np.arange(n) * 60_000,
            100 + rng.normal(0, 0.2, n).cumsum(),
            100 + rng.normal(0, 0.2, n).cumsum(),
            101 + rng.normal(0, 0.2, n).cumsum(),
            99 + rng.normal(0, 0.2, n).cumsum(),
            rng.uniform(1, 10, n),
        ]
    )
    # enforce OHLC sanity
    candles[:, 3] = np.maximum(candles[:, 1], np.maximum(candles[:, 2], candles[:, 3]))
    candles[:, 4] = np.minimum(candles[:, 1], np.minimum(candles[:, 2], candles[:, 4]))
    seq = compute_latest_sequence(candles, seq_len=32)
    assert seq.shape == (32, len(FEATURE_NAMES))
    assert seq.dtype == np.float32
    assert not np.isnan(seq).any()
