#!/usr/bin/env python3
"""CPU inference helpers for joblib-wrapped EventLSTM artifacts (`*.pt`).

GPU trainers dump `{state_dict, metrics, ...}` via joblib. Serving reconstructs
`EventLSTM` from tensor shapes so jesse-app does not need the training grid.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Mapping, Optional

import numpy as np

_VENDOR = os.environ.get("JESSE_TORCH_VENDOR", "/home/vendor/cpu-torch")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


SEQ_LEN = 32


def is_rejected_artifact(path: str) -> bool:
    name = os.path.basename(path)
    return ".rejected." in name or name.endswith(".rejected.pt") or name.endswith(".rejected.joblib")


def infer_lstm_arch(state_dict: Mapping[str, Any]) -> Dict[str, int]:
    """Recover input size, hidden size, and layer count from an LSTM state_dict."""
    if "lstm.weight_ih_l0" not in state_dict:
        raise ValueError("state_dict missing lstm.weight_ih_l0")
    weight = state_dict["lstm.weight_ih_l0"]
    shape = tuple(weight.shape)
    if len(shape) != 2 or shape[0] % 4 != 0:
        raise ValueError(f"unexpected lstm.weight_ih_l0 shape {shape}")
    hidden = int(shape[0]) // 4
    n_features = int(shape[1])
    layers = 1
    while f"lstm.weight_ih_l{layers}" in state_dict:
        layers += 1
    return {"hidden": hidden, "n_features": n_features, "layers": layers}


class EventLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Must stay aligned with `train_sequence.EventLSTM`."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        if nn is None:
            raise RuntimeError("PyTorch is required for EventLSTM")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class LSTMClassifier:
    """sklearn-like wrapper so predict_server can call `predict_proba`."""

    def __init__(self, model: Any, seq_len: int = SEQ_LEN):
        self.model = model
        self.seq_len = int(seq_len)

    def predict_proba(self, X: Any) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required to serve LSTM artifacts")
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"LSTM expects (batch, seq, features); got shape {arr.shape}")
        tensor = torch.from_numpy(arr)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()


def wrap_lstm_payload(payload: Dict[str, Any]) -> LSTMClassifier:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to serve LSTM artifacts. "
            "Install CPU torch under /home/vendor/cpu-torch (bind-mounted)."
        )
    state = payload.get("state_dict")
    if not state:
        raise ValueError("LSTM payload missing state_dict")
    arch = infer_lstm_arch(state)
    metrics = payload.get("metrics") or {}
    best = metrics.get("best_params") or {}
    dropout = float(best.get("dropout") or 0.0)
    model = EventLSTM(
        n_features=arch["n_features"],
        hidden=arch["hidden"],
        layers=arch["layers"],
        dropout=dropout,
    )
    model.load_state_dict(state)
    model.eval()
    seq_len = int(metrics.get("seq_len") or SEQ_LEN)
    return LSTMClassifier(model, seq_len=seq_len)


def sidecar_meta_from_payload(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    timeframe: str,
    model_type: str = "lstm",
) -> Dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "model_type": model_type,
        "trained_at": payload.get("trained_at"),
        "feature_hash": payload.get("feature_hash")
        or payload.get("feature_schema_hash")
        or metrics.get("feature_schema_hash"),
        "pt_mult": payload.get("pt_mult", metrics.get("pt_mult")),
        "sl_mult": payload.get("sl_mult", metrics.get("sl_mult")),
        "promotion_ok": bool(payload.get("promotion_ok", metrics.get("promotion_ok"))),
        "metrics": metrics,
    }


def parse_artifact_stem(filename: str) -> Optional[Dict[str, str]]:
    name = os.path.basename(filename)
    if is_rejected_artifact(name):
        return None
    for ext in (".pt", ".joblib"):
        if name.endswith(ext):
            stem = name[: -len(ext)]
            parts = stem.split("_")
            if len(parts) == 3 and "." not in parts[2]:
                return {"symbol": parts[0], "timeframe": parts[1], "model_type": parts[2]}
            return None
    return None
