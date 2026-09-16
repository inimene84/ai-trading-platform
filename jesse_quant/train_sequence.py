#!/usr/bin/env python3
"""Small GPU sequence meta-labeler (LSTM) for QuantumAI events.

Trains on the same 26-feature schema and 5.5/1.75 ATR triple-barrier labels as
the LightGBM meta-labeler. Promotion still goes through DSR/PBO/geometry gates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gpu_device import TrainingDevice, detect_training_device, torch_device_string
from sklearn.metrics import classification_report

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


SEQ_LEN = 32
LSTM_GRID: List[Dict[str, Any]] = [
    {"hidden": 32, "layers": 1, "dropout": 0.1, "lr": 1e-3, "epochs": 12, "batch": 64},
    {"hidden": 64, "layers": 1, "dropout": 0.2, "lr": 8e-4, "epochs": 16, "batch": 64},
    {"hidden": 64, "layers": 2, "dropout": 0.2, "lr": 5e-4, "epochs": 20, "batch": 64},
]
B200_GRID: List[Dict[str, Any]] = [
    {"hidden": 256, "layers": 2, "dropout": 0.30, "lr": 5e-4, "epochs": 60, "batch": 1024},
    {"hidden": 512, "layers": 3, "dropout": 0.30, "lr": 3e-4, "epochs": 80, "batch": 512},
    {"hidden": 768, "layers": 3, "dropout": 0.35, "lr": 2e-4, "epochs": 100, "batch": 256},
]


def _select_grid(n_events: int, vram_mb: Optional[int]) -> List[Dict[str, Any]]:
    if vram_mb is not None and vram_mb >= 80_000 and n_events >= 4_000:
        return B200_GRID
    return LSTM_GRID


class EventLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
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


def build_event_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], pd.Index]:
    """For each labeled event, take the preceding seq_len feature rows (inclusive)."""
    full_index = X.index
    loc_map = {ts: i for i, ts in enumerate(full_index)}
    seqs: List[np.ndarray] = []
    labels: List[int] = []
    weights: List[float] = []
    kept: List[Any] = []
    values = X.to_numpy(dtype=np.float32)
    for ts, label in y.items():
        i = loc_map.get(ts)
        if i is None or i < seq_len - 1:
            continue
        window = values[i - seq_len + 1 : i + 1]
        if window.shape[0] != seq_len or np.isnan(window).any():
            continue
        seqs.append(window)
        labels.append(int(label))
        if sample_weights is not None:
            weights.append(float(sample_weights.loc[ts]))
        kept.append(ts)
    if not seqs:
        raise ValueError("No sequence windows could be built from event labels")
    w_arr = np.asarray(weights, dtype=np.float32) if sample_weights is not None else None
    return np.stack(seqs), np.asarray(labels, dtype=np.int64), w_arr, pd.Index(kept)


def _class_weights(y: np.ndarray, device: str):
    assert torch is not None
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one(
    model: EventLSTM,
    train_loader: DataLoader,
    device: str,
    lr: float,
    epochs: int,
    class_weights,
) -> None:
    assert torch is not None
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            xb, yb, wb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = (loss_fn(logits, yb) * wb).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


def train_lstm_meta(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    test_size: float = 0.20,
    device_info: Optional[TrainingDevice] = None,
) -> Tuple[Optional[Any], Dict[str, Any], List[np.ndarray]]:
    """Returns (state_dict or None, metrics, holdout return columns per trial)."""
    if torch is None:
        return None, {"error": "torch_not_installed", "promotion_ok": False}, []

    device_info = device_info or detect_training_device()
    device = torch_device_string(device_info)
    seq_x, seq_y, seq_w, kept = build_event_sequences(X, y, sample_weights)
    split_idx = int(len(seq_x) * (1.0 - test_size))
    if split_idx < 32 or (len(seq_x) - split_idx) < 16:
        return None, {"error": "too_few_sequence_events", "n_events": int(len(seq_x)), "promotion_ok": False}, []

    x_train, x_test = seq_x[:split_idx], seq_x[split_idx:]
    y_train, y_test = seq_y[:split_idx], seq_y[split_idx:]
    if seq_w is None:
        w_train = np.ones(len(x_train), dtype=np.float32)
    else:
        w_train = seq_w[:split_idx]

    grid = _select_grid(len(seq_x), device_info.vram_mb)
    class_w = _class_weights(y_train, device)
    # Inverse-frequency sampler so minority TP-hit class is seen every epoch.
    counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sample_w = (counts.sum() / (2.0 * counts[y_train])).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(y_train),
        replacement=True,
    )

    holdout_columns: List[np.ndarray] = []
    trial_sharpes: List[float] = []
    fitted: List[Tuple[Dict[str, Any], Dict[str, Any], float, np.ndarray]] = []

    n_features = int(x_train.shape[-1])
    for params in grid:
        model = EventLSTM(
            n_features=n_features,
            hidden=int(params["hidden"]),
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
        )
        ds = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        )
        loader = DataLoader(
            ds,
            batch_size=int(params.get("batch", 64)),
            sampler=sampler,
            drop_last=False,
        )
        _train_one(
            model,
            loader,
            device,
            lr=float(params["lr"]),
            epochs=int(params["epochs"]),
            class_weights=class_w,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_test).to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        pred_signal = np.where(preds == 1, 1.0, -1.0)
        actual_signal = np.where(y_test == 1, 1.0, -1.0)
        rets = pred_signal * actual_signal * 0.01
        holdout_columns.append(rets)
        sr = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-12) * np.sqrt(365.0))
        trial_sharpes.append(sr)
        fitted.append((params, {k: v.cpu() for k, v in model.state_dict().items()}, sr, preds))
        print(f"    lstm trial hidden={params['hidden']} layers={params['layers']} Sharpe={sr:.3f}")

    best_idx = int(np.argmax(trial_sharpes))
    best_params, best_state, _, best_preds = fitted[best_idx]
    report = classification_report(y_test, best_preds, output_dict=True, zero_division=0)
    metrics = {
        "model_type": "lstm",
        "device": device,
        "n_events": int(len(seq_x)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_grid": int(len(grid)),
        "best_params": best_params,
        "seq_len": SEQ_LEN,
        "test_accuracy": float(report.get("accuracy", 0.0)),
        "bullish_precision": float(report.get("1", {}).get("precision", 0.0)),
        "bullish_recall": float(report.get("1", {}).get("recall", 0.0)),
        "bearish_precision": float(report.get("0", {}).get("precision", 0.0)),
        "bearish_recall": float(report.get("0", {}).get("recall", 0.0)),
        "kept_events": int(len(kept)),
        "gpu_name": device_info.name,
    }
    return best_state, metrics, holdout_columns
