"""Research trust scores for logged Jev decisions.

Accuracy, multiclass Brier, Wilson intervals, and ECE describe the log.
They do not authorize size or a live order. Only a FACE_VALUE verdict may
pass the existing influence gate, and even then sizing stays at zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.serialization import load_pem_private_key

from backend.jev.config import calibration_min_labels

VERDICTS = ("UNVERIFIED", "FACE_VALUE", "DISCOUNT", "DOWNGRADE")
_MIN_VERDICT_LABELS = 30


def wilson_interval(hits: int, samples: int, z: float = 1.96) -> tuple[float, float] | None:
    if samples <= 0:
        return None
    proportion = hits / samples
    z2 = z * z
    denominator = 1.0 + z2 / samples
    centre = proportion + z2 / (2.0 * samples)
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * samples)) / samples)
    low = max(0.0, (centre - margin) / denominator)
    high = min(1.0, (centre + margin) / denominator)
    return (low, high)


def multiclass_brier(rows: list[tuple[dict[str, float], str]]) -> float | None:
    """Mean squared error of a probability vector against a one-hot outcome."""
    if not rows:
        return None
    total = 0.0
    counted = 0
    for probabilities, outcome in rows:
        if outcome not in probabilities:
            continue
        weight = sum(float(value) for value in probabilities.values()) or 1.0
        for name, value in probabilities.items():
            target = 1.0 if name == outcome else 0.0
            predicted = float(value) / weight
            total += (predicted - target) ** 2
        counted += 1
    if counted == 0:
        return None
    return total / counted


def expected_calibration_error(pairs: list[tuple[float, int]], bins: int = 5) -> float | None:
    """ECE of P(event) against a 0/1 outcome. None when there is nothing to bin."""
    usable = [(min(1.0, max(0.0, float(prob))), 1 if int(label) else 0) for prob, label in pairs]
    if len(usable) < _MIN_VERDICT_LABELS:
        return None
    width = 1.0 / bins
    error = 0.0
    for index in range(bins):
        lo = index * width
        hi = 1.0 if index == bins - 1 else (index + 1) * width
        bucket = [item for item in usable if (lo <= item[0] < hi) or (index == bins - 1 and item[0] == 1.0)]
        if not bucket:
            continue
        mean_p = sum(item[0] for item in bucket) / len(bucket)
        mean_y = sum(item[1] for item in bucket) / len(bucket)
        error += (len(bucket) / len(usable)) * abs(mean_p - mean_y)
    return error


def trust_verdict(labels: int, ece: float | None, min_labels: int | None = None) -> str:
    required = calibration_min_labels() if min_labels is None else min_labels
    if labels < _MIN_VERDICT_LABELS or ece is None:
        return "UNVERIFIED"
    if ece > 0.15:
        return "DOWNGRADE"
    if ece > 0.05 or labels < required:
        return "DISCOUNT"
    return "FACE_VALUE"


def allows_book_influence(verdict: str) -> bool:
    """FACE_VALUE is necessary and still not sufficient for a book vote."""
    return verdict == "FACE_VALUE"


def calibration_currency(ece: float | None) -> float | None:
    if ece is None:
        return None
    return round(1.0 - ece, 4)


def replay_hysteresis(scores: list[float], enter: float, exit_below: float) -> list[str]:
    """Replay an enter/exit band over recorded scores. Does not call Jev."""
    if exit_below >= enter:
        raise ValueError("exit threshold must sit strictly below the enter threshold")
    state = "out"
    path: list[str] = []
    for score in scores:
        value = float(score)
        if state == "out" and value >= enter:
            state = "in"
        elif state == "in" and value <= exit_below:
            state = "out"
        path.append(state)
    return path


def trust_report(labeled: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the research verdict from journal rows. Sizing stays disabled."""
    probability_rows: list[tuple[dict[str, float], str]] = []
    event_pairs: list[tuple[float, int]] = []
    hits = 0
    for row in labeled:
        answers = row.get("answers") or {}
        probabilities = answers.get("direction_probabilities") or answers.get("trade_probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            continue
        label = int(row.get("label") or 0)
        if "direction_probabilities" in answers:
            outcome = {1: "UP", 0: "FLAT", -1: "DOWN"}.get(label)
        else:
            outcome = answers.get("trade_action")
        if outcome in probabilities:
            probability_rows.append((probabilities, str(outcome)))
            if max(probabilities, key=probabilities.get) == outcome:
                hits += 1
            event_pairs.append((float(probabilities[str(outcome)]), 1))
        else:
            event_pairs.append((float(max(probabilities.values())), 0))
    samples = len(probability_rows)
    ece = expected_calibration_error(event_pairs)
    brier = multiclass_brier(probability_rows)
    interval = wilson_interval(hits, samples) if samples else None
    verdict = trust_verdict(samples, ece)
    return {
        "labels": samples,
        "hits": hits,
        "accuracy": None if samples == 0 else round(hits / samples, 4),
        "wilson95": None if interval is None else [round(interval[0], 4), round(interval[1], 4)],
        "brier": None if brier is None else round(brier, 4),
        "ece": None if ece is None else round(ece, 4),
        "calibration_currency": calibration_currency(ece),
        "verdict": verdict,
        "allows_book_influence": allows_book_influence(verdict),
        "sizing_allowed": False,
        "note": "A trust verdict never authorizes order size. FACE_VALUE only unlocks the existing influence flag.",
    }


def export_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)


def sign_jsonl(payload: str, private_key_path: str) -> dict[str, Any]:
    """Sign an export when an operator supplies an ed25519 PEM path.

    No key is generated or stored by this function. A missing path leaves the
    export unsigned.
    """
    if not private_key_path:
        return {"signed": False, "reason": "JEV_TRUST_PRIVATE_KEY_PATH is unset"}
    path = Path(private_key_path)
    if not path.is_file():
        return {"signed": False, "reason": "signing key file is missing"}
    key = load_pem_private_key(path.read_bytes(), password=None)
    signature = key.sign(payload.encode("utf-8"))
    return {"signed": True, "algorithm": "ed25519", "signature_hex": signature.hex()}
