#!/usr/bin/env python3
"""Detect CUDA / LightGBM GPU availability for Jesse training jobs.

The GPU Hostinger node is an ephemeral training accelerator. Inference on the
trading VPS stays CPU-only. This module never talks to brokers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TrainingDevice:
    kind: str
    name: str
    cuda_available: bool
    vram_mb: Optional[int]
    driver: Optional[str]
    cuda_version: Optional[str]
    torch_version: Optional[str]
    torch_cuda: Optional[str]
    lightgbm_device: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _nvidia_smi_query() -> Optional[Dict[str, str]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3:
        return None
    return {"name": parts[0], "vram_mb": parts[1], "driver": parts[2]}


def _parse_vram_mb(raw: str) -> Optional[int]:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _lightgbm_cuda_supported() -> bool:
    if lgb is None:
        return False
    # LightGBM 4.x exposes CUDA via device='cuda'. Probe without allocating a booster.
    try:
        params = getattr(lgb, "BasicParameters", None)
        if params is not None and hasattr(params, "device"):
            return True
    except Exception:
        pass
    version = getattr(lgb, "__version__", "0")
    try:
        major = int(str(version).split(".", 1)[0])
    except ValueError:
        major = 0
    return major >= 4


def detect_training_device(prefer: str = "auto") -> TrainingDevice:
    """
    prefer: auto | cuda | cpu

    auto uses CUDA when torch.cuda.is_available() or nvidia-smi reports a GPU.
    """
    prefer = (prefer or "auto").strip().lower()
    smi = _nvidia_smi_query()
    torch_ok = bool(torch is not None and torch.cuda.is_available())
    torch_name = None
    torch_cuda = None
    torch_ver = getattr(torch, "__version__", None) if torch is not None else None
    vram_mb = _parse_vram_mb(smi["vram_mb"]) if smi else None
    driver = smi["driver"] if smi else None
    name = smi["name"] if smi else "cpu"
    if torch_ok:
        try:
            torch_name = torch.cuda.get_device_name(0)
            torch_cuda = str(torch.version.cuda) if torch.version.cuda else None
            name = torch_name or name
            if vram_mb is None:
                vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            torch_ok = False

    force_cpu = prefer == "cpu" or os.getenv("JESSE_TRAIN_DEVICE", "").lower() == "cpu"
    want_cuda = prefer == "cuda" or prefer == "auto"
    cuda_available = bool((torch_ok or smi) and want_cuda and not force_cpu)

    lgbm_device = "cpu"
    if cuda_available and _lightgbm_cuda_supported():
        lgbm_device = "cuda"
    elif cuda_available:
        lgbm_device = "gpu"

    kind = "cuda" if cuda_available else "cpu"
    source = "torch" if torch_ok else ("nvidia-smi" if smi else "cpu")
    return TrainingDevice(
        kind=kind,
        name=name if kind == "cuda" else "cpu",
        cuda_available=cuda_available,
        vram_mb=vram_mb if kind == "cuda" else None,
        driver=driver,
        cuda_version=torch_cuda,
        torch_version=torch_ver,
        torch_cuda=torch_cuda,
        lightgbm_device=lgbm_device if kind == "cuda" else "cpu",
        source=source,
    )


def lightgbm_device_kwargs(device: Optional[TrainingDevice] = None) -> Dict[str, Any]:
    """Extra LGBMClassifier kwargs. Safe on CPU (empty)."""
    device = device or detect_training_device()
    if device.kind != "cuda":
        return {}
    if device.lightgbm_device == "cuda":
        return {"device": "cuda", "gpu_device_id": 0}
    return {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}


def torch_device_string(device: Optional[TrainingDevice] = None) -> str:
    device = device or detect_training_device()
    return "cuda:0" if device.kind == "cuda" and torch is not None and torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    info = detect_training_device()
    print(json.dumps(info.to_dict(), indent=2))
    print("lightgbm kwargs:", lightgbm_device_kwargs(info))
    print("torch device:", torch_device_string(info))
