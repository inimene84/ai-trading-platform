#!/usr/bin/env bash
# Bootstrap a GPU training venv on the Hostinger training node.
# Does not deploy QuantumTrade live. Never prints or writes secrets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${JESSE_GPU_VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

echo "[*] GPU inventory"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "[!] nvidia-smi not found — CPU fallback"
fi

mkdir -p "$ROOT/storage/models" "$ROOT/storage/candles" "$ROOT/logs"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[*] Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install -U pip wheel setuptools
echo "[*] Installing PyTorch CUDA 12.8 wheels (Blackwell / B200)"
"$VENV/bin/pip" install --timeout 180 torch --index-url https://download.pytorch.org/whl/cu128
echo "[*] Installing tabular ML stack"
"$VENV/bin/pip" install --timeout 180 numpy pandas scikit-learn joblib "lightgbm>=4.3" pyarrow scipy psycopg2-binary

echo "[*] Device probe"
"$VENV/bin/python" "$ROOT/gpu_device.py"
echo "[✓] GPU training env ready. Use:"
echo "    $VENV/bin/python $ROOT/train_gpu.py --candles storage/candles/BTC-USDT_1m.csv.gz --model both"
