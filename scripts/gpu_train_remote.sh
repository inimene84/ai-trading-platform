#!/usr/bin/env bash
# Copy Jesse GPU training scripts to the GPU node and run inventory/train.
# Required env (never commit these): GPU_SSH_HOST, GPU_SSH_USER, GPU_SSH_PASSWORD
# Optional: GPU_SSH_PORT (default 22), GPU_REMOTE_DIR (default /home/ubuntu/jesse-gpu)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_SSH_HOST="${GPU_SSH_HOST:-}"
GPU_SSH_USER="${GPU_SSH_USER:-ubuntu}"
GPU_SSH_PORT="${GPU_SSH_PORT:-22}"
GPU_REMOTE_DIR="${GPU_REMOTE_DIR:-/home/ubuntu/jesse-gpu}"

if [[ -z "$GPU_SSH_HOST" ]]; then
  echo "Error: GPU_SSH_HOST is required (do not hardcode the host in this script)." >&2
  exit 1
fi
if [[ -z "${GPU_SSH_PASSWORD:-}" && -z "${GPU_SSH_PRIVATE_KEY:-}" ]]; then
  echo "Error: set GPU_SSH_PASSWORD or GPU_SSH_PRIVATE_KEY." >&2
  exit 1
fi

command -v sshpass >/dev/null || { echo "Install sshpass for password auth"; exit 1; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -p "$GPU_SSH_PORT")
SCP_OPTS=(-o StrictHostKeyChecking=accept-new -P "$GPU_SSH_PORT")
SSH_PREFIX=(sshpass -e ssh)
SCP_PREFIX=(sshpass -e scp)
export SSHPASS="${GPU_SSH_PASSWORD:-}"

remote() {
  "${SSH_PREFIX[@]}" "${SSH_OPTS[@]}" "${GPU_SSH_USER}@${GPU_SSH_HOST}" "$@"
}

echo "[*] Syncing jesse_quant scripts to GPU node (no secrets, no model binaries)"
remote "mkdir -p ${GPU_REMOTE_DIR}/storage/models ${GPU_REMOTE_DIR}/storage/candles ${GPU_REMOTE_DIR}/logs"
"${SCP_PREFIX[@]}" "${SCP_OPTS[@]}" -r \
  "$ROOT/jesse_quant/"*.py \
  "$ROOT/jesse_quant/manage.sh" \
  "$ROOT/jesse_quant/bootstrap_gpu.sh" \
  "$ROOT/jesse_quant/requirements-gpu.txt" \
  "$ROOT/jesse_quant/geometry.json" \
  "${GPU_SSH_USER}@${GPU_SSH_HOST}:${GPU_REMOTE_DIR}/"

CMD="${1:-inventory}"
case "$CMD" in
  inventory)
    remote "nvidia-smi; ${GPU_REMOTE_DIR}/.venv/bin/python ${GPU_REMOTE_DIR}/gpu_device.py"
    ;;
  bootstrap)
    remote "cd ${GPU_REMOTE_DIR} && bash bootstrap_gpu.sh"
    ;;
  train)
    SYMBOL="${2:-BTC-USDT}"
    MODEL="${3:-both}"
    CANDLES="${GPU_REMOTE_DIR}/storage/candles/${SYMBOL}_1m.csv.gz"
    remote "cd ${GPU_REMOTE_DIR} && .venv/bin/python train_gpu.py --symbol ${SYMBOL} --candles ${CANDLES} --model ${MODEL} --pt-mult 5.5 --sl-mult 1.75"
    ;;
  download)
    CLASS="${2:-all}"
    remote "cd ${GPU_REMOTE_DIR} && .venv/bin/python download_ohlcv.py --asset-class ${CLASS} --out ${GPU_REMOTE_DIR}/storage/candles"
    ;;
  train-universe)
    CLASS="${2:-all}"
    MODEL="${3:-both}"
    remote "cd ${GPU_REMOTE_DIR} && .venv/bin/python train_gpu.py --asset-class ${CLASS} --candles-dir ${GPU_REMOTE_DIR}/storage/candles --model ${MODEL} --pt-mult 5.5 --sl-mult 1.75"
    ;;
  *)
    echo "Usage: GPU_SSH_HOST=... GPU_SSH_PASSWORD=... $0 [inventory|bootstrap|train [SYMBOL] [both|lightgbm|lstm]|download [CLASS]|train-universe [CLASS] [both|lightgbm|lstm]]"
    exit 1
    ;;
esac
