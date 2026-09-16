#!/usr/bin/env bash
# Sync Jesse GPU trainers to the Hostinger GPU node and run inventory/train.
#
# The trading VPS (SSH_HOST) is CPU-only. Train on the dedicated GPU instance
# (NVIDIA B200), then copy promoted artifacts back to Jesse ML (:9003).
#
# Required env (never commit these):
#   GPU_SSH_HOST
#   GPU_SSH_PASSWORD  or  GPU_SSH_PRIVATE_KEY
# Optional:
#   GPU_SSH_USER      default ubuntu
#   GPU_SSH_PORT      default 32408 (Hostinger GPU SSH port)
#   GPU_REMOTE_DIR    default /home/ubuntu/jesse-gpu
#
# Usage:
#   ./scripts/gpu_train_remote.sh inventory
#   ./scripts/gpu_train_remote.sh bootstrap
#   ./scripts/gpu_train_remote.sh train BTC-USDT both
#   ./scripts/gpu_train_remote.sh train ETH-USDT lstm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_SSH_HOST="${GPU_SSH_HOST:-}"
GPU_SSH_USER="${GPU_SSH_USER:-ubuntu}"
GPU_SSH_PORT="${GPU_SSH_PORT:-32408}"
GPU_REMOTE_DIR="${GPU_REMOTE_DIR:-/home/ubuntu/jesse-gpu}"
KEY_FILE=""

cleanup() { [[ -n "$KEY_FILE" ]] && rm -f "$KEY_FILE"; }
trap cleanup EXIT

if [[ ! "$GPU_SSH_USER" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Error: invalid GPU_SSH_USER" >&2
  exit 1
fi
if [[ ! "$GPU_SSH_PORT" =~ ^[0-9]+$ ]] || (( GPU_SSH_PORT < 1 || GPU_SSH_PORT > 65535 )); then
  echo "Error: invalid GPU_SSH_PORT" >&2
  exit 1
fi
if [[ ! "$GPU_REMOTE_DIR" =~ ^/home/ubuntu/jesse-gpu(/[A-Za-z0-9._-]+)*$ ]]; then
  echo "Error: GPU_REMOTE_DIR must stay under /home/ubuntu/jesse-gpu" >&2
  exit 1
fi

if [[ -z "$GPU_SSH_HOST" ]]; then
  echo "Error: GPU_SSH_HOST is required (do not hardcode the host in this script)." >&2
  echo "This is the Hostinger GPU instance, not SSH_HOST (trading VPS)." >&2
  exit 1
fi
if [[ -z "${GPU_SSH_PASSWORD:-}" && -z "${GPU_SSH_PRIVATE_KEY:-}" ]]; then
  echo "Error: set GPU_SSH_PASSWORD or GPU_SSH_PRIVATE_KEY." >&2
  exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$GPU_SSH_PORT")
SCP_OPTS=(-o StrictHostKeyChecking=accept-new -P "$GPU_SSH_PORT")
SSH_PREFIX=(ssh)
SCP_PREFIX=(scp)

if [[ -n "${GPU_SSH_PRIVATE_KEY:-}" ]]; then
  umask 077
  KEY_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu_ssh_key.XXXXXX")"
  BEGIN_MARKER="-----BEGIN OPENSSH PRIVATE KEY-----"
  END_MARKER="-----END OPENSSH PRIVATE KEY-----"
  if [[ "$GPU_SSH_PRIVATE_KEY" != *$'\n'* && "$GPU_SSH_PRIVATE_KEY" == *"$BEGIN_MARKER"* ]]; then
    body="${GPU_SSH_PRIVATE_KEY//$BEGIN_MARKER/}"
    body="${body//$END_MARKER/}"
    body="${body// /$'\n'}"
    printf '%s\n%s\n%s\n' "$BEGIN_MARKER" "$body" "$END_MARKER" > "$KEY_FILE"
  else
    printf '%s\n' "$GPU_SSH_PRIVATE_KEY" > "$KEY_FILE"
  fi
  chmod 600 "$KEY_FILE"
  SSH_OPTS+=(-i "$KEY_FILE" -o BatchMode=yes)
  SCP_OPTS+=(-i "$KEY_FILE" -o BatchMode=yes)
elif [[ -n "${GPU_SSH_PASSWORD:-}" ]]; then
  command -v sshpass >/dev/null || { echo "Install sshpass for password auth, or set GPU_SSH_PRIVATE_KEY"; exit 1; }
  export SSHPASS="$GPU_SSH_PASSWORD"
  SSH_PREFIX=(sshpass -e ssh)
  SCP_PREFIX=(sshpass -e scp)
fi

remote() {
  "${SSH_PREFIX[@]}" "${SSH_OPTS[@]}" "${GPU_SSH_USER}@${GPU_SSH_HOST}" "$@"
}

echo "[*] Syncing jesse_quant scripts to GPU node (no secrets, no model binaries)"
remote "mkdir -p ${GPU_REMOTE_DIR}/storage/models ${GPU_REMOTE_DIR}/storage/candles ${GPU_REMOTE_DIR}/logs"
tar -C "$ROOT/jesse_quant" -cf - \
  --exclude='storage' --exclude='.venv' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='examples' --exclude='.env' \
  --exclude='*.key' --exclude='secrets.*' \
  . | remote "tar -C ${GPU_REMOTE_DIR} -xf -"

CMD="${1:-inventory}"
case "$CMD" in
  inventory)
    remote "nvidia-smi; echo '---'; ${GPU_REMOTE_DIR}/.venv/bin/python ${GPU_REMOTE_DIR}/gpu_device.py"
    ;;
  bootstrap)
    remote "cd ${GPU_REMOTE_DIR} && bash bootstrap_gpu.sh"
    ;;
  train)
    SYMBOL="${2:-BTC-USDT}"
    MODEL="${3:-both}"
    if [[ ! "$SYMBOL" =~ ^[A-Z0-9-]+$ ]]; then
      echo "Error: invalid symbol '$SYMBOL'" >&2
      exit 1
    fi
    if [[ ! "$MODEL" =~ ^(both|lightgbm|lstm)$ ]]; then
      echo "Error: model must be both|lightgbm|lstm" >&2
      exit 1
    fi
    remote "cd ${GPU_REMOTE_DIR} && JESSE_TRAIN_DEVICE=auto .venv/bin/python -u train_gpu.py --symbol ${SYMBOL} --candles ${GPU_REMOTE_DIR}/storage/candles/${SYMBOL}_1m.csv.gz --model ${MODEL} --pt-mult 5.5 --sl-mult 1.75 --holding 48"
    ;;
  status)
    remote "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; echo '--- models ---'; ls -lh ${GPU_REMOTE_DIR}/storage/models"
    ;;
  *)
    echo "Usage: GPU_SSH_HOST=... GPU_SSH_PASSWORD=... $0 [inventory|bootstrap|status|train [SYMBOL] [both|lightgbm|lstm]]"
    exit 1
    ;;
esac
