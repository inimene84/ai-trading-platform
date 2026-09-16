#!/usr/bin/env bash
# Train Jesse ML models on the VPS with strategy-aligned triple-barrier geometry.
# Uses auto-retrain-promote: writes artifact ONLY when DSR >= 0.95 and PBO < 0.30.
#
# Required env: SSH_HOST, SSH_USER, SSH_PRIVATE_KEY (or SSH_PASSWORD)
#
# Usage:
#   ./scripts/jesse_vps_train.sh [SYMBOL] [TIMEFRAME] [MODEL_TYPE] [EXTRA_ARGS...]
#
# Examples:
#   ./scripts/jesse_vps_train.sh BTC-USDT 1h lightgbm
#   ./scripts/jesse_vps_train.sh ETH-USDT 1h lightgbm --n-configs 6 --folds 5
#   N_CONFIGS=6 ./scripts/jesse_vps_train.sh SOL-USDT 1h
#
# Env overrides:
#   JESSE_DIR          Jesse repo on VPS (default: /root/jesse-trading)
#   N_CONFIGS          Hyperparameter grid size (default: 6; fewer trials → lower PBO)
#   FOLDS              PurgedKFold folds (default: 5)
#   RUN_BACKTEST       Run backtest after promote (default: true)
#   RUN_PREDICT_TEST   Smoke-test /predict after promote (default: true)
set -euo pipefail

SYMBOL="${1:-BTC-USDT}"
TIMEFRAME="${2:-1h}"
MODEL_TYPE="${3:-lightgbm}"
shift $(( $# >= 3 ? 3 : $# )) || true
EXTRA_ARGS=("$@")

N_CONFIGS="${N_CONFIGS:-6}"
FOLDS="${FOLDS:-5}"
RUN_BACKTEST="${RUN_BACKTEST:-true}"
RUN_PREDICT_TEST="${RUN_PREDICT_TEST:-true}"

SSH_HOST="${SSH_HOST:-}"
if [[ -z "$SSH_HOST" ]]; then
  echo "Error: SSH_HOST environment variable is required." >&2
  exit 1
fi
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
JESSE_DIR="${JESSE_DIR:-/root/jesse-trading}"
KEY_FILE="${TMPDIR:-/tmp}/vps_ssh_key_$$"

cleanup() { rm -f "$KEY_FILE"; }
trap cleanup EXIT

if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
  BEGIN_MARKER="-----BEGIN OPENSSH PRIVATE KEY-----"
  END_MARKER="-----END OPENSSH PRIVATE KEY-----"
  if [[ "$SSH_PRIVATE_KEY" != *$'\n'* && "$SSH_PRIVATE_KEY" == *"$BEGIN_MARKER"* ]]; then
    body="${SSH_PRIVATE_KEY//$BEGIN_MARKER/}"
    body="${body//$END_MARKER/}"
    body="${body// /$'\n'}"
    printf '%s\n%s\n%s\n' "$BEGIN_MARKER" "$body" "$END_MARKER" > "$KEY_FILE"
  else
    printf '%b\n' "$SSH_PRIVATE_KEY" > "$KEY_FILE"
  fi
  chmod 600 "$KEY_FILE"
  SSH_OPTS=(-i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -p "$SSH_PORT")
elif [[ -n "${SSH_PASSWORD:-}" ]]; then
  command -v sshpass >/dev/null || { echo "Install sshpass or use SSH_PRIVATE_KEY"; exit 1; }
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -p "$SSH_PORT")
  SSH_PASS_CMD=(sshpass -p "$SSH_PASSWORD")
else
  echo "Set SSH_PRIVATE_KEY or SSH_PASSWORD" >&2
  exit 1
fi

# Build extra train_ml.py args (skip if caller already passed --n-configs / --folds)
HAS_N_CONFIGS=false
HAS_FOLDS=false
for arg in "${EXTRA_ARGS[@]}"; do
  [[ "$arg" == "--n-configs" ]] && HAS_N_CONFIGS=true
  [[ "$arg" == "--folds" ]] && HAS_FOLDS=true
done
TRAIN_ARGS=()
[[ "$HAS_N_CONFIGS" == "false" ]] && TRAIN_ARGS+=(--n-configs "$N_CONFIGS")
[[ "$HAS_FOLDS" == "false" ]] && TRAIN_ARGS+=(--folds "$FOLDS")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

REMOTE_CMD=$(cat <<EOF
set -e
cd ${JESSE_DIR}
echo "[*] Auto-retrain-promote ${SYMBOL} ${TIMEFRAME} ${MODEL_TYPE} (n-configs=${N_CONFIGS}, folds=${FOLDS})"
./manage.sh auto-retrain-promote ${SYMBOL} ${TIMEFRAME} ${MODEL_TYPE} ${TRAIN_ARGS[*]}
echo "[*] Model metadata:"
./manage.sh model-metadata ${SYMBOL} ${TIMEFRAME}
EOF
)

if [[ "$RUN_PREDICT_TEST" == "true" ]]; then
  REMOTE_CMD+=$(cat <<EOF

echo "[*] Predict smoke test:"
./manage.sh predict-ml ${SYMBOL} ${TIMEFRAME}
EOF
)
fi

if [[ "$RUN_BACKTEST" == "true" ]]; then
  REMOTE_CMD+=$(cat <<EOF

echo "[*] Strategy backtest:"
./manage.sh backtest QuantumAIStrategy
EOF
)
fi

echo "Connecting to ${SSH_USER}@${SSH_HOST}:${SSH_PORT} ..."
echo "Training ${SYMBOL} ${TIMEFRAME} with auto-retrain-promote (DSR>=0.95, PBO<0.30 gate)"
if [[ -n "${SSH_PASSWORD:-}" ]]; then
  "${SSH_PASS_CMD[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
else
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
fi

echo "Jesse VPS training finished."
