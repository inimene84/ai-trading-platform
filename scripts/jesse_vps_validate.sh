#!/usr/bin/env bash
# Validate Jesse ML models on VPS: DSR/PBO gates, metadata, predict smoke test.
#
# Usage:
#   ./scripts/jesse_vps_validate.sh [SYMBOL] [TIMEFRAME]
#   ./scripts/jesse_vps_validate.sh --all          # BTC, ETH, SOL @ 1h
#
# Required env: SSH_HOST, SSH_USER, SSH_PRIVATE_KEY (or SSH_PASSWORD)
set -euo pipefail

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

run_remote() {
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    "${SSH_PASS_CMD[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$1"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$1"
  fi
}

validate_symbol() {
  local sym="$1"
  local tf="$2"
  echo "============================================================"
  echo " Validating ${sym} ${tf}"
  echo "============================================================"
  run_remote "cd ${JESSE_DIR} && ./manage.sh model-metadata ${sym} ${tf} 2>&1"
  echo ""
  run_remote "cd ${JESSE_DIR} && ./manage.sh predict-ml ${sym} ${tf} 2>&1 | head -20"
  echo ""
}

if [[ "${1:-}" == "--all" ]]; then
  validate_symbol "BTC-USDT" "1h"
  validate_symbol "ETH-USDT" "1h"
  validate_symbol "SOL-USDT" "1h"
  echo "============================================================"
  echo " ML service health"
  echo "============================================================"
  run_remote "cd ${JESSE_DIR} && ./manage.sh ml-status 2>&1"
else
  SYMBOL="${1:-BTC-USDT}"
  TIMEFRAME="${2:-1h}"
  validate_symbol "$SYMBOL" "$TIMEFRAME"
fi

echo "Validation complete."
