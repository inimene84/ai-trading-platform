#!/usr/bin/env bash
# Copy jesse_quant inference/import scripts onto the trading VPS Jesse workspace.
# Required env: SSH_HOST, SSH_USER, SSH_PRIVATE_KEY (same as ssh_vps_remote.sh)
# Does not copy .env, models, or candle dumps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="${SSH_HOST:-}"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
JESSE_DIR="${JESSE_DIR:-/root/jesse-trading}"
KEY_FILE="${TMPDIR:-/tmp}/vps_ssh_key_jesse_$$"

if [[ -z "$SSH_HOST" ]]; then
  echo "Error: SSH_HOST is required." >&2
  exit 1
fi

cleanup() { rm -f "$KEY_FILE"; }
trap cleanup EXIT

BEGIN_MARKER="-----BEGIN OPENSSH PRIVATE KEY-----"
END_MARKER="-----END OPENSSH PRIVATE KEY-----"
if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
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
  SCP_OPTS=(-i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -P "$SSH_PORT")
else
  echo "SSH_PRIVATE_KEY is required" >&2
  exit 1
fi

echo "[*] Syncing jesse_quant scripts to ${SSH_USER}@${SSH_HOST}:${JESSE_DIR}"
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "mkdir -p ${JESSE_DIR}/storage/models ${JESSE_DIR}/storage/candles"
# shellcheck disable=SC2086
scp "${SCP_OPTS[@]}" \
  "$ROOT/jesse_quant/"*.py \
  "$ROOT/jesse_quant/manage.sh" \
  "$ROOT/jesse_quant/bootstrap_gpu.sh" \
  "$ROOT/jesse_quant/requirements-gpu.txt" \
  "$ROOT/jesse_quant/geometry.json" \
  "${SSH_USER}@${SSH_HOST}:${JESSE_DIR}/"

echo "[*] Restarting jesse-app to reload /predict"
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "cd ${JESSE_DIR} && docker compose -f docker/docker-compose.yml restart jesse"
echo "Jesse sync finished."
