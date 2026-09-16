#!/usr/bin/env bash
# Reset Jesse ML trial registry for a symbol scope (lowers DSR penalty on next train).
#
# Usage:
#   ./scripts/jesse_vps_reset_trials.sh BTC-USDT 1h
#   ./scripts/jesse_vps_reset_trials.sh --all
set -euo pipefail

SSH_HOST="${SSH_HOST:-}"
if [[ -z "$SSH_HOST" ]]; then
  echo "Error: SSH_HOST environment variable is required." >&2
  exit 1
fi
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
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

if [[ "${1:-}" == "--all" ]]; then
  SCOPES_JSON='["ml:BTC-USDT:1h","ml:ETH-USDT:1h","ml:SOL-USDT:1h"]'
else
  SYMBOL="${1:-BTC-USDT}"
  TIMEFRAME="${2:-1h}"
  SCOPES_JSON="[\"ml:${SYMBOL}:${TIMEFRAME}\"]"
fi

REMOTE_CMD="docker exec jesse-app python3 -c \"
import json
scopes = ${SCOPES_JSON}
path = '/home/storage/trial_registry.json'
with open(path) as f:
    reg = json.load(f)
for scope in scopes:
    reg['scopes'][scope] = {'total_trials': 0, 'runs': 0}
    print(f'Reset {scope}')
with open(path, 'w') as f:
    json.dump(reg, f, indent=2)
\""

if [[ -n "${SSH_PASSWORD:-}" ]]; then
  "${SSH_PASS_CMD[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
else
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
fi

echo "Trial registry reset complete."
