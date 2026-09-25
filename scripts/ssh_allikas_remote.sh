#!/usr/bin/env bash
# SSH to the Allikas VPS (OmniRoute / OpenViking / Hermes).
# Same key as trading. Do not deploy QuantumTrade here.
#
# Required env: SSH_PRIVATE_KEY (or SSH_PASSWORD)
# Host (first set wins): SSH_HOST_HERMES, SSH_HOST_ALLIKAS, SSH_HOST_CONSTRUCTION
# Optional: SSH_USER (default root), SSH_PORT (default 22)
# Usage: ./scripts/ssh_allikas_remote.sh [remote command]
set -euo pipefail

SSH_HOST="${SSH_HOST_HERMES:-${SSH_HOST_ALLIKAS:-${SSH_HOST_CONSTRUCTION:-}}}"
if [[ -z "$SSH_HOST" ]]; then
  echo "Error: set SSH_HOST_HERMES (or SSH_HOST_ALLIKAS / SSH_HOST_CONSTRUCTION)." >&2
  echo "This is the Allikas OmniRoute host, not trading Qdrant." >&2
  exit 1
fi
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
KEY_FILE="${TMPDIR:-/tmp}/allikas_ssh_key_$$"

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
  SSH_OPTS=(-i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p "$SSH_PORT")
elif [[ -f "${HOME}/.ssh/cursor_cloud_agent" ]]; then
  SSH_OPTS=(-i "${HOME}/.ssh/cursor_cloud_agent" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$SSH_PORT")
elif [[ -n "${SSH_PASSWORD:-}" ]]; then
  command -v sshpass >/dev/null || { echo "Install sshpass or use SSH_PRIVATE_KEY"; exit 1; }
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -p "$SSH_PORT")
  SSH_PASS_CMD=(sshpass -p "$SSH_PASSWORD")
else
  echo "Set SSH_PRIVATE_KEY, SSH_PASSWORD, or add ~/.ssh/cursor_cloud_agent.pub to Allikas authorized_keys" >&2
  exit 1
fi

REMOTE_CMD="${*:-hostname; docker ps --format 'table {{.Names}}\t{{.Status}}' | head -40}"

echo "Connecting to ${SSH_USER}@${SSH_HOST}:${SSH_PORT} (Allikas / OmniRoute) ..."
if [[ -n "${SSH_PASSWORD:-}" ]]; then
  "${SSH_PASS_CMD[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
else
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
fi
