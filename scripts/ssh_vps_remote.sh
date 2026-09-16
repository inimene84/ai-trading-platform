#!/usr/bin/env bash
# Run from Cursor Cloud Agent when SSH secrets are configured.
# Required env: SSH_HOST, SSH_USER, SSH_PRIVATE_KEY (or SSH_PASSWORD)
# Optional: SSH_PORT (default 22), PROJECT_DIR (default /root/ai-trading-platform-v3)
# Optional: DEPLOY_REF (git branch or tag to deploy; default main)
set -euo pipefail

SSH_HOST="${SSH_HOST:-}"
if [[ -z "$SSH_HOST" ]]; then
  echo "Error: SSH_HOST environment variable is required." >&2
  exit 1
fi
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
PROJECT_DIR="${PROJECT_DIR:-/root/ai-trading-platform-v3}"
DEPLOY_REF="${DEPLOY_REF:-main}"
KEY_FILE="${TMPDIR:-/tmp}/vps_ssh_key_$$"

cleanup() { rm -f "$KEY_FILE"; }
trap cleanup EXIT

if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
  # Secrets may store the key as one line (spaces instead of newlines)
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
elif [[ -f "${HOME}/.ssh/cursor_cloud_agent" ]]; then
  SSH_OPTS=(-i "${HOME}/.ssh/cursor_cloud_agent" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$SSH_PORT")
elif [[ -n "${SSH_PASSWORD:-}" ]]; then
  command -v sshpass >/dev/null || { echo "Install sshpass or use SSH_PRIVATE_KEY"; exit 1; }
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -p "$SSH_PORT")
  SSH_PASS_CMD=(sshpass -p "$SSH_PASSWORD")
else
  echo "Set SSH_PRIVATE_KEY, SSH_PASSWORD, or add ~/.ssh/cursor_cloud_agent.pub to VPS authorized_keys"
  exit 1
fi

REMOTE_CMD="cd ${PROJECT_DIR} && git fetch origin ${DEPLOY_REF} && git checkout ${DEPLOY_REF} && git pull origin ${DEPLOY_REF} && chmod +x scripts/hostinger_vps_apply.sh scripts/vps_realtime_watchdog.sh scripts/vps_remote_oneliner.sh 2>/dev/null; PROJECT_DIR=${PROJECT_DIR} DEPLOY_REF=${DEPLOY_REF} bash scripts/hostinger_vps_apply.sh"

echo "Connecting to ${SSH_USER}@${SSH_HOST}:${SSH_PORT} ..."
if [[ -n "${SSH_PASSWORD:-}" ]]; then
  "${SSH_PASS_CMD[@]}" ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
else
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$REMOTE_CMD"
fi

echo "Remote apply finished."
