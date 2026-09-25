#!/usr/bin/env bash
# Wire SSH + OmniRoute + GrokBOT connections between the two Hostinger VPSes.
#
# Trading VPS  (SSH_HOST / srv1071801) — QuantumTrade backend, n8n, A0, MCP
# Allikas VPS  (SSH_HOST_HERMES / 76.13.78.71 / srv1364509) — OmniRoute, Hermes, OCE
#
# Run on EITHER host as root. Detects role from hostname / VPS_THIS_ROLE.
# Safe to re-run (idempotent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/vps_ssh_hygiene.sh
if [[ -f "${SCRIPT_DIR}/lib/vps_ssh_hygiene.sh" ]]; then
  source "${SCRIPT_DIR}/lib/vps_ssh_hygiene.sh"
else
  vps_ssh_hygiene() {
    chown root:root /root 2>/dev/null || true
    chmod 700 /root 2>/dev/null || true
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  }
fi

VPS_TRADING_IP="${VPS_TRADING_IP:-${SSH_HOST_TRADING:-${SSH_HOST:-}}}"
VPS_HERMES_IP="${VPS_HERMES_IP:-${SSH_HOST_HERMES:-${SSH_HOST_CONSTRUCTION:-76.13.78.71}}}"
VPS_TRADING_HOSTNAME="${VPS_TRADING_HOSTNAME:-srv1071801}"
VPS_HERMES_HOSTNAME="${VPS_HERMES_HOSTNAME:-srv1364509}"
OMNI_PORT="${OMNI_PORT:-20128}"
OMNI_IMAGE="${OMNI_IMAGE:-diegosouzapw/omniroute:latest}"

_detect_role() {
  local h
  h="$(hostname -s 2>/dev/null || hostname)"
  if [[ "$h" == "$VPS_HERMES_HOSTNAME" ]] || [[ "$(curl -4 -sf --max-time 3 ifconfig.me 2>/dev/null || true)" == "$VPS_HERMES_IP" ]]; then
    echo "hermes"
  else
    echo "trading"
  fi
}

_setup_ssh_aliases() {
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  if [[ ! -f /root/.ssh/id_ed25519 ]]; then
    ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 -q
  fi
  chmod 600 /root/.ssh/id_ed25519
  chmod 644 /root/.ssh/id_ed25519.pub

  local trading_ip="$VPS_TRADING_IP"
  local hermes_ip="$VPS_HERMES_IP"
  if [[ -z "$trading_ip" ]] && [[ "$ROLE" == "trading" ]]; then
    trading_ip="$(curl -4 -sf --max-time 3 ifconfig.me 2>/dev/null || true)"
  fi
  if [[ -z "$trading_ip" ]]; then
    echo "WARN: VPS_TRADING_IP / SSH_HOST not set — skip SSH config aliases"
    return 0
  fi

  cat > /root/.ssh/config <<EOF
# QuantumTrade two-VPS layout (see docs/ops/TWO_VPS_CONNECTIONS.md)
Host vps-trading ${VPS_TRADING_HOSTNAME}
    HostName ${trading_ip}
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new

Host vps-hermes vps-allikas vps-construction ${VPS_HERMES_HOSTNAME}
    HostName ${hermes_ip}
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
EOF
  chmod 600 /root/.ssh/config
  echo "SSH aliases: vps-trading (${trading_ip}), vps-hermes (${hermes_ip})"
}

_exchange_host_keys() {
  local peer_ip peer_name local_pub
  local_pub="$(cat /root/.ssh/id_ed25519.pub)"
  if [[ "$ROLE" == "trading" ]]; then
    peer_ip="$VPS_HERMES_IP"
    peer_name="vps-hermes"
  else
    peer_ip="$VPS_TRADING_IP"
    peer_name="vps-trading"
  fi
  if [[ -z "$peer_ip" ]]; then
    echo "WARN: peer IP unknown — skip key exchange"
    return 0
  fi
  if ! grep -qF "$local_pub" /root/.ssh/authorized_keys 2>/dev/null; then
    echo "$local_pub" >> /root/.ssh/authorized_keys
  fi
  chmod 600 /root/.ssh/authorized_keys
  echo "Local pubkey present in authorized_keys"
  if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      "root@${peer_ip}" "grep -qF '$local_pub' /root/.ssh/authorized_keys 2>/dev/null || echo '$local_pub' >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys" 2>/dev/null; then
    echo "Peer key exchange OK ($peer_name)"
  else
    echo "WARN: could not push pubkey to ${peer_name} (${peer_ip}) — add manually if needed"
  fi
}

_fix_omniroute_on_hermes() {
  echo "=== Stabilize OmniRoute on Allikas (${VPS_HERMES_IP}) ==="

  # Redis sidecar (rate limiter backend — without it OmniRoute logs warnings and can flap)
  if ! docker ps --format '{{.Names}}' | grep -qx omniroute-redis; then
    docker run -d --name omniroute-redis --restart unless-stopped \
      -v omniroute-redis-data:/data \
      redis:8-alpine redis-server --save 60 1 --loglevel warning
    echo "Started omniroute-redis"
  fi

  local qdrant_net=""
  qdrant_net="$(docker inspect openconstructionerp-qdrant-1 --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | awk '{print $1}')"
  [[ -z "$qdrant_net" ]] && qdrant_net="bridge"

  docker network connect "$qdrant_net" omniroute-redis 2>/dev/null || true

  local existing_env
  existing_env="$(docker inspect omniroute --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null || true)"

  # OmniRoute loads ~9k model catalog entries at boot — 1GB Node heap OOMs (#4437).
  if ! swapon --show | grep -q /swapfile-omni; then
    fallocate -l 4G /swapfile-omni 2>/dev/null || dd if=/dev/zero of=/swapfile-omni bs=1M count=4096
    chmod 600 /swapfile-omni && mkswap /swapfile-omni && swapon /swapfile-omni
    grep -q swapfile-omni /etc/fstab || echo '/swapfile-omni none swap sw 0 0' >> /etc/fstab
  fi

  docker rm -f omniroute 2>/dev/null || true

  docker run -d \
    --name omniroute \
    --restart unless-stopped \
    -p "127.0.0.1:${OMNI_PORT}:${OMNI_PORT}" \
    -v omniroute-data:/app/data \
    --network "$qdrant_net" \
    -e BASE_URL="http://127.0.0.1:${OMNI_PORT}" \
    -e NODE_ENV=production \
    -e PORT="${OMNI_PORT}" \
    -e DASHBOARD_PORT="${OMNI_PORT}" \
    -e API_HOST=0.0.0.0 \
    -e HOSTNAME=0.0.0.0 \
    -e DATA_DIR=/app/data \
    -e NEXT_PUBLIC_BASE_URL=https://omni.allikas.online \
    -e AUTH_COOKIE_SECURE=true \
    -e NODE_OPTIONS=--max-old-space-size=4096 \
    -e OMNIROUTE_MEMORY_MB=4096 \
    -e REDIS_URL=redis://omniroute-redis:6379 \
    -e QDRANT_HOST=openconstructionerp-qdrant-1 \
    -e QDRANT_PORT=6333 \
    -e QDRANT_COLLECTION=omniroute_memory \
    -e QDRANT_EMBEDDING_MODEL=openai/text-embedding-3-small \
    -e OMNIROUTE_MIGRATIONS_DIR=/app/migrations \
    "$OMNI_IMAGE" node dev/run-standalone.mjs

  docker network connect "$qdrant_net" omniroute-redis 2>/dev/null || true

  echo "Waiting for OmniRoute health..."
  for i in $(seq 1 24); do
    if curl -sf --max-time 3 "http://127.0.0.1:${OMNI_PORT}/api/health/ping" >/dev/null 2>&1; then
      echo "OmniRoute healthy on :${OMNI_PORT}"
      break
    fi
    sleep 5
  done

  nginx -t && systemctl reload nginx
  echo "nginx reloaded for omni.allikas.online"
}

_wire_hermes_to_trading() {
  local trading_ip="$VPS_TRADING_IP"
  [[ -z "$trading_ip" ]] && trading_ip="$(ssh -o BatchMode=yes -o ConnectTimeout=8 vps-trading 'curl -4 -sf --max-time 3 ifconfig.me' 2>/dev/null || true)"
  [[ -z "$trading_ip" ]] && { echo "WARN: trading IP unknown — skip Hermes QT_BASE"; return 0; }

  local hermes_env="/var/lib/docker/volumes/hermes-webui_hermes-home/_data/.env"
  local skill_dir="/var/lib/docker/volumes/hermes-webui_hermes-home/_data/skills/trading/quantumtrade-grokbot"
  mkdir -p "$skill_dir"

  if [[ -f "$hermes_env" ]]; then
    if grep -q '^QT_BASE=' "$hermes_env" 2>/dev/null; then
      sed -i "s|^QT_BASE=.*|QT_BASE=http://${trading_ip}:8081/api|" "$hermes_env"
    else
      echo "QT_BASE=http://${trading_ip}:8081/api" >> "$hermes_env"
    fi
    chmod 600 "$hermes_env"
    echo "Hermes QT_BASE -> http://${trading_ip}:8081/api"
  fi

  cat > "${skill_dir}/SKILL.md" <<'SKILL'
# GrokBOT / Hermes — QuantumTrade overseer (read-only default)

Trading VPS REST (nginx): `QT_BASE` + `QT_KEY` (ADMIN_API_KEY from trading `.env`).

| Endpoint | Purpose |
|----------|---------|
| `GET $QT_BASE/health` | Backend health |
| `GET $QT_BASE/trading/loop/status` | Trading loop |
| `GET $QT_BASE/trading/positions` | Open positions |
| `GET $QT_BASE/sentry/status` | Kill switch / sentry |
| `GET $QT_BASE/api/agents/grok-overseer/overview` | GrokBOT snapshot (needs X-API-Key) |
| `POST $QT_BASE/api/agents/grok-overseer/analyze` | Grok LLM summary |

OmniRoute (this host): `https://omni.allikas.online/v1` — primary LLM for trading personas.
Do not use OmniRoute for live order placement; use REST oversight tools only.
SKILL

  docker restart hermes-agent 2>/dev/null || true
  docker restart hermes-webui 2>/dev/null || true
  echo "Hermes GrokBOT skill + QT_BASE wired"
}

_verify_from_trading() {
  local key
  key="$(grep '^OMNIROUTE_API_KEY=' /root/ai-trading-platform-v3/.env 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
  echo "=== Verify OmniRoute from trading VPS ==="
  if [[ -n "$key" ]]; then
    curl -sS --max-time 30 https://omni.allikas.online/v1/chat/completions \
      -H "Authorization: Bearer ${key}" \
      -H 'Content-Type: application/json' \
      -d '{"model":"auto/fast","messages":[{"role":"user","content":"ping"}],"max_tokens":5}' \
      | head -c 300
    echo
  else
    curl -sS --max-time 10 -o /dev/null -w "omni models HTTP %{http_code}\n" \
      -H "Authorization: Bearer ${key}" https://omni.allikas.online/v1/models || true
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 vps-hermes 'hostname' 2>/dev/null && echo "SSH vps-hermes OK" || echo "WARN: SSH vps-hermes failed"
}

ROLE="${VPS_THIS_ROLE:-$(_detect_role)}"
echo "=== Two-VPS connection fix (role=${ROLE}) ==="
vps_ssh_hygiene
_setup_ssh_aliases

case "$ROLE" in
  hermes|allikas|construction)
    _fix_omniroute_on_hermes
    _exchange_host_keys
    _wire_hermes_to_trading
    ;;
  trading|*)
    _exchange_host_keys
    _verify_from_trading
    ;;
esac

cat <<EOF

=== Done (role=${ROLE}) ===
Trading:  ssh vps-trading  (QuantumTrade / GrokBOT n8n workflows)
Allikas:  ssh vps-hermes   (OmniRoute https://omni.allikas.online/v1)
Docs:     docs/ops/TWO_VPS_CONNECTIONS.md
EOF
