# Two-VPS Connection Map (Trading + Allikas)

QuantumTrade uses **two separate Hostinger VPS machines**. Do not mix them.

| Role | IP | Hostname | Env var | What runs there |
|------|-----|----------|---------|-----------------|
| **Trading** | `$SSH_HOST` | `srv1071801` | `SSH_HOST` / `SSH_HOST_TRADING` | QuantumTrade backend, n8n, A0, MCP, nginx `:8081` |
| **Allikas / Hermes** | `76.13.78.71` | `srv1364509` | `SSH_HOST_HERMES` | **OmniRoute** (`omni.allikas.online`), Hermes agent, OCE, Qdrant (OCE) |

> **Note:** `76.13.78.71` is the **Allikas VPS** (OmniRoute + Hermes). It also runs an OCE Qdrant container — that is separate from the trading-stack Qdrant on the trading VPS. DNS `omni.allikas.online` → `76.13.78.71` is correct.

## SSH (same deploy key on both)

```bash
# After running scripts/vps_fix_two_vps_connections.sh on each host:
ssh vps-trading    # QuantumTrade VPS
ssh vps-hermes     # Allikas / OmniRoute VPS (76.13.78.71)
```

Cloud agents / CI:

```bash
SSH_HOST=<trading-ip>           # default target for vps_ssh_common.py
SSH_HOST_HERMES=76.13.78.71     # Allikas / OmniRoute
```

## OmniRoute (Allikas → Trading)

| Setting | Value |
|---------|-------|
| Public URL | `https://omni.allikas.online/v1` |
| Upstream | nginx `:443` → `127.0.0.1:20128` (OmniRoute container) |
| Trading `.env` | `OMNIROUTE_BASE_URL=https://omni.allikas.online/v1` |
| Trading `.env` | `OMNIROUTE_API_KEY=<from OmniRoute dashboard>` |
| Timeout | `LLM_OMNIROUTE_TIMEOUT_SECONDS=25` (do not lower) |

n8n workflows use credential **OmniRoute Trading v3** (`OmniRtTradV3n801`).

### Common failure: HTTP 502

nginx returns 502 when the OmniRoute container restarts or crashes mid-request (usually Node.js heap OOM during model-catalog warmup — needs `NODE_OPTIONS=--max-old-space-size=4096` and optional 4G host swap). Check on Allikas:

```bash
ssh vps-hermes
docker ps --filter name=omniroute
docker logs omniroute --tail 40
curl -sf http://127.0.0.1:20128/api/health/ping
```

Fix (idempotent):

```bash
ssh vps-hermes 'bash /root/ai-trading-platform-v3/scripts/vps_fix_two_vps_connections.sh'
```

## GrokBOT overseer (Trading VPS)

GrokBOT is **not** a separate container — it is an API route + n8n workflow on the **trading VPS**.

| Component | Location |
|-----------|----------|
| API overview | `GET /api/agents/grok-overseer/overview` (needs `X-API-Key`) |
| API analyze | `POST /api/agents/grok-overseer/analyze` (xAI Grok summary) |
| n8n workflow | `20 - QuantumTrade GrokBOT Daily Overseer (Telegram)` |
| Webhook | `qt-grok-overseer-daily` |
| LLM | `XAI_API_KEY` on trading `.env`; falls back via router if missing |

From inside Docker (n8n):

```
POST http://ai-trading-backend:8000/api/agents/grok-overseer/analyze
```

From Hermes on Allikas (REST via nginx):

```bash
export QT_BASE=http://<trading-ip>:8081/api
export QT_KEY=<ADMIN_API_KEY>
curl -s -X POST "$QT_BASE/agents/grok-overseer/analyze" \
  -H "X-API-Key: $QT_KEY" -H 'Content-Type: application/json' \
  -d '{"focus":"risk and operational health"}'
```

Agent connect manifest: `GET /api/agents/connect`

## Hermes on Allikas (construction overseer)

Hermes on `76.13.78.71` is primarily for **OpenConstructionERP**. For trading oversight it should use REST to the trading VPS (`QT_BASE`), not SSH.

Broken MCP to fix separately: `a0.thorinvest.org` (Agent Zero remote) — logs show HTTP 500; use trading MCP only from the trading VPS itself (`docs/ops/A0_HERMES_MCP.md`).

## One-shot repair

Run on **both** hosts (or from cloud agent with `SSH_PRIVATE_KEY`):

```bash
# Trading VPS
cd /root/ai-trading-platform-v3
bash scripts/vps_fix_two_vps_connections.sh

# Allikas VPS (same repo path if present, or copy script)
ssh vps-hermes 'curl -fsSL ... | bash'   # or git pull + run
```

## Inventory source of truth

On Allikas: `/opt/hermes-workspace/VPS-INVENTORY.md`
