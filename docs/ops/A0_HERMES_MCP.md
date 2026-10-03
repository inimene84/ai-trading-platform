# Connect Agent Zero (A0) + Hermes to QuantumTrade

Goal: let **A0** (MCP/A2A client) and **Hermes** oversee live trades — inspect positions, adjust SL/TP, halt/resume, and stop the loop — without giving them exchange API keys.

## Architecture

```text
A0 (a0-instance) ──MCP HTTP──► ai-trading-mcp:9100/mcp ──HTTP──► ai-trading-backend:8000
                 ──MCP HTTP──► ai-trading-scrapling:8000/mcp  (web research)
Hermes           ──REST/MCP──► 127.0.0.1:8001  /  127.0.0.1:9100/mcp
Grok bots        ──REST──────► POST /api/research/fetch  ──► scrapling:8080/fetch
```

- MCP server is already in `docker-compose.yml` / `docker-compose.prod.yml` as `ai-trading-mcp`.
- Bound to `127.0.0.1:9100` on the host (not public).
- `MCP_API_TOKEN` = `ADMIN_API_KEY` so write tools work.

A2A is **not** required for this platform: A0 should use MCP. Hermes can use MCP or REST.

## One-shot VPS wiring

```bash
cd /root/ai-trading-platform-v3
chmod +x scripts/vps_connect_a0_hermes_mcp.sh scripts/vps_apply_small_restart.sh
./scripts/vps_connect_a0_hermes_mcp.sh
```

That script:
1. Finds the backend Docker network (`*trading-net*`)
2. Ensures `mcp-server` is up
3. Attaches `a0-instance` + `hermes-webui` to that network
4. Writes `quantumtrade` into A0 `settings.json` and Hermes `config.yaml`
5. Restarts clients and prints REST fallback for a remote Hermes VPS

## Two-VPS layout

| Host | Role | Connections we use |
| --- | --- | --- |
| Trading VPS (`$SSH_HOST`) | QuantumTrade live money, local `vps-qdrant`, MCP `:9100`, nginx `:8081` | Do not expose trading Qdrant off-box |
| Allikas / Hermes VPS (`$SSH_HOST_HERMES` / `$SSH_HOST_ALLIKAS`) | OmniRoute `https://omni.allikas.online`, OpenViking `https://viking.allikas.online`, Hermes, OCE | HTTPS OmniRoute + Viking. Same SSH key as trading. |

The Allikas host also runs **OCE's** Qdrant for construction ERP memory. That is **not** a standalone Qdrant VPS and is **not** `QDRANT_URL` for QuantumTrade. Trading Qdrant stays `http://vps-qdrant:6333` on the trading Docker network.

- **A0 / hermes-webui on trading VPS** → MCP `http://ai-trading-mcp:9100/mcp` on `trading-net`
- **Hermes on Allikas** → REST `http://$SSH_HOST:8081/api/...` (MCP port is not public)
- **OmniRoute from trading** → `https://omni.allikas.online/v1` (`OMNIROUTE_BASE_URL`)

## A0 configuration

The connect script writes this automatically. Manual equivalent inside Agent Zero:

| Field | Value |
| --- | --- |
| Name | `quantumtrade` |
| URL | `http://ai-trading-mcp:9100/mcp` |
| Transport | streamable-http / HTTP |
| Auth | Bearer / API key = `ADMIN_API_KEY` (if client supports headers) |

| Name | `scrapling` |
| URL | `http://ai-trading-scrapling:8000/mcp` |
| Transport | streamable-http / HTTP |
| Auth | Bearer = `ADMIN_API_KEY` |
| Use | Web/news/calendar research for Grok bots. Do not use for live prices. |

A0 stores servers under `mcp_servers.mcpServers.<name>` (not as a sibling of `mcpServers`).

Suggested A0 system prompt snippet:

> You are a trading oversight agent for QuantumTrade. Prefer `list_positions`, `list_binance_positions`, `get_loop_status`, and `sentry_status` before intervening. Use `modify_position` for SL/TP fixes, `close_position` only when necessary, and `sentry_halt` / `emergency_halt` for risk events. Never ask for Binance keys.

## Hermes configuration

### Option A — MCP (trading VPS only)

On the trading VPS, Hermes `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  quantumtrade:
    url: http://ai-trading-mcp:9100/mcp
    timeout: 120
    connect_timeout: 60
```

Host loopback also works from the host namespace: `http://127.0.0.1:9100/mcp`.

### Option B — REST (local or remote Hermes)

```bash
# On trading VPS host:
export QT_BASE=http://127.0.0.1:8001
# From remote Hermes VPS (nginx):
export QT_BASE=http://$SSH_HOST:8081/api
export QT_KEY=<ADMIN_API_KEY>

curl -s "$QT_BASE/trading/positions"
curl -s "$QT_BASE/trading/loop/status"
curl -s "$QT_BASE/sentry/status"

curl -s -X POST "$QT_BASE/trading/positions/<ID>/close" -H "X-API-Key: $QT_KEY"
curl -s -X PUT  "$QT_BASE/trading/positions/<ID>/modify" \
  -H "X-API-Key: $QT_KEY" -H 'Content-Type: application/json' \
  -d '{"stop_loss":12345.0,"take_profit":13000.0}'
```

### Verify

```bash
# A0 logs should show quantumtrade tools loaded
docker logs a0-instance 2>&1 | grep -i quantumtrade | tail

# Platform
curl -sf http://127.0.0.1:8001/health
curl -sf http://127.0.0.1:8001/trading/loop/status
```

## SSH recovery (trading VPS)

If SSH suddenly returns `Permission denied (publickey)` while HTTP `:8081` still works, `/root` ownership/modes are usually wrong (often after a container bind-mounts host `/root`). From Hostinger **Browser Terminal**:

```bash
chown root:root /root && chmod 700 /root
chown -R root:root /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
```

Do **not** bind-mount host `/root` into `hermes-webui` — mount only `/root/.hermes`.

Also **do not** mount host `/root/.ssh` into Hermes. A container process can change
ownership/modes on that bind and break host SSH again (`Permission denied (publickey)`
while HTTP still works). Hermes should oversee trading via MCP/REST on `trading-net`,
not via SSH into localhost.

One-shot fix (Browser Terminal or SSH) for hermes mount + Kronos + MCP rewire:

```bash
cd /root/ai-trading-platform-v3
git fetch origin && git checkout cursor/a0-hermes-mcp-oversight-523e && git pull
chmod +x scripts/vps_fix_hermes_kronos_gaps.sh
./scripts/vps_fix_hermes_kronos_gaps.sh
```

## Curated MCP oversight tools

| Tool | Purpose |
| --- | --- |
| `list_positions` / `list_binance_positions` / `list_trades` | Inspect |
| `close_position` / `modify_position` | Fix / flatten |
| `get_loop_status` / `start_trading_loop` / `stop_trading_loop` | Loop control |
| `sentry_status` / `sentry_halt` / `emergency_halt` / `sentry_resume` | Safety |
| `get_portfolio` / `get_trading_config` / `backend_health` | Health |
| `get_trading_opinion` / trade-memory / skills tools | Research |

## Small-amount restart (after flattening)

```bash
./scripts/vps_apply_small_restart.sh
```

Sets:
- Symbols: `BTCUSDC,ETHUSDC,SOLUSDC,BNBUSDC`
- `TRADE_USDT_AMOUNT=25`, `MAX_POSITIONS=2`, no pyramids
- `TIMING_GATE_SHADOW=true` (log timing vetoes, do not hard-block yet)
- Restarts backend / MCP / Kronos sidecar when present

## A0 memory / forced re-login (OOM)

If A0 randomly asks you to log in again while the container stays “Up”, it is usually **cgroup OOM**, not internet:

- Host hard-limits `a0-instance` (historically 8–10 GiB, no swap)
- `run_ui.py` can grow past ~6–7 GiB under multi-subagent load
- OOM kills the UI process → Flask session/CSRF resets → forced re-login

Mitigation on the trading VPS (~15 GiB host):

```bash
cd /root/ai-trading-platform-v3
chmod +x scripts/vps_fix_a0_memory.sh
./scripts/vps_fix_a0_memory.sh
```

That script:

1. Enables a persistent **4 GiB** host swapfile (`/swapfile-a0`)
2. Raises A0 to **12G RAM / 14G memswap** in `/docker/agent-zero/docker-compose.yml`
3. Applies live `docker update` + `compose up -d a0`
4. Checks UI HTTP, cgroup `oom`/`oom_kill`, and trading health/loop

Scheduled trading overview tasks live in the `a0-data` volume (`/a0/usr/scheduler/tasks.json`) and survive recreate. Prefer read-only scheduled jobs that do **not** spawn many parallel subagents while RAM is tight.

## Security notes

1. Do **not** publish `:9100` to the public internet without auth/VPN.
2. Never put exchange secrets into A0/Hermes — only `ADMIN_API_KEY`.
3. Prefer `sentry_halt` before mass closes when unsure.
4. Keep `TIMING_GATE_SHADOW=true` until WP5 calibration looks sane.
