# AGENTS.md — QuantumTrade Pro

Guidance for AI coding agents working in this repository. Read this before making changes.

## Project Overview

**QuantumTrade Pro** (`pyproject.toml` name: `quantumtrade-pro`) is an AI-powered, multi-agent crypto-futures / forex trading platform. It began as a fork of `virattt/ai-hedge-fund` but has been heavily extended: Binance Futures and cTrader broker integrations, a Kronos deep-learning forecasting sidecar, an opinion layer of 8+ LLM analyst agents, n8n webhook pipelines, InfluxDB time-series storage, and Qdrant vector storage for news sentiment and trade memory.

**This is a LIVE, real-money trading system.** Treat every change to trading logic, risk gates, order routing, or deployment scripts with production-level care. Safety gates are fail-closed by default; never weaken them without an explicit request.

Git remote: `github.com/inimene84/ai-trading-platform` (main branch; feature work happens on branches like `fix/...`).

## Repository Layout

```
backend/          FastAPI application (Python 3.11+) — the core of the system
  main.py         App entrypoint: lifespan manager, supervised background tasks
  routes/         REST API routers (trading, agents, backtest, news, signals, ...)
  services/       Business logic: trading_loop.py, unified_trading.py, decision_engine.py,
                  risk_guard.py, position_manager.py, opinion_layer.py, risk_reviewer.py,
                  binance_* / ctrader_* broker services, kronos_*, skill_miner.py, ...
  agents/         LLM analyst personas (warren_buffett, technicals, sentiment, ...)
  strategies/     Technical strategies (trend_following, mean_reversion, breakout, combined)
  brokers/        Broker abstraction (base, registry, auth)
  database/       SQLAlchemy models + connection (SQLite default, Postgres optional)
  alembic/        DB migrations
  llm/            Multi-provider LLM router (xAI, Kie.ai, OpenAI, Anthropic, Ollama, ...)
  backtesting/    Backtesting engine + CLI (`backtester` poetry script)
  tests/          pytest suite (unit/, integration/, mocks/)
  requirements.txt  PRODUCTION dependency source of truth (see below)
frontend/         React 19 + Vite 8 + TypeScript + Tailwind 4 dashboard
  server.ts       Express server (dev: vite middleware; prod: serves dist/)
  src/            App.tsx, components/, services/ (API clients), lib/
mcp_server/       FastMCP server exposing backend tools to MCP agents (Claude, Cursor, ...)
kronos-infer/     Standalone inference sidecar for NeoQuasar/Kronos-mini (PyTorch isolated)
grafana/          Monitoring dashboards/provisioning
workflows/        n8n workflow JSON exports (market scanner, news scanner, executor)
scripts/          Ops/deployment scripts, mostly for the Hostinger VPS (vps_*.sh/py)
docs/             Design docs (docs/ops/ contains operational runbooks)
docker-compose.yml        Full stack: backend, litellm, qdrant, influxdb, nginx, mcp, kronos
docker-compose.prod.yml   Production variant
Dockerfile.backend        Production backend image (python:3.11-slim, uv-based install)
litellm-config.yaml       LiteLLM proxy model routing config
nginx.conf                Reverse proxy / static frontend hosting
```

Note: the checkout on this machine may sit inside a wrapper directory; the git repository root is the directory containing this file.

## Tech Stack

- **Backend:** Python 3.11+ (CI runs 3.12), FastAPI + Uvicorn, SQLAlchemy 2 + Alembic, Pydantic v2, LangChain/LangGraph multi-agent framework, ccxt / python-binance / ctrader-open-api for brokers, structlog, sentry-sdk.
- **Frontend:** React 19, Vite 8, TypeScript ~5.8, Tailwind CSS 4, Express dev/prod server via `tsx`, lightweight-charts, recharts, @xyflow/react (workflow builder).
- **Datastores:** SQLite (default dev DB: `backend/hedge_fund.db`), PostgreSQL (optional), InfluxDB (time-series metrics/sentiment), Qdrant (news + trade-memory vectors), Redis (cache, on VPS).
- **Infra:** Docker Compose, nginx, LiteLLM proxy, n8n, Grafana, Sentry.

## Dependency Management — IMPORTANT

`backend/requirements.txt` is the **single source of truth for production** (`Dockerfile.backend` installs it with `uv`, CI installs it with pip). `pyproject.toml` (Poetry) is kept in sync for local dev. **When bumping or adding a dependency, update BOTH files** and re-run `pytest backend/tests`.

## Build and Run Commands

### Backend

```bash
# Install (from repo root)
poetry install                      # dev flow
# or: pip install -r backend/requirements.txt   # mirrors production/CI

# Run (dev)
poetry run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080
# STARTUP.md native flow uses port 8000; run.sh/run.bat use 8080
```

API docs at `/docs`; health check at `/health` (also `/api/health`).

### Frontend

```bash
cd frontend
npm install
npm run dev      # tsx server.ts — Express + Vite dev server (port 5173 by default)
npm run build    # production build into frontend/dist
npm run lint     # TypeScript type-check (tsc --noEmit)
```

### One-shot local start

`./run.sh` (Linux/macOS) or `run.bat` (Windows) installs deps and starts both services. `start_docker_local.sh` runs the Docker stack locally.

### Docker (full stack)

```bash
cp .env.example .env   # then fill in API keys
docker compose up -d --build
```

Services: backend (host `127.0.0.1:8001` → container 8000), nginx (`:8081`), litellm (`127.0.0.1:4100`), qdrant (`127.0.0.1:6333`), influxdb (`127.0.0.1:8086`), mcp-server (`127.0.0.1:9100`), kronos-infer (`127.0.0.1:8002`). Requires pre-created external Docker networks `trading-net` and `n8n_default`.

## Testing

```bash
# From repo root — this is exactly what CI runs:
ruff check backend
PYTHONPATH=. pytest backend/tests          # or: cd backend && pytest
```

- pytest config: `backend/pytest.ini` (`testpaths = tests`, verbose, `pythonpath = .` — so run from either the repo root with `PYTHONPATH=.` or from `backend/`).
- Layout: `backend/tests/unit/` (the bulk — brokers, risk guards, decision engine, trading loop safety, security), `backend/tests/integration/` (`test_trading_cycle.py`), `backend/tests/mocks/`.
- `backend/tests/conftest.py` provides an `auth_headers` fixture that reads `ADMIN_API_KEY` / `API_AUTH_TOKEN` / `BACKEND_API_KEY` from env — tests that POST to protected routes need it when auth is enabled.
- Frontend has no test suite; the CI gate is `npm run build` + `npm run lint`.

## Code Style Guidelines

- **Python:** Ruff is the enforced linter (line-length 420, target py311). Only high-confidence rules are selected: `E9`, `F401`, `F811`, `F821`, `F841`, `W6`; `E501` is ignored. Black and isort (black profile) are configured for formatting but not enforced in CI. Ruff excludes `backend/alembic/versions`, `scripts`, `scratch`. Per-file ignores: `backend/tests/*` may have unused imports/variables.
- **TypeScript:** strict type-check via `tsc --noEmit` (`npm run lint`); ESLint is not configured.
- Follow the existing module organization: API endpoints in `backend/routes/`, business logic in `backend/services/`, LLM personas in `backend/agents/`. New background loops should be registered under the task-restart supervisor in `backend/main.py`.
- Logging: use `structlog`/`logging` patterns already present; JSON logs are enabled via `JSON_LOGS=true`.

## Runtime Architecture (what agents must not break)

- `backend/main.py` starts supervised background tasks (auto-restart after 10s on crash): wallet poller, order poller, trading loop, sentiment loop, trade-memory recorder, skill miner. Graceful shutdown cancels and awaits all of them.
- The trading loop enforces a fail-closed stack of safety gates before any entry: risk guard (drawdown/daily-loss/position limits), kill switch (equity floor), emergency position manager, broker position sync, margin gate, symbol quality gate, and an execution lock serializing order placement.
- Per-symbol pipeline: combined technical strategy → Kronos foundation-model gate → weighted AI opinion layer (8+ agents) → LLM risk reviewer → order execution via the unified router (paper fill engine or live Binance/cTrader).
- Key env flags live in `.env` / `docker-compose.yml`: `PAPER_TRADING`, `ACTIVE_BROKER`, `DRY_RUN_ALL`, `DISABLE_RISK_GUARD` (must stay `false` in prod), `TRADING_SYMBOLS`, `SL_ATR_MULT`/`TP_ATR_MULT`/trailing params, `MAKER_ENTRY_ENABLED`, risk limits (`RISK_MAX_DRAWDOWN_PCT`, etc.).

## CI/CD and Deployment

- **GitHub Actions** (`.github/workflows/`):
  - `backend-ci.yml` — on backend changes: pip install from `backend/requirements.txt`, `ruff check backend`, `pytest backend/tests`.
  - `frontend-ci.yml` — on frontend changes: `npm ci`, `npm run build`, `npm run lint`.
  - `vps-deploy.yml` — **manual-only** (`workflow_dispatch`); requires typing `DEPLOY` and the `production` environment. Deliberately not triggered on push — this is a live-money system. Deploys are serialized via a concurrency group and fail closed if `SSH_PRIVATE_KEY` is missing.
- Deploy target: Hostinger VPS, project dir `/root/ai-trading-platform-v3`. The workflow pulls `main` and runs `scripts/hostinger_vps_apply.sh`. `deploy.sh` and `scripts/vps_*` are operator scripts for the same host.

## VPS Access & Deployment Guidelines

Deployments should be executed via standard CI/CD pipelines or automated deploy hooks.
Direct SSH access requires private keys configured in secure runner environments.

### Required Secrets
- `SSH_HOST`: Target server hostname or IP address (configured in secure environment variables only)
- `SSH_USER`: Deployment user (use a dedicated non-root deploy user with restricted Docker permissions)
- `SSH_PRIVATE_KEY`: Deployment key with pass-phrase protection

Never commit credentials, private keys, or raw IP addresses into git-tracked repositories.

- `SSH_HOST` - VPS IP address
- `SSH_USER` - `root`
- `SSH_PRIVATE_KEY` - ed25519 private key (may be stored as single line with spaces; `scripts/ssh_vps_remote.sh` reformats OpenSSH keys automatically)

### Quick SSH Test

```python
import os, subprocess, tempfile

raw = os.environ.get("SSH_PRIVATE_KEY", "").strip()
begin = "-----BEGIN OPENSSH PRIVATE KEY-----"
end = "-----END OPENSSH PRIVATE KEY-----"
body = raw.replace(begin, "").replace(end, "").strip().replace(" ", "\n")
key_content = f"{begin}\n{body}\n{end}\n"

with tempfile.NamedTemporaryFile(mode='w', suffix='_key', delete=False) as f:
    f.write(key_content)
    key_path = f.name
os.chmod(key_path, 0o600)

host = os.environ["SSH_HOST"]
user = os.environ["SSH_USER"]
cmd = ["ssh", "-i", key_path, "-o", "StrictHostKeyChecking=accept-new",
       "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
       f"{user}@{host}", "hostname && docker ps --format 'table {{.Names}}\t{{.Status}}'"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout or r.stderr)
os.unlink(key_path)
```

### SSH Troubleshooting

If SSH fails with "Permission denied", check VPS auth log:
```bash
tail -20 /var/log/auth.log | grep ssh
```

Common issues:
| Error | Fix |
|-------|-----|
| `bad ownership or modes for directory /root` | `chown root:root /root && chmod 700 /root` |
| `bad ownership or modes for file authorized_keys` | `chmod 600 /root/.ssh/authorized_keys` |
| Key not in authorized_keys | Add public key to `/root/.ssh/authorized_keys` |
| Key stored as single line | Script handles this automatically |

### VPS Services

The VPS runs these Docker containers:
- `ai-trading-backend` - FastAPI backend on port 8001
- `ai-trading-nginx` - Reverse proxy on port 8081
- `ai-trading-litellm` - LLM proxy
- `ai-trading-redis` - Cache
- `vps-influxdb` - Time series DB
- `vps-qdrant` - Vector DB
- `grafana-*` - Monitoring
- `n8n` - Workflows at `/docker/n8n` (`127.0.0.1:5678`). Assistant sandbox overlay: `docker-compose.sandbox.yml` (privileged runner, not published). Search: `http://ai-trading-searxng:8080`.

### Testing Backend API on the VPS

```bash
curl -sf "http://${SSH_HOST}:8001/health"
curl -sf "http://${SSH_HOST}:8001/openapi.json" | head -c 500
```

## Security Considerations

- **Secrets:** all credentials live in `.env` (gitignored; `.env.example` documents every variable). Never commit `.env`, API keys, or SSH keys. Broker keys, LLM provider keys, `ADMIN_API_KEY`, `QDRANT_API_KEY`, and `INFLUXDB_TOKEN` are the sensitive ones.
- **Admin auth:** sensitive routes are protected by an API-key check (`X-API-Key` header, see `backend/security.py` and `validate_admin_request`); auth primitives use PyJWT + bcrypt/passlib.
- **Production ports are bound to `127.0.0.1`** in docker-compose (except nginx on 8081) — keep it that way; public exposure goes through nginx.
- **Fail-closed safety:** risk guard, kill switch, and margin gates default ON. `DISABLE_RISK_GUARD=true` exists only for explicit local testing and must never reach the VPS.
- Sentry DSN validation skips placeholder values so a bad `SENTRY_DSN` cannot crash startup.
- This project is for educational/research purposes; see the disclaimer in `README.md`.
