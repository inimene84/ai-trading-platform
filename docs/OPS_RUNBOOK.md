# Ops Runbook — QuantumTrade Pro (Hostinger VPS)

> **This system trades real money on the VPS.** The live-trading posture is
> **authoritative in the VPS `.env`**, not hardcoded in compose. Prod compose
> interpolates `${PAPER_TRADING:-true}`, `${DRY_RUN_ALL:-true}`,
> `${TRADING_MODE:-paper}` — so the file on `/root/ai-trading-platform-v3/.env`
> is what the container actually runs. Every command below can still move
> real capital if that `.env` is live. Read the [Safety warnings](#safety-warnings)
> first.
>
> **Soak risk before the next VPS pull (2026-09-16).** `main` recently merged
> Dependabot majors `fastapi[standard]>=0.141.1` (PR #74) and `bcrypt>=5.0.0`
> (PR #77). These have not been soaked on the live trading host. **Pin the
> running deploy tag / image digest before `git pull`** so a FastAPI or
> password-hash regression can be rolled back without taking the loop through
> an untested auth stack.

Scope: this document inventories the operational scripts and entrypoints that
exist in this repo and describes what each one *actually does*, based on reading
the scripts. It does **not** describe observed server state — there is no SSH
access from here.

- Target host: `72.60.18.113`, user `root`
- Deploy dir on VPS: `/root/ai-trading-platform-v3`
- Canonical deploy script: `scripts/hostinger_vps_apply.sh`
- Companion docs already in repo: `docs/ops/PRODUCTION_DEPLOY.md`,
  `docs/ops/HOSTINGER_QUICKFIX.md`, `scripts/README.md`, `DEPLOYMENT.md`

---

## 1. Architecture / deploy overview

The production stack is Docker Compose (`docker-compose.prod.yml`), started on
the VPS. Services and container names, with host port bindings as declared in
the compose file:

| Service | Container | Host binding | Notes |
|---|---|---|---|
| `backend` | `ai-trading-backend` | `127.0.0.1:8001 -> 8000` | FastAPI (`uvicorn backend.main:app`), built from `Dockerfile.backend` |
| `litellm` | `ai-trading-litellm` | `127.0.0.1:4001 -> 4000` | LLM router, config `litellm-config.yaml` |
| `nginx` | `ai-trading-nginx` | `8081 -> 80` | Serves `frontend/dist` + proxies `/api/backend/*` and `/grafana/*` |
| `qdrant` | `vps-qdrant` | `127.0.0.1:6333` | Vector store (`crypto-news` collection) |
| `influxdb` | `vps-influxdb` | `127.0.0.1:8086` | Metrics; buckets created by `scripts/ensure_influx_buckets.sh` |
| `grafana` | `ai-trading-grafana` | via nginx `/grafana/` | An unrelated older Grafana also exists on `:3000` per `scripts/vps_rewire_old_grafana.sh` |
| `sentry-watchdog` | `ai-trading-sentry-watchdog` | — | Heartbeat/halt watchdog (`sentry_watchdog/Dockerfile`) |
| `mcp-server` | `ai-trading-mcp` | `127.0.0.1:9100` | MCP server (`mcp_server/Dockerfile`) |

Docker network: `trading-net` (created by the deploy script if missing).

The only supported public entry point is nginx on `:8081`; the backend itself is
bound to loopback (`127.0.0.1:8001`).

---

## 2. How the code actually gets to the VPS today

**Short version: it does not, automatically. Deploys are hand-run on the VPS.**

### GitHub workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/backend-ci.yml` | `push` / `pull_request` on `backend/**`, `pyproject.toml` | Python 3.12, `pip install -r backend/requirements.txt`, `ruff check backend`, `pytest backend/tests` |
| `.github/workflows/frontend-ci.yml` | `push` / `pull_request` on `frontend/**` | Node 22, `npm ci`, `npm run build`, `npm run lint` |
| `.github/workflows/vps-deploy.yml` | **`workflow_dispatch` only** | SSHes to the VPS and runs `scripts/hostinger_vps_apply.sh` |

`vps-deploy.yml` is deliberately manual (comment at lines 3–6). Its gates:

- `on: workflow_dispatch` with a required `confirm` input (default `"no"`), and
  `if: ${{ github.event.inputs.confirm == 'DEPLOY' }}` (line 26) — the job is
  skipped unless the operator literally types `DEPLOY`.
- `environment: production` (line 25), so a GitHub environment with required
  reviewers can gate it.
- `concurrency: group: vps-deploy, cancel-in-progress: false` (lines 16–18) —
  deploys are serialized.
- A **fail-closed step** (lines 31–38): if `secrets.SSH_PRIVATE_KEY` is empty
  the job errors out before touching the VPS.

### The SSH-secret gap

`SSH_PRIVATE_KEY` is **not configured** in the repository secrets (also stated
in `docs/ops/PRODUCTION_DEPLOY.md`, "Current state"). Consequences:

1. Pushes to `main` run CI and land in the repo, but **never reach the VPS**.
2. Even a manual `Run workflow` → `DEPLOY` run fails at the "Fail closed if SSH
   key is missing" step. Nothing is deployed, nothing is broken.
3. Therefore **the VPS only changes when a human runs the deploy on the box**
   (Hostinger browser terminal or SSH) — the repo and the running system can
   drift arbitrarily far apart.

To close the gap: add `SSH_PRIVATE_KEY` (and optionally `SSH_HOST`, `SSH_USER`)
under Settings → Secrets and variables → Actions. `SSH_HOST`/`SSH_USER` fall
back to the hardcoded `'72.60.18.113'` / `'root'` (lines 42, 46–47).

### Note on the `github.repository` guard

`docs/ops/PRODUCTION_DEPLOY.md` ("Repo name caveat") refers to a
`github.repository` guard, but **no such condition currently exists** in any
file under `.github/workflows/` (verified by grep). The only job-level guard in
`vps-deploy.yml` is the `confirm == 'DEPLOY'` check plus `environment:
production`. If a fork-safety guard is wanted, it still has to be added — and
note the repository name literally ends with a period
(`inimene84/ai-trading-platform.`), which any such condition must match exactly.

### What the deploy actually executes

`vps-deploy.yml` → over SSH → in `/root/ai-trading-platform-v3`:
`git fetch/checkout/pull main` → `./scripts/hostinger_vps_apply.sh`, which then:

1. `vps_ssh_hygiene` (from `scripts/lib/vps_ssh_hygiene.sh`) — fixes `/root`
   perms and **appends a hardcoded SSH public key to
   `/root/.ssh/authorized_keys`**.
2. Requires `.env`; requires `ADMIN_API_KEY`, `LITELLM_API_KEY`,
   `QDRANT_API_KEY`; aborts if `PAPER_TRADING=false` + `DRY_RUN_ALL=false`
   without `CONFIRM_LIVE_DEPLOY=true`.
3. **Rewrites ~20 risk keys in `.env`** (`_upsert_env`): drawdown/daily-loss
   gates, pyramid limits, trailing-stop geometry, `SYMBOL_BLACKLIST`, and bumps
   `TRADING_KILL_FLOOR_USDT` 20 → 65. Also strips ADA/ARB/DOGE/APT from
   `TRADING_SYMBOLS`.
4. `npm ci && npm run build` in `frontend/`; aborts if `frontend/dist/index.html`
   is missing.
5. `docker network create trading-net` (idempotent).
6. `docker compose -f docker-compose.prod.yml up -d --build backend litellm
   nginx influxdb grafana qdrant mcp-server`.
7. `./scripts/ensure_influx_buckets.sh`, health-wait on `:8001/health`,
   `docker restart ai-trading-nginx`, smoke tests, Grafana datasource fix.
8. Posts runtime LLM toggles, then **stops and restarts the live trading loop**
   with the symbols from `.env` at `interval_minutes: 15`, `strategy: combined`.

---

## 3. Script inventory

Danger levels: **HIGH** = can move money, halt trading, rewrite `.env`, or kill
processes/containers. **MED** = restarts services or mutates config/state.
**LOW** = read-only or local-dev only.

### Deploy / apply (production)

| Path | Purpose | Canonical? | Danger |
|---|---|---|---|
| `scripts/hostinger_vps_apply.sh` | Full production deploy on `main`: SSH hygiene, `.env` sanity + risk-key rewrite, frontend build, `docker compose up -d --build` (7 services), Influx buckets, health/smoke tests, Grafana fix, **stop+start trading loop** | **YES** — referenced by `vps-deploy.yml`, `vps_remote_oneliner.sh`, `ssh_vps_remote.sh`, `vps_deploy_p0_oneliner.sh`, `vps_deploy_risk_reviewer_fix.sh`, and the ops docs | **HIGH** |
| `scripts/lib/vps_ssh_hygiene.sh` | Sourced by the deploy; chmods `/root`, `/root/.ssh`, and appends a hardcoded `ssh-ed25519` key to `authorized_keys` | yes (library) | **HIGH** (grants persistent SSH access) |
| `scripts/vps_remote_oneliner.sh` | Paste-on-VPS wrapper: `git pull main` then run the canonical apply (falls back to `docker compose restart nginx`) | yes (wrapper) | MED |
| `scripts/vps_bootstrap_and_deploy.sh` | `curl \| bash` bootstrap that fetches and execs `vps_remote_oneliner.sh` from raw.githubusercontent | wrapper | **HIGH** (pipes remote code into root shell) |
| `scripts/ssh_vps_remote.sh` | Runs the deploy *from* a workstation/agent using `SSH_PRIVATE_KEY`/`SSH_PASSWORD`; default host `72.60.18.113` | yes (client side) | MED |
| `scripts/vps_deploy_p0_oneliner.sh` | Deprecated shim → `exec hostinger_vps_apply.sh` | no — retire | LOW (delegates) |
| `scripts/vps_deploy_risk_reviewer_fix.sh` | Deprecated shim → `exec hostinger_vps_apply.sh` | no — retire | LOW (delegates) |
| `deploy.sh` (repo root) | Older standalone deploy: installs Docker via `curl \| sh` if absent, `docker-compose -f docker-compose.prod.yml down`, `build --no-cache`, `up -d`, health-checks `localhost:8000` | **no** — superseded; port and compose-v1 syntax no longer match prod (`127.0.0.1:8001`) | **HIGH** (full `down` of the live stack) |
| `scripts/deploy_kie_sonnet.sh` | `git pull main` + `docker compose up -d --build backend litellm`, verifies LiteLLM/AI routing | partial (LLM-only deploy) | MED |
| `update_vps_env.py` (repo root) | One-off: SSHes in, rewrites 5 LLM keys in the remote `.env`, `git pull`, restarts backend. Referenced by nothing | **no** — stale | **HIGH** (remote `.env` rewrite + restart) |

### Local dev entrypoints

| Path | Purpose | Canonical? | Danger |
|---|---|---|---|
| `start.sh` | `./start.sh` = native (uvicorn `:8000` + Vite); `./start.sh docker` = `docker compose down` + `up -d --build` using **`docker-compose.yml`** | yes for local | MED–**HIGH** — never run on the VPS; local compose interpolates `${TRADING_MODE:-paper}` |
| `run.sh` | Poetry-based launcher: backend `:8080`, frontend `:3000`, traps Ctrl-C | duplicate of `start.sh` native mode | LOW |
| `start_docker_local.sh` / `.bat`, `run.bat` | Windows/local Docker helpers | duplicates | LOW |
| `Dockerfile.backend`, `mcp_server/Dockerfile`, `sentry_watchdog/Dockerfile` | Image builds (backend is `python:3.11-slim`, `uv pip install -r requirements.txt`) | yes | LOW |
| `docker-compose.yml` / `docker-compose.prod.yml` | Dev vs prod stack; prod interpolates paper/live flags from the VPS `.env` | `prod` is canonical on VPS | **HIGH** |

There is **no Makefile and no systemd unit** in the repo. `scripts/vps_realtime_watchdog.sh`
documents itself as a **cron** job instead.

### Status / health / diagnostics (read-only unless noted)

| Path | Purpose | Canonical? | Danger |
|---|---|---|---|
| `scripts/vps_health_check.sh` | `/health`, `/sentry/status`, key trading-loop fields, Binance wallet, error grep over `ai-trading-backend` logs, watchdog logs, market alerts | **YES** for "is it healthy?" | LOW |
| `scripts/vps_live_trading_check.sh` | Deeper live check: containers, `/trading/status`, wallet, open positions, live env flags, 2h log grep, nginx checks, recent Influx write | **YES** for "is it trading safely?" | LOW (reads `INFLUXDB_TOKEN` from `.env`) |
| `scripts/vps_check_protection.sh` | `docker cp` + run `scripts/vps_check_protection.py` in the backend container — SL/TP protection status | yes | LOW |
| `scripts/verify_endpoints.sh` | Container list + backend/Influx/Qdrant endpoint tests; POSTs a test sentiment record | partial | LOW |
| `scripts/vps_diagnose_providers.sh` | Reports which LLM/Telegram keys are set/placeholder, then live-calls Telegram, OpenRouter, xAI, Gemini, Kie | yes | LOW (sources secrets into env; may print API responses) |
| `scripts/check_logs.sh`, `scripts/check_qdrant_logs.sh` | One-line `docker logs ai-trading-backend \| grep` helpers | trivial duplicates | LOW |
| `scripts/qdrant_debug.sh` | Not a script so much as a command sheet (health, create collection, archive test, logs) | no — notes | LOW |
| `scripts/test_archive.sh`, `scripts/test_archive_full.sh` | POST a synthetic 1536-dim embedding to `/api/news/archive` | duplicate pair | LOW (writes test data) |
| `docs/ops/verify_fix.sh` | Post-deploy curl checks against `:8000` / `:8081` | stale (wrong backend port) | LOW |
| `docs/ops/vps-diagnose.sh` | Prints `/root/.ssh/authorized_keys`, **restarts `ssh.socket`**, docker ps, Qdrant check, then `git pull` + `docker compose up -d --build backend` | no — stale | **HIGH** (SSH restart = lockout risk; also deploys) |

### Trading-loop / risk control (all touch live money)

| Path | Purpose | Canonical? | Danger |
|---|---|---|---|
| `scripts/vps_start_loop.sh` | POST `/trading/loop/start` with `.env` symbols, 15-min interval; prints status. **No `X-API-Key` header** | yes (simplest start) | **HIGH** |
| `scripts/vps_apply_recovery_mode.sh` | "Green-day" mode: rewrites ~15 risk keys, **market-closes every open position outside BTC/ETH/SOL/BNB**, recreates backend, `/sentry/resume`, stop+start loop, prints RiskConfig | yes for de-risking | **HIGH** (realizes P&L immediately) |
| `scripts/vps_apply_quality_expand.sh` | Opposite direction: 6 max positions, wider symbol set, `RISK_PER_TRADE_PCT=0.007`, daily loss 3%; recreates backend, resumes sentry, stop+start loop | yes for re-expanding | **HIGH** (loosens risk limits) |
| `scripts/vps_apply_protection_fix.sh` | `docker cp` two service files into the running container, `docker restart ai-trading-backend`, run `vps_restore_protection.py`, verify | one-off | **HIGH** (hot-patches container; drifts from image) |
| `scripts/vps_apply_binance_poll.sh` | Sets `BINANCE_*_POLL_INTERVAL=180` in `.env`, restarts backend | one-off | MED |
| `scripts/vps_sentry_resume.sh` | POST `/sentry/resume` with `SENTRY_WATCHDOG_TOKEN` | yes | **HIGH** (re-enables trading after a halt) |
| `scripts/vps_realtime_watchdog.sh` | Cron watchdog: on failed `/health` or nginx check, `docker compose -f docker-compose.prod.yml restart backend` / `nginx`; only *logs* if the loop is stopped | yes | MED (auto-restarts prod containers) |
| `scripts/close_all_positions.py` | Emergency: MARKET-closes all DB open/filled trades on Binance Futures and updates rows | yes (break-glass) | **HIGH** |

### Observability plumbing (Influx / Grafana / Qdrant / n8n)

| Path | Purpose | Canonical? | Danger |
|---|---|---|---|
| `scripts/ensure_influx_buckets.sh` | Idempotent create **or update** of the 6 buckets (`trading-system` **90d** … `news-sentiment` 90d) in container `vps-influxdb`. Snapshot Influx before changing retention on a live host. | **YES** — called by the deploy | LOW |
| `docs/ops/setup_influx_buckets.sh` | Same job, hardcoded token (now `***REMOVED***`), 5 buckets | no — superseded, **retire** | MED |
| `docs/ops/fix_influxdb_complete.sh` | Buckets + `docker compose down backend` + rebuild | no — superseded, **retire** | **HIGH** (stops backend) |
| `docs/ops/fix_influx_token.sh`, `docs/ops/fix_influxdb_token.sh` | Sign in to Influx with admin creds from env, mint a token, print it / `sed -i` it into `.env`, restart backend | no — near-duplicates | **HIGH** (`.env` rewrite; prints token prefix) |
| `scripts/fix_grafana_influx.sh` | Repoints every Grafana InfluxDB datasource at `http://vps-influxdb:8086` with the org/token from `.env`, then calls `deploy_grafana.sh` | **YES** — called by the deploy | MED |
| `scripts/deploy_grafana.sh` | Uploads `grafana/dashboards/ai_trading_dashboard.json` with `overwrite: true`; default auth `admin:admin` | yes (helper) | MED (overwrites dashboard) |
| `scripts/vps_fix_grafana_datasources.sh` | Reconciles `INFLUXDB_ORG` in `.env`, ensures buckets, fixes the **old** Grafana on `:3000` using a password read from `/docker/grafana-k9xk/.env`, rewrites `GRAFANA_URL`, recreates backend | overlapping one-off | **HIGH** (`.env` rewrite + reads another stack's secret) |
| `scripts/vps_rewire_old_grafana.sh` | Connects `grafana-k9xk-grafana-1` to `trading-net` and runs the datasource fix against `:3000` | one-off | MED |
| `scripts/vps_fix_grafana_remote.sh` | Read-only comparison of old/new Grafana datasources and Influx buckets | diagnostic | LOW |
| `scripts/vps_probe_old_grafana.sh` | Tries `GRAFANA_PROBE_AUTHS` credential pairs against `:3000` | diagnostic | LOW (credential probing) |
| `scripts/vps_restore_influx_history.sh` | `docker compose up -d --force-recreate influxdb`, query 30d, rewire old Grafana | one-off | MED |
| `scripts/create_qdrant_collection.sh`, `docs/ops/init_qdrant.sh` | `PUT /collections/crypto-news` (1536-dim, Cosine) | duplicates; keep one | LOW |
| `docs/ops/fix_qdrant.sh` | `fuser -k 6333/tcp`, `docker rm -f vps-qdrant qdrant-13fq-qdrant-1`, `git pull`, `docker compose up -d qdrant`, create collection | no — **retire** | **HIGH** (kills processes, removes containers) |
| `docs/ops/deploy_qdrant_fix.sh` | Bigger version of the same: `fuser -k`, `docker rm -f` every `qdrant`-named container, `docker compose down --remove-orphans`, `chmod 777 ./qdrant-storage`, redeploy | no — **retire** | **HIGH** (`down --remove-orphans` stops the whole stack) |
| `docs/ops/VPS_FIX_SCRIPT.sh`, `docs/ops/VPS_FULL_FIX.sh` | Historical network fixes referencing the container name `qdrant-13fq-qdrant-1` and rebuilding nginx/backend | no — **retire** | MED–HIGH |
| `scripts/connect_n8n_network.sh` | Attaches the n8n container to `trading-net`, restarts it, connectivity tests (contains a `YOUR_TOKEN_HERE` placeholder and a hardcoded Influx org ID) | one-off | MED |
| `scripts/fix_n8n_db_corruption.sh` | Stops n8n, backs up `database.sqlite`, `CAST(nodes AS TEXT)` repair, `chown`, restarts, prunes old backups with `rm -f` | yes (n8n-specific) | MED (unrelated to trading) |

### Repo maintenance

| Path | Purpose | Canonical? | Danger |
|---|---|---|---|
| `scripts/cleanup_stale_branches.sh` | `git push origin --delete` for 9 hardcoded merged branches, then lists unmerged ones | yes | MED (deletes remote branches) |

### The Python `vps_*.py` fleet

`scripts/` also contains ~45 `vps_*.py` operator scripts that SSH in via
`scripts/vps_ssh_common.py` (default host `72.60.18.113`, key
`~/.ssh/id_vps_bot`). `scripts/README.md` already tags them
active / one-off / deprecated. The ones to know:

- Deprecated deploys (use `hostinger_vps_apply.sh` instead):
  `vps_deploy_latest.py`, `vps_deploy_universe.py`, `vps_clean_deploy.py`
  (the last one runs `git checkout -- .` on the VPS, discarding local drift, and
  strips a short `GOOGLE_API_KEY` from `.env` — **HIGH**).
- Break-glass / risk: `vps_restore_protection.py`, `vps_fix_naked_positions.py`,
  `vps_resume_trading.py`, `vps_restart_loop.py`,
  `vps_reconcile_stale_trades.py`, `repair_corrupt_trades.py` — all **HIGH**.
- Read-only checks: `vps_check_exposure.py`, `vps_check_binance_status.py`,
  `vps_verify_health.py`, `vps_env_audit.py`, `vps_error_logs.py`.

---

## 4. Overlap / duplication map

| Job | Scripts that do it | Canonical |
|---|---|---|
| Full deploy | `scripts/hostinger_vps_apply.sh`, `deploy.sh`, `scripts/vps_deploy_p0_oneliner.sh`, `scripts/vps_deploy_risk_reviewer_fix.sh`, `scripts/vps_deploy_latest.py`, `vps_deploy_universe.py`, `vps_clean_deploy.py`, `docs/ops/vps-diagnose.sh`, `update_vps_env.py` | `scripts/hostinger_vps_apply.sh` |
| Remote invocation | `scripts/ssh_vps_remote.sh`, `scripts/vps_remote_oneliner.sh`, `scripts/vps_bootstrap_and_deploy.sh`, `.github/workflows/vps-deploy.yml` | workflow (once `SSH_PRIVATE_KEY` exists), else `vps_remote_oneliner.sh` on the box |
| Influx buckets | `scripts/ensure_influx_buckets.sh`, `docs/ops/setup_influx_buckets.sh`, `docs/ops/fix_influxdb_complete.sh` | `scripts/ensure_influx_buckets.sh` |
| Influx token repair | `docs/ops/fix_influx_token.sh`, `docs/ops/fix_influxdb_token.sh` | neither is current; fold into one documented procedure |
| Grafana datasources | `scripts/fix_grafana_influx.sh`, `scripts/vps_fix_grafana_datasources.sh`, `scripts/vps_rewire_old_grafana.sh`, `scripts/vps_fix_grafana_remote.sh`, `scripts/vps_probe_old_grafana.sh` | `scripts/fix_grafana_influx.sh` |
| Qdrant collection | `scripts/create_qdrant_collection.sh`, `docs/ops/init_qdrant.sh`, `docs/ops/fix_qdrant.sh`, `docs/ops/deploy_qdrant_fix.sh`, `scripts/qdrant_debug.sh` | `scripts/create_qdrant_collection.sh` |
| Health check | `scripts/vps_health_check.sh`, `scripts/vps_live_trading_check.sh`, `scripts/verify_endpoints.sh`, `docs/ops/verify_fix.sh` | `vps_health_check.sh` (quick) + `vps_live_trading_check.sh` (deep) |
| Log grep | `scripts/check_logs.sh`, `scripts/check_qdrant_logs.sh` | inline `docker logs` (see §5) |
| Archive smoke test | `scripts/test_archive.sh`, `scripts/test_archive_full.sh` | `test_archive_full.sh` |
| Local dev launch | `start.sh`, `run.sh`, `start_docker_local.sh`, `run.bat`, `start_docker_local.bat` | `start.sh` |

---

## 5. Common operations

All VPS commands assume `ssh root@72.60.18.113` (or the Hostinger browser
terminal) and `cd /root/ai-trading-platform-v3`.

### Deploy the current `main`

```bash
cd /root/ai-trading-platform-v3
git fetch origin main && git checkout main && git pull origin main
chmod +x scripts/hostinger_vps_apply.sh scripts/lib/vps_ssh_hygiene.sh scripts/ensure_influx_buckets.sh
./scripts/hostinger_vps_apply.sh
```

Expect it to rewrite risk keys in `.env`, rebuild the frontend, rebuild/recreate
7 containers, and **stop and restart the trading loop** at the end.

Via GitHub (only works once `SSH_PRIVATE_KEY` is set): Actions → **VPS Deploy**
→ Run workflow → type `DEPLOY`.

### Restart just the backend (no rebuild)

```bash
docker compose -f docker-compose.prod.yml restart backend
# or, to pick up changed .env / image:
docker compose -f docker-compose.prod.yml up -d backend
curl -sf http://127.0.0.1:8001/health
```

### Restart nginx / refresh the dashboard upstream

```bash
docker restart ai-trading-nginx
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8081/api/backend/health
```

### Check status

```bash
./scripts/vps_health_check.sh          # health, sentry, loop, wallet, error grep
./scripts/vps_live_trading_check.sh    # + positions, live env flags, Influx writes
docker ps --format 'table {{.Names}}\t{{.Status}}'
curl -sf http://127.0.0.1:8001/trading/loop/status
```

### View logs

```bash
docker logs ai-trading-backend --tail 200
docker logs ai-trading-backend --since 2h 2>&1 | grep -E 'Trading Cycle|ERROR|Exception|filled @|KILL SWITCH'
docker logs ai-trading-sentry-watchdog --tail 50
docker logs ai-trading-nginx --tail 50
docker compose -f docker-compose.prod.yml logs -f backend
```

### Start / stop the trading loop

```bash
ADMIN=$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)

# stop
curl -sf -X POST http://127.0.0.1:8001/trading/loop/stop \
  -H 'Content-Type: application/json' -H "X-API-Key: $ADMIN"

# start (symbols taken from .env, as hostinger_vps_apply.sh does)
./scripts/vps_start_loop.sh
```

### Resume after a sentry halt

```bash
SENTRY_WATCHDOG_TOKEN=<token from .env> ./scripts/vps_sentry_resume.sh
# or, with the admin key:
curl -sf -X POST http://127.0.0.1:8001/sentry/resume \
  -H "X-API-Key: $ADMIN" -H 'Content-Type: application/json' \
  -d '{"note":"manual resume"}'
```

### De-risk fast

```bash
./scripts/vps_apply_recovery_mode.sh     # majors only + market-closes other legs
# absolute break-glass, closes everything:
docker exec -e PYTHONPATH=/app ai-trading-backend python3 /app/scripts/close_all_positions.py
```

(`close_all_positions.py` lives in `scripts/`, which is **not** copied into the
backend image by `Dockerfile.backend` — `docker cp scripts/close_all_positions.py
ai-trading-backend:/tmp/` first, mirroring how
`scripts/vps_check_protection.sh` ships its Python helper.)

### Roll back

```bash
cd /root/ai-trading-platform-v3
git log --oneline -10
git checkout <good-commit>
docker compose -f docker-compose.prod.yml up -d --build backend litellm nginx
curl -sf http://127.0.0.1:8001/health
```

> **Do not** follow a `git checkout <good-commit>` with
> `./scripts/hostinger_vps_apply.sh`: step 1 of that script runs
> `git checkout main && git pull origin main`, which silently undoes the
> rollback and redeploys the bad commit. `docs/ops/PRODUCTION_DEPLOY.md`
> currently documents that sequence — treat the compose rebuild above as the
> correct rollback until that doc is fixed.

---

## 6. Safety warnings

1. **VPS `.env` is authoritative.** `docker-compose.prod.yml` interpolates
   `${PAPER_TRADING:-true}`, `${DRY_RUN_ALL:-true}`, `${TRADING_MODE:-paper}`
   (and the matching sentry-watchdog flags). Compose no longer hardcodes
   live-trading booleans that silently override `.env`. The VPS file at
   `/root/ai-trading-platform-v3/.env` is what the container runs. Do not
   change those compose defaults in a drive-by edit. The
   `CONFIRM_LIVE_DEPLOY=true` gate in `hostinger_vps_apply.sh` still inspects
   the same keys — they now match the container. **Before the next pull**,
   pin the running deploy tag: FastAPI ≥0.141.1 and bcrypt ≥5.0.0 landed on
   `main` and have not been soaked on this host.
2. **The deploy restarts the trading loop.** Step 10 of
   `hostinger_vps_apply.sh` stops the loop and starts a fresh one. Never deploy
   in the middle of managing an open position without checking
   `./scripts/vps_live_trading_check.sh` first.
3. **The deploy rewrites `.env`.** Any manual tuning of drawdown, daily-loss,
   pyramid, trailing-stop, blacklist, or kill-floor values is reset to the
   values hardcoded in the script. Tune via the script (or via
   `vps_apply_recovery_mode.sh` / `vps_apply_quality_expand.sh` afterwards), not
   by hand-editing `.env`.
4. **`DISABLE_RISK_GUARD`** defaults to `false` in both compose files. Never set
   it to `true` on the VPS.
5. **Position-closing scripts realize losses immediately** at market:
   `vps_apply_recovery_mode.sh` (anything outside BTC/ETH/SOL/BNB) and
   `close_all_positions.py` (everything).
6. **`scripts/vps_start_loop.sh` sends no `X-API-Key`** — it only works if
   `ADMIN_API_KEY` protection is absent or the endpoint is unauthenticated.
   Treat that as a finding, not a feature.
7. **Never run `deploy.sh`, `start.sh docker`, or the `docs/ops/*.sh` fix
   scripts on the VPS.** They `down` the stack, `fuser -k` ports, `docker rm -f`
   containers, restart `ssh.socket`, or target container names
   (`qdrant-13fq-qdrant-1`) from a past topology.
8. **Curl-piped bootstrap.** `scripts/vps_bootstrap_and_deploy.sh` executes code
   fetched from `raw.githubusercontent.com` as root. Convenient, but it means a
   repo compromise is a root compromise.
9. **Every deploy grants SSH access.** `scripts/lib/vps_ssh_hygiene.sh` appends a
   hardcoded public key to `/root/.ssh/authorized_keys` unless
   `CLOUD_AGENT_PUBKEY` is overridden.

---

## 7. Hardcoded values that should not be in git

Values are redacted here; see the file and line for the full string.

| File:line | What | Assessment |
|---|---|---|
| `scripts/lib/vps_ssh_hygiene.sh:4` | Default `CLOUD_AGENT_PUBKEY` = `ssh-ed25519 AAAAC3Nz...j1d7t valgutom@gmail.com`, appended to `/root/.ssh/authorized_keys` on every deploy | Public key (not secret) but it is a **standing root-access grant plus a personal email**. Move to a required env var / GitHub secret and audit `authorized_keys`. |
| `docs/ops/setup_influx_buckets.sh:5` | `TOKEN="***REMOVED***"` | A real InfluxDB admin token was committed here and later scrubbed in place. Scrubbing does not remove it from upstream git history — **rotate the InfluxDB token**. |
| `docs/ops/fix_influxdb_complete.sh:8` | `TOKEN="***REMOVED***"` | Same finding, same rotation. |
| `scripts/fix_grafana_influx.sh:18` | Influx org id `819d4...1bd6` compared as a literal | Org ID, low sensitivity; still config that belongs in `.env`. |
| `scripts/connect_n8n_network.sh:35-36` | Same org id in a write URL plus `Authorization: Token YOUR_TOKEN_HERE` | Placeholder, not a leak, but the script cannot work as written. |
| `scripts/deploy_grafana.sh:6`, `scripts/fix_grafana_influx.sh:7`, `scripts/vps_fix_grafana_remote.sh:13,21,29`, `scripts/vps_probe_old_grafana.sh:6` | Default credentials `admin:admin` | Remove the default; require explicit credentials. |
| `scripts/vps_rewire_old_grafana.sh:4`, `scripts/vps_restore_influx_history.sh:15`, `scripts/vps_fix_grafana_datasources.sh:28` | Read `GF_SECURITY_ADMIN_PASSWORD` out of `/docker/grafana-k9xk/.env` (an unrelated stack) | Cross-stack secret read; document or drop. |
| `scripts/vps_fix_grafana_datasources.sh:50` | Echoes "Login: admin / (see /docker/grafana-k9xk/.env)" | Points an operator at another stack's secret file. |
| VPS IP `72.60.18.113` — `.github/workflows/vps-deploy.yml:42,46`; `scripts/ssh_vps_remote.sh:7`; `scripts/vps_ssh_common.py:19`; `update_vps_env.py:18`; `scripts/vps_fix_grafana_datasources.sh:33,49`; `scripts/vps_restore_influx_history.sh:17`; `docs/ops/VPS_FIX_SCRIPT.sh:38,42`; `.env.example:73`; plus ~12 markdown docs | Production host address of a live money system | Not a credential, but it advertises the target. Prefer `SSH_HOST` / a repo variable and strip it from docs. |

No live API keys, passwords, or private keys were found in the current tree:
`.env.example` contains only `your_*_here` / `change-me-*` placeholders,
`litellm-config.yaml` uses `os.environ/OPENROUTER_API_KEY`, and
`docker-compose*.yml` reference `${VAR}` only. `update_vps_env.py` and
`scripts/ssh_vps_remote.sh` handle a private key but read it from
`SSH_PRIVATE_KEY`.

---

## 8. Cleanup recommendations

Nothing below has been deleted or edited by this document — these are proposals.

**Retire (move to an `archive/` dir or delete), superseded and dangerous:**

| Script | Why |
|---|---|
| `deploy.sh` | Superseded by `scripts/hostinger_vps_apply.sh`; uses `docker-compose` v1, `down`s the live stack, health-checks the wrong port (`8000` vs `127.0.0.1:8001`). |
| `update_vps_env.py` | Unreferenced one-off that rewrites the remote `.env` and restarts the backend. |
| `docs/ops/vps-diagnose.sh` | Restarts `ssh.socket` (lockout risk) and deploys as a side effect. |
| `docs/ops/fix_qdrant.sh`, `docs/ops/deploy_qdrant_fix.sh` | `fuser -k`, `docker rm -f`, `docker compose down --remove-orphans`, `chmod 777`. |
| `docs/ops/VPS_FIX_SCRIPT.sh`, `docs/ops/VPS_FULL_FIX.sh` | Target the retired container name `qdrant-13fq-qdrant-1`. |
| `docs/ops/setup_influx_buckets.sh`, `docs/ops/fix_influxdb_complete.sh` | Superseded by `scripts/ensure_influx_buckets.sh`; carry scrubbed-token placeholders. |
| `docs/ops/verify_fix.sh` | Checks `localhost:8000`, which prod does not expose. |
| `scripts/vps_deploy_p0_oneliner.sh`, `scripts/vps_deploy_risk_reviewer_fix.sh` | Now pure shims; the docs already point at the canonical script. |

**Consolidate:**

1. **Influx token repair** — merge `docs/ops/fix_influx_token.sh` and
   `docs/ops/fix_influxdb_token.sh` into one `scripts/fix_influx_token.sh` that
   never prints token material and calls `ensure_influx_buckets.sh`.
2. **Grafana** — keep `scripts/fix_grafana_influx.sh` + `scripts/deploy_grafana.sh`;
   fold `vps_fix_grafana_datasources.sh`, `vps_rewire_old_grafana.sh`,
   `vps_fix_grafana_remote.sh`, `vps_probe_old_grafana.sh` into one
   `scripts/grafana_ops.sh <fix|probe|rewire>`, and drop the `admin:admin`
   defaults.
3. **Qdrant** — keep `scripts/create_qdrant_collection.sh`; delete the four other
   variants and move `scripts/qdrant_debug.sh` into `docs/ops/` as notes (it is a
   command sheet, not a script).
4. **Log helpers** — delete `scripts/check_logs.sh` and
   `scripts/check_qdrant_logs.sh`; the equivalent `docker logs … | grep` lines
   are in §5 and inside `vps_health_check.sh`.
5. **Archive smoke tests** — keep `scripts/test_archive_full.sh`, drop
   `scripts/test_archive.sh`.
6. **Local launchers** — keep `start.sh` (+ the `.bat` for Windows users); drop
   `run.sh` / `start_docker_local.sh` or reduce them to one-line wrappers.
7. **Risk profiles** — `vps_apply_recovery_mode.sh` and
   `vps_apply_quality_expand.sh` are 80% identical and both duplicate the
   `_upsert_env` helper from `hostinger_vps_apply.sh`. Extract
   `scripts/lib/env_upsert.sh` and express the profiles as data
   (`scripts/profiles/recovery.env`, `quality_expand.env`) applied by one script.

**Fix (correctness / safety, beyond cleanup):**

- Paper/live flags are already `${PAPER_TRADING:-true}`-style in prod compose;
  treat the VPS `.env` as the source of truth. Do not re-hardcode live
  booleans into `environment:`.
- Correct the rollback procedure in `docs/ops/PRODUCTION_DEPLOY.md` (the
  `hostinger_vps_apply.sh` re-checkout of `main` defeats it) and remove the
  reference to a `github.repository` guard that does not exist in
  `.github/workflows/`.
- Add `X-API-Key: $ADMIN_API_KEY` to `scripts/vps_start_loop.sh`.
- Configure `SSH_PRIVATE_KEY` (plus `SSH_HOST`, `SSH_USER`) so `vps-deploy.yml`
  becomes the single audited deploy path, and add required reviewers on the
  `production` environment.
- Move the SSH public key out of `scripts/lib/vps_ssh_hygiene.sh` into a
  required `CLOUD_AGENT_PUBKEY` env var, and rotate the InfluxDB token that was
  once committed to `docs/ops/`.
