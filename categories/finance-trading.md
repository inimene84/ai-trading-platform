# Finance & Trading

Projects and systems in the **finance & trading** category: algorithmic execution,
market data, risk management, quantitative research, and broker integrations.

---

## QuantumTrade Pro

**Institutional-grade, autonomous multi-broker quantitative trading and risk-execution platform.**

QuantumTrade Pro runs continuous 24/7 autonomous trading across **Binance Futures**
(crypto perpetuals) and **cTrader** (institutional FX and metals), combining a
technical strategy stack, a deep-learning forecasting gate, a multi-agent LLM opinion
layer, and a fail-closed risk stack.

- **Repository:** `github.com/inimene84/ai-trading-platform`
- **License:** Apache-2.0
- **Status:** Live, real-money system (fail-closed by default)

### Capabilities

- **Execution** — Non-blocking async event loop, serialized per-symbol execution locks,
  GTX maker order routing, strict book partitioning (`broker + account_id + mode`).
- **Quant ML** — LightGBM meta-labeling on Triple-Barrier events with purged
  cross-validation, sample-uniqueness weighting, and split-conformal uncertainty
  intervals; a Kronos deep-learning forecasting sidecar as a foundation-model gate.
- **AI opinion layer** — 8+ LLM analyst personas plus an LLM risk reviewer, routed
  through a multi-provider LLM router.
- **Fail-closed safety** — Double-locked live-deploy authorization, risk guard
  (drawdown / daily-loss / position limits), kill switch (equity floor), emergency
  position manager, broker position sync, margin gate, and symbol-quality gate.
- **Data & telemetry** — InfluxDB time-series metrics/sentiment, Qdrant vector storage
  for news sentiment and trade memory, n8n webhook pipelines, Grafana dashboards, Sentry.

### Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Python 3.11+, FastAPI + Uvicorn, SQLAlchemy 2 + Alembic, Pydantic v2, LangChain/LangGraph |
| Brokers | ccxt, python-binance, ctrader-open-api |
| Frontend | React 19, Vite 8, TypeScript, Tailwind CSS 4, lightweight-charts |
| Datastores | SQLite (dev) / PostgreSQL (opt), InfluxDB, Qdrant, Redis |
| Infra | Docker Compose, nginx, LiteLLM proxy, n8n, Grafana |

### Architecture (per-symbol pipeline)

```
combined technical strategy
  → Kronos foundation-model gate
  → weighted AI opinion layer (8+ agents)
  → LLM risk reviewer
  → fail-closed risk stack
  → unified order router (paper fill engine or live Binance/cTrader)
```

### Deployment

- **Production:** Hostinger VPS, Docker Compose (`docker-compose.prod.yml`), reverse-proxied
  through nginx; dashboard served on port `8081`. Deploys are manual-only via the
  `vps-deploy.yml` GitHub Actions workflow (typed `DEPLOY` confirmation, `production`
  environment).
- **Local:** `./run.sh` (native) or `start_docker_local.sh` (Docker); see `STARTUP.md`.

---

## Live deployment snapshot

_Verified 2026-09-23 on the production VPS (`srv1071801`)._

- **Version:** `main` @ `dee8db2` (fast-forwarded from 13 commits behind; stack rebuilt,
  containers recreated).
- **Backend API:** `GET /health` → `status: ok`, `trading_status: ACTIVE`.
- **Frontend:** dashboard on port `8081` → HTTP 200.
- **Trading loop:** running on the 17-symbol universe
  (`BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, AVAXUSDT, LINKUSDT, NEARUSDT, LTCUSDT,
  DOTUSDT, ATOMUSDT, OPUSDT, INJUSDT, SUIUSDT, UNIUSDT, POLUSDT, BTCUSDC`);
  sentiment loop cycle completing across 17 symbols.
- **Known gap:** Grafana InfluxDB datasource provisioning has no working token —
  `INFLUXDB_TOKEN` is not set in the host/Grafana provisioning environment (the value is
  present inside the backend container but not exported to the Grafana provisioning env).
  Set `INFLUXDB_TOKEN` in the server environment and re-provision to restore InfluxDB
  panels.
