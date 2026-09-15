# QuantumTrade Pro: Research Plane Architecture & Operations Guide

## Overview

The **Research Plane** is a dedicated, read-only analytics and search subsystem for QuantumTrade Pro. It aggregates market context (news, economic calendar releases, MetaTrader OHLCV bars) and records all trading decisions and vetoes into OpenSearch.

```
SearXNG ──json──> research_scheduler ──bulk──> OpenSearch (qt-news)
MT5 sidecar ────> research_scheduler ──bulk──> OpenSearch (qt-mt-bars)
Calendar ───────> research_scheduler ──bulk──> OpenSearch (qt-events)
DecisionEngine ─> ingest_decision ───bulk──> OpenSearch (qt-decisions)
                                                    │
                                                    ▼
                                            GET /research/*
                                         (Admin API-gated)
                                                    │
                                         Dashboards / MCP Agent
```

> [!IMPORTANT]
> **Core Architectural Invariant: NOT A LIVE GATE**
> The Research Plane is strictly non-gating for live trading. Hits from SearXNG, OpenSearch indices, or MetaTrader bars are NEVER converted into live order vetoes or live position sizing inputs.
> All ingestion and retrieval paths are fail-soft: network timeouts or cluster outages log warnings and return empty results without interrupting the active trading loop.

---

## Configuration & Environment Variables

Add to your `.env` on the VPS:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OPENSEARCH_URL` | `http://opensearch:9200` | OpenSearch HTTP endpoint on `trading-net` |
| `SEARXNG_URL` | `http://searxng:8080` | Self-hosted SearXNG search endpoint |
| `MT5_BRIDGE_URL` | `http://mt5-bridge:8001` | MetaTrader 5 Bridge sidecar |
| `RESEARCH_INGEST_ENABLED` | `false` | Master toggle for index bootstrap and scheduler ingestion |
| `RESEARCH_SEARXNG_INTERVAL_SEC` | `900` | Ingestion poll interval in seconds (default: 15 minutes) |

---

## OpenSearch Indices & Templates

The research plane maintains four time-based indices with hot-to-warm retention:

1. **`qt-news`**: Web news hits fetched via SearXNG.
   - Fields: `ts` (date), `symbol` (keyword array), `source` (keyword), `title` (text), `url` (keyword), `snippet` (text), `query` (keyword), `engine` (keyword), `sentiment` (object).
2. **`qt-events`**: High-impact economic calendar events from `calendar_service`.
   - Fields: `ts` (date), `currency` (keyword), `impact` (keyword), `title` (text), `source` (keyword).
3. **`qt-decisions`**: Ingested evaluations and gate vetoes from `evaluate_symbol`.
   - Fields: `ts` (date), `symbol` (keyword), `broker` (keyword), `mode` (keyword), `signal` (keyword), `confidence` (float), `promotion_verdict` (keyword), `promotion_reason` (text), `gate_id` (keyword), `reason` (text), `shadow` (bool), `rejected` (bool), `prompt_version` (keyword), `model_id` (keyword).
4. **`qt-mt-bars`**: Multi-timeframe closed bars from the MT5 bridge sidecar.
   - Fields: `ts` (date), `symbol` (keyword), `tf` (keyword), `o`, `h`, `l`, `c`, `v` (float).

Templates are located in `opensearch/templates/` and automatically applied on backend boot when `RESEARCH_INGEST_ENABLED=true`.

---

## VPS Deployment & Network Attach Guide

To ensure high performance and strict isolation:
1. **Never add research services to root `docker-compose.yml` or `docker-compose.prod.yml`**.
2. Run OpenSearch, SearXNG, and the MT5 bridge in their dedicated project folders.
3. **Attach network, don’t recreate cluster**: The existing OpenSearch cluster (`opensearch/docker-compose.yml`) runs single-node with `DISABLE_SECURITY_PLUGIN=true` on loopback `127.0.0.1:9200`. Do not recreate or rebuild the cluster; simply attach the running container to `trading-net`:
   ```bash
   docker network connect trading-net ai-trading-opensearch
   docker network connect trading-net searxng
   docker network connect trading-net mt5-bridge
   ```
4. **Security Model & Boundaries**:
   - Single-node OpenSearch has the security plugin disabled (`DISABLE_SECURITY_PLUGIN=true`).
   - Loopback binding (`127.0.0.1:9200`) ensures OpenSearch is never exposed on the public internet interface (`0.0.0.0`).
   - However, any service attached to Docker's internal `trading-net` bridge network (e.g. n8n, MCP) can reach `http://ai-trading-opensearch:9200`.
   - External access via HTTP `/research/*` is strictly gated behind the admin API key (`X-API-Key`).
5. **Strict Host Bindings**: All host ports must bind strictly to loopback (`127.0.0.1`):
   - OpenSearch: `127.0.0.1:9200:9200`
   - SearXNG: `127.0.0.1:8888:8080`
   - MT5 Bridge: `127.0.0.1:8001:8001`
   - **Never** expose `0.0.0.0:9200` or `0.0.0.0:8888` on the public network interface.
6. **Scope Boundary (v1)**:
   - Decision ingestion instruments `evaluate_symbol` on crypto assets.
   - n8n candidate ingestion is scheduled for v2 upon n8n webhook standardization.

---

## API Endpoints (Admin Gated)

All endpoints reside under `/research` and require the `X-API-Key` or `Authorization: Bearer <key>` header:

- `GET /research/health`: Booleans for OpenSearch, SearXNG, and MT5 ping health.
- `GET /research/news?q=&symbol=&from=`: Filter news articles by ticker or text query.
- `GET /research/events?currency=&from=`: Filter high-impact macro releases.
- `GET /research/decisions?symbol=&verdict=&from=`: Audit why trades were taken or vetoed (e.g. `verdict=REJECT`).
- `GET /research/mt/bars?symbol=&tf=&from=`: Retrieve historical/recent MT5 bars.
