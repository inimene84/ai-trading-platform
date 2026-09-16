# QuantumTrade n8n Workflows

Three automation flows for forex + crypto market scanning and execution.
Import these into n8n at **https://n8n1.thorinvest.org**.

## Files

| File | Schedule | Backend endpoint | Purpose |
|------|----------|------------------|---------|
| `01_market_scanner_workflow.json` | Every 5 min | `POST /api/signals/scan-markets` | Technical scan across EURUSD, XAUUSD, crypto, etc. |
| `02_news_macro_scanner_workflow.json` | Every 1 min | `POST /api/signals/scan-news` | Macro calendar + news sentiment candidates |
| `03_execution_scheduler_workflow.json` | Every 30 sec | `GET /api/signals/ready-for-execution` → `POST /api/signals/execute-candidate` | Execute timed candidates via smart router |
| `04_forex_scanner_workflow.json` | Every 5 min | `POST /api/signals/scan-markets` | Forex/metals only (no crypto) |
| `05_forex_sentiment_ingestion_workflow.json` | Every 15 min | `POST /api/sentiment/ingest` | Forex & Macro news ingestion with Gemini scoring |
| `20_grok_overseer_daily_telegram.json` | Daily 15:00 UTC (18:00 EEST) + webhook `qt-grok-overseer-daily` | `POST /api/agents/grok-overseer/analyze` | GrokBOT daily overseer summary to Telegram |


## Where to find them

1. **Git repo:** `workflows/` (this folder)
2. **VPS:** `/root/ai-trading-platform-v3/workflows/`
3. **Dashboard:** Strategy Timing Control → **n8n Workflows** tab
4. **n8n UI:** Search **QuantumTrade** after import

## Import (VPS)

```bash
cd /root/ai-trading-platform-v3
chmod +x scripts/import_quantumtrade_n8n_workflows.sh
./scripts/import_quantumtrade_n8n_workflows.sh
```

Requires `BACKEND_API_KEY` in the n8n container (same value as `ADMIN_API_KEY` in the trading platform `.env`).
The safe default imports workflows as drafts. Publish only after a successful
paper-mode test:

```bash
ACTIVATE=1 ./scripts/import_quantumtrade_n8n_workflows.sh
```

Publish only safe scanners (01, 02, 04) while keeping the execution scheduler off:

```bash
chmod +x scripts/publish_n8n_scanners.sh
./scripts/publish_n8n_scanners.sh
```

## Upgrade n8n

```bash
chmod +x scripts/upgrade_n8n_vps.sh
N8N_TARGET_VERSION=2.36.8 ./scripts/upgrade_n8n_vps.sh
```

Stack lives at `/docker/n8n/docker-compose.yml`. n8n Assistant sandbox (certs, API, privileged runner) is an overlay: `n8n/docker-compose.sandbox.yml`. Apply with `./scripts/apply_n8n_sandbox.sh`. Web search uses the existing SearXNG at `http://ai-trading-searxng:8080`, not a second copy. Do not migrate this instance to Postgres in place — SQLite holds live workflows.

## Building a forex-only scanner

Duplicate workflow **01** in n8n and change the JSON body universe, e.g.:

```json
{
  "universe": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "USDCAD", "AUDUSD"],
  "timeframe": "M5"
}
```

On weekends forex is closed — expect zero new candidates until Sunday ~22:00 UTC.
Use **BTCUSDT** in the universe to test the pipeline 24/7.

## Manual import in n8n UI

1. Open n8n → **Workflows** → **Import from file**
2. Select each JSON file
3. Activate the workflow (toggle top-right)
4. Run **Execute workflow** once to verify

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| HTTP 401 on scan-markets | Set `BACKEND_API_KEY` in n8n env; re-import workflows |
| Cannot reach backend | n8n must be on `trading-net` (already configured) |
| Workflow not listed | Search **QuantumTrade** — they were not auto-imported before Aug 2026 |
