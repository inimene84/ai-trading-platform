# n8n Enrichment Workflows

Four scheduled workflows write extra signals into the InfluxDB `news-sentiment` bucket. The native 30m sentiment loop stays the reliability floor.

## Backend ingest (Docker network)

| Workflow | Method | URL |
|---|---|---|
| 1. On-Chain Whale Activity Monitor | POST | `http://ai-trading-backend:8000/api/market-data/on-chain` |
| 2. Macro Cross-Asset Correlation Monitor | POST | `http://ai-trading-backend:8000/api/market-data/macro` |
| 3. Technical Divergence Alert Workflow | POST | `http://ai-trading-backend:8000/api/market-data/technical` |
| 4. Sentiment-Price Divergence Alerts | GET technical + POST | `http://ai-trading-backend:8000/api/market-data/technical` then `.../divergence` (alias: `/api/alerts/divergence`). Manual webhook `qt-agentzero-div-manual` ACKs immediately (`onReceived`). |

POSTs send `X-API-Key: {{ $env.BACKEND_API_KEY }}`. Backend logs `Stored on-chain|macro|technical|divergence ...`.

Measurements: `onchain_signal`, `macro_signal`, `technical_signal`, `divergence_alert`.

## Activate (only after a green manual run)

1. Confirm the schedule node is the live trigger.
2. Confirm HTTP nodes use the internal Docker hostname above, not the public `:8001` bind.
3. Execute once. Check backend logs for the `Stored ...` lines.
4. Activate in order 1 → 2 → 3 → 4.

Import on the VPS with `n8n import:workflow --input=<file.json>` (same workflow id updates the existing copy). Default import leaves workflows inactive unless `--activeState=fromJson`.

## Follow-up

The opinion layer still reads `news_sentiment` / `market_alert` / `global_sentiment`. Wiring it to the four new measurements is a separate change.
