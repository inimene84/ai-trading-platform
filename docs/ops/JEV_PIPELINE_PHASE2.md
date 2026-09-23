# JEV Phase 2 pipeline (shadow / paper)

Additive ingest + evaluate + optional paper path. **Does not replace**
`workflows/11_jev_ensemble_v2_workflow.json` or
`workflows/12_jev_outcome_filler_workflow.json`. Those remain the live n8n
Jev Ensemble / outcome-filler flows.

This PR does **not** deploy to a VPS and does **not** enable live order
placement.

## Enable shadow

1. Set `JEV_EXECUTION_MODE=off` (default) to ingest + evaluate + log only.
2. Set `JEV_EXECUTION_MODE=shadow` to also write `jev_pipeline_trades` rows
   for what *would* have filled. No `UnifiedTrading.place_order` call.
3. Restart the backend process you already operate. Do not flip live flags.

Auth is the existing admin token (`X-API-Key` / Bearer) used by `/api/jev`.

## Later: enable paper

1. Confirm the platform paper/sandbox book is the one you want
   (`TRADING_MODE=paper` or the existing Binance paper-parallel path).
2. Set `JEV_EXECUTION_MODE=paper`.
3. Paper fills go through `UnifiedTrading` session `jev_pipeline_paper`
   (same engine as `POST /trading/paper/order`). Risk guard + Sentry
   `is_trading_allowed()` still apply. Confidence threshold is not enough.
4. Optional: `JEV_TRADE_THRESHOLD` (default `0.65`), `JEV_PAPER_QUANTITY`
   (default `0.001`).

## Do not set live without owner confirm

`JEV_EXECUTION_MODE=live` is stubbed and **blocked in code**
(`jev_live_execution_allowed()` is always false in this PR). A later ops
change plus explicit human confirm is required before any live venue call.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/jev/ingest/market` | Store a market scan |
| POST | `/api/jev/ingest/news` | Store news/sentiment context |
| POST | `/api/data/collect-market` | Alias (zip-shaped scanner payload) |
| POST | `/api/data/collect-news` | Alias |
| POST | `/api/jev/pipeline/evaluate` | Run existing `evaluate_symbol` + log |
| POST | `/api/jev/pipeline/orchestrate` | Oldest-pending scan → eval → mode gate |
| GET | `/api/jev/pipeline/calibration` | SQLite-friendly calibration snapshot |
| GET | `/api/jev/pipeline/news/{symbol}` | Recent ingested headlines |
| GET | `/api/jev/pipeline/status` | Current mode / live-blocked flag |

Legacy `GET /jev/evaluate` is unchanged and still advisory.

Uses existing `OPENROUTER_*` / `JEV_OPENROUTER_MODEL` / `JEV_PROVIDER`.
A new `TYPESAFE_API_KEY` is not required unless you already run the
direct TypeSafe provider.

## n8n

`workflows/21_jev_evaluation_pipeline.json` is a **draft**, `active: false`.
Do not assume a production import. Prefer appending HTTP ingest nodes to
workflows 11/12 rather than replacing them.

## Migration

Alembic revision `e4b7c1a9d2f0` creates four new tables with JSON columns
(SQLite-safe). `Base.metadata.create_all` also creates them on startup.
Existing `trades` / `hedge_fund_*` / paper / `api_keys` models are untouched.
