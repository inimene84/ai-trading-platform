# VPS Ops Scripts — Inventory & Runbook

Operational scripts for the Hostinger VPS running QuantumTrade Pro. This
directory accumulated many one-off scripts during incident response. Nothing
here is imported by the application at runtime — these are operator tools run
manually (or via cron) against the VPS.

> **Canonical deploy:** `hostinger_vps_apply.sh` (also used by the GitHub
> Action `VPS Deploy`). Prefer it over the older `vps_deploy_*` scripts.
> See `docs/ops/PRODUCTION_DEPLOY.md` for the full deploy/rollback procedure.

## Shared module

| File | Purpose |
|---|---|
| `vps_ssh_common.py` | Shared SSH connection settings used by ~36 `vps_*` scripts. Supports `SSH_HOST_HERMES` (Allikas/OmniRoute) alongside `SSH_HOST` (trading). |
| `vps_fix_two_vps_connections.sh` | One-shot SSH + OmniRoute + GrokBOT wiring between trading and Allikas VPSes. See `docs/ops/TWO_VPS_CONNECTIONS.md`. |

## Status tags

- **active** — still the recommended way to do this.
- **one-off** — written for a specific past incident; keep for reference, not for routine use.
- **deprecated** — superseded; use `hostinger_vps_apply.sh` instead.

## Status & health checks

| Script | Status | Purpose |
|---|---|---|
| `vps_health_check.sh` | active | Container/service health overview. |
| `vps_live_trading_check.sh` | active | Live-trading safety-state check. |
| `vps_check_protection.py` / `.sh` | active | Open-position protection status from the API. |
| `vps_check_exposure.py` | active | Compare DB open trades vs live exchange positions + notionals. |
| `vps_check_binance_status.py` | active | Binance IP-ban state, env tuning, trading data. |
| `vps_check_litellm.py` | active | LiteLLM config, git state, kie/risk-reviewer logs. |
| `vps_check_batch_logs.py` | active | Symbol batch-scan logs. |
| `vps_check_antigravity.py` | active | Compare VPS working tree vs origin for Antigravity fixes. |
| `vps_verify_health.py` | active | End-to-end health verification. |
| `vps_status_parse.py` | active | Parse status output. |
| `vps_error_logs.py` | active | Scan container logs for recent errors/warnings. |
| `vps_pkg_versions.py` | active | Package versions inside the backend container. |
| `vps_env_audit.py` | active | Compare local vs VPS env key lengths; find env backups. |
| `vps_diagnose_providers.py` / `.sh` | active | LLM provider diagnostics via SSH. |
| `vps_final_check2.py` | one-off | Upload final router, rebuild, full-system verify. |
| `vps_final_verify.py` | one-off | Final verification (no docstring). |

## Deploy

| Script | Status | Purpose |
|---|---|---|
| `hostinger_vps_apply.sh` | **canonical** | Production deploy on main (git pull → rebuild → recreate → health check). |
| `vps_deploy_latest.py` | deprecated | Deploy latest main + restart loop. Use `hostinger_vps_apply.sh`. |
| `vps_deploy_universe.py` | deprecated | Pull, rebuild, restart with 20-coin universe. Use `hostinger_vps_apply.sh`. |
| `vps_clean_deploy.py` | deprecated | Reset to clean main + rebuild. Use `hostinger_vps_apply.sh`. |
| `vps_deploy_cap_fix.py` | one-off | Deploy direction-cap fix (past incident). |
| `vps_deploy_kie_fix.py` | one-off | Deploy Kie JSON fixes (past incident). |
| `vps_deploy_p0_oneliner.sh` | deprecated | Use `hostinger_vps_apply.sh`. |
| `vps_deploy_risk_reviewer_fix.sh` | deprecated | Use `hostinger_vps_apply.sh`. |
| `vps_bootstrap_and_deploy.sh` | active | One-line bootstrap for Hostinger web terminal. |
| `vps_remote_oneliner.sh` | active | Paste/run on VPS as root (no Cloud Agent SSH needed). |

## Fix / restore / reconcile / resume

| Script | Status | Purpose |
|---|---|---|
| `vps_fix_naked_positions.py` | active | Restore missing SL/TP on open positions. |
| `vps_restore_protection.py` | active | Restore missing Binance SL/TP from open DB trades (runs in container). |
| `vps_restore_influx_history.sh` | active | Restore InfluxDB history. |
| `vps_sentry_resume.sh` | active | Resume after a sentry (emergency-halt) event. |
| `vps_resume_trading.py` | active | Resume trading after a halt. |
| `vps_apply_protection_fix.sh` | active | Apply protection fixes. |
| `vps_fix_grafana_datasources.sh` | active | Fix Grafana datasource env. |
| `vps_fix_grafana_remote.sh` | active | Fix remote Grafana. |
| `vps_reconcile_stale_trades.py` | one-off | Close DB trade rows verified flat on exchange (past incident). |
| `vps_apply_recovery_mode.sh` | active | Recovery / "green-day" mode. |

## KIE / LLM provider ops

| Script | Status | Purpose |
|---|---|---|
| `vps_kie_diagnose.py` | active | Diagnose Kie truncation; verify OpenAI/xAI keys. |
| `vps_kie_raw.py` | active | Inspect raw Kie response to find invalid JSON. |
| `vps_kie_post_cycle.py` | active | Post-cycle Kie verification. |
| `vps_poll_kie_logs.py` | active | Poll logs for Kie errors during cycles. |
| `vps_watch_kie_logs.py` | active | Watch backend logs for Kie prefill + router activity. |
| `vps_loop_kie_status.py` | active | Detailed loop + Kie log status. |
| `vps_test_kie_cache.py` | active | Verify Kie proxy prompt-caching support. |
| `vps_find_openrouter_model.py` | active | Find working OpenRouter models on VPS. |
| `vps_hermes_keys.py` | active | Hermes key operations. |
| `vps_sync_env_from_hermes.py` | active | Sync valid API keys from hermes `.env` to platform `.env`. |

## Env / config

| Script | Status | Purpose |
|---|---|---|
| `vps_mask_env.py` | active | Mask secrets in env output. |
| `vps_patch_batch_env.py` | active | Set rate-limit batching env vars. |
| `vps_scan_other_envs.py` | active | Scan for other env files. |
| `vps_expand_universe.py` | active | 20 scan coins, 10 max positions, equity-based sizing. |
| `vps_apply_quality_expand.sh` | active | More concurrent positions on positive-edge symbols. |

## Watch / poll / loop

| Script | Status | Purpose |
|---|---|---|
| `vps_realtime_watchdog.sh` | active | Lightweight real-time health watchdog (cron every 2–5 min). |
| `vps_restart_loop.py` | active | Restart trading loop to pick up env changes. |
| `vps_start_loop.sh` | active | Start the trading loop. |
| `vps_apply_binance_poll.sh` | active | Apply Binance polling config. |

## Other (non-vps) scripts in this directory

`check_logs.sh`, `check_qdrant_logs.sh`,
`cleanup_stale_branches.sh`, `close_all_positions.py`,
`connect_n8n_network.sh`, `create_qdrant_collection.sh`, `deploy_grafana.sh`,
`deploy_kie_sonnet.sh`, `e2e_ai_workflow_test.py`, `ensure_influx_buckets.sh`,
`fix_grafana_influx.sh`, `fix_n8n_db_corruption.sh`, `hostinger_vps_apply.sh`,
`qdrant_debug.py` / `.sh`, `reconcile_check.py`,
`repair_corrupt_trades.py`, `ssh_vps_remote.sh`, `test_archive*.sh`,
`verify_endpoints.sh`, plus `lib/vps_ssh_hygiene.sh` (sourced by the deploy).

## Maintenance notes

- These scripts SSH to the VPS using `vps_ssh_common.py`; the GitHub deploy
  uses the `SSH_PRIVATE_KEY` secret (see `vps-deploy.yml`).
- When adding a new operational script, add a row here and prefer extending an
  existing category over a new one-off.
- Deprecated scripts are kept for reference (recovery knowledge); do not delete
  without confirming the underlying fix is merged into `main`.
