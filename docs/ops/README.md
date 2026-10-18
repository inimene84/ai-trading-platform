# Ops & Incident Notes

Historical operational runbooks, one-off fix scripts, and VPS deployment notes
that used to live in the repo root. Kept for reference; **not** part of the
normal build/run flow.

Canonical, still-current docs live at the repo root:
- `../../README.md` — project overview
- `../../STARTUP.md` — how to start the stack
- `../../DEPLOYMENT.md` — deployment architecture
- `../../deploy.sh`, `../../run.sh`, `../../start.sh` — entrypoint scripts

## Contents

| File | Purpose |
|------|---------|
| `VENUE_FUSION.md` | Binance + cTrader venue-fusion flags (all default off/shadow) |
| `ACTION_PLAN.md` | Historical action plan |
| `DIAGNOSIS_SUMMARY.md` | Past incident diagnosis |
| `DEBUG_PORT.md` | Debug-port notes |
| `INTEGRATION.md` | Integration notes |
| `NETWORK_FIX.md` | Docker network isolation fix |
| `HOSTINGER_QUICKFIX.md` | Hostinger VPS quick fixes |
| `FINAL_DEPLOY.md`, `FINAL_QDRANT_DEPLOY.md` | Deploy runbooks |
| `VPS_*.md` | VPS command/runbook snapshots |
| `VPS_FIX_SCRIPT.sh`, `VPS_FULL_FIX.sh` | VPS fix scripts |
| `fix_influx*.sh`, `setup_influx_buckets.sh` | InfluxDB token/bucket fixes |
| `fix_qdrant.sh`, `init_qdrant.sh`, `deploy_qdrant_fix.sh` | Qdrant fixes |
| `verify_fix.sh`, `vps-diagnose.sh` | Verification / diagnostics |

> These are point-in-time artifacts. Prefer the root docs and `docker-compose`
> for current operations.
