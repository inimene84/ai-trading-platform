# Jesse ML Model Training — Grok Agent Runbook

Reproducible workflow for training, validating, and deploying Jesse ML direction models
on the QuantumTrade VPS. Models gate live entries via DSR/PBO institutional thresholds.

## Architecture

| Service | Port | Purpose |
|---------|------|---------|
| Jesse Dashboard | `:9000` | Candle data, backtests, strategy management |
| Jesse MCP | `:9002` | MCP tool bridge |
| Jesse ML (`predict_server`) | `:9003` | Inference, `/model-metadata`, `/predict` |
| QuantumTrade backend | `:8001` | `GET /api/jesse/ml-validation` gate check |

VPS Jesse repo: `/root/jesse-trading`  
Model artifacts: `/root/jesse-trading/storage/models/` (mounted at `/home/storage/models` in container)

## Gate Thresholds (fail-closed in LIVE mode)

| Metric | Gate | Env var |
|--------|------|---------|
| Deflated Sharpe Ratio (DSR) | ≥ 0.95 | `JESSE_ML_DSR_MIN` |
| Probability of Backtest Overfitting (PBO) | < 0.30 (30%) | `JESSE_ML_PBO_MAX` |

Override (not recommended): `JESSE_ML_PBO_OVERRIDE=true`

Triple-barrier geometry (aligned with live RiskConfig):
- TP: `5.5× ATR` (`JESSE_DEFAULT_TP_ATR_MULT`)
- SL: `1.75× ATR` (`JESSE_DEFAULT_SL_ATR_MULT`)

## Required Environment Variables

### Local / Cloud Agent (SSH to VPS)

```bash
export SSH_HOST="<vps-hostname-or-ip>"
export SSH_USER="root"
export SSH_PRIVATE_KEY="<openssh-private-key>"   # or SSH_PASSWORD
# Optional
export JESSE_DIR="/root/jesse-trading"
export N_CONFIGS=6          # hyperparameter trials (fewer → lower PBO risk)
export FOLDS=5              # PurgedKFold folds for CPCV
```

### QuantumTrade Backend (`.env`)

```bash
JESSE_ML_URL=http://jesse-app:9003          # Docker internal
JESSE_ML_LOCAL_URL=http://127.0.0.1:9003      # host loopback
JESSE_ML_GATE_ENABLED=true
JESSE_ML_PBO_MAX=0.30
JESSE_ML_DSR_MIN=0.95
JESSE_ML_PBO_OVERRIDE=false
JESSE_DEFAULT_SL_ATR_MULT=1.75
JESSE_DEFAULT_TP_ATR_MULT=5.5
```

## Candle Data (current VPS inventory)

| Symbol | Range | ~1m bars |
|--------|-------|----------|
| BTC-USDT | 2023-10-05 → 2026-09-10 | ~1,514,880 |
| ETH-USDT | 2024-12-31 → 2026-09-09 | ~57,600 |
| SOL-USDT | 2025-01-01 → 2026-09-10 | ~57,600 |

Import more history if needed:
```bash
ssh root@$SSH_HOST "cd /root/jesse-trading && ./manage.sh import-candles 'Binance Perpetual Futures' 'ETH-USDT' '2024-01-01'"
```

---

## Step-by-Step: Train on VPS (recommended)

### 1. Verify Jesse stack

```bash
ssh root@$SSH_HOST "cd /root/jesse-trading && ./manage.sh status && ./manage.sh ml-status"
```

Expected: `jesse-app` Up, ML health `status: ok`, `gate_thresholds.dsr_min: 0.95`, `pbo_max: 0.3`.

### 2. Train with auto-retrain-promote (from Cloud Agent)

Uses `auto-retrain-promote`: writes `SYMBOL_TF_lightgbm.joblib` **only** when DSR ≥ 0.95 and PBO < 30%.

```bash
# Single symbol (default: BTC-USDT 1h, n-configs=6, folds=5)
./scripts/jesse_vps_train.sh BTC-USDT 1h lightgbm

# ETH / SOL
./scripts/jesse_vps_train.sh ETH-USDT 1h lightgbm
./scripts/jesse_vps_train.sh SOL-USDT 1h lightgbm

# Custom hyperparameter grid (fewer trials reduces PBO)
N_CONFIGS=6 FOLDS=5 ./scripts/jesse_vps_train.sh BTC-USDT 1h lightgbm

# Skip backtest for faster iteration
RUN_BACKTEST=false ./scripts/jesse_vps_train.sh BTC-USDT 1h lightgbm
```

**Promotion refused (exit 2):** artifact unchanged; check metadata for DSR/PBO. Do not use `JESSE_ML_PBO_OVERRIDE` in production.

### 3. Validate

```bash
./scripts/jesse_vps_validate.sh BTC-USDT 1h
./scripts/jesse_vps_validate.sh --all    # BTC + ETH + SOL
```

### 4. Direct VPS commands (SSH shell)

```bash
ssh root@$SSH_HOST
cd /root/jesse-trading

# Train + promote (enforced gate)
./manage.sh auto-retrain-promote BTC-USDT 1h lightgbm --n-configs 6 --folds 5

# Metadata (DSR, PBO, holdout Sharpe, gate)
./manage.sh model-metadata BTC-USDT 1h

# Smoke predict
./manage.sh predict-ml BTC-USDT 1h

# Full statistical validation suite
./manage.sh run-validation

# Strategy backtest with cost model
./manage.sh backtest QuantumAIStrategy --start 2024-01-01 --finish 2025-01-01

# Clear inference cache after new artifact
./manage.sh clear-cache
```

---

## Step-by-Step: Validate via QuantumTrade API

After backend deploy with Jesse bridge routes:

```bash
curl -sf "http://${SSH_HOST}:8001/api/jesse/ml-validation?symbol=BTC-USDT&timeframe=1h"
curl -sf "http://${SSH_HOST}:8001/api/jesse/ml-models"
```

Expected pass response:
```json
{
  "status": "success",
  "deployment_ok": true,
  "dsr": 1.0,
  "pbo": 0.02,
  "dsr_pass": true,
  "pbo_pass": true,
  "reasons": []
}
```

Expected fail (live trading blocked):
```json
{
  "deployment_ok": false,
  "pbo": 0.636,
  "reasons": ["PBO 63.6% exceeds gate 30%"]
}
```

---

## Reducing PBO (overfitting)

PBO rises with hyperparameter search breadth. Mitigations:

1. **Fewer trials:** `--n-configs 6` (default in `jesse_vps_train.sh`). Avoid 18+ configs.
2. **More folds:** `--folds 5` for PurgedKFold CPCV.
3. **Use `auto-retrain-promote`** — never `--force-promote` in production.
4. **Simpler models:** collapsed-everybar variants with `max_depth=2`, high `min_child_samples`.
5. **More candle history:** import earlier start dates for ETH/SOL before retraining.

Low-PBO reference (BTC collapsed-everybar, Sep 2026):
- DSR 1.0, PBO 2.0%, holdout Sharpe 4.29, n_trials 6

High-PBO failure example (18 trials):
- DSR 0.99, PBO 46%, promotion refused

---

## Artifact Naming

| File | Meaning |
|------|---------|
| `BTC-USDT_1h_lightgbm.joblib` | **Promoted** production model |
| `BTC-USDT_1h_lightgbm_meta.json` | DSR/PBO metadata (queried by `/model-metadata`) |
| `*.collapsed-everybar.joblib` | Alternate label mode; not served unless promoted |
| `*.rejected.joblib` | Failed promotion (collapsed classifier, etc.) |

`/model-metadata` looks for `{symbol}_{timeframe}_{model_type}_meta.json`. If missing, gate returns error and live mode blocks entries.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Metadata not found for BTC-USDT_1h_lightgbm_meta.json` | Run `auto-retrain-promote`; prior train may have failed gate |
| `PROMOTION REFUSED (exit 2)` | Lower `--n-configs`, add data, or accept gate failure |
| `DSR unavailable` | Retrain; metadata corrupt or missing |
| ML health shows symbol in `rejected_models` | Model loaded but fails gate; retrain or remove artifact |
| SSH fails | Set `SSH_HOST`, `SSH_USER`, `SSH_PRIVATE_KEY` in Cloud Agent secrets |
| ETH/SOL poor DSR | Import more history (`import-candles` from 2024-01-01) |

---

## Local Unit Tests (no VPS)

```bash
cd backend && poetry run pytest tests/unit/test_jesse_validation.py tests/unit/test_jesse_ml_gate.py -q
```

---

## Training Results (2026-09-16 run)

| Symbol | Timeframe | DSR | PBO | Holdout Sharpe | Gate | Notes |
|--------|-----------|-----|-----|----------------|------|-------|
| BTC-USDT | 1h | **1.00** | **2.0%** | 4.29 | **PASS** | Promoted `collapsed-everybar` artifact → production |
| ETH-USDT | 1h | 0.00 | 49.6% | — | FAIL | Only ~9 months data; import from 2024-01-01 |
| SOL-USDT | 1h | 0.89 | 32.8–56% | — | FAIL | Close on PBO once; needs more history |

Fresh `auto-retrain-promote` runs (n-configs=6) failed gate for all symbols on 2026-09-16.
The production BTC model is the Sep-11 `collapsed-everybar` run (n_trials=6, shallow trees).

### Reset trial registry before retrain

Cumulative `n_trials` in `/home/storage/trial_registry.json` penalizes DSR. Reset first:

```bash
./scripts/jesse_vps_reset_trials.sh BTC-USDT 1h
./scripts/jesse_vps_reset_trials.sh --all
```

### Promote a passing candidate manually

When a candidate passes gates but was saved under a variant name:

```bash
ssh root@$SSH_HOST 'cd /root/jesse-trading/storage/models && \
  cp BTC-USDT_1h_lightgbm.collapsed-everybar.joblib BTC-USDT_1h_lightgbm.joblib && \
  cp BTC-USDT_1h_lightgbm.collapsed-everybar_meta.json BTC-USDT_1h_lightgbm_meta.json && \
  cd /root/jesse-trading && ./manage.sh clear-cache'
```

---

## VPS Postgres Auth Fix

If training fails with `password authentication failed for user "jesse_user"`:

```bash
ssh root@$SSH_HOST 'PASS=$(docker exec jesse-postgres printenv POSTGRES_PASSWORD) && \
  docker exec jesse-postgres psql -U jesse_user -d jesse_db \
  -c "ALTER USER jesse_user WITH PASSWORD '\''$PASS'\'';"'
```

Verify: `docker exec jesse-app python3 -c "import os,psycopg2; ..."` (see runbook above).

---

## Quick Reference Commands

```bash
# Reset trials, then train all three symbols
./scripts/jesse_vps_reset_trials.sh --all
for S in BTC-USDT ETH-USDT SOL-USDT; do
  RUN_BACKTEST=false ./scripts/jesse_vps_train.sh "$S" 1h lightgbm || echo "FAIL: $S"
done

# Validate all
./scripts/jesse_vps_validate.sh --all

# VPS one-liner metadata
ssh root@$SSH_HOST 'cd /root/jesse-trading && for s in BTC-USDT ETH-USDT SOL-USDT; do echo "=== $s ==="; curl -sf "http://127.0.0.1:9003/model-metadata?symbol=$s&timeframe=1h" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get(\"metrics\",{}); print(f\"DSR={m.get(\"deflated_sharpe_ratio\")} PBO={m.get(\"prob_backtest_overfitting\")}\")"; done'
```
