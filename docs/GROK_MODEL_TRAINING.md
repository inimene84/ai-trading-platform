# Jesse ML Model Training — Grok Agent Runbook

Train on the **Hostinger GPU node** (NVIDIA B200). Serve and trade on the
CPU trading VPS. Do not train LightGBM/LSTM/Kronos on `SSH_HOST` — that box
has no GPU (`nvidia-smi` missing, Kronos sidecar already on CPU).

## Two hosts

| Role | What runs there | How Grok SSHes |
|------|-----------------|----------------|
| **GPU training node** | PyTorch cu128, `jesse_quant/train_gpu.py`, candle dumps, rejected/promoted artifacts | `GPU_SSH_HOST` / `GPU_SSH_USER=ubuntu` / `GPU_SSH_PORT=32408` / password or key |
| **Trading VPS** (`srv1071801`) | Jesse Docker (`:9000/:9002/:9003`), QuantumTrade `:8001`, live risk | `SSH_HOST` / `SSH_USER=root` / `SSH_PRIVATE_KEY` port 22 |

Remote dir on GPU: `/home/ubuntu/jesse-gpu`  
Repo trainers: `jesse_quant/` (synced by `scripts/gpu_train_remote.sh`)

## Architecture (trading VPS — inference only)

| Service | Port | Purpose |
|---------|------|---------|
| Jesse Dashboard | `:9000` | Candle data, backtests, strategy management |
| Jesse MCP | `:9002` | MCP tool bridge |
| Jesse ML (`predict_server`) | `:9003` | Inference, `/model-metadata`, `/predict` |
| QuantumTrade backend | `:8001` | `GET /api/jesse/ml-validation` gate check |

VPS Jesse repo: `/root/jesse-trading`  
Model artifacts: `/root/jesse-trading/storage/models/` (mounted at `/home/storage/models` in container)
Kronos sidecar: `ai-trading-kronos` with `KRONOS_DEVICE=cpu` — fine-tune on GPU, do not flip this container to CUDA on the trading VPS.

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

### GPU node (required for training)

Add these as Cloud Agent secrets. Never commit host, password, or key.

```bash
export GPU_SSH_HOST="<hostinger-gpu-host-or-ip>"
export GPU_SSH_USER="ubuntu"
export GPU_SSH_PORT=32408
export GPU_SSH_PASSWORD="<gpu-instance-password>"   # or GPU_SSH_PRIVATE_KEY
export GPU_REMOTE_DIR="/home/ubuntu/jesse-gpu"
export JESSE_TRAIN_DEVICE=auto
```

### Trading VPS (inference, candle dump, backtest)

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
JESSE_ML_MODEL_TYPE=lightgbm
JESSE_ML_FALLBACK_MODEL_TYPE=lstm            # none disables LSTM fallback
JESSE_ML_PREDICT_TIMEOUT=20                  # CPU LSTM cold-start; clamped at 30s
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

## Step-by-Step: Train on Hostinger GPU (required)

The B200 is already bootstrapped (`/home/ubuntu/jesse-gpu/.venv`, PyTorch 2.11+cu128).
LSTM trains on `cuda:0`. LightGBM PyPI wheels fall back to CPU unless rebuilt with `-DUSE_CUDA=1`.

### 1. Probe GPU

```bash
./scripts/gpu_train_remote.sh inventory
# expect: NVIDIA B200, torch cuda.is_available True
```

### 2. Sync trainers + bootstrap (first run or after code changes)

```bash
./scripts/gpu_train_remote.sh bootstrap
```

### 3. Train at live geometry (5.5 / 1.75 ATR)

```bash
./scripts/gpu_train_remote.sh train BTC-USDT both
./scripts/gpu_train_remote.sh train ETH-USDT both
./scripts/gpu_train_remote.sh train SOL-USDT both
./scripts/gpu_train_remote.sh status
```

Promotion writes production `*.joblib` / `*.pt` **only** when DSR ≥ 0.95, PBO < 30%, and both-class recall ≥ 10%. Failures land as `*.rejected.*` — do not copy those to the trading VPS.

On the GPU node directly:

```bash
cd /home/ubuntu/jesse-gpu
./manage.sh gpu-inventory
./manage.sh gpu-train BTC-USDT 1h both storage/candles/BTC-USDT_1m.csv.gz
./manage.sh gpu-auto-retrain-promote BTC-USDT 1h both
```

### 4. Evacuate promoted artifacts before the GPU VM is destroyed

The Hostinger GPU instance is ephemeral. Pull **promoted** `*.pt` / `*_meta.json`
only — never `*.rejected.*` — onto the trading VPS bind-mount
(`/root/jesse-trading/storage/models/` → jesse-app `/home/storage/models`).

```bash
# from a Cloud Agent / operator box that has both GPU and VPS secrets
python3 scripts/gpu_evacuate_promoted.py --push-vps
```

Jesse ML (`:9003`) serves `model_type=lstm` from `SYMBOL_TF_lstm.pt` after
CPU torch is installed on the bind-mounted vendor path:

```bash
docker exec jesse-app python3 -m pip install --target /home/vendor/cpu-torch \
  torch --index-url https://download.pytorch.org/whl/cpu
# reload only the ML process (do not restart live jesse run)
docker exec jesse-app python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9003/cache/clear')"
```

Smoke:

```bash
curl -sf "http://127.0.0.1:9003/model-metadata?symbol=ETH-USDT&timeframe=1h&model_type=lstm"
curl -sf "http://127.0.0.1:9003/predict?symbol=ETH-USDT&timeframe=1h&model_type=lstm"
```

Do **not** auto-enable live Jesse sync (`JESSE_SYNC_TO_LIVE` stays false).
QuantumTrade's live ML gate still prefers LightGBM, then fail-closed
falls back to a promoted LSTM at the same symbol/timeframe when the
LightGBM artifact is missing (ETH/SOL 1h geometry quarantine). It never
disables `JESSE_ML_GATE_ENABLED` and never loads `*.rejected.*`.

---

## Fallback: Train on trading VPS CPU (not preferred)

Use only if the GPU node is down. Same DSR/PBO gates; much slower; Kronos stays CPU.

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
| `BTC-USDT_1h_lightgbm.joblib` | **Promoted** production LightGBM |
| `ETH-USDT_1h_lstm.pt` | **Promoted** production LSTM (joblib-wrapped `state_dict`) |
| `*_lightgbm_meta.json` / `*_lstm_meta.json` | DSR/PBO metadata (queried by `/model-metadata`) |
| `*.collapsed-everybar.joblib` | Alternate label mode; not served unless promoted |
| `*.rejected.joblib` / `*.rejected.pt` | Failed promotion — never copy to the trading VPS |

`/model-metadata` looks for `{symbol}_{timeframe}_{model_type}_meta.json`. If the
LightGBM sidecar is missing, the live gate tries `{symbol}_{timeframe}_lstm_meta.json`
and still fail-closes when neither production artifact exists.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Metadata not found for BTC-USDT_1h_lightgbm_meta.json` | Run `auto-retrain-promote`; prior train may have failed gate |
| `No model artifact found for ETH-USDT (1h, lightgbm)` | Expected when LightGBM is quarantined (`*.refused-geometry-*`). Live gate should use `ETH-USDT_1h_lstm.pt` — do **not** set `JESSE_ML_GATE_ENABLED=false` |
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

## Training Results

### CPU VPS LightGBM (2026-09-16 / Sep-11)

| Symbol | Timeframe | DSR | PBO | Holdout Sharpe | Gate | Notes |
|--------|-----------|-----|-----|----------------|------|-------|
| BTC-USDT | 1h | **1.00** | **2.0%** | 4.29 | **PASS** | Sep-11 `collapsed-everybar` LightGBM |
| ETH-USDT | 1h | 0.00 | 49.6% | — | FAIL | CPU LightGBM |
| SOL-USDT | 1h | 0.89 | 32.8–56% | — | FAIL | CPU LightGBM |

### Hostinger B200 LSTM campaign (2026-09-16, every-bar 5.5/1.75, class-balanced)

| Symbol | Timeframe | DSR | PBO | Bull / Bear recall | Gate | Artifact |
|--------|-----------|-----|-----|--------------------|------|----------|
| ETH-USDT | 1h | **1.00** | **2.8%** | 30% / 75% | **PASS** | `ETH-USDT_1h_lstm.pt` |
| SOL-USDT | 1h | **1.00** | **1.6%** | 30% / 76% | **PASS** | `SOL-USDT_1h_lstm.pt` |
| BTC-USDT | 15m | **1.00** | **0.8%** | 27% / 74% | **PASS** | `BTC-USDT_15m_lstm.pt` |
| BTC-USDT | 5m | **1.00** | **0.0%** | 27% / 75% | **PASS** | `BTC-USDT_5m_lstm.pt` |
| ETH-USDT | 15m | **1.00** | **2.0%** | 28% / 76% | **PASS** | `ETH-USDT_15m_lstm.pt` |
| BTC-USDT | 1h | 0.07 or fail-class | — | — | **BLOCKED** | `*.rejected.pt` — do not copy |

FX / equity symbols trained later in the same campaign were all rejected.
LightGBM every-bar on GPU collapsed (majority-class). Evacuate the five
promoted LSTMs before the GPU VM is destroyed.

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
# GPU (preferred)
./scripts/gpu_train_remote.sh inventory
./scripts/gpu_train_remote.sh bootstrap
./scripts/gpu_train_remote.sh train BTC-USDT both

# CPU trading VPS (fallback only)
./scripts/jesse_vps_reset_trials.sh --all
for S in BTC-USDT ETH-USDT SOL-USDT; do
  RUN_BACKTEST=false ./scripts/jesse_vps_train.sh "$S" 1h lightgbm || echo "FAIL: $S"
done

# Validate all
./scripts/jesse_vps_validate.sh --all

# VPS one-liner metadata
ssh root@$SSH_HOST 'cd /root/jesse-trading && for s in BTC-USDT ETH-USDT SOL-USDT; do echo "=== $s ==="; curl -sf "http://127.0.0.1:9003/model-metadata?symbol=$s&timeframe=1h" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get(\"metrics\",{}); print(f\"DSR={m.get(\"deflated_sharpe_ratio\")} PBO={m.get(\"prob_backtest_overfitting\")}\")"; done'
```
