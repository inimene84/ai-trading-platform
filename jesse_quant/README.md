# Jesse quant research scripts (VPS `/root/jesse-trading`)

These files are the promotion-gated training and serving contract for the
Jesse stack. Training can run on a dedicated **GPU node** (Hostinger GPU
instance). Live execution, risk, and broker connectivity stay on the
trading VPS. Huge joblib/pt binaries and candle dumps stay on those hosts
and are gitignored.

## Live geometry (do not drift)

- Stop-loss: **1.75 ATR**
- Take-profit: **5.5 ATR**
- Kelly payoff fallback \(b = 5.5 / 1.75 \approx 3.14\)
- Promote only if **DSR > 0.95**, **PBO < 0.30**, and **both-class recall ≥ 10%**
- `n_trials` for DSR is the cumulative hyperparameter-grid count (trial registry), not `5`
- Collapsed classifiers (always-SELL / 0% bullish recall) are never promoted

## GPU training node

The GPU box is an ephemeral accelerator: pull candles, train, write
`geometry.json` / `metrics.json` / `contract.json`, copy a promoted artifact
back to Jesse ML (`:9003`) only if the contract verdict is not `REJECT`.

```bash
# On the GPU node (venv created by bootstrap_gpu.sh)
./bootstrap_gpu.sh
./manage.sh gpu-inventory
./manage.sh gpu-train BTC-USDT 1h both storage/candles/BTC-USDT_1m.csv.gz
./manage.sh gpu-auto-retrain-promote BTC-USDT 1h both
```

From this repo, with GPU SSH settings in the environment (never commit them):

```bash
GPU_SSH_HOST=... GPU_SSH_PORT=... GPU_SSH_USER=ubuntu GPU_SSH_PASSWORD=... \
  ./scripts/gpu_train_remote.sh inventory
GPU_SSH_HOST=... GPU_SSH_PORT=... GPU_SSH_USER=ubuntu GPU_SSH_PASSWORD=... \
  ./scripts/gpu_train_remote.sh train BTC-USDT both
```

`gpu_device.py` prefers CUDA (PyTorch cu128 wheels cover Blackwell / B200).
LightGBM uses `device=cuda` when the GPU probe succeeds and falls back to CPU
if the CUDA booster cannot initialize. The LSTM meta-labeler trains with AMP
on `cuda:0`.

## Promotion contract

`promotion_contract.py` implements QTP Promotion Contract v1.0.0. The GPU job
writes metrics; the Decision Engine re-reads them and fail-closes live `/predict`
unless geometry, DSR, PBO, and class-recall gates pass. Sealed-holdout
`PROMOTE` is reserved for a later spend of `holdout_id`; a clean hard-pass
without that spend is `SHADOW`.

## Deploy inference to the trading VPS

Copy onto the Jesse workspace (bind-mounted at `/home` in `jesse-app`):

```text
promotion_gates.py
promotion_contract.py
gpu_device.py
train_ml.py
train_gpu.py
train_sequence.py
predict_server.py
triple_barrier.py
barrier_config.py
ml_features.py
feature_schema.py
validation_metrics.py
trial_registry.py
manage.sh
strategies/QuantumAIStrategy/__init__.py
```

Then:

```bash
./manage.sh auto-retrain-promote BTC-USDT 1h lightgbm
./manage.sh backtest QuantumAIStrategy
docker compose -f docker/docker-compose.yml restart jesse
```

Rejected artifacts are written as `*.rejected.joblib` / `*.rejected.pt` and
**do not** overwrite the production model. Inference refuses models that fail
the promotion gate unless `JESSE_ML_ALLOW_OVERFIT=true`.
