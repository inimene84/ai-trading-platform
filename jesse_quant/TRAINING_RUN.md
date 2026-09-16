# Jesse GPU training run (2026-09-16)

Artifacts stayed on the GPU training node (`jesse-gpu/storage/models/`). Joblib/pt
binaries and candle dumps are not committed. This GPU instance is a **new**
Hostinger training box, not an upgrade of the trading VPS (that host remains
CPU-only with Jesse Postgres + live QuantumTrade).

## GPU inventory

| Item | Value |
|---|---|
| GPU | NVIDIA B200 |
| VRAM | 182631 MiB (~178 GiB) |
| Driver | 580.159.03 (CUDA driver 13.0) |
| Toolkit on image | CUDA 12.6 (`nvcc`) |
| Training runtime | PyTorch **2.11.0+cu128** (sm_100 / Blackwell) |
| CPU / RAM / disk | AMD EPYC 9575F (224 threads), 64 GiB, 89 GiB overlay |
| Docker / Jesse | not installed on this node (by design) |

`gpu_device.py` reported `kind=cuda`, `lightgbm_device=cuda`. The PyPI LightGBM
4.7 wheel is **not** built with `-DUSE_CUDA=1`, so LightGBM fell back to CPU
after one failed CUDA Tree Learner init. LSTM training ran on `cuda:0`.

## Data

Candles were dumped from the trading VPS Jesse Postgres (no credentials stored
in this repo) and copied as gzip CSV onto the GPU node:

| Symbol | 1m rows | 1h bars | QuantumAI events (5.5/1.75 ATR, 24h vertical) |
|---|---|---|---|
| BTC-USDT | 1,543,680 | 25,728 (2023-10-05 → 2026-09-10) | 1,898 (675 TP / 1,223 fail) |
| ETH-USDT | 888,600 | — | 787 |
| SOL-USDT | 889,920 | — | 733 |

Event count is up from the 2026-09-11 CPU run (1,861 BTC events).

## Models trained (live geometry 5.5 / 1.75 ATR)

Gates kept: DSR > 0.95, PBO < 0.30, both-class recall ≥ 10%, collapsed-classifier
block. **No production `*.joblib` was written.** Rejected artifacts only.

### BTC-USDT 1h

| Model | DSR | PBO | Bullish recall | Fail-class recall | Verdict |
|---|---|---|---|---|---|
| LightGBM (CPU fallback, 6-trial grid, n_trials ledger=41) | ~0 | 59.2% | 0.8% | 96.9% | **BLOCKED** collapsed classifier |
| LSTM (GPU AMP, 3-trial grid) | 0.99998 | 15.6% | 0.0% | 100% | **BLOCKED** collapsed classifier |

BTC LSTM would have *passed* DSR and PBO if we ignored class balance. That is
exactly the majority-class trap: always predicting fail/SELL on a 64% negative
holdout. The recall gate is doing its job.

### ETH-USDT 1h

| Model | DSR | PBO | Bullish recall | Fail-class recall | Verdict |
|---|---|---|---|---|---|
| LightGBM | ~0 | 76.0% | 12.5% | 89.1% | **BLOCKED** DSR < 0.95 |
| LSTM | ~0 | 22.8% | 6.2% | 98.1% | **BLOCKED** bullish recall < 10% |

ETH LightGBM cleared the 10% recall floor and was still refused on DSR/PBO.

### SOL-USDT 1h

| Model | DSR | PBO | Bullish recall | Fail-class recall | Verdict |
|---|---|---|---|---|---|
| LightGBM | ~0 | 21.6% | 0.0% | 100% | **BLOCKED** collapsed classifier |
| LSTM | 0.99987 | 41.2% | 0.0% | 100% | **BLOCKED** collapsed classifier |

Live `/predict` on the trading VPS must keep fail-closing (`No model artifact`
or promotion-gate error). Do not copy rejected GPU artifacts into production.

## What was installed on the GPU node

- `python3-venv`, build-essential, tmux
- venv at `jesse-gpu/.venv`
- `torch==2.11.0+cu128` from the official cu128 index
- pandas, numpy, scikit-learn, lightgbm 4.7, pyarrow, scipy, joblib
- Trainer scripts synced from this repo (`train_gpu.py`, gates, contract)

CUDA matmul on B200 succeeded. LightGBM CUDA requires a from-source build
(`-DUSE_CUDA=1`); not done this run (tabular N≈2k does not need it).

## Next training steps

- Collect more QuantumAI-event samples or a less one-sided primary filter
- Rebuild LightGBM with CUDA only if event counts grow into the 100k+ range
- Kronos-small fine-tune is still blocked on Qlib-format corpus + HF weights
- CPCV path reconstruction with `purgedcv` once N_events supports 10–20 partitions
- Shadow-log predictions until PBO < 0.30 **and** both-class recall ≥ 10%

## B200 9h campaign (2026-09-16 03:30–04:12Z) — evacuate before VM destroy

Every-bar labels + class-balanced LSTM (WeightedRandomSampler, CE class weights,
B200_GRID hidden 256/512/768). Five artifacts **passed** gates and were copied
to the trading VPS `storage/models/` (rejected files stayed on the GPU only):

| Artifact | DSR | PBO | Bull / Bear recall |
|---|---|---|---|
| `ETH-USDT_1h_lstm.pt` | 1.0 | 2.8% | 30% / 75% |
| `SOL-USDT_1h_lstm.pt` | 1.0 | 1.6% | 30% / 76% |
| `BTC-USDT_15m_lstm.pt` | 1.0 | 0.8% | 27% / 74% |
| `BTC-USDT_5m_lstm.pt` | 1.0 | 0.0% | 27% / 75% |
| `ETH-USDT_15m_lstm.pt` | 1.0 | 2.0% | 28% / 76% |

BTC 1h LSTM and all later FX/equity symbols were rejected. Jesse ML now loads
`model_type=lstm` from `*.pt` on CPU torch (`/home/vendor/cpu-torch`). Live
QuantumTrade still defaults to LightGBM until an operator points a symbol at
LSTM. Re-run `python3 scripts/gpu_evacuate_promoted.py --push-vps` if the
staging dir is lost before the GPU is destroyed.
