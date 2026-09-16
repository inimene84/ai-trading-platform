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

---

# Multi-asset GPU run (2026-09-16, same B200 node)

Same live geometry and gates. Artifacts stayed on the GPU node. **Nothing
promoted.** Exit code 2 (fail-closed). BTC/ETH/SOL were excluded from this
batch because they were already trained and rejected on QuantumAI events
earlier the same day.

## Inventory (what actually exists)

| Source | What is there | Used for training? |
|---|---|---|
| Jesse Postgres (`candle`) | BTC-USDT 1,543,680 / SOL-USDT 889,920 / ETH-USDT 888,600 1m rows | Yes (prior run). No FX, metals, oil, or stocks in Jesse. |
| Live `TRADING_SYMBOLS` | 16 USDT/USDC perps (BTC, ETH, SOL, XRP, BNB, AVAX, LINK, NEAR, LTC, DOT, ATOM, OP, INJ, SUI, UNI, POL) | Extra crypto via public Binance Vision 1h klines (~32.5k bars from 2023-01-01). `api.binance.com` is geo-blocked (HTTP 451); Vision works. |
| `unified_feed` / `multi_asset_bars` | Forex majors, XAU/XAG/XPT/XPD, USOIL/UKOIL (CL=F/BZ=F), AAPL/MSFT/NVDA, SPX | yfinance 1h (~730d). SPX is an index — not trained. |
| cTrader | Live FX/metal quotes on the trading VPS | No historical dump available to copy. |
| Influx | Ops/trade metrics buckets | Not OHLCV. |
| Qlib / CSV corpora | None on either host | Skipped. |
| Copper / NG / industrial minerals | Not in platform maps | **Not invented. Skip.** |

Minerals/commodities in this run = WTI (`USOIL` / `CL=F`) and Brent (`UKOIL` / `BZ=F`) only.

## Batch

- CLI: `train_gpu.py --asset-class all --events quantum_ai --exclude BTC-USDT,ETH-USDT,SOL-USDT --model both`
- Geometry: 5.5 ATR TP / 1.75 ATR SL, 24-bar vertical, feature hash `cd15d2380809b247`
- Device: NVIDIA B200, PyTorch 2.11.0+cu128; LightGBM CPU fallback (no CUDA Tree Learner wheel)
- Wall time: ~60s for 30 symbols × LightGBM+LSTM (event N is tabular-small)
- Production `*.joblib` written this batch: **none**

### Extra crypto (Binance Vision 1h)

| Symbol | Events | LGBM DSR / PBO / bull / fail | LSTM DSR / PBO / bull / fail | Verdict |
|---|---|---|---|---|
| BNB-USDT | 2332 | ~0 / 49.6% / 0.6% / 99.7% | 0.999 / 0% / 0% / 100% | **BLOCKED** collapsed |
| XRP-USDT | 1918 | ~0 / 24.8% / 0% / 99.3% | ~0 / 0% / 0% / 100% | **BLOCKED** collapsed |
| AVAX-USDT | 1454 | ~0 / 62.0% / 2.9% / 99.1% | ~0 / 0% / 0% / 100% | **BLOCKED** collapsed |
| LINK-USDT | 1692 | ~0 / 38.8% / 0% / 99.6% | 0.77 / 0% / 0% / 100% | **BLOCKED** collapsed |
| UNI-USDT | 1648 | ~0 / 17.6% / 1.0% / 99.6% | ~0 / 8.8% / 0% / 100% | **BLOCKED** collapsed |
| NEAR-USDT | 1467 | ~0 / 26.4% / 0% / 100% | 0.06 / 26.4% / 6.2% / 97.4% | **BLOCKED** collapsed |
| LTC-USDT | 1933 | ~0 / 57.6% / 0.8% / 99.2% | 1.00 / 0% / 0% / 100% | **BLOCKED** collapsed |
| DOT-USDT | 1446 | ~0 / 0.4% / 0% / 100% | ~0 / 0% / 2.4% / 100% | **BLOCKED** collapsed |
| ATOM-USDT | 1503 | ~0 / 68.4% / 0% / 98.9% | ~0 / 7.2% / 13.5% / 95.7% | **BLOCKED** LGBM collapsed; LSTM DSR |
| OP-USDT | 1326 | ~0 / 87.2% / 1.4% / 99.5% | 1.00 / 0% / 1.4% / 100% | **BLOCKED** collapsed |
| INJ-USDT | 1509 | ~0 / 30.8% / 0% / 100% | 0.86 / 0% / 0% / 100% | **BLOCKED** collapsed |
| SUI-USDT | 1270 | ~0 / 74.4% / 0% / 100% | 1.00 / 0% / 0% / 100% | **BLOCKED** collapsed |
| POL-USDT | 717 | ~0 / 21.2% / 0% / 100% | 0.75 / 0% / 4.1% / 100% | **BLOCKED** collapsed |

LSTM on BNB/LTC/OP/SUI can print DSR≈1.0 with PBO≈0 while bullish recall is 0% — majority-class trap. Recall gate holds.

### Forex (yfinance `=X`, ~17.2k 1h bars)

| Symbol | Events | Closest failure |
|---|---|---|
| EURUSD | 1046 | collapsed (bull 1.9%) |
| GBPUSD | 1105 | collapsed (bull 0%) |
| USDJPY | 1264 | LGBM bullish recall **9.5%** (just under 10% floor), DSR ~0 |
| EURJPY | 1277 | collapsed (bull 2.2%) |
| AUDUSD | 1142 | collapsed (bull 1.2%) |
| USDCAD | 1122 | LGBM collapsed (bull 3.8%); LSTM both-class 59.6% / 42.6% but DSR ~0, PBO 69.6% |
| USDCHF | 1107 | collapsed (bull 0%) |
| NZDUSD | 943 | collapsed (bull 0%) |

### Metals (yfinance futures proxies)

| Symbol | Events | Closest failure |
|---|---|---|
| XAUUSD | 1057 | collapsed (bull 0%) |
| XAGUSD | 1010 | LGBM bullish recall **10.0%** (floor met), fail 93.9%, **DSR ~0 / PBO 48%** |
| XPTUSD | 822 | LGBM collapsed; LSTM bull 11.5% / fail 90.7%, DSR ~0 |
| XPDUSD | 715 | collapsed (bull 0%) |

### Minerals / commodities (oil only)

| Symbol | Events | Closest failure |
|---|---|---|
| USOIL | 742 | collapsed (bull 2.0%) |
| UKOIL | 733 | LGBM PBO 9.2% would pass PBO; recall 3.2% collapsed; DSR ~0 |

### Stocks (yfinance 1h, ~5.1k RTH bars)

| Symbol | Events | Closest failure |
|---|---|---|
| AAPL | 289 | collapsed (bull 0%); LSTM bull 9.5% |
| MSFT | 302 | collapsed (bull 0%) |
| NVDA | 281 | LGBM collapsed; LSTM bull 26.7% / fail 91.4%, DSR ~0, PBO 33.3% |

Equity event counts are the smallest (RTH-only 1h history). Still above the train floor of 20; none promoted.

## Data gaps / next data needed

- Jesse-quality 1m (or true cTrader H1) history for FX and metals — Yahoo futures/FX proxies are not the live cTrader book
- Copper, NG, industrial minerals: **no platform feed** — do not train until wired
- Qlib/Kronos corpus still absent
- Extra Binance 1m dumps for the 13 non-BTC/ETH/SOL perps (current extra-crypto run is 1h Vision klines)
- A less one-sided primary than the QuantumAI EMA/RSI long filter, or a short-side event set, before DSR/PBO can mean anything

Do not copy GPU `*.rejected.joblib` / `*.rejected.pt` onto the trading VPS. An overlapping every-bar campaign on this node wrote a few unsuffixed `*.pt` files for BTC/ETH/SOL timeframes; those are **not** this QuantumAI-event promotion batch and must not be live-deployed.

