# Bayesian optimization for QuantumTrade Pro

Exploration of which Bayesian optimization (BO) methods fit this repository, and which search surfaces they are allowed to touch. This document does not add a tuner, does not change live gates, and does not place orders.

The promotion contract already names the sampler correction. `purgedcv` 0.1.6 (the training pin in `backend/ml/requirements-train.txt`) ships `effective_n_trials` and `TrialSharpeRecorder` specifically because sequential samplers such as Optuna TPE and CMA-ES draw each trial conditioned on the previous ones. The Decision Engine refuses a raw Optuna count (`N_TRIALS_NOT_EFFECTIVE` in `backend/ml/promotion_gates.py`). Any BO work in this repo is a GPU-job search that writes `metrics.json` through `backend/ml/gpu_job.py` / `backend/ml/purgedcv_metrics.py`.

## Recommendation

| Surface | Where it lives today | Method | Runs where |
|---|---|---|---|
| LightGBM meta-labeler hyperparameters | `LGBM_GRID` (6 hand points) in `jesse_quant/train_ml.py` | Single-fidelity Optuna TPE, callback `TrialSharpeRecorder` | GPU training job only |
| Fractional-difference `d` | feature schema / Phase 1 fracdiff | 1-D Gaussian process with noisy expected improvement, still recorded in trial order | Same GPU job, its own study |
| Opinion-agent weights | `backend/services/opinion_layer.py` floors 0.02, cap 0.35, learning floors | Offline TPE on the softmax simplex, objective = shadow costed Sharpe | Research artifact, shadow log |
| Decision Brain weights (funding, OI, volume, TA, sentiment) | Phase 6 proposal workflow | Regime-bucketed offline TPE once `signal_outcomes` exist | Shadow proposals only |
| Live four-number thresholds (`p_win_min` 0.55, `width_max` 0.35, `min_costed_edge_bps`) | `backend/ml/geometry.py`, `backend/ml/live_signal.py` | Reliability diagram + conformal coverage already in the contract | Leave the thresholds frozen |
| Triple-barrier geometry (SL 1.75 ATR / PT 5.5 ATR) | `HOUSE_SL_ATR_MULT` / `HOUSE_PT_ATR_MULT` | Leave it locked for `strategy_id` `qtp-lgbm-meta-v1` | A new geometry is a new `strategy_id` with its own full study |
| Online tuning of risk-guard, kill-switch, or size | `risk_guard`, live loop | No BO method | Stays fail-closed and out of the search |

TPE is the default because the pinned library's trial correction was written for it, the LightGBM space is mixed (integers, log-scale rates, conditionals such as `num_leaves` given `max_depth`), and Optuna is already the optional extra `purgedcv[optuna]` on the training node. A Gaussian process is the right surrogate only for a handful of continuous knobs. Neither sampler belongs in the VPS image: `backend/requirements.txt` stays free of Optuna, BoTorch, and GPyTorch. The engine re-reads `metrics.json`; it does not run a study.

## How this repo counts a sequential search

Bailey & López de Prado's deflated Sharpe ratio treats `n_trials` as the number of **independent** configurations. A TPE study of 200 trials is not 200 independent bets. `purgedcv.effective_n_trials` (0.1.6, method `"autocorr"`) estimates that with the integrated autocorrelation time of the **trial-order** performance series:

```
tau   = 1 + 2 * sum_k rho_k     # Geyer 1992 initial-positive-sequence:
                                # stop at the first lag with rho_k <= 0
n_eff = round(n / tau)          # integer clamped to [1, n]
```

Fewer than 3 trials returns the raw count. A constant series returns 1. The library documents this as a heuristic that assumes trial order is the sampler's dependence order.

`TrialSharpeRecorder` (`purgedcv.optuna_integration`) is an Optuna callback. It appends each trial's finite user attribute `"sharpe"` (falling back to the objective value) and exposes `n_trials()`, `n_effective()`, and `var_sharpe(ddof=1)`. QTP passes `n_effective()` into `deflated_sharpe_ratio`. The library's own usage example that passes `recorder.n_trials()` is the raw count this contract rejects.

Measured with that function, seed 0:

| Series | Raw n | n_eff |
|---|---:|---:|
| Independent normal draws | 200 | 160 |
| AR(1) with lag-1 correlation 0.85 (8 seeds) | 200 | 12–49 |
| 40 wide draws then 160 local draws | 200 | 12 |
| Same AR(1) series sorted by Sharpe | 200 | 4 |
| Constant series | 200 | 1 |
| The current 6-point `LGBM_GRID` values | 6 | 6 |

Two consequences follow.

1. Replacing the 6-point grid with a 200-trial TPE study does **not** automatically collapse DSR, provided `n_trials` is `n_effective()` in execution order. The raw count would over-deflate (the failure mode the library calls out: DSR driven to zero). The contract already forbids that raw count.
2. The same estimator is easy to bias downward, which would inflate DSR. Sorting trials by Sharpe before recording dropped the AR(1) example from the 12–49 band to 4. Dropping failed trials changes both `n` and the autocorrelation. The recorder silently skips non-finite values, so a trial that raises or returns NaN disappears from `n_effective`.

Rules that keep the heuristic honest:

- Record Sharpe in **execution order**. Never sort, shuffle, or resume into a different order before `n_effective()`.
- Every evaluation the sampler used records a finite per-period Sharpe, including poor configs. A crashed trial is a large negative sentinel, not a skip.
- `n_trials_raw` is the number of evaluations the sampler consumed, including sentinels. The gate already rejects `n_trials_effective > n_trials_raw`.
- `trials_returns.parquet` keeps every completed return path. Winner-only matrices are `PBO_MATRIX_INCOMPLETE`.
- Prior studies on the same bars stay in the ledger (`jesse_quant/trial_registry.py`). Add the prior raw count on top of this study's `n_effective()`. Do not drop the ledger, and do not splice old Sharpe scalars into this study's autocorrelation series (they were not drawn by this sampler, so they are extra independent trials, not extra lags). Passing only the new study's `n_eff` repeats the bug where a small `n_trials` pushes DSR to 1.
- `var_sharpe` is the variance of the same per-period Sharpe series. `deflated_sharpe_ratio` compares it to the per-period Sharpe of the selected costed returns. An annualized trial Sharpe against per-period returns is a unit error.
- Do not replace `effective_n_trials` with a homegrown pairwise-correlation formula. The contract pins this library the same way it pins DSR.

The independent-draw row landed at 160 rather than 200 because a chance positive lag stops the sum early. Treat `n_eff` as an order-of-magnitude correction, then let the rest of the gates (PBO, min trades, costed edge, sealed holdout) carry the promotion decision.

## Methods

### TPE (Tree-structured Parzen Estimator)

Bergstra, Bardenet, Bengio, and Kégl (2011). Optuna's default sampler. It models two densities, `p(x | y good)` and `p(x | y bad)`, and proposes points that maximize their ratio. There is no Gaussian process and no acquisition-function inner loop.

Use it for the LightGBM study. The space is the one `LGBM_GRID` already sketches, widened rather than hand-picked:

- `n_estimators`, `max_depth`, `num_leaves` (conditional on depth), `min_child_samples`: integers
- `learning_rate`, `reg_lambda`: log-uniform
- `colsample_bytree`, `subsample`: uniform on (0, 1]

A 6-point grid has `n_eff = 6` and teaches the ledger almost nothing about sampler dependence. TPE starts to matter once the study is large enough for the autocorrelation correction to move (tens to a few hundred trials). Cap the study. The sealed holdout is not one of the trials.

Multivariate TPE (Optuna's `multivariate=True`) is appropriate here because `num_leaves` and `max_depth` move together. Constant-liar / constant-liar batch proposals are unnecessary while the GPU job evaluates one config at a time.

### Gaussian process with noisy expected improvement

Jones, Schonlau, and Welch (1998) for expected improvement. Letham et al. and BoTorch's qNoisyEI for the noisy case.

The surrogate is a GP, default kernel Matérn-5/2 with a noise term. Expected improvement assumes the observed objective is exact. A Sharpe ratio over a finite purged fold is not exact, so the acquisition is noisy EI, which integrates the improvement over the posterior of the incumbent as well as the candidate.

Use a GP for one to about eight continuous parameters where each evaluation is a full purged backtest: fractional-difference `d` on a bounded interval, or a small weight vector. Past that dimension the GP's lengthscales are weakly identified on the trial budgets this project can afford, and TPE is the more stable proposal engine.

Knowledge gradient (Frazier) values information for a final freeze, which matches "search, then write one artifact." It costs an inner optimization per suggestion. Use it only as a last batch before the holdout is spent, and only if noisy EI has already plateaued. Max-value entropy search targets the location of the maximizer; the contract promotes on the **value** of the selected costed path (DSR, PBO, path Sharpe), so noisy EI matches the decision better.

### Random-forest surrogates (SMAC)

Hutter's sequential model-based algorithm configuration. A random forest handles mixed and conditional spaces at higher dimension than a GP. The LightGBM space is small enough that TPE and SMAC will propose similar points, and SMAC would be a second sampler whose trial dependence `effective_n_trials` was not written against. Stay on TPE unless a future space grows past a few dozen conditional dimensions.

### CMA-ES

Hansen's covariance-matrix adaptation is an evolution strategy, not a Bayesian model. `purgedcv` names it next to TPE because its trial series is also autocorrelated, so the same `n_effective()` correction applies. It wants a continuous box. LightGBM's conditional integers and the opinion-weight simplex need a transform before CMA-ES is even defined. TPE already covers those spaces. Adding CMA-ES would spend a second ledger budget to learn a sampler the contract does not need.

### Trust-region BO (TuRBO)

Eriksson, Pearce, Gardner, Turner, and Poloczek (2019). Several local GPs, each inside a trust region, for roughly 10–100 continuous dimensions. The opinion stack is about ten weights with a cap of 0.35 and a floor of 0.02. A softmax (or stick-breaking) reparameterization brings that simplex back to a low-dimensional continuous box that ordinary TPE or a single GP can search. TuRBO becomes relevant only if a later study jointly tunes model hyperparameters and weights in one vector. Keep those studies separate so each `strategy_id` has one search and one ledger scope.

### Multi-objective: qNEHVI

Daulton, Balandat, and Bakshy. The hypervolume improvement acquisition over a Pareto front of costed Sharpe, maximum drawdown, and turnover.

A Sharpe-only objective is how a search discovers a high-variance config that PBO later rejects. For opinion weights and Decision Brain weights, evaluate the vector of objectives and pre-commit the selection rule in the study config before the first trial: for example, among configs with enough trades and path-Sharpe p10 at or above the geometry floor, take the best costed Sharpe. That rule is one configuration submitted to the gates. Every Pareto evaluation still enters the Sharpe series and the PBO matrix. Changing the knee rule after seeing the front is another search and needs a new `strategy_id`.

### Outcome constraints

Gardner et al. (ICML 2014) constrained BO, and Sui et al. SafeOpt. Constrained BO avoids proposing configs that violate a cheap constraint (too few trades, weight outside the cap) when the constraint can be predicted. Those rejected-before-evaluation points were never scored on the bars, so they do not enter `n_trials`. Points that were evaluated and then failed the constraint do enter.

SafeOpt's guarantee is for **online** queries that must stay inside a safe set with high probability before they are run. That is the wrong tool for this system. Live SL/TP, risk limits, kill-switch thresholds, and order size are not query points. The research plane is already non-gating; a BO loop is the same kind of subsystem.

### Multi-fidelity: BOHB, Hyperband, ASHA

Falkner, Klein, and Hutter (2018) for BOHB. Successive halving stops unpromising configs at a lower fidelity. In this codebase the only honest fidelity is a shorter but still **purged and embargoed** path (fewer CPCV paths, or a shorter training prefix with the embargo audit in `train_ml.py` still green). A random row subsample leaks.

BOHB is deferred. A rung that the sampler used and then pruned has no full column for `trials_returns.parquet`. Omitting it trips `PBO_MATRIX_INCOMPLETE` or, if the column is dropped quietly, understates the search. Until a pruned rung has a defined complete return path at that fidelity, the GPU job stays single-fidelity TPE. One full purged evaluation per trial is the setting `TrialSharpeRecorder` and PBO already agree on.

### Contextual and regime-bucketed search

Krause and Ong (2011) contextual BO models the objective as a function of both the decision and a context vector. Phase 6's Decision Brain is exactly that shape: weights on funding, open interest, volume, technicals, and sentiment, and the right weights can differ in a high-funding regime versus a quiet one.

A full contextual kernel is more machinery than the shadow log will support at first. Split the shadow outcomes into a few regime buckets (funding sign, realized-vol tercile), run one small TPE or GP study per bucket, and give each bucket its own ledger scope. The deployed object is a frozen map from regime to weight vector. It proposes a decision. The risk guard still disposes, which is the same safety rule Phase 6 already states.

Do this after Phase 5 has real `signal_outcomes`. Tuning Decision Brain weights on an empty window repeats the Phase 4 problem of proposing from `candles: []`.

### Threshold search on the live four numbers

`p_win_min`, `width_max`, and `min_costed_edge_bps` are gates, not model hyperparameters. The contract forbids post-hoc gate edits without a new `strategy_id`. Searching them on the same history that trained the model is a second multiple-testing layer aimed at the veto itself.

Phase 5 calibration stays with the tools already in the training path:

- `CalibratedClassifierCV` under `PurgedKFold` (isotonic, then sigmoid) in `jesse_quant/train_ml.py`
- the reliability diagram and `calibration_deploy_control` written next to those probabilities
- conformal coverage against `gates.min_conformal_coverage` (default 0.70)
- the live check in `backend/ml/live_signal.py`, which vetoes when any of `p_win`, `conformal_width`, or `costed_edge_bps` is missing

Jev confidence is not a substitute for those three numbers. Filling `p_win` from a Jev score would walk through the live four-number gate. BO does not get to invent them either.

## What the surrogate is allowed to see

The objective of every trial is the mean costed Sharpe on an **inner** `PurgedKFold` (embargo at least the vertical barrier and the longest feature lookback). Costs are the geometry block: taker fee, slippage, funding. Zero-cost Sharpe can be logged under `metrics.zero_cost` and is not the objective and not the DSR input.

The sealed holdout is spent once, after the study is frozen, and then `holdout.spent=true` for that `holdout_id`. It is not a trial, not an acquisition observation, and not the series used to pick `best_params`. The current `train_ml.py` grid scores Sharpe on the same holdout it uses to choose the winner and then computes DSR on that holdout. A TPE port that keeps that structure searches harder against the only out-of-sample slice. The port moves selection onto the purged inner score first, and only then replaces the grid.

Geometry for `qtp-lgbm-meta-v1` stays out of the search box: SL 1.75 ATR, PT 5.5 ATR, ATR period 14, vertical timeout 48, bar `1h`. Putting the barriers in the box and shipping the winner under the same `strategy_id` is a train/serve mismatch (`GEOMETRY_LIVE_LOCK`) and a large extra trial dimension.

Per-trial bookkeeping passed to `write_promotion_artifacts`:

- `trial_sharpes`: execution-order per-period costed Sharpes, one per evaluation
- `recorder`: the `TrialSharpeRecorder` instance, so `n_trials_source` is `TrialSharpeRecorder.n_effective`
- `raw_n_trials`: sampler evaluations plus prior ledger count
- `trial_returns_obs_by_config`: shape `(n_obs, n_configs)` including losers
- `costed_selected_returns`: the pre-committed winner's costed path, not a zero-cost path
- `promote_requested=false` until a human spends the holdout

`resolve_effective_n_trials` prefers the recorder when it is passed. The Sharpe series handed to the recorder has to be the same series the function would have used, in the same order.

## Where this sits in the combined plan

**Phase 3 (ML config).** Replace `LGBM_GRID` with one TPE study inside the GPU job. Wire the callback to the existing artifact writer. Keep the Decision Engine ignorant of Optuna. The study id and the ledger scope stay `ml:{symbol}:{timeframe}`.

**Phase 5 (calibration and telemetry).** Build `signal_log` / `signal_outcomes`, hit-rate against stated confidence, and per-strategy Sharpe from history. Use that to **audit** the frozen gates. Changing a gate number is a new `strategy_id` plus a new study, not an update to the live geometry file.

**Phase 6 (Decision Brain).** After outcomes exist, regime-bucketed offline TPE on the five proposal weights. The n8n workflow stays a proposer (`active: false`, no execute node). Optimized weights do not set `executable`.

**Opinion layer.** Same offline treatment. Projection onto the existing floor, cap, and learning-agent floors happens inside the objective, and the projected point is what gets scored. The live route that serves weights does not call a sampler.

**Phase 4 (Jev).** Unchanged. Jev proposes a `qtp.jev.signal.v1` envelope. BO is not a source of `p_win`, `conformal_width`, or `costed_edge_bps`.

## Definition of done for a later implementation

A patch that introduces the TPE study is done when all of the following hold.

- The inner objective is purged and costed. The holdout slice does not appear in any trial.
- `metrics.json` reports `n_trials_source = TrialSharpeRecorder.n_effective`, with `n_trials_effective` equal to this study's `n_effective()` plus the prior ledger count, and `n_trials_effective <= n_trials_raw`.
- `trials_returns.parquet` has one column per completed trial, losers included.
- Sorting or filtering the recorder series is covered by a unit test that would fail the promotion gate.
- House geometry is unchanged for `qtp-lgbm-meta-v1`.
- Optuna is imported only on the training path (`requirements-train.txt`). `backend/requirements.txt` and the runtime image do not gain it.
- No live loop, risk guard, Jev route, or n8n workflow reads the study.

## Sources

- Jones, Schonlau, Welch (1998), Efficient Global Optimization.
- Bergstra, Bardenet, Bengio, Kégl (2011), Algorithms for Hyper-Parameter Optimization (TPE).
- Bailey and López de Prado (2014), The Deflated Sharpe Ratio.
- Bailey, Borwein, López de Prado, Zhu (2014), The Probability of Backtest Overfitting.
- Snoek, Larochelle, Adams (2012), Practical Bayesian Optimization.
- Gardner et al. (2014), Bayesian Optimization with Inequality Constraints.
- Falkner, Klein, Hutter (2018), BOHB.
- Eriksson et al. (2019), TuRBO.
- Krause and Ong (2011), Contextual Gaussian Process Bandit Optimization.
- Daulton, Balandat, Bakshy, qNEHVI (BoTorch).
- Geyer (1992), Practical Markov Chain Monte Carlo — initial positive sequence, as implemented by `purgedcv.effective_n_trials` 0.1.6.
- López de Prado, Advances in Financial Machine Learning, chapters on purged CV and deflated Sharpe.
