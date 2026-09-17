# Trade Logic Review (v2)

**Supersedes:** informal review dated 4 Sep 2025  
**Verified against:** `origin/main` + `cursor/live-sync-ae57` (17 Sep 2026)  
**Scope:** Per-cycle loop guards, per-entry decision pipeline, exit path, veto matrix, known gaps

---

## 1. Per-cycle flow (`trading_loop.py`)

Each loop iteration, in order:

1. **Sentry halt** — if `is_trading_allowed()` is false, the entire cycle is skipped.
2. **Risk Guard** — `enforce_risk_limits()` (open positions, exposure, 72h peak drawdown, UTC daily loss). Fail-closed. `DISABLE_RISK_GUARD=true` is **ignored in LIVE** (paper only).
3. **Kill switch / margin pre-gate** — equity snapshot once per cycle. Blocks **new entries and pyramids** when equity is unavailable, below floor, or margin caps trip. **Exits always run** (SL/TP/trail/partial-TP/emergency).
4. **Symbol-quality gate** — blacklist, Binance `TRADING` status, 24h quote-volume floor, per-symbol negative-expectancy (30d / ≥20 closed trades). Volume fetch failure → **fail open** (keep candidates). Empty ticker map → **fail open** in code (log warns). Unknown ticker for a symbol with no open leg → reject new entry.
5. **New-bar gate** — when `eval_on_new_bar_only=true` (default), re-evaluation within the same bar is skipped (avoids duplicate entries on a 5‑min poll against 1h signals).
6. **Per-symbol evaluation** — `DecisionEngine.evaluate_symbol()` → execution with per-broker circuit breaker.

---

## 2. Per-entry flow (`decision_engine._evaluate_symbol_internal`)

Strict order — each step can kill the trade; nothing downstream resurrects it.

| Step | Gate | Notes |
|------|------|-------|
| 0 | Strategy signal | `CombinedStrategy` + cached `MarketRegimeDetector` per symbol |
| 1 | Pyramiding + RANGING | Pyramids blocked in RANGING/CHOP |
| 2 | Early RANGING block | Hard block unless config path matches (see §4) |
| 3 | Funding nudge | Adjusts confidence; cap `funding_conf_adj_cap` (default 0.05) |
| 3b | Confidence threshold | `min_signal_strength`; **+0.15 in RANGING** |
| 3c | Funding-rate cap | Blocks BUYs when `current_funding_rate > funding_rate_cap` |
| 4 | Kronos pre-execution | Timing guard, optional vision, path metrics (`apply_kronos_gate`) |
| 4d | Sentiment gate | Only if `SENTIMENT_FILTER_ENABLED=true` (default **off**) |
| 4e | Jesse ML gate | QTP promotion contract, conformal HIGH, opposing prob ≥0.50, Kelly cap |
| 5 | AI opinion layer | Only if `enable_personas=true` (default **off** for entries) |
| 6 | Max positions | Flatten/reverse of existing leg is not a new slot |
| 7 | Decision build | `_create_entry_decision`: RANGING re-check, notional caps, **min-edge fee gate** |
| 8 | LLM Risk Reviewer | Veto on ticket (incl. pyramid adds when that path runs reviewer) |
| 9 | Event-Risk Filter | Only if `EVENT_RISK_FILTER_ENABLED=true` (default **off**) |

---

## 3. Exit path (`position_manager.py` + loop)

Priority (simplified):

1. **Emergency drawdown** — `pnl_pct <= emergency_drawdown_pct` (default **−8%**). Exempt from min-hold and most other gates.
2. **Min hold** — `min_position_hold_min` (default **20 min**). Blocks AI reversal and technical deterioration exits; **not** emergency, SL/TP, trail, partial-TP.
3. **Time-based exit** — max hold hours (+ funding hold bonus on shorts).
4. **AI reversal** — opinion layer on exits when `opinion_layer_fn` is wired (uses `exit_opinion_threshold`; personas optional).
5. **Technical deterioration** — indicator-based exit signal.

Loop-level **SL/TP / trailing / partial-TP** (incl. `#100` BE+fees ratchet after confirmed partial) continue even when entries are margin-blocked.

---

## 4. Veto matrix

| Gate | Vetoes | Fail mode | Default |
|------|--------|-----------|---------|
| Sentry halt | entire loop | — | on |
| Risk Guard | whole cycle | fail-closed; `DISABLE_RISK_GUARD` ignored in LIVE | on |
| Kill switch / margin gate | entries + pyramids only | equity unavailable → block entries | on |
| New-bar gate | re-eval same bar | — | on (`eval_on_new_bar_only`) |
| Symbol gate | new entries on symbol | volume fetch fail → **fail open**; empty snapshot → **fail closed**; no ticker → reject if no open leg | on |
| RANGING block | new entries + pyramids | hard unless `allow_ranging_entries` + setup match + (`allow_ranging_in_live` in LIVE) | **blocked** |
| Funding-rate cap | BUYs above cap | — | env (`funding_rate_cap`) |
| Kronos gate | entries opposing forecast | **shadow** logs only (`TIMING_GATE_SHADOW=true`) | shadow |
| Sentiment gate | extreme divergence | error → neutral (no veto) | **off** |
| Jesse ML gate | entries | **fail-closed in LIVE** on errors, 5xx, promotion REJECT; **`no_model` → veto in LIVE** (paper skip + `jesse_ml_gap`) | on (`JESSE_ML_GATE_ENABLED`) |
| Jesse four-number | entries when QTP `promote` | missing telemetry → veto (paper fail-open on missing only) | when promotion bundle active |
| AI opinion layer | entries when weak | error → requires conf +0.1 vs `min_signal_strength` | only if `enable_personas` |
| Max positions / notional caps | entries | — | on |
| Min-edge fee gate | entries | paper: fail open on error; **live: fail closed** when enabled; `min_edge_fee_mult=0` disables | on (`MIN_EDGE_FEE_MULT`) |
| LLM Risk Reviewer | entry ticket | fail-closed in LIVE unless `RISK_REVIEWER_FAIL_OPEN=true` | on |
| Event-Risk Filter | entries ±30m/−15m macro | stale calendar → fail-closed in LIVE when enabled | **off** |
| Circuit breaker | execution on one broker | — | on |

### Jesse ML sub-gates (live path, when gate enabled)

```
signal → QTP REJECT? → live veto
      → Jesse /predict
           no_model / 5xx / error     → live veto (paper skip)
           success + promote          → four-number check (mapped telemetry on live-sync branch)
           uncertainty HIGH / gated     → veto
           opposing ML direction        → veto
           Kelly                        → ≤ 1.0 × configured risk (thin-book clip)
```

LSTM fallback (`JESSE_ML_FALLBACK_MODEL_TYPE=lstm`): when LightGBM is missing/quarantined, bridge retries `/predict` with LSTM before returning `no_model`.

---

## 5. Fixed since Sept 4 review (verified in code)

| P0 item | Status |
|---------|--------|
| FX sizing uses live equity | `_resolve_equity()` in `signal_candidate_engine.py` (cTrader/Binance; $150 conservative fallback logged) |
| Candidate store TTL | `_prune_terminal_candidates()` (24h terminal eviction) |
| Min hold on exits | `position_manager` enforces `min_position_hold_min` on AI/technical exits |
| Regime detector cache | `_regime_detectors` in `trading_loop.py`, injected into `DecisionEngine` |

### Fixed since Sept 4 — trading safety (#100, on `main`)

| Issue | Fix |
|-------|-----|
| Partial TP left SL at −1.0 ATR | `_ratchet_stop_to_be_fees` after confirmed partial (paper + live) |
| Live `no_model` skipped gate | Live: **veto**; paper: skip + `jesse_ml_gap` |
| Missing four-number telemetry skipped | Live: fail-closed; paper: fail-open on missing telemetry only |
| Kelly uncapped | Cap 1.0 (`clip_kelly_for_thin_book`) |
| Min-edge ignored 50% partial | Blend when partial TP on |
| `/jesse/sync` on REJECT | Restore `.env` / HTTP 409 on REJECT, exception, ROLLBACK |

### Live-sync branch additions (17 Sep 2026)

- Four-number field mapping: Jesse `conformal_margin` / `expected_value_r` → `conformal_width` / `costed_edge_bps` in bridge
- Enrichment ingest routes + n8n workflows (ingest-only; opinion layer not yet reading new measurements)
- `qtp-gate` service in `docker-compose.prod.yml` (Traefik ForwardAuth)

---

## 6. Fixed 17 Sep 2026 (code hardening)

| Item | Fix |
|------|-----|
| Gated NEUTRAL four-number pass | Jesse telemetry omitted when `gated` or final signal is NEUTRAL |
| Empty volume snapshot | Fail **closed** for new entries; open legs kept |
| Min-edge in LIVE | Fail **closed** on error or missing inputs when gate enabled |
| Volume floor code default | `RiskConfig` default **$3M** (matches `.env.example` for USDC perps) |

## 7. Still open / watch list

1. **RANGING hard-blocked by default** — opt-in path exists (`allow_ranging_entries`, `allow_ranging_in_live`, mean-reversion / mined-skill match). Mined-edge vs ranging contradiction unresolved unless flags flipped and shadow-measured.

2. **Kronos veto unscored in production** — default shadow mode; `shadow_tracker.py` exists but must be run to measure would-veto edge.

3. **Confidence is heuristic** — additive boosts/dampens feed fixed thresholds; no calibration map.

4. **SL/TP clamp** — `compute_sl_tp_levels()` can override strategy levels with generic ATR min/max distances (TPs pushed farther).

5. **Opinion layer asymmetry** — exits can use opinion when wired; **entries** skip opinion unless `enable_personas=true` (heuristics-only by default).

6. **Fail-open vs fail-closed asymmetry** — sentiment errors still allow trades; Jesse ML and Risk Reviewer fail closed in live. Min-edge now fail-closed in live when enabled.

7. **QTP promotion bundle** — `QTP_PROMOTION_*` unset on VPS → Jesse artifact gates apply; first real PROMOTE needs telemetry mapping (live-sync) and process restart for promotion snapshot at engine construct.

8. **No promoted LightGBM for several majors** — ETH/SOL use LSTM fallback; altcoins without 1h LSTM → `no_model` live veto (quieter book by design).

9. **Dual FX/crypto brains, research ingest on uvicorn loop, credential rotation** — operational; not resolved in trading logic.

---

## 8. Configuration quick reference

| Variable | Default | Effect |
|----------|---------|--------|
| `TRADING_MODE` | paper | live vs paper fail modes |
| `JESSE_ML_GATE_ENABLED` | true | Jesse ML veto chain |
| `JESSE_ML_FALLBACK_MODEL_TYPE` | lstm | Retry LSTM when LightGBM missing |
| `TIMING_GATE_SHADOW` | true | Kronos shadow (no block) |
| `SENTIMENT_FILTER_ENABLED` | false | News sentiment entry gate |
| `enable_personas` | false | AI opinion on **entries** |
| `EVENT_RISK_FILTER_ENABLED` | false | Macro event window gate |
| `MIN_EDGE_FEE_MULT` | 2.5 | Fee-churn gate (0 = off) |
| `MIN_24H_QUOTE_VOLUME_USDT` | 3M | Liquidity floor (code default matches `.env.example`) |
| `allow_ranging_entries` | false | RANGING opt-in |
| `RISK_REVIEWER_FAIL_OPEN` | false | LLM reviewer errors |
| `DISABLE_RISK_GUARD` | false | Ignored in LIVE |

---

## 9. Related files

| Area | Primary modules |
|------|-----------------|
| Loop | `backend/services/trading_loop.py`, `trading_loop_helpers.py` |
| Entries | `backend/services/decision_engine.py` |
| Exits | `backend/services/position_manager.py` |
| Jesse ML | `backend/services/jesse_bridge.py`, `jesse_ml_gates.py` |
| Kronos | `backend/services/kronos_gate.py`, `kronos_service.py` |
| Risk | `backend/services/risk_guard.py`, `risk_config.py` |
| Promotion | `backend/ml/promotion_service.py`, `live_signal.py` |
| Sizing | `backend/services/signal_candidate_engine.py` |

---

*Last verified: 17 Sep 2026. Re-verify after merges to `main` or VPS deploys.*
