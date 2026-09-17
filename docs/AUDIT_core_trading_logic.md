# Core Trading Logic Audit — `ai-trading-platform`

**Scope:** Does the system do what it should — trade crypto, profitably?
**Audited commit:** `98ec16b` (current `origin/main`).
**Nothing in the live trading path was modified.** No broker credentials were touched.

> **Correction notice.** A first pass of this audit was performed against a
> working copy that was **119 commits stale**. Two of its conclusions did not
> survive re-verification against current `main` and have been corrected below
> (see "Corrections"). Every finding that remains has been re-confirmed at
> `98ec16b` with the line numbers shown.

---

## Verdict

The execution mechanics are **correct**, and the newer ML promotion layer is
genuinely rigorous. I could not find a way to make the system place a wrong-sided
order, mis-direct a stop, or bypass the kill switch.

Profitability is **still not demonstrated**, but the reason is narrower than it
first appeared. The serious evaluation path already refuses to promote a model on
zero-cost metrics. What remains is one real hole in the exit ladder (Finding 1) and
one stale evaluation path that can still produce misleadingly good numbers
(Finding 2).

---

## Finding 1 — HIGH (confirmed at `98ec16b`): no breakeven ratchet after partial TP

The exit ladder has an unprotected window between 1.0 and 2.0 ATR.

| Level | What happens |
|---|---|
| **1.0 ATR** (`partial_tp_atr_mult`) | 50% closed for profit (`partial_tp_close_pct = 0.50`) |
| 1.0 → 2.0 ATR | **stop is still at the original −1.0 ATR** |
| **2.0 ATR** (`trail_activation_atr`) | trail arms, locks ≈ +1.2 ATR (`− trail_atr_mult = 0.8`) |

Verified by tracing every write to `trade.stop_loss` in the repo. There are exactly
four, all inside `TrailingStopManager.apply_trailing_stop`
(`trading_loop_helpers.py:538, 552, 568, 581`). Both partial-TP functions —
`apply_partial_tp` (`:629`, paper) and `apply_partial_tp_live` (`:709`, live) —
update `trade.quantity` and `trade.notes` and **never touch `trade.stop_loss`**.

So a trade can run +1.0 ATR in your favour, bank half, then hand the remaining half
back for a full −1.0 ATR loss. That is consistent with the signature recorded in
`risk_config.py:248-251`:

> "Live week: 69/72 wins closed near a raised SL and 0 hits of full TP — early
> trail/STEP-TRAIL clipped winners while losers paid full risk
> (avg win $0.34 vs avg loss $0.66)."

The one mechanism that would close the window, `step_trail_enabled`, is **off by
default**, and `risk_config.py:266-267` explains why: it "was the dominant 'win near
SL' path and destroyed payoff asymmetry." That reasoning is sound — but note it
applies to ratcheting at **0.75× activation regardless of profit taken**. Ratcheting
only *after the partial has actually filled* is a different trigger: at that point
realized profit is already banked, so protecting the remainder at breakeven cannot
clip a winner that hasn't already paid for itself.

**Suggested fix:** in both partial-TP functions, after a confirmed fill, ratchet the
stop to entry ± (round-trip cost in price terms), monotonic in the favourable
direction only, then call the existing `TrailingStopManager._sync_exchange_stop`.
Strictly tightening; cannot widen a stop.

**Honest caveat on magnitude.** I built a zero-edge random-walk simulation to size
this. It suggested only a **modest** gain (+6% to +11% of the per-trade loss), and
it **failed to reproduce the live signature** (0% scratches simulated vs 52%
recorded live). The model is therefore not validated and I am not presenting its
numbers as a projection. Treat this as a real structural hole worth closing, not as
a quantified profit opportunity.

---

## Finding 2 — MEDIUM (confirmed at `98ec16b`): the legacy backtester models zero costs

`backend/backtesting/crypto_backtester.py` computes PnL as a raw price difference
with **no fee, slippage, or funding term anywhere in the file**:

```python
if direction == 'BUY':
    pnl = (exit_price - entry_price) * qty
else:
    pnl = (entry_price - exit_price) * qty
self.cash += pnl
```

It also does not model partial TP at all (it does model trailing), so its exit
mechanics differ from live on top of being uncosted.

This matters because the live cost is the dominant PnL term. `roundtrip_cost_rate`
= `2 * (taker_fee_rate + slippage_rate)` = **0.12% of notional per round trip**, and
`risk_config.py` records the live outcome:

> "81-day history: 52% of trades were scratches (|PnL| <= $0.20) and commission drag
> was **-$193.54 on a +$64 net**"

Commissions were roughly three times the net result.

**Why this is MEDIUM and not CRITICAL.** The engine is *not* being tuned on this
path any more. `backend/ml/promotion_gates.py` implements a fail-closed
`ZERO_COST_ONLY` gate (`:333-338`) that **rejects promotion** unless a costed metric
series is supplied, and `backend/ml/costs.py` computes returns net of fees,
slippage, and funding — its own docstring states "promoting on zero-cost DSR/Sharpe
alone is ZERO_COST_ONLY." That is exactly the right design.

The risk is that the uncosted engine is **still reachable** and will still hand back
flattering numbers: `backend/routes/backtest.py:5` and `backend/cli/crypto_backtest.py:4`
both import `CryptoBacktestEngine`. Anyone reading a number off that route or CLI is
reading a gross figure.

**Suggested fix:** either charge `roundtrip_cost_rate` per fill inside
`_execute_trade`/`_close_trade` and implement partial TP so it matches live, or mark
the route/CLI output explicitly as gross-of-cost and diagnostics-only. Until one of
those, no number from that path should be read as evidence of edge.

---

## Corrections to the first pass

Both of these were wrong. Recording them so no one spends time on them.

**1. The min-edge gate is NOT overestimating edge against the unreachable full TP.**
This was my leading hypothesis and it is false. `_passes_min_edge`
(`decision_engine.py:1075-1095`) already accounts for the trail:

```python
captured_atr = max(0.0, activation - trail_mult)   # 2.0 - 0.8 = 1.2 ATR
expected_move = min(tp_distance, captured_atr * atr)
```

It gates on the 1.2 ATR the trail actually locks, not the 2.5 ATR `tp_atr_mult` the
config notes say is never reached ("0% TP rate over 500 trades"). Correct,
well-reasoned code.
*Residual, minor:* it does not blend in the 50% closed at 1.0 ATR. True blended
capture ≈ `0.5*1.0 + 0.5*1.2 = 1.1` ATR vs the gate's 1.2 — about **8% optimistic**.
Worth tidying for correctness; not a profit driver.

**2. The kill floor IS enforced.** I initially reported `kill_floor_usdt` as dead
config. That was my error, caused by a truncated grep. It is enforced in the loop,
halts trading, and requires a manual restart.

---

## Checked and found correct

- **SL/TP geometry** (`compute_sl_tp_levels`) — correct for both directions; the
  `min`/`max` clamps make a signal-supplied level only ever *tighter* than the ATR
  level, never looser.
- **Order side / hedge mode** — `positionSide` is always the position side (`LONG`
  for BUY, `SHORT` for SELL); protective orders take the opposing side
  (`sl_side = 'SELL' if position_side == 'LONG' else 'BUY'`). The comment "In Hedge
  Mode, we DO NOT send reduceOnly as positionSide handles it" is correct Binance
  behaviour.
- **Protective orders** use `closePosition: "true"` with `STOP_MARKET` /
  `TAKE_PROFIT_MARKET`, and the code explicitly handles Binance `-4130` (one
  `closePosition` order per side) plus the pyramid case where an existing stop
  already covers added size.
- **Fee accounting is not understated.** Three fills occur (entry 100% + partial 50%
  + final 50% = 200% of notional) while the gate models two legs — arithmetically
  equivalent.
- **`trail_atr_mult` (0.8) < `trail_activation_atr` (2.0)**, so the documented
  "trail locks a LOSS at activation" failure mode (leak #4) is not active.
- **Liquidity and blacklist filters are enforced**, addressing the recorded
  meme-coin loss tail (`SIREN, AIGENSYN, MAGMA, SPK`, ≈ −$300).
- **Trailing never clamps the stop to the mark** — an earlier bug, now explicitly
  guarded (`if candidate >= current_price: continue`).
- **Live exits are exchange-held** and therefore tick-precise; `_check_sl_tp`
  correctly defers full closes to the exchange in live mode, and the close-price
  comparison path is paper-only.
- **ML promotion gates** are fail-closed and cost-aware, including guards against
  second-peek holdout overwrite (`holdout_registry.py`) and zero-cost promotion.
  This is the strongest part of the codebase.

## Risk posture, for the record

`BINANCE_LEVERAGE` defaults to **10x** with `risk_per_trade_pct = 0.01`,
`max_trade_notional_equity_mult = 2.0`, `max_positions = 4`,
`max_directional_exposure_usdt = 500.0`, `emergency_drawdown_pct = -8.0`,
`max_portfolio_drawdown_pct = 20.0`, `max_daily_loss_pct = 5.0`,
`kill_floor_usdt = 65.0`. Layered and enforced — but 10x leverage on a strategy
whose net-of-cost edge is not yet demonstrated is the main standing exposure.

---

## Separately: a live-config issue found during the ops inventory

> **Addendum (2026-09-16, verified at commit `bd3a7a3`).** This specific
> finding is **resolved in compose**: `docker-compose.prod.yml` now
> interpolates `PAPER_TRADING: ${PAPER_TRADING:-true}`,
> `DRY_RUN_ALL: ${DRY_RUN_ALL:-true}`, and
> `BINANCE_TESTNET: ${BINANCE_TESTNET:-false}`, so `.env` on the VPS is
> authoritative for these flags (with safe paper-mode defaults when unset).
> The rest of this audit remains as written at the audited commit `98ec16b`.

`docker-compose.prod.yml` sets `PAPER_TRADING: false`, `DRY_RUN_ALL: false`, and
`BINANCE_TESTNET: false` in the `environment:` block. In Compose, `environment:`
**overrides** `env_file:`, so setting `PAPER_TRADING=true` in `.env` on the VPS has
**no effect** — the container still trades live. The deploy script's
`CONFIRM_LIVE_DEPLOY` gate is checking flags the container ignores.

This is the single highest-risk item found anywhere in this pass, because it means
the most obvious "put it in paper mode" action silently does nothing. Details and
the rest of the script inventory are in `docs/OPS_RUNBOOK.md`.

---

## Recommended order of work

1. **Fix the Compose paper-mode override** so `PAPER_TRADING` in `.env` is
   authoritative. Safety-critical and independent of everything else.
2. **Add the BE ratchet after partial TP** (Finding 1) — small, safe, strictly
   tightening.
3. **Cost the legacy backtester or label it gross-only** (Finding 2).
4. **Blend the partial into the min-edge gate** (1.1 vs 1.2 ATR) — correctness tidy-up.
5. Route strategy evaluation through the cost-aware ML promotion path, walk-forward
   and out-of-sample. Expect far fewer qualifying trades; that is the correct
   outcome, not a regression.
6. Only once (5) shows positive expectancy **net of costs**, consider raising size
   or leverage.

**On the current live deployment:** `PAPER_TRADING=false` with real keys, and the
recorded 81-day net was positive but dominated by commissions (+$64 net against
−$193.54 of fees) — within noise of zero. My recommendation is to reduce leverage or
pause live sizing until (5) produces a cost-aware expectancy estimate. That is your
call; I have not changed any runtime config.
