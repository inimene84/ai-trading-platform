# Venue fusion (Binance USDC-M + cTrader / IC Markets)

Crypto stays on Binance USDC-M futures (24/7). Forex and metals stay on
cTrader (IC Markets) during the Sun 21:00 – Fri 21:00 UTC session.

Every new live behavior is behind an env flag. Anything that can change
order placement or risk defaults to **off** or **shadow**, so merging this
branch does not change the live book until an operator flips a flag.

## `CTRADER_INVERT_SIDE`

Unchanged. Do **not** flip `CTRADER_INVERT_SIDE` as part of venue fusion.
The invert was calibrated to the Sep 2026 IC demo closed book. Re-measure
that edge on a fresh sample before turning it off.

cTrader still refuses crypto symbols in `place_order` even if
`VENUE_CRYPTO=ctrader` — that refusal is the backstop.

## Flag table

| Flag | Default | Effect when flipped |
|---|---|---|
| `BINANCE_MAKER_TP_ENABLED` | `false` | Reduce-only GTX LIMIT at TP instead of `TAKE_PROFIT_MARKET`. Stops stay `STOP_MARKET`. Market flatten only if last/mark has already traded through (never on clock timeout). |
| `BINANCE_MAKER_TP_TIMEOUT_SEC` | `120` | Observational only: log when a resting maker TP has aged this long. Never flattens. |
| `BINANCE_MAKER_CLOSE_ENABLED` | `false` | Maker-first GTX for reduce-only / time-based closes. |
| `BINANCE_MAKER_REQUOTE_ENABLED` | `false` | Re-join the touch when the book walks away instead of a single GTX then MARKET. |
| `BINANCE_MAKER_REQUOTE_MAX` | `4` | Max re-quotes per entry attempt. |
| `BINANCE_COMMISSION_RATE_TTL_SEC` | `3600` | Cache TTL for `GET /fapi/v1/commissionRate` (logging only). |
| `FUNDING_FILTER_MODE` | `shadow` | `off` / `shadow` / `enforce`. Veto or size-down entries whose next funding opposes the side. |
| `FUNDING_FILTER_THRESHOLD` | `0.0005` | Absolute funding rate that counts as a veto. |
| `FUNDING_FILTER_SIZE_MULT` | `0.5` | Size multiplier when funding opposes but is below the veto threshold. |
| `VENUE_CRYPTO` | `binance` | Asset-class → venue. Aliases: `binance_futures`, `ctrader`, `ic`. |
| `VENUE_FOREX` | `ctrader` | |
| `VENUE_METALS` | `ctrader` | |
| `VENUE_OIL` | `ctrader` | |
| `VENUE_INDEX` | `ctrader` | |
| `VENUE_STOCK` | `ctrader` | |
| `CTRADER_ENTRY_CUTOFF_MODE` | `shadow` | `off` / `shadow` / `enforce`. Skip new IC entries inside the pre-close / daily-break window. |
| `CTRADER_ENTRY_CUTOFF_MIN` | `30` | Minutes before venue close that count as the cutoff. |
| `CTRADER_DAILY_BREAK_START_UTC` | `21:00` | Weekday daily-break start. |
| `CTRADER_DAILY_BREAK_END_UTC` | `22:00` | Weekday daily-break end. |
| `CTRADER_WEEKEND_PROTECT_MODE` | `shadow` | `off` / `shadow` / `tighten` / `flatten`. Friday pre-close job. |
| `CTRADER_WEEKEND_PROTECT_MINUTES` | `60` | Lead window before Fri 21:00 UTC. |
| `CTRADER_SWAP_HOLD_MODE` | `shadow` | `off` / `shadow` / `enforce`. Count swap in EV for holds > 1 day. |
| `CTRADER_PIP_VALUE_PER_LOT` | `10` | Swap USD conversion helper. |
| `SHARED_RISK_MODE` | `shadow` | `off` / `shadow` / `enforce`. Combined open-risk / drawdown / position / net-USD caps. |
| `SHARED_RISK_MAX_POSITIONS` | `4` | Combined distinct symbols. Per-venue `MAX_POSITIONS` still applies first. |
| `SHARED_RISK_MAX_OPEN_RISK_PCT` | `6.0` | Combined stop-risk / equity. |
| `SHARED_RISK_MAX_DRAWDOWN_PCT` | `15.0` | Combined drawdown (only when peak+equity are supplied). |
| `SHARED_RISK_MAX_NET_USD` | `0` | `0` disables the net-USD cap. |
| `BACKTEST_VENUE_COSTS` | `false` | Apply spread / swap (triple Wed FX, Fri crypto) / funding / maker-taker in backtests. |

Split books (cTrader demo + Binance live) are never unioned. Shared risk
reports both books with a warning and leaves per-venue caps in place.
