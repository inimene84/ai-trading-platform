# MetaTrader 5 Bridge Contract (Data Only)

This sidecar provides a read-only market data bridge between MetaTrader 5 and the QuantumTrade Pro research plane.

## Architectural Boundaries

- **Data Only**: Pulls bars and prices for analysis, backtesting, and research indexing (`qt-mt-bars`).
- **Never an Execution Broker**: MT5 is never configured as `ACTIVE_BROKER` or allowed into live order execution paths. Live orders are routed strictly via Binance and cTrader.
- **Separate from Trading Stack**: Runs as a separate sidecar container on `trading-net` with host binds on `127.0.0.1`.

## Endpoints

- `GET /health`: Health and terminal connection status.
- `GET /bars?symbol={symbol}&tf={tf}&limit={limit}`: Returns historical/recent bars. When no terminal is attached, returns `[]` with status `"no_terminal"`.

## Expected JSONL Dump Format for EA / File Ingest

When an MQL5 Expert Advisor dumps closed bars to file or HTTP stream, each line is an NDJSON record formatted as:

```json
{"ts": "2026-09-15T20:00:00Z", "symbol": "XAUUSD", "tf": "H1", "o": 2580.50, "h": 2585.10, "l": 2578.20, "c": 2583.40, "v": 12450.0}
```

Fields:
- `ts`: ISO-8601 UTC timestamp of bar open
- `symbol`: Ticker symbol (e.g. `XAUUSD`, `EURUSD`, `GBPUSD`)
- `tf`: Timeframe string (`M15`, `H1`)
- `o`: Open price (float)
- `h`: High price (float)
- `l`: Low price (float)
- `c`: Close price (float)
- `v`: Tick/real volume (float)
