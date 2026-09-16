#!/usr/bin/env python3
"""Load the GPU training universe into Jesse Postgres on the trading VPS.

Reuses `asset_universe.py` / `download_ohlcv.py` so live candles match GPU dumps.
Does not talk to brokers. POSTGRES_PASSWORD must come from the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]

from asset_universe import UniverseSymbol, resolve_universe
from download_ohlcv import DownloadError, fetch_binance_klines, fetch_yfinance_ohlcv

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_NAME", "jesse_db")
DB_USER = os.getenv("POSTGRES_USERNAME", "jesse_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

BINANCE_EXCHANGE = "Binance Perpetual Futures"
YFINANCE_EXCHANGE = "Yahoo Finance"

INSERT_SQL = """
INSERT INTO candle (id, timestamp, open, close, high, low, volume, exchange, symbol, timeframe)
VALUES %s
ON CONFLICT (exchange, symbol, timeframe, timestamp) DO UPDATE SET
    open = EXCLUDED.open,
    close = EXCLUDED.close,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    volume = EXCLUDED.volume
"""

Kline = Tuple[int, float, float, float, float, float]


def _connect():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed")
    if not DB_PASS:
        raise RuntimeError("POSTGRES_PASSWORD is not set — refusing to connect")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def rows_from_klines(
    klines: Sequence[Kline],
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> List[Tuple[Any, ...]]:
    rows: List[Tuple[Any, ...]] = []
    for ts, o, h, low, c, vol in klines:
        rows.append(
            (
                str(uuid.uuid4()),
                int(ts),
                float(o),
                float(c),
                float(h),
                float(low),
                float(vol if vol == vol else 0.0),
                exchange,
                symbol,
                timeframe,
            )
        )
    return rows


def rows_from_ohlcv(
    frame: pd.DataFrame,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> List[Tuple[Any, ...]]:
    """Test helper: convert an OHLCV DataFrame into Jesse candle tuples."""
    if frame is None or frame.empty:
        return []
    work = frame.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("OHLCV frame must be indexed by timestamp")
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")
    klines: List[Kline] = []
    colmap = {str(c).lower(): c for c in work.columns}
    for ts, row in work.iterrows():
        klines.append(
            (
                int(ts.timestamp() * 1000),
                float(row[colmap["open"]]),
                float(row[colmap["high"]]),
                float(row[colmap["low"]]),
                float(row[colmap["close"]]),
                float(row[colmap["volume"]]) if "volume" in colmap else 0.0,
            )
        )
    return rows_from_klines(klines, exchange=exchange, symbol=symbol, timeframe=timeframe)


def upsert_candles(conn, rows: Sequence[Tuple[Any, ...]]) -> int:
    if not rows:
        return 0
    if execute_values is None:
        raise RuntimeError("psycopg2 is not installed")
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, page_size=1000)
    conn.commit()
    return len(rows)


def exchange_for(item: UniverseSymbol) -> str:
    if item.source in ("binance", "jesse_dump"):
        return BINANCE_EXCHANGE
    return YFINANCE_EXCHANGE


def fetch_item(item: UniverseSymbol) -> Sequence[Kline]:
    if item.source == "binance":
        try:
            return fetch_binance_klines(item.ticker, interval=item.native_tf)
        except DownloadError:
            yf_ticker = item.ticker.replace("USDT", "-USD").replace("USDC", "-USD")
            return fetch_yfinance_ohlcv(yf_ticker, interval=item.native_tf)
    if item.source == "yfinance":
        return fetch_yfinance_ohlcv(item.ticker, interval=item.native_tf)
    raise DownloadError(f"skip source {item.source}")


def import_universe(conn, asset_class: str = "all") -> Dict[str, int]:
    results: Dict[str, int] = {}
    for item in resolve_universe(asset_class):
        if item.source == "jesse_dump":
            print(f"[*] skip {item.symbol} (1m already in Jesse Postgres)")
            results[item.symbol] = -1
            continue
        print(f"[*] {item.asset_class} {item.symbol} via {item.source} ({item.ticker})")
        try:
            klines = fetch_item(item)
        except Exception as exc:
            print(f"    [!] download failed: {exc}")
            results[item.symbol] = 0
            continue
        tf = item.native_tf
        rows = rows_from_klines(
            klines,
            exchange=exchange_for(item),
            symbol=item.symbol,
            timeframe=tf,
        )
        n = upsert_candles(conn, rows)
        print(f"    [✓] upserted {n} {tf} bars")
        results[item.symbol] = n
    return results


def inventory(conn) -> List[Tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT exchange, symbol, timeframe, COUNT(*), "
            "to_timestamp(MIN(timestamp)/1000.0), to_timestamp(MAX(timestamp)/1000.0) "
            "FROM candle GROUP BY 1,2,3 ORDER BY 1,2,3"
        )
        return cur.fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import multi-asset candles into Jesse Postgres")
    parser.add_argument("--asset-class", default="all")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    conn = _connect()
    if args.inventory_only:
        for row in inventory(conn):
            print(row)
        conn.close()
        return 0

    results = import_universe(conn, args.asset_class)
    print("\n=== upsert summary ===")
    for sym, n in results.items():
        print(f"  {sym:<12} {n}")
    print("\n=== candle inventory ===")
    for row in inventory(conn):
        print(row)
    conn.close()
    failed = [s for s, n in results.items() if n == 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
