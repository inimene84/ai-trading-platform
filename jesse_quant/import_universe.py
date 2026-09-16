#!/usr/bin/env python3
"""
Jesse crypto candle importer — queues 1-minute history via Jesse's REST API.

Does not store credentials. JESSE_PASSWORD / PASSWORD must be set in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

from multi_asset_universe import CRYPTO_IMPORT, JESSE_BINANCE_EXCHANGE

API_URL = os.getenv("JESSE_API_URL", "http://127.0.0.1:9000")
PASSWORD = os.getenv("JESSE_PASSWORD") or os.getenv("PASSWORD") or ""
DEFAULT_SYMBOLS = [row["symbol"] for row in CRYPTO_IMPORT if row["symbol"] not in ("BTC-USDT", "ETH-USDT", "SOL-USDT")]


def get_token() -> str:
    if not PASSWORD:
        raise RuntimeError("JESSE_PASSWORD / PASSWORD is not set — refusing to authenticate")
    req = urllib.request.Request(
        f"{API_URL}/auth/login",
        data=json.dumps({"password": PASSWORD}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        token = json.loads(resp.read().decode("utf-8")).get("auth_token")
    if not token:
        raise RuntimeError("Jesse login returned no auth_token")
    return token


def import_symbol_candles(symbol: str, start_date: str, exchange: str = JESSE_BINANCE_EXCHANGE) -> bool:
    token = get_token()
    task_id = str(uuid.uuid4())
    payload = {
        "id": task_id,
        "exchange": exchange,
        "symbol": symbol,
        "start_date": start_date,
    }
    req = urllib.request.Request(
        f"{API_URL}/candles/import",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": token},
    )
    print(f"\n[*] Initiating candle import for {symbol} ({exchange}) from {start_date}...")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            json.loads(resp.read().decode("utf-8"))
            print(f"    [✓] Import task queued ({task_id[:8]}...)")
    except urllib.error.HTTPError as exc:
        print(f"    [!] Error queuing import for {symbol}: {exc.code} {exc.read().decode('utf-8')}")
        return False

    dots = 0
    while True:
        time.sleep(2)
        status_req = urllib.request.Request(
            f"{API_URL}/candles/import-status",
            data=json.dumps({"id": task_id}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": token},
        )
        try:
            with urllib.request.urlopen(status_req, timeout=10) as status_resp:
                s_data = json.loads(status_resp.read().decode("utf-8"))
                status = s_data.get("status")
                progress = s_data.get("progress", 0)
                errors = s_data.get("errors")
                if status == "finished":
                    print(f"\n    [✓] {symbol} candle import completed")
                    return True
                if status == "failed":
                    print(f"\n    [!] {symbol} candle import failed: {errors or 'Unknown error'}")
                    return False
                dots += 1
                pct_str = f"{progress:.1f}%" if progress else "processing..."
                sys.stdout.write(f"\r    -> Downloading candles: {pct_str} [{'=' * (dots % 20)}]")
                sys.stdout.flush()
        except Exception:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Jesse crypto candlestick importer")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--exchange", default=JESSE_BINANCE_EXCHANGE)
    args = parser.parse_args()

    print("=========================================================")
    print("         JESSE CRYPTO CANDLESTICK IMPORTER               ")
    print("=========================================================")
    print(f"Exchange:   {args.exchange}")
    print(f"Start Date: {args.start}")
    print(f"Symbols:    {', '.join(args.symbols)}")
    print("=========================================================")

    results = {}
    for sym in args.symbols:
        results[sym] = "SUCCESS" if import_symbol_candles(sym, args.start, args.exchange) else "FAILED"
        time.sleep(1)

    print("\n=========================================================")
    print("                 IMPORT SUMMARY REPORT                   ")
    print("=========================================================")
    for sym, status in results.items():
        print(f" {sym:<12} : {status}")
    failed = [s for s, st in results.items() if st != "SUCCESS"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
