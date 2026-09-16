#!/usr/bin/env python3
"""Download public OHLCV for the multi-asset training universe.

Sources:
  - binance: public spot klines (no API key)
  - yfinance: FX / metals / oil / equities (Yahoo)

Jesse Postgres dumps (BTC/ETH/SOL 1m) are NOT fetched here — copy those gzip
CSVs onto the GPU node separately. Never writes credentials.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]

from asset_universe import (
    ALL_CLASSES,
    UniverseSymbol,
    iter_downloadable,
    resolve_universe,
)

BINANCE_VISION_KLINES = "https://data-api.binance.vision/api/v3/klines"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
USER_AGENT = "jesse-quant-ohlcv/1.0"


class DownloadError(RuntimeError):
    pass


def _http_get_json(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _write_gzip_csv(path: str, rows: Sequence[Tuple[int, float, float, float, float, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("timestamp,open,high,low,close,volume\n")
        for ts, o, h, l, c, v in rows:
            fh.write(f"{int(ts)},{o},{h},{l},{c},{v}\n")


def fetch_binance_klines(
    ticker: str,
    interval: str = "1h",
    start_ms: Optional[int] = None,
    limit_pages: int = 40,
) -> List[Tuple[int, float, float, float, float, float]]:
    """Paginate public klines. Prefer spot; fall back to USDT-M futures."""
    if start_ms is None:
        start_ms = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    out: List[Tuple[int, float, float, float, float, float]] = []
    for base in (BINANCE_VISION_KLINES, BINANCE_KLINES, BINANCE_FAPI_KLINES):
        cursor = start_ms
        pages = 0
        out = []
        try:
            while pages < limit_pages:
                qs = urllib.parse.urlencode(
                    {
                        "symbol": ticker,
                        "interval": interval,
                        "startTime": cursor,
                        "limit": 1000,
                    }
                )
                payload = _http_get_json(f"{base}?{qs}")
                if not isinstance(payload, list) or not payload:
                    break
                for row in payload:
                    out.append(
                        (
                            int(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                        )
                    )
                last_open = int(payload[-1][0])
                pages += 1
                if len(payload) < 1000:
                    break
                nxt = last_open + 1
                if nxt <= cursor:
                    break
                cursor = nxt
                time.sleep(0.12)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, TypeError, IndexError):
            out = []
            continue
        if out:
            return out
    raise DownloadError(f"binance returned no klines for {ticker}")


def fetch_yfinance_ohlcv(
    ticker: str,
    interval: str = "1h",
    period: str = "730d",
) -> List[Tuple[int, float, float, float, float, float]]:
    if yf is None:
        raise DownloadError("yfinance is not installed")
    hist = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if hist is None or getattr(hist, "empty", True):
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
    if hist is None or hist.empty:
        raise DownloadError(f"yfinance empty history for {ticker}")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    colmap = {str(c).lower(): c for c in hist.columns}
    def col(name: str) -> str:
        if name in colmap:
            return colmap[name]
        for key, orig in colmap.items():
            if key.startswith(name):
                return orig
        raise DownloadError(f"yfinance missing column {name} for {ticker}: {list(hist.columns)}")

    rows: List[Tuple[int, float, float, float, float, float]] = []
    for idx, rec in hist.iterrows():
        ts = idx
        if hasattr(ts, "timestamp"):
            ms = int(ts.timestamp() * 1000)
        else:
            ms = int(ts)
        try:
            o = float(rec[col("open")])
            h = float(rec[col("high")])
            l = float(rec[col("low")])
            c = float(rec[col("close")])
            v = float(rec[col("volume")]) if "volume" in colmap else 0.0
        except (TypeError, ValueError, KeyError):
            continue
        if o != o:  # NaN
            continue
        rows.append((ms, o, h, l, c, v if v == v else 0.0))
    if len(rows) < 200:
        raise DownloadError(f"yfinance too few bars for {ticker}: {len(rows)}")
    return rows


def download_one(item: UniverseSymbol, out_dir: str, skip_existing: bool = True) -> Dict[str, Any]:
    path = os.path.join(out_dir, item.filename())
    result: Dict[str, Any] = {
        "symbol": item.symbol,
        "asset_class": item.asset_class,
        "source": item.source,
        "ticker": item.ticker,
        "path": path,
        "ok": False,
        "rows": 0,
        "error": None,
    }
    if item.source == "jesse_dump":
        if os.path.exists(path) or os.path.exists(
            os.path.join(out_dir, f"{item.symbol}_1m.csv.gz")
        ):
            result["ok"] = True
            result["error"] = "jesse_dump_present"
            return result
        result["error"] = "jesse_dump_missing — copy gzip CSV from trading VPS"
        return result
    if skip_existing and os.path.exists(path) and os.path.getsize(path) > 64:
        result["ok"] = True
        result["error"] = "exists"
        return result
    try:
        if item.source == "binance":
            try:
                rows = fetch_binance_klines(item.ticker, interval=item.native_tf)
            except DownloadError:
                yf_ticker = item.ticker.replace("USDT", "-USD").replace("USDC", "-USD")
                print(f"    binance blocked; yfinance fallback {yf_ticker}")
                rows = fetch_yfinance_ohlcv(yf_ticker, interval=item.native_tf)
        elif item.source == "yfinance":
            rows = fetch_yfinance_ohlcv(item.ticker, interval=item.native_tf)
        else:
            raise DownloadError(f"unsupported source {item.source}")
        _write_gzip_csv(path, rows)
        result["ok"] = True
        result["rows"] = len(rows)
    except Exception as exc:  # noqa: BLE001 — CLI must keep going across symbols
        result["error"] = str(exc)
    return result


def download_universe(
    asset_class: str,
    out_dir: str,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    items = iter_downloadable(resolve_universe(asset_class))
    jesse_only = [r for r in resolve_universe(asset_class) if r.source == "jesse_dump"]
    reports: List[Dict[str, Any]] = []
    for item in jesse_only:
        reports.append(download_one(item, out_dir, skip_existing=skip_existing))
    for item in items:
        print(f"[*] {item.asset_class} {item.symbol} via {item.source} ({item.ticker})")
        rec = download_one(item, out_dir, skip_existing=skip_existing)
        status = "ok" if rec["ok"] else "FAIL"
        print(f"    [{status}] rows={rec['rows']} {rec.get('error') or rec['path']}")
        reports.append(rec)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download multi-asset OHLCV for GPU training")
    parser.add_argument(
        "--asset-class",
        default="all",
        help="crypto|forex|metals|minerals|stocks|all",
    )
    parser.add_argument("--out", default="storage/candles", help="output directory")
    parser.add_argument("--force", action="store_true", help="re-download even if gzip exists")
    parser.add_argument("--list", action="store_true", help="print universe and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for cls in ALL_CLASSES:
            print(f"## {cls}")
            for item in resolve_universe(cls):
                print(f"  {item.symbol:12} {item.source:12} {item.ticker}  {item.filename()}")
        return 0
    reports = download_universe(args.asset_class, args.out, skip_existing=not args.force)
    ok = sum(1 for r in reports if r["ok"])
    print(json.dumps({"downloaded_ok": ok, "n": len(reports), "rows": reports}, indent=2, default=str))
    # jesse_dump_missing is not a hard fail of this downloader
    hard_fail = [r for r in reports if not r["ok"] and r["source"] != "jesse_dump"]
    return 1 if hard_fail and ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
