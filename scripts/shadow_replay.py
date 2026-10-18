"""Shadow replay: re-run logged Decision Engine snapshots from Supabase
``public.jev_signals`` through alternate OmniRoute model presets and score
against realized point-in-time Binance Futures klines.

Usage (on VPS or locally with .env):
    python -m scripts.shadow_replay --days 30 --preset auto/smart --limit 200
    python -m scripts.shadow_replay --days 7 --preset auto/fast --limit 25 --dry-run
    python -m scripts.shadow_replay --settle

Key fixes over the prototype pack:
  - Queries the REAL ``public.jev_signals`` schema (``id``, ``created_at``, ``symbol``,
    ``timeframe``, ``decision``, ``allowed_side``, ``confidence``, ``signal_strength``,
    ``probabilities``, ``funding_annualized``, ``oi_change_pct_1h``, ``raw_answers``).
  - Point-in-time historical scoring via Binance Futures ``/fapi/v1/klines`` at
    ``decision_ts`` (entry) and ``decision_ts + holding_hours`` (exit), rather than
    comparing old signals against today's live ticker price.
  - Filters out non-Binance FX/commodity symbols (cTrader pairs) cleanly.
  - Scores BOTH the shadow preset and the original recorded decision on the same
    forward return window so you get an apples-to-apples baseline.
  - Handles ``neutral`` predictions as coverage rather than false losses.
  - Uses ``backend.utils.rate_limit`` and idempotent upsert on ``(signal_id, preset)``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from backend.services.supabase_admin import service_role_credentials
from backend.utils.rate_limit import acquire

logger = logging.getLogger("shadow_replay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HOLDING_HOURS = int(os.getenv("SHADOW_HOLDING_HOURS", "24"))
OMNI_URL = (os.getenv("OMNIROUTE_BASE_URL") or "https://omni.allikas.online/v1").strip().rstrip("/")
OMNI_KEY = (os.getenv("OMNIROUTE_API_KEY") or "").strip()


def _service_creds() -> tuple[str, str]:
    """Service-role only. Never falls back to the anon key."""
    return service_role_credentials()


SUPABASE_URL, SUPABASE_KEY = _service_creds()

_JEV_SELECT_COLS = (
    "id,symbol,timeframe,decision,allowed_side,confidence,signal_strength,"
    "probabilities,funding_annualized,oi_change_pct_1h,raw_answers,created_at"
)


def _is_binance_perp_symbol(symbol: str) -> bool:
    s = (symbol or "").upper().replace("/", "").replace("-", "")
    return s.endswith(("USDT", "USDC")) and len(s) >= 6


def _normalize_original_direction(sig: dict[str, Any]) -> str:
    side = str(sig.get("allowed_side") or sig.get("direction") or "").upper()
    dec = str(sig.get("decision") or "").upper()
    if side in ("BUY", "LONG", "UP") or dec in ("BUY", "LONG"):
        return "up"
    if side in ("SELL", "SHORT", "DOWN") or dec in ("SELL", "SHORT"):
        return "down"
    return "neutral"


def sb(
    path: str,
    method: str = "GET",
    payload: Any = None,
    params: dict[str, str] | None = None,
    prefer: str = "return=minimal",
) -> httpx.Response:
    url, key = _service_creds()
    if not (url and key):
        raise RuntimeError(
            "shadow_replay requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(or SUPABASE_SERVICE_KEY). The anon key is never used for writes."
        )
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }
    return httpx.request(
        method,
        f"{url}/rest/v1/{path}",
        headers=headers,
        json=payload,
        params=params,
        timeout=15.0,
    )


def fetch_snapshots(days: int, limit: int) -> list[dict[str, Any]]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    r = sb(
        "jev_signals",
        params={
            "select": _JEV_SELECT_COLS,
            "created_at": f"gte.{since}",
            "order": "created_at.asc",
            "limit": str(limit),
        },
    )
    r.raise_for_status()
    return r.json()


def build_replay_prompt(sig: dict[str, Any]) -> str:
    context = {
        "symbol": sig.get("symbol"),
        "timeframe": sig.get("timeframe"),
        "signal_strength": sig.get("signal_strength"),
        "probabilities": sig.get("probabilities"),
        "funding_annualized": sig.get("funding_annualized"),
        "oi_change_pct_1h": sig.get("oi_change_pct_1h"),
        "evidence": sig.get("raw_answers") or sig.get("payload") or {},
    }
    return (
        "Replay this historical crypto perpetuals market snapshot.\n"
        'Reply as strict JSON: {"direction": "up"|"down"|"neutral", "confidence": 0.0}\n'
        f"symbol={sig.get('symbol')} time={sig.get('created_at')}\n"
        f"snapshot={json.dumps(context, default=str)[:2500]}"
    )


def replay_one(sig: dict[str, Any], preset: str) -> dict[str, Any] | None:
    if not acquire("omniroute", max_wait=30.0):
        logger.warning("rate limit skipped omniroute for %s", sig.get("id"))
        return None
    prompt = build_replay_prompt(sig)
    try:
        r = httpx.post(
            f"{OMNI_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OMNI_KEY}"},
            json={
                "model": preset,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            timeout=30.0,
        )
        r.raise_for_status()
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        verdict = json.loads(content)
        resolved = body.get("model") or r.headers.get("x-model") or preset
        verdict["_model"] = resolved
        direction = str(verdict.get("direction") or "neutral").strip().lower()
        if direction not in ("up", "down", "neutral"):
            direction = "neutral"
        verdict["direction"] = direction
        verdict["confidence"] = max(0.0, min(1.0, float(verdict.get("confidence") or 0.0)))
        return verdict
    except Exception as e:
        logger.warning("replay failed for %s: %s", sig.get("id"), e)
        return None


def _parse_iso_ms(ts_str: str) -> int | None:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def get_historical_price_at(symbol: str, ts_iso: str) -> float | None:
    """Fetch 1m kline open/close price on Binance Futures at ``ts_iso``."""
    clean_sym = symbol.upper().replace("/", "").replace("-", "")
    if not _is_binance_perp_symbol(clean_sym):
        return None
    start_ms = _parse_iso_ms(ts_iso)
    if start_ms is None:
        return None
    if not acquire("binance", max_wait=15.0):
        return None
    try:
        r = httpx.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={
                "symbol": clean_sym,
                "interval": "1m",
                "startTime": str(start_ms),
                "limit": "1",
            },
            timeout=8.0,
        )
        r.raise_for_status()
        klines = r.json()
        if klines and isinstance(klines, list):
            return float(klines[0][4])  # close of that 1m candle
    except Exception as e:
        logger.debug("historical kline lookup failed for %s at %s: %s", clean_sym, ts_iso, e)
    return None


def score_direction(direction: str | None, forward_return_pct: float) -> bool | None:
    """Return True/False for directional calls, or None when direction is neutral."""
    d = (direction or "neutral").strip().lower()
    if d == "up":
        return forward_return_pct > 0.0
    if d == "down":
        return forward_return_pct < 0.0
    return None


def run(days: int, limit: int, preset: str, dry_run: bool = False) -> dict[str, int]:
    sigs = fetch_snapshots(days, limit)
    logger.info("fetched %d snapshots (days=%d, preset=%s, dry_run=%s)", len(sigs), days, preset, dry_run)
    done = skip = fx_skip = 0
    for s in sigs:
        sym = str(s.get("symbol") or "")
        if not _is_binance_perp_symbol(sym):
            fx_skip += 1
            continue
        verdict = replay_one(s, preset)
        if not verdict:
            skip += 1
            continue
        entry_px = get_historical_price_at(sym, str(s.get("created_at") or ""))
        orig_dir = _normalize_original_direction(s)
        row = {
            "signal_id": str(s["id"]),
            "symbol": sym.upper().replace("/", "").replace("-", ""),
            "preset": preset,
            "shadow_model": verdict.get("_model", preset),
            "shadow_direction": verdict.get("direction"),
            "shadow_confidence": verdict.get("confidence"),
            "original_direction": orig_dir,
            "entry_price": entry_px,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "decision_ts": s["created_at"],
        }
        if dry_run:
            logger.info("[DRY-RUN] %s", json.dumps(row))
        else:
            sb(
                "shadow_replays",
                method="POST",
                payload=row,
                params={"on_conflict": "signal_id,preset"},
                prefer="resolution=merge-duplicates,return=minimal",
            )
        done += 1
    logger.info("done=%d skipped=%d fx_skipped=%d", done, skip, fx_skip)
    return {"done": done, "skipped": skip, "fx_skipped": fx_skip}


def settle() -> dict[str, int]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=HOLDING_HOURS)).isoformat()
    r = sb(
        "shadow_replays",
        params={
            "select": "*",
            "settled": "is.false",
            "decision_ts": f"lte.{cutoff}",
            "limit": "500",
        },
    )
    r.raise_for_status()
    rows = r.json()
    logger.info("settling %d shadow verdicts (holding=%dh)", len(rows), HOLDING_HOURS)
    settled_count = skipped = 0
    for row in rows:
        sym = str(row.get("symbol") or "")
        dec_ts = str(row.get("decision_ts") or "")
        entry = row.get("entry_price")
        if not entry:
            entry = get_historical_price_at(sym, dec_ts)
        dec_ms = _parse_iso_ms(dec_ts)
        if not entry or dec_ms is None:
            skipped += 1
            continue
        exit_dt = datetime.fromtimestamp(dec_ms / 1000.0, tz=timezone.utc) + timedelta(hours=HOLDING_HOURS)
        exit_px = get_historical_price_at(sym, exit_dt.isoformat())
        if not exit_px:
            skipped += 1
            continue
        ret = (float(exit_px) - float(entry)) / float(entry) * 100.0
        shadow_correct = score_direction(row.get("shadow_direction"), ret)
        orig_correct = score_direction(row.get("original_direction"), ret)
        sb(
            "shadow_replays",
            method="PATCH",
            payload={
                "entry_price": float(entry),
                "exit_price": float(exit_px),
                "settled": True,
                "settle_ts": exit_dt.isoformat(),
                "forward_return_pct": round(ret, 4),
                "correct": shadow_correct,
                "original_correct": orig_correct,
            },
            params={"id": f"eq.{row['id']}"},
        )
        settled_count += 1
    logger.info("settle complete: settled=%d skipped=%d", settled_count, skipped)
    return {"settled": settled_count, "skipped": skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--preset", default="auto/smart")
    args = ap.parse_args()
    if args.settle:
        settle()
    else:
        run(args.days, args.limit, args.preset, dry_run=args.dry_run)
