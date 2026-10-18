"""Bear-veto debate layer (LangGraph).

Runs AFTER the Decision Engine proposes a trade, BEFORE order dispatch.
Default: SHADOW mode (logs verdict, never blocks, and can be dispatched
asynchronously via ``schedule_shadow_bear_veto`` so it adds zero latency to
live/paper entries). Flip ``BEAR_VETO_MODE=enforce`` after ~2 weeks of logged
data prove the bear adds positive expectancy.

Key guarantees over the prototype pack:
  - Fully async nodes invoking ``await call_llm_resilient_meta(...)``.
  - Dedicated system prompts for bull, bear, and judge (no f-string brace bugs).
  - If the bear node fails or returns empty text, the judge abstains
    (``abstain=True, veto=False``) rather than hallucinating a verdict.
  - Bounded total debate budget via ``asyncio.wait_for(BEAR_VETO_TIMEOUT_SEC)``.
  - Every verdict is projected to Supabase ``public.bear_verdicts`` (with a
    JSONL fallback) so the bear's hit rate is directly measurable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, StateGraph

from backend.llm.router import call_llm_resilient_meta
from backend.services.supabase_admin import service_role_credentials

logger = logging.getLogger(__name__)

BEAR_VETO_MODE = os.getenv("BEAR_VETO_MODE", "shadow").strip().lower()  # off | shadow | enforce
BEAR_VETO_MIN_SIGNAL = float(os.getenv("BEAR_VETO_MIN_SIGNAL", "0.55"))
BEAR_VETO_TIMEOUT_SEC = float(os.getenv("BEAR_VETO_TIMEOUT_SEC", "30.0"))
BEAR_VETO_TASK_TYPE = os.getenv("BEAR_VETO_TASK_TYPE", "general")


class DebateState(TypedDict, total=False):
    symbol: str
    direction: str
    proposed: dict[str, Any]
    market_snapshot: dict[str, Any]
    bull_case: str
    bear_case: str
    judge_verdict: dict[str, Any]
    veto: bool
    abstain: bool
    resolved_model: str
    resolved_provider: str


_ADVOCATE_SYS_TEMPLATE = (
    "You are the {role} analyst for a risk-averse quantitative trading desk. "
    "Argue ONLY your assigned side using the evidence given. "
    'Reply as strict JSON: {{"points": ["..."], "strongest": "...", "confidence": 0.0}}'
)

_JUDGE_SYS_PROMPT = (
    "You are the independent Chief Risk Judge for a quantitative trading desk. "
    "Weigh the bull and bear arguments objectively. Only veto when the bear "
    "demonstrates a concrete structural flaw (fee churn, adverse regime, "
    "stop vulnerability, funding/liquidity trap, or weak statistical edge). "
    'Reply as strict JSON: {{"veto": false, "bear_strength": 0.0, "reason": "..."}}'
)


def _sys(role: str) -> str:
    if role == "judge":
        return _JUDGE_SYS_PROMPT
    return _ADVOCATE_SYS_TEMPLATE.format(role=role)


async def bull_node(state: DebateState) -> DebateState:
    prompt = (
        f"Trade proposal: {state.get('direction')} {state.get('symbol')}.\n"
        f"Decision Engine evidence: {json.dumps(state.get('proposed') or {}, default=str)[:3000]}\n"
        f"Market snapshot: {json.dumps(state.get('market_snapshot') or {}, default=str)[:2000]}\n"
        "Make the strongest data-backed case FOR executing this trade."
    )
    try:
        txt, meta = await call_llm_resilient_meta(
            BEAR_VETO_TASK_TYPE,
            prompt,
            system=_sys("bull"),
            response_json=True,
        )
        state["bull_case"] = txt or ""
        state["resolved_model"] = meta.get("model", "?")
        state["resolved_provider"] = meta.get("provider", "?")
    except Exception as e:
        logger.warning("bull_node failed for %s: %s", state.get("symbol"), e)
        state["bull_case"] = ""
    return state


async def bear_node(state: DebateState) -> DebateState:
    prompt = (
        f"Trade proposal: {state.get('direction')} {state.get('symbol')}.\n"
        f"Evidence supporting it: {json.dumps(state.get('proposed') or {}, default=str)[:3000]}\n"
        f"Bull case already made: {(state.get('bull_case') or '')[:1500]}\n"
        f"Market snapshot: {json.dumps(state.get('market_snapshot') or {}, default=str)[:2000]}\n"
        "Your job: KILL this trade. Attack fees/edge, regime mismatch, "
        "correlation, timing, stop placement, liquidity, and model uncertainty."
    )
    try:
        txt, meta = await call_llm_resilient_meta(
            BEAR_VETO_TASK_TYPE,
            prompt,
            system=_sys("bear"),
            response_json=True,
        )
        state["bear_case"] = txt or ""
        state["resolved_model"] = meta.get("model", state.get("resolved_model", "?"))
        state["resolved_provider"] = meta.get("provider", state.get("resolved_provider", "?"))
    except Exception as e:
        logger.warning("bear_node failed for %s: %s", state.get("symbol"), e)
        state["bear_case"] = ""
    return state


async def judge_node(state: DebateState) -> DebateState:
    bear_txt = (state.get("bear_case") or "").strip()
    if not bear_txt:
        state["abstain"] = True
        state["veto"] = False
        state["judge_verdict"] = {
            "veto": False,
            "bear_strength": 0.0,
            "abstain": True,
            "reason": "bear_node produced no argument; abstaining (fail-open)",
        }
        return state

    prompt = (
        "Read BOTH cases and deliver a strict-JSON verdict.\n"
        f"Proposal: {state.get('direction')} {state.get('symbol')}\n"
        f"BULL: {(state.get('bull_case') or '')[:2000]}\n"
        f"BEAR: {bear_txt[:2000]}\n"
        'Reply: {"veto": true|false, "bear_strength": 0.0, "reason": "..."}'
    )
    try:
        txt, meta = await call_llm_resilient_meta(
            BEAR_VETO_TASK_TYPE,
            prompt,
            system=_sys("judge"),
            response_json=True,
        )
        verdict = json.loads(txt) if isinstance(txt, str) else dict(txt)
        verdict.setdefault("veto", False)
        verdict["bear_strength"] = max(0.0, min(1.0, float(verdict.get("bear_strength", 0.0) or 0.0)))
        verdict["abstain"] = False
        state["judge_verdict"] = verdict
        state["abstain"] = False
        state["veto"] = bool(verdict["veto"] and verdict["bear_strength"] >= BEAR_VETO_MIN_SIGNAL)
        state["resolved_model"] = meta.get("model", state.get("resolved_model", "?"))
        state["resolved_provider"] = meta.get("provider", state.get("resolved_provider", "?"))
    except Exception as e:
        logger.warning("judge_node failed for %s (fail-open, logged): %s", state.get("symbol"), e)
        state["judge_verdict"] = {"veto": False, "bear_strength": 0.0, "abstain": True, "error": str(e)}
        state["abstain"] = True
        state["veto"] = False
    return state


try:
    _graph = StateGraph(DebateState)
    _graph.add_node("bull", bull_node)
    _graph.add_node("bear", bear_node)
    _graph.add_node("judge", judge_node)
    _graph.set_entry_point("bull")
    _graph.add_edge("bull", "bear")
    _graph.add_edge("bear", "judge")
    _graph.add_edge("judge", END)
    bear_veto_app = _graph.compile()
except Exception as _graph_err:  # pragma: no cover - import-time fail-open
    logger.warning("bear_veto graph compile failed (fail-open): %s", _graph_err)
    bear_veto_app = None


def _current_mode() -> str:
    return (BEAR_VETO_MODE or os.getenv("BEAR_VETO_MODE", "shadow") or "shadow").strip().lower()


def _supabase_credentials() -> tuple[str, str]:
    """Service-role only. Never falls back to the anon key."""
    return service_role_credentials()


async def _persist_verdict(row: dict[str, Any]) -> bool:
    """Persist bear verdict to Supabase ``bear_verdicts`` with local JSONL fallback."""
    url, key = _supabase_credentials()
    if url and key:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.post(
                    f"{url}/rest/v1/bear_verdicts",
                    json=row,
                    headers={
                        "apikey": key,
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                )
                if r.status_code in (200, 201, 204):
                    return True
                logger.debug("bear_verdicts supabase post HTTP %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.debug("bear_verdicts supabase post failed: %s", e)
    elif url and not key:
        logger.debug("bear_verdicts supabase post skipped: service-role credentials not configured")

    try:
        fallback_path = Path(os.getenv("BEAR_VETO_LOG_PATH", "scratch/bear_verdicts.jsonl"))
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with fallback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        logger.debug("bear_verdicts local fallback write failed: %s", e)
    return False


async def evaluate_bear_veto(trade_proposal: dict[str, Any], persist: bool = True) -> dict[str, Any]:
    """Async evaluation of the Bull -> Bear -> Judge LangGraph.

    Returns ``{veto, veto_effective, abstain, verdict, model, provider, latency_ms, mode}``.
    """
    mode = _current_mode()
    if mode == "off":
        return {"veto": False, "veto_effective": False, "abstain": False, "mode": "off"}

    started = time.monotonic()
    state: DebateState = {
        "symbol": str(trade_proposal.get("symbol", "")),
        "direction": str(trade_proposal.get("direction", "")),
        "proposed": dict(trade_proposal.get("signal_detail") or {}),
        "market_snapshot": dict(trade_proposal.get("market_snapshot") or {}),
    }
    try:
        if bear_veto_app is None:
            raise RuntimeError("bear_veto graph unavailable")
        out = await asyncio.wait_for(bear_veto_app.ainvoke(state), timeout=BEAR_VETO_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            "BEAR_VETO timeout (%.1fs) for %s %s; failing open",
            BEAR_VETO_TIMEOUT_SEC,
            state["symbol"],
            state["direction"],
        )
        out = {
            "veto": False,
            "abstain": True,
            "judge_verdict": {"veto": False, "bear_strength": 0.0, "abstain": True, "error": "timeout"},
            "resolved_model": "?",
            "resolved_provider": "?",
        }
    except Exception as e:
        logger.warning("BEAR_VETO error for %s: %s; failing open", state["symbol"], e)
        out = {
            "veto": False,
            "abstain": True,
            "judge_verdict": {"veto": False, "bear_strength": 0.0, "abstain": True, "error": str(e)},
            "resolved_model": "?",
            "resolved_provider": "?",
        }

    latency_ms = round((time.monotonic() - started) * 1000.0, 1)
    verdict = out.get("judge_verdict") or {}
    veto = bool(out.get("veto", False))
    abstain = bool(out.get("abstain", False))
    veto_effective = bool(veto and mode == "enforce")

    result = {
        "veto": veto,
        "veto_effective": veto_effective,
        "abstain": abstain,
        "verdict": verdict,
        "model": out.get("resolved_model"),
        "provider": out.get("resolved_provider"),
        "latency_ms": latency_ms,
        "mode": mode,
    }
    logger.info(
        "BEAR_VETO %s %s veto=%s effective=%s strength=%.2f model=%s latency=%.0fms mode=%s",
        state["symbol"],
        state["direction"],
        veto,
        veto_effective,
        float(verdict.get("bear_strength", 0.0) or 0.0),
        result["model"],
        latency_ms,
        mode,
    )

    if persist:
        row = {
            "symbol": state["symbol"],
            "direction": state["direction"],
            "mode": mode,
            "veto": veto,
            "veto_effective": veto_effective,
            "abstain": abstain,
            "bear_strength": float(verdict.get("bear_strength", 0.0) or 0.0),
            "reason": str(verdict.get("reason") or verdict.get("error") or "")[:500],
            "resolved_model": result["model"],
            "resolved_provider": result["provider"],
            "latency_ms": latency_ms,
            "entry_price": (state["proposed"] or {}).get("entry_price"),
            "confidence": (state["proposed"] or {}).get("confidence"),
            "regime": (state["market_snapshot"] or {}).get("regime"),
            "bull_case": (out.get("bull_case") or "")[:1500],
            "bear_case": (out.get("bear_case") or "")[:1500],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await asyncio.wait_for(_persist_verdict(row), timeout=8.0)
        except Exception as e:
            logger.debug("bear_veto persist skipped: %s", e)

    return result


async def _safe_evaluate_bear_veto(trade_proposal: dict[str, Any]) -> dict[str, Any]:
    try:
        return await evaluate_bear_veto(trade_proposal, persist=True)
    except Exception as e:
        logger.warning("shadow bear veto failed open: %s", e)
        return {"veto": False, "veto_effective": False, "abstain": True, "error": str(e), "mode": _current_mode()}


def schedule_shadow_bear_veto(trade_proposal: dict[str, Any]) -> asyncio.Task[dict[str, Any]] | None:
    """Fire-and-forget background task for SHADOW mode so live/paper entry latency is 0ms."""
    if _current_mode() == "off":
        return None
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(_safe_evaluate_bear_veto(trade_proposal))
    except RuntimeError:
        return None


def evaluate_bear_veto_sync(trade_proposal: dict[str, Any], persist: bool = False) -> dict[str, Any]:
    """Synchronous convenience wrapper for unit tests and CLI scripts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("evaluate_bear_veto_sync cannot be called inside a running event loop; await evaluate_bear_veto instead")
    return asyncio.run(evaluate_bear_veto(trade_proposal, persist=persist))
