"""
QuantumTrade Pro — MCP Server
=============================
Exposes the live trading backend to MCP-compatible agents (Claude Desktop,
Cursor, Agent Zero / A0, Hermes) as a curated set of tools.

Runs as its own lightweight process/container and talks to the backend over
HTTP, so it needs no database, exchange, or model credentials of its own.

Env:
  BACKEND_BASE_URL   default http://ai-trading-backend:8000
  BACKEND_API_PREFIX default /api
  MCP_API_TOKEN      bearer token sent to the backend (ADMIN_API_KEY)
  MCP_TRANSPORT      default http   (stdio | http | sse)
  MCP_HOST / MCP_PORT
  FULL_API           default false
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

try:
    from fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "fastmcp is required: pip install fastmcp httpx  (see mcp_server/requirements.txt)"
    ) from e


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://ai-trading-backend:8000")
API_PREFIX = os.getenv("BACKEND_API_PREFIX", "/api").rstrip("/")
API_TOKEN = os.getenv("MCP_API_TOKEN", "")
TIMEOUT = float(os.getenv("MCP_HTTP_TIMEOUT", "20"))


def _headers() -> Dict[str, str]:
    h = {"Accept": "application/json", "Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
        # Some routes also accept x-api-key
        h["X-API-Key"] = API_TOKEN
    return h


def _url(path: str) -> str:
    path = path if path.startswith("/") else "/" + path
    if API_PREFIX and not path.startswith(API_PREFIX + "/") and path != API_PREFIX:
        path = API_PREFIX + path
    return f"{BACKEND_BASE_URL.rstrip('/')}{path}"


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return {"status_code": r.status_code, "text": r.text[:2000]}


async def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    m = method.upper()
    async with httpx.AsyncClient(timeout=TIMEOUT) as cx:
        r = await cx.request(
            m, _url(path), params=params, json=json_body, headers=_headers(),
        )
        r.raise_for_status()
        return _safe_json(r)


async def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return await _request("GET", path, params=params)


async def _post(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    return await _request("POST", path, params=params, json_body=json_body)


async def _put(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    return await _request("PUT", path, params=params, json_body=json_body)


mcp = FastMCP(
    name="QuantumTrade Pro",
    instructions=(
        "Tools for the QuantumTrade Pro AI trading backend. Use them to inspect "
        "positions and loop health, review multi-agent opinions / trade memory, "
        "and intervene safely (close a position, adjust SL/TP, stop/start the "
        "loop, or sentry halt/resume). Prefer curated tools over call_backend. "
        "Never request exchange API keys — use MCP/API token only."
    ),
)


# ---- Opinion / learning ------------------------------------------------------
@mcp.tool
async def get_trading_opinion(symbol: str, bars: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Analyze a symbol with the multi-agent opinion layer.

    Prefer providing recent OHLCV bars when available. If bars is omitted, the
    backend may reject the request — use get_loop_status / call_backend to fetch
    market context first when needed."""
    body: Dict[str, Any] = {"symbol": symbol.upper()}
    if bars:
        body["bars"] = bars
    return await _post("/trading/opinion/analyze", json_body=body)


@mcp.tool
async def get_opinion_weights() -> dict:
    """Get the current agent voting weights used to aggregate the opinion."""
    return await _get("/trading/opinion/weights")


@mcp.tool
async def set_opinion_weight(agent: str, weight: float) -> dict:
    """Set the voting weight for one agent (e.g. learned_skill=0.15)."""
    return await _post("/trading/opinion/weights", json_body={"weights": {agent: weight}})


@mcp.tool
async def trade_memory_status() -> dict:
    """Status of the semantic trade-memory vector store."""
    return await _get("/trading/trade-memory/status")


@mcp.tool
async def recall_similar_trades(
    context: Dict[str, Any],
    symbol: Optional[str] = None,
    same_symbol_only: bool = False,
    limit: int = 8,
) -> dict:
    """Recall similar past trades for a market context and summarise outcomes."""
    body = {
        "context": context,
        "symbol": symbol,
        "same_symbol_only": same_symbol_only,
        "limit": limit,
    }
    return await _post("/trading/trade-memory/recall", json_body=body)


@mcp.tool
async def trade_memory_backfill(limit: int = 1000) -> dict:
    """Vectorise existing closed trades from SQL into trade-memory."""
    return await _post("/trading/trade-memory/backfill", params={"limit": limit})


@mcp.tool
async def list_strategy_skills(active_only: bool = True, limit: int = 50) -> dict:
    """List learned strategy skills."""
    return await _get("/trading/skills", params={"active_only": active_only, "limit": limit})


@mcp.tool
async def skills_leaderboard(limit: int = 10) -> dict:
    """Top learned strategy skills by edge score."""
    return await _get("/trading/skills/leaderboard", params={"limit": limit})


@mcp.tool
async def mine_skills(limit: Optional[int] = None) -> dict:
    """Trigger a skill-mining pass over closed-trade history."""
    params = {"limit": limit} if limit is not None else None
    return await _post("/trading/skills/mine", params=params)


@mcp.tool
async def match_skill(context: Dict[str, Any]) -> dict:
    """Match a market context to the best learned strategy skill."""
    return await _post("/trading/skills/match", json_body={"context": context})


# ---- Portfolio / positions / trades (oversight) ------------------------------
@mcp.tool
async def get_portfolio() -> dict:
    """Get portfolio snapshot (positions, equity, PnL)."""
    return await _get("/trading/portfolio")


@mcp.tool
async def list_positions() -> dict:
    """List open DB/exchange-tracked positions."""
    return await _get("/trading/positions")


@mcp.tool
async def list_binance_positions() -> dict:
    """List live Binance Futures positions (exchange truth)."""
    return await _get("/trading/binance/positions")


@mcp.tool
async def list_trades(limit: int = 50) -> dict:
    """List recent trades (open + closed)."""
    return await _get("/trading/trades", params={"limit": limit})


@mcp.tool
async def close_position(position_id: int) -> dict:
    """Close an open position by DB id (market reduce-only on exchange)."""
    return await _post(f"/trading/positions/{position_id}/close")


@mcp.tool
async def modify_position(
    position_id: int,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> dict:
    """Modify stop-loss and/or take-profit for an open position."""
    body: Dict[str, Any] = {}
    if stop_loss is not None:
        body["stop_loss"] = stop_loss
    if take_profit is not None:
        body["take_profit"] = take_profit
    if not body:
        raise ValueError("Provide stop_loss and/or take_profit")
    return await _put(f"/trading/positions/{position_id}/modify", json_body=body)


# ---- Loop / config control ---------------------------------------------------
@mcp.tool
async def get_loop_status() -> dict:
    """Trading loop status (running, interval, symbols, cycle count)."""
    return await _get("/trading/loop/status")


@mcp.tool
async def get_trading_config() -> dict:
    """Active trading configuration (mode, symbols, risk limits)."""
    return await _get("/trading/config")


@mcp.tool
async def update_trading_config(updates: Dict[str, Any]) -> dict:
    """Update trading config keys (risk/sizing/symbols). Use carefully."""
    return await _post("/trading/config/update", json_body=updates)


@mcp.tool
async def start_trading_loop(
    interval_minutes: int = 15,
    strategy: str = "combined",
    symbols: Optional[List[str]] = None,
) -> dict:
    """Start (or restart) the trading loop."""
    body: Dict[str, Any] = {
        "interval_minutes": interval_minutes,
        "strategy": strategy,
    }
    if symbols:
        body["symbols"] = symbols
    return await _post("/trading/loop/start", json_body=body)


@mcp.tool
async def stop_trading_loop() -> dict:
    """Stop the trading loop (positions remain open until closed)."""
    return await _post("/trading/loop/stop")


# ---- Sentry safety -----------------------------------------------------------
@mcp.tool
async def sentry_status() -> dict:
    """Sentry halt state (trading_allowed, reason, timestamps)."""
    return await _get("/sentry/status")


@mcp.tool
async def sentry_halt(reason: str = "operator MCP halt", source: str = "a0-hermes-mcp") -> dict:
    """Soft halt: block new entries without cancelling open orders."""
    return await _post(
        "/sentry/halt",
        json_body={"reason": reason, "source": source, "manual": True},
    )


@mcp.tool
async def emergency_halt(
    reason: str = "operator MCP emergency halt",
    source: str = "a0-hermes-mcp",
) -> dict:
    """Emergency halt: stop trading and cancel open orders."""
    return await _post(
        "/sentry/emergency-halt",
        json_body={"reason": reason, "source": source, "manual": True},
    )


@mcp.tool
async def sentry_resume(note: str = "resumed via MCP", reconcile: bool = True) -> dict:
    """Clear sentry halt and optionally reconcile protections."""
    return await _post(
        "/sentry/resume",
        json_body={"note": note, "reconcile": reconcile},
    )


# ---- Research (Scrapling sidecar via backend) --------------------------------
@mcp.tool
async def research_fetch(
    url: str,
    mode: str = "http",
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
) -> dict:
    """Fetch a public web page through Scrapling for Grok/A0 research.

    Use this for news, calendars, and filings — not live prices (use feed/binance).
    mode=http for normal pages; mode=stealth for JS-heavy sites.
    """
    return await _post(
        "/research/fetch",
        json_body={
            "url": url,
            "mode": mode,
            "extraction_type": extraction_type,
            "css_selector": css_selector,
        },
    )


@mcp.tool
async def research_health() -> dict:
    """Check whether the Scrapling research sidecar is reachable."""
    return await _get("/research/health")


# ---- Sentiment ---------------------------------------------------------------
@mcp.tool
async def sentiment_loop_status() -> dict:
    """Native sentiment loop status."""
    return await _get("/news/sentiment-loop/status")


@mcp.tool
async def run_sentiment_loop(dry_run: bool = True) -> dict:
    """Run one sentiment-loop pass. dry_run=True writes nothing."""
    return await _post("/news/sentiment-loop/run", params={"dry_run": dry_run})


# ---- Research Plane (read-only search) ---------------------------------------
@mcp.tool
async def search_research(
    kind: str = "decisions",
    symbol: Optional[str] = None,
    q: Optional[str] = None,
    verdict: Optional[str] = None,
    from_date: Optional[str] = None,
    limit: int = 50,
) -> Any:
    """Search the read-only Research Plane for news or trading decisions.

    kind: 'news' or 'decisions' (news and decisions search only).
    symbol: filter by ticker symbol (e.g. BTCUSDC).
    q: keyword search query (for news).
    verdict: filter by decision or promotion verdict (e.g. REJECT, SHADOW, PROMOTE, BUY, SELL).
    from_date: ISO datetime start filter.
    """
    kind_lower = kind.strip().lower()
    params: Dict[str, Any] = {"limit": min(100, max(1, limit))}
    if from_date:
        params["from"] = from_date

    if kind_lower == "news":
        if symbol:
            params["symbol"] = symbol.upper()
        if q:
            params["q"] = q
        return await _get("/research/news", params=params)
    elif kind_lower in ("decisions", "decision"):
        if symbol:
            params["symbol"] = symbol.upper()
        if verdict:
            params["verdict"] = verdict.upper()
        return await _get("/research/decisions", params=params)
    else:
        return {"error": f"Unsupported research search kind '{kind}'. Must be 'news' or 'decisions'."}


# ---- Health / escape hatch ---------------------------------------------------

@mcp.tool
async def backend_health() -> dict:
    """Backend health check."""
    return await _get("/health")


@mcp.tool
async def call_backend(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> dict:
    """Call an arbitrary backend endpoint. method=GET|POST|PUT.

    Prefer curated tools. Path like "/trading/strategies"."""
    m = method.upper()
    if m not in {"GET", "POST", "PUT"}:
        raise ValueError("method must be GET, POST, or PUT")
    return {"result": await _request(m, path, params=params, json_body=body)}


def _maybe_full_api() -> "FastMCP":
    if os.getenv("FULL_API", "false").lower() != "true":
        return mcp
    try:
        openapi_url = f"{BACKEND_BASE_URL.rstrip('/')}/openapi.json"
        spec = httpx.get(openapi_url, timeout=TIMEOUT, headers=_headers()).json()
        client = httpx.AsyncClient(
            base_url=BACKEND_BASE_URL.rstrip("/"), headers=_headers(),
        )
        return FastMCP.from_openapi(
            openapi_spec=spec, client=client, name="QuantumTrade Pro (full API)",
        )
    except Exception as e:  # pragma: no cover
        print(f"[mcp] FULL_API requested but failed ({e}); using curated tools")
        return mcp


def main():
    server = _maybe_full_api()
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "stdio":
        server.run()
    else:
        server.run(
            transport=transport,
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "9100")),
        )


if __name__ == "__main__":
    main()
