"""External agent connectivity — OAuth metadata, tool manifest, Grok overseer."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.security import validate_admin_request

router = APIRouter(prefix="/agents", tags=["agents"])


def _public_base_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8081"
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"


@router.get("/connect")
async def agent_connect_manifest(request: Request) -> Dict[str, Any]:
    """Connection manifest for MCP clients, Cursor agents, and GrokBOT overseer."""
    base = _public_base_url(request)
    mcp_port = os.getenv("MCP_PORT", "9100")
    return {
        "name": "QuantumTrade Pro Agent API",
        "version": "1.0",
        "backend_health": f"{base}/api/backend/health",
        "openapi": f"{base}/api/backend/openapi.json",
        "mcp": {
            "transport": os.getenv("MCP_TRANSPORT", "http"),
            "url": f"http://{request.headers.get('host', 'localhost').split(':')[0]}:{mcp_port}",
            "auth": "Bearer <ADMIN_API_KEY or AGENT_API_KEY>",
        },
        "auth_methods": [
            {"type": "api_key", "header": "X-API-Key", "env": "ADMIN_API_KEY"},
            {"type": "bearer", "header": "Authorization", "value": "Bearer <token>"},
            {"type": "oauth2_client_credentials", "token_url": f"{base}/api/agents/oauth/token"},
        ],
        "tool_routes": {
            "trading_status": "/api/backend/trading/status",
            "loop_status": "/api/backend/trading/loop/status",
            "opinion_analyze": "/api/backend/trading/opinion/analyze",
            "signals_scan": "/api/signals/scan-markets",
            "grok_overseer": "/api/agents/grok-overseer/overview",
        },
        "scopes": ["read:trading", "write:trading", "read:signals", "overseer:grok"],
    }


@router.get("/oauth")
async def agent_oauth_metadata(request: Request) -> Dict[str, Any]:
    """OAuth2 discovery for agent tool integrations (client credentials)."""
    base = _public_base_url(request)
    return {
        "issuer": base,
        "token_endpoint": f"{base}/api/agents/oauth/token",
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"],
        "scopes_supported": ["read:trading", "write:trading", "read:signals", "overseer:grok"],
        "service_documentation": f"{base}/api/backend/docs",
    }


class AgentTokenRequest(BaseModel):
    grant_type: str = Field(default="client_credentials")
    client_id: str
    client_secret: str
    scope: Optional[str] = None


@router.post("/oauth/token")
async def agent_oauth_token(body: AgentTokenRequest) -> Dict[str, Any]:
    """Issue a bearer token for external agents (client_id + client_secret)."""
    expected_id = os.getenv("AGENT_CLIENT_ID", os.getenv("ADMIN_API_KEY", ""))
    expected_secret = os.getenv("AGENT_CLIENT_SECRET", os.getenv("ADMIN_API_KEY", ""))
    if not expected_id or not expected_secret:
        raise HTTPException(status_code=503, detail="Agent OAuth not configured on server")

    if body.client_id != expected_id or body.client_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid client credentials")

    # Static long-lived token — same as admin key for single-tenant VPS deploys
    token = os.getenv("AGENT_API_KEY", os.getenv("ADMIN_API_KEY", ""))
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 86400,
        "scope": body.scope or "read:trading write:trading read:signals overseer:grok",
    }


@router.post("/validate")
async def validate_agent_token(request: Request) -> Dict[str, Any]:
    """Validate the caller's agent/admin token."""
    try:
        validate_admin_request(request)
        return {"valid": True, "message": "Token accepted"}
    except HTTPException as exc:
        return {"valid": False, "message": exc.detail}


@router.get("/grok-overseer/overview")
async def grok_overseer_overview(request: Request) -> Dict[str, Any]:
    """Read-only platform overview for GrokBOT overseer / supervisor jobs."""
    validate_admin_request(request)
    import httpx

    admin_token = (
        request.headers.get("x-api-key", "").strip()
        or (
            request.headers.get("authorization", "").strip()[7:].strip()
            if request.headers.get("authorization", "").strip().lower().startswith("bearer ")
            else ""
        )
        or os.getenv("ADMIN_API_KEY", "").strip()
        or os.getenv("API_AUTH_TOKEN", "").strip()
        or os.getenv("BACKEND_API_KEY", "").strip()
    )
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}

    base = "http://127.0.0.1:8000"
    paths = {
        "health": "/health",
        "sentry": "/sentry/status",
        "loop": "/trading/loop/status",
        "portfolio": "/trading/portfolio",
        "positions": "/trading/positions",
        "signals_candidates": "/api/signals/candidates",
        "ctrader": "/trading/ctrader/status",
        "brokers": "/trading/brokers",
    }
    out: Dict[str, Any] = {"timestamp": int(time.time()), "sections": {}}
    async with httpx.AsyncClient(timeout=12.0) as client:
        for key, path in paths.items():
            try:
                r = await client.get(f"{base}{path}", headers=headers)
                out["sections"][key] = r.json() if r.is_success else {"error": r.text[:300], "status": r.status_code}
            except Exception as exc:
                out["sections"][key] = {"error": str(exc)}

    out["public_url"] = _public_base_url(request)
    out["grok_model"] = os.getenv("XAI_MODEL", "grok-beta")
    out["overseer_note"] = (
        "Use this snapshot for supervisor decisions. POST /api/agents/grok-overseer/analyze "
        "for an LLM summary when XAI_API_KEY is configured. "
        "For web/news/calendar research use POST /api/research/fetch or MCP scrapling / research_fetch."
    )
    out["research"] = {
        "fetch": "/api/research/fetch",
        "health": "/api/research/health",
    }
    return out


class GrokOverseerAnalyzeRequest(BaseModel):
    focus: str = Field(default="risk and operational health", max_length=500)
    include_positions: bool = True


@router.post("/grok-overseer/analyze")
async def grok_overseer_analyze(body: GrokOverseerAnalyzeRequest, request: Request) -> Dict[str, Any]:
    """Run a Grok-powered overseer summary over the live platform snapshot."""
    validate_admin_request(request)
    overview = await grok_overseer_overview(request)

    import json
    from backend.llm.router import call_llm_resilient

    prompt = (
        f"You are GrokBOT, the trading platform overseer. Focus: {body.focus}.\n"
        f"Platform snapshot JSON:\n{json.dumps(overview, default=str)[:12000]}\n"
        "Return: 1) Health verdict 2) Top risks 3) Recommended actions (max 5 bullets each)."
    )
    try:
        summary = await call_llm_resilient(
            "grok_overseer",
            prompt=prompt,
            system="You are a concise trading infrastructure overseer. No fluff.",
            temperature=0.2,
            max_tokens=900,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Grok overseer failed: {exc}") from exc

    return {"summary": summary, "overview": overview}
