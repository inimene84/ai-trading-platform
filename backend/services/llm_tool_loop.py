"""
LLM Tool Execution Loop
Ported from FinceptTerminal LlmService.cpp (do_request, do_tool_loop)

This lets your LLM call functions (get_price, place_order, etc.) in a loop
until the task is done. Works with ANY OpenAI-compatible endpoint:
OpenAI, xAI, Groq, OpenRouter, etc.

Usage:
    from backend.services.llm_tool_loop import LlmToolClient, ToolRegistry

    client = LlmToolClient(api_key="sk-...", base_url="https://api.x.ai/v1", model="grok-4")
    client.tools = build_trading_tools()   # register your functions

    result = client.chat(
        user_message="Analyze BTC and place a paper trade if it looks bullish",
        system_prompt="You are a trading AI. Use tools to act."
    )
    print(result["content"])
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    content: str
    data: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"success": self.success, "content": self.content, "data": self.data}


class ToolRegistry:
    """
    Ported from Fincept McpService + McpProvider.
    Register functions the LLM is allowed to call.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, dict] = {}

    def register(self, name: str, schema: dict, fn: Callable):
        """
        Schema = OpenAI function schema dict with keys:
            description: str
            properties: dict
            required: list[str]
        """
        self._tools[name] = fn
        self._schemas[name] = schema

    def get_schemas(self) -> List[dict]:
        """Format for OpenAI-compatible tool calling."""
        tools = []
        for name, schema in self._schemas.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                }
            })
        return tools

    def execute(self, name: str, arguments: dict) -> ToolResult:
        if name not in self._tools:
            return ToolResult(False, f"Tool not found: {name}")
        try:
            result = self._tools[name](**arguments)
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, dict):
                return ToolResult(True, json.dumps(result), result)
            return ToolResult(True, str(result))
        except Exception as e:
            logger.exception(f"Tool execution failed: {name}")
            return ToolResult(False, f"Error: {str(e)}")


class LlmToolClient:
    """
    Multi-provider LLM client with tool-loop support.
    Ported from Fincept LlmService::do_request + do_tool_loop.
    """

    def __init__(self, api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o",
                 provider: str = "openai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.tools = ToolRegistry()
        self.timeout = 120.0

    def chat(self, user_message: str,
             history: Optional[List[dict]] = None,
             system_prompt: Optional[str] = None,
             max_tool_rounds: int = 5,
             temperature: float = 0.7) -> dict:
        """
        Send message to LLM. If the model requests tool calls, execute them
        and send follow-up requests automatically (max 5 rounds).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_schemas = self.tools.get_schemas()
        total_tool_calls = 0

        for round_num in range(max_tool_rounds):
            response = self._raw_chat(messages, tools=tool_schemas if round_num == 0 else None, temperature=temperature)
            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})

            # Check for tool_calls (OpenAI format)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # Normal text response
                return {
                    "content": msg.get("content", ""),
                    "usage": response.get("usage", {}),
                    "tool_calls_used": total_tool_calls > 0,
                    "tool_rounds": total_tool_calls,
                }

            # --- Execute tool calls ---
            logger.info(f"Tool round {round_num + 1}: {len(tool_calls)} tool call(s)")
            messages.append(msg)  # assistant message with tool_calls
            total_tool_calls += len(tool_calls)

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                call_id = tc["id"]

                logger.info(f"Executing tool: {fn_name}({fn_args})")
                result = self.tools.execute(fn_name, fn_args)

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result.to_json())
                })

        # Max rounds reached
        return {
            "content": "[Max tool rounds reached — incomplete]",
            "usage": {},
            "tool_calls_used": True,
            "tool_rounds": total_tool_calls,
        }

    def _raw_chat(self, messages: List[dict],
                  tools: Optional[List[dict]] = None,
                  temperature: float = 0.7) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {"choices": [{"message": {"content": f"Error: {e}"}}]}


# ═══════════════════════════════════════════════════════════════════════════════
# Trading Tool Builder — wires LLM to your trading system
# ═══════════════════════════════════════════════════════════════════════════════

def build_trading_tools(
    unified_trading,
    market_data_service=None,
    paper_session_id: str | None = None,
) -> ToolRegistry:
    """
    Register the tools your AI needs to trade.
    Pass your UnifiedTrading singleton and optional market data fetcher.
    """

    registry = ToolRegistry()

    # ── Market Data ──
    registry.register("get_price", {
        "description": "Get the current market price for a symbol",
        "properties": {"symbol": {"type": "string", "description": "Trading pair symbol, e.g. BTCUSDT"}},
        "required": ["symbol"]
    }, lambda symbol: _get_price_helper(symbol, market_data_service))

    registry.register("get_portfolio", {
        "description": "Get current paper trading portfolio state including cash and positions",
        "properties": {},
        "required": []
    }, lambda: _get_portfolio_helper(unified_trading, paper_session_id))

    # ── Order Management ──
    registry.register("place_paper_order", {
        "description": (
            "Place a simulated (paper) order. "
            "side='buy' or 'sell'. order_type='market' or 'limit'. "
            "For market orders, price is optional. "
            "For limit orders, price is required."
        ),
        "properties": {
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "quantity": {"type": "number"},
            "order_type": {"type": "string", "enum": ["market", "limit"]},
            "price": {"type": "number", "description": "Required for limit orders"},
            "stop_loss": {"type": "number"},
            "take_profit": {"type": "number"},
        },
        "required": ["symbol", "side", "quantity"]
    }, lambda symbol, side, quantity, order_type="market", price=0.0, stop_loss=0.0, take_profit=0.0:
        _place_order_helper(
            unified_trading, symbol, side, quantity, order_type, price,
            stop_loss, take_profit, paper_session_id,
        ))

    registry.register("cancel_paper_order", {
        "description": "Cancel a paper order by its order ID",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"]
    }, lambda order_id: _cancel_order_helper(unified_trading, order_id, paper_session_id))

    registry.register("get_paper_orders", {
        "description": "List paper orders, optionally filtered by status",
        "properties": {"status": {"type": "string", "enum": ["", "pending", "partial", "filled", "cancelled"]}},
        "required": []
    }, lambda status="": _get_orders_helper(unified_trading, status, paper_session_id))

    return registry


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_price_helper(symbol: str, market_data_service) -> dict:
    """Try DataHub first, then market data service, then fallback."""
    from backend.services.data_hub import DataHub
    cached = DataHub().peek(f"market:quote:{symbol}")
    if cached and isinstance(cached, dict) and "price" in cached:
        return {"symbol": symbol, "price": cached["price"], "source": "cache"}
    if market_data_service and hasattr(market_data_service, "get_price"):
        price = market_data_service.get_price(symbol)
        return {"symbol": symbol, "price": price, "source": "market_data_service"}
    return {"symbol": symbol, "price": None, "error": "No price source available"}


def _get_portfolio_helper(unified_trading, session_id=None) -> dict:
    pf = unified_trading.get_paper_portfolio(session_id)
    positions = unified_trading.get_paper_positions(session_id)
    stats = unified_trading.get_paper_stats(session_id)
    return {
        "cash": pf["cash"] if pf else 0.0,
        "positions": [
            {"symbol": p.symbol, "side": p.side, "quantity": p.quantity, "avg_price": p.avg_price}
            for p in positions
        ],
        "stats": stats,
    }


def _place_order_helper(
    ut, symbol, side, quantity, order_type, price, stop_loss, take_profit,
    session_id=None,
) -> dict:
    from backend.services.unified_trading import UnifiedOrder, OrderSide, OrderType
    try:
        order = UnifiedOrder(
            symbol=symbol.upper(),
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            quantity=float(quantity),
            price=float(price) if price else 0.0,
            stop_loss=float(stop_loss) if stop_loss else 0.0,
            take_profit=float(take_profit) if take_profit else 0.0,
        )
        resp = ut.place_order(order, session_id=session_id)
        return {
            "success": resp.success,
            "order_id": resp.order_id,
            "message": resp.message,
            "mode": resp.mode,
            "filled_price": resp.filled_price,
            "filled_qty": resp.filled_qty,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _cancel_order_helper(ut, order_id: str, session_id=None) -> dict:
    resp = ut.cancel_order(order_id, session_id=session_id)
    return {"success": resp.success, "message": resp.message, "mode": resp.mode}


def _get_orders_helper(ut, status: str, session_id=None) -> dict:
    orders = ut.get_paper_orders(status, session_id=session_id)
    return {"count": len(orders), "orders": orders}
