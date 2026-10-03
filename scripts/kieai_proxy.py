"""OpenAI-compat proxy that maps Agent Zero chat calls onto Kie.ai family endpoints.

Mounted on the VPS at ``/docker/kieai-proxy/proxy.py``. Agent Zero talks
``POST /v1/chat/completions``; Kie requires Claude Messages, Responses API, or
per-slug OpenAI chat depending on the model (docs.kie.ai).
"""

from __future__ import annotations

import json
import os
import time
import uuid

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

KIE_KEY = os.environ.get("KIE_API_KEY", "").replace("Bearer ", "").strip()
OMNI_KEY = os.environ.get("OMNIROUTE_API_KEY", "").replace("Bearer ", "").strip()
OMNI_URL = "https://omni.allikas.online/v1/chat/completions"
KIE_HOST = "https://api.kie.ai"
PORT = int(os.environ.get("KIEAI_PROXY_PORT", "11434"))

_ALIASES = {
    "astra": "gpt-6-astra",
    "gpt-astra": "gpt-6-astra",
    "gpt6-astra": "gpt-6-astra",
    "gpt-6.astra": "gpt-6-astra",
    "claude-fable-5.1": "claude-fable-5-1",
    "fable-5.1": "claude-fable-5-1",
    "fable-5-1": "claude-fable-5-1",
    "fable-5": "claude-fable-5",
    "claude-fable": "claude-fable-5",
    "gpt-5.2": "gpt-5-2",
    "gemini-3-8-flash": "gemini-3-8-flash-openai",
    "gemini-3-6-flash": "gemini-3-6-flash-openai",
}
_OMNI_PRESETS = {
    "auto/smart",
    "auto/fast",
    "auto/cheap",
    "auto/reasoning",
    "auto/coding",
    "auto/best-free",
    "auto/chat",
    "auto/best-coding",
}

_GEMINI_SLUGS = {
    "gemini-3-8-flash": "gemini-3-8-flash-openai",
    "gemini-3-8-flash-openai": "gemini-3-8-flash-openai",
    "gemini-3-pro": "gemini-3-pro",
    "gemini-3-pro-openai": "gemini-3-pro",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3-flash-openai": "gemini-3-flash-openai",
    "gemini-3-6-flash": "gemini-3-6-flash-openai",
    "gemini-3-6-flash-openai": "gemini-3-6-flash-openai",
}

MODEL_IDS = [
    "gpt-6-astra",
    "gpt-5-6-luna",
    "gpt-5-6-sol",
    "gpt-5-6-terra",
    "gpt-5-2",
    "gpt-5.4-codex",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex",
    "gpt-5-codex",
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5",
    "gemini-3-8-flash-openai",
    "gemini-3-6-flash-openai",
    "gemini-3-flash",
    "gemini-3-pro",
    "grok-4-6",
    "grok-4-5",
    "deepseek-chat",
    "deepseek-reasoner",
    "auto/fast",
    "auto/chat",
    "auto/cheap",
    "auto/best-coding",
]


def canonical_model(name: str) -> str:
    mid = (name or "").strip().lower().split("/")[-1]
    if not mid:
        return "gpt-5-6-luna"
    return _ALIASES.get(mid, mid)


def resolve_route(model: str) -> tuple[str, str, str]:
    """Return (kind, url, canonical_model)."""
    raw = (model or "").strip()
    lower = raw.lower()
    if lower.startswith("auto/") or lower in _OMNI_PRESETS:
        preset = lower if lower.startswith("auto/") else f"auto/{lower}"
        return "omni", OMNI_URL, preset
    mid = canonical_model(raw)
    if mid.startswith("auto") or mid in {"best-coding", "fast", "cheap", "chat"}:
        preset = mid if mid.startswith("auto/") else f"auto/{mid}"
        return "omni", OMNI_URL, preset
    if mid.startswith("claude"):
        return "claude", f"{KIE_HOST}/claude/v1/messages", mid
    if mid.startswith("grok"):
        return "responses", f"{KIE_HOST}/grok/v1/responses", mid
    if mid.startswith("gemini"):
        slug = _GEMINI_SLUGS.get(mid, mid if mid.endswith("-openai") else mid)
        if "flash" in mid and mid not in _GEMINI_SLUGS and not mid.endswith("-openai"):
            slug = f"{mid}-openai"
        return "openai_chat", f"{KIE_HOST}/{slug}/v1/chat/completions", slug
    if "codex" in mid:
        return "responses", f"{KIE_HOST}/api/v1/responses", mid
    if mid.startswith("deepseek"):
        sub = "deepseek-reasoner" if "reason" in mid or "r1" in mid else "deepseek-chat"
        return "openai_chat", f"{KIE_HOST}/api/v1/chat/completions", sub
    if (
        mid.startswith("gpt-5-6")
        or mid.startswith("gpt-5-5")
        or mid.startswith("gpt-5-4")
        or mid.startswith("gpt-6")
    ):
        return "responses", f"{KIE_HOST}/codex/v1/responses", mid
    if mid in {"gpt-5-2"} or mid.startswith("gpt-5-2"):
        return "openai_chat", f"{KIE_HOST}/gpt-5-2/v1/chat/completions", "gpt-5-2"
    return "responses", f"{KIE_HOST}/codex/v1/responses", mid


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
        return "\n".join(p for p in parts if p)
    return str(content or "")


def split_system(messages: list) -> tuple[str, list]:
    system_parts = []
    rest = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            system_parts.append(_message_text(msg.get("content")))
        else:
            rest.append(msg)
    return "\n".join(p for p in system_parts if p), rest


def to_responses_input(messages: list, system: str) -> list:
    out = []
    if system:
        out.append({"role": "system", "content": [{"type": "input_text", "text": system}]})
    for msg in messages:
        role = msg.get("role") or "user"
        content = msg.get("content")
        part_type = "output_text" if role == "assistant" else "input_text"
        parts = []
        if isinstance(content, str):
            parts.append({"type": part_type, "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": part_type, "text": item})
                elif isinstance(item, dict):
                    itype = item.get("type")
                    if itype in ("text", "input_text", "output_text"):
                        parts.append({"type": part_type, "text": item.get("text", "")})
                    elif itype in ("image_url", "input_image"):
                        url = item.get("image_url") or item.get("url") or ""
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        parts.append({"type": "input_image", "image_url": url})
                    else:
                        parts.append({"type": part_type, "text": item.get("text", "")})
        else:
            parts.append({"type": part_type, "text": str(content or "")})
        out.append({"role": role, "content": parts})
    if not out:
        out.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})
    return out


def claude_content(content):
    if isinstance(content, list):
        return content
    return _message_text(content)


def claude_messages(messages: list) -> list:
    out = []
    for msg in messages:
        role = msg.get("role") or "user"
        if role not in ("user", "assistant"):
            role = "user"
        out.append({"role": role, "content": claude_content(msg.get("content"))})
    if not out:
        out.append({"role": "user", "content": ""})
    return out


def extract_text(kind: str, data: dict) -> str:
    if kind == "claude":
        text = ""
        for block in data.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text") or ""
        return text
    if kind == "responses":
        if isinstance(data.get("output_text"), str) and data["output_text"]:
            return data["output_text"]
        chunks = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                    chunks.append(part.get("text") or "")
        if chunks:
            return "".join(chunks)
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            return (msg.get("content") if isinstance(msg, dict) else "") or ""
        return ""
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            return msg.get("content") or msg.get("reasoning_content") or ""
    return data.get("output_text") or ""


def openai_completion(model: str, text: str) -> dict:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse_wrap(model: str, text: str) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    done = {
        "id": chunk["id"],
        "object": "chat.completion.chunk",
        "created": chunk["created"],
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(done)}\n\ndata: [DONE]\n\n"


def kie_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


def _copy_tools(src: dict, dest: dict) -> None:
    if src.get("tools"):
        dest["tools"] = src["tools"]
    if src.get("tool_choice") is not None:
        dest["tool_choice"] = src["tool_choice"]


def build_payload(kind: str, model: str, body: dict) -> dict:
    messages = body.get("messages") or []
    system, rest = split_system(messages)
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system = "\n".join(p for p in (instructions.strip(), system) if p)
    tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
    if kind == "claude":
        payload = {
            "model": model,
            "messages": claude_messages(rest),
            "max_tokens": tokens,
            "stream": False,
        }
        if system:
            payload["system"] = system
        _copy_tools(body, payload)
        return payload
    if kind == "responses":
        payload = {
            "model": model,
            "stream": False,
            "input": to_responses_input(rest, system),
            "reasoning": {"effort": "low"},
        }
        _copy_tools(body, payload)
        return payload
    chat = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + rest,
        "max_tokens": tokens,
        "stream": False,
    }
    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    _copy_tools(body, chat)
    return chat


def post_kie(kind: str, url: str, model: str, body: dict, api_key: str) -> tuple[int, dict | str]:
    payload = build_payload(kind, model, body)
    try:
        resp = requests.post(url, headers=kie_headers(api_key), json=payload, timeout=120)
    except requests.RequestException as exc:
        return 502, {"error": {"message": str(exc), "type": "proxy_error"}}
    try:
        data = resp.json()
    except ValueError:
        data = {"error": {"message": resp.text[:800], "type": "invalid_json"}}
    return resp.status_code, data


def execute(body: dict) -> tuple[int, dict]:
    req_model = body.get("model") or ""
    kind, url, model = resolve_route(req_model)
    api_key = OMNI_KEY if kind == "omni" else KIE_KEY
    status, data = post_kie(kind, url, model, body, api_key)
    text = extract_text(kind, data if isinstance(data, dict) else {}) if status < 400 else ""
    # Fable 5.1 is listed on Kie as coming soon; use documented Fable 5 if empty/4xx.
    if canonical_model(req_model) == "claude-fable-5-1" and (status >= 400 or not text):
        status, data = post_kie(
            "claude", f"{KIE_HOST}/claude/v1/messages", "claude-fable-5", body, KIE_KEY
        )
        kind = "claude"
        model = "claude-fable-5"
        text = extract_text(kind, data if isinstance(data, dict) else {}) if status < 400 else ""
    if status >= 400:
        err = data if isinstance(data, dict) else {"error": {"message": str(data)}}
        return status, err
    if not text:
        return 502, {"error": {"message": f"Empty Kie response for {model}", "type": "empty_response"}}
    return 200, openai_completion(req_model or model, text)


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    stream = bool(body.get("stream"))
    status, payload = execute(body)
    if status != 200:
        return jsonify(payload), status
    if stream:
        model = (body.get("model") or payload.get("model") or "").strip()
        text = payload["choices"][0]["message"]["content"]
        return Response(sse_wrap(model, text), mimetype="text/event-stream")
    return jsonify(payload)


@app.route("/v1/responses", methods=["POST"])
@app.route("/responses", methods=["POST"])
def responses():
    body = request.get_json(silent=True) or {}
    if "messages" not in body and "input" in body:
        inp = body.get("input")
        messages = []
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            messages.extend(inp)
        body = dict(body)
        body["messages"] = messages
    status, payload = execute(body)
    if status != 200:
        return jsonify(payload), status
    text = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    shaped = {
        "id": str(payload.get("id") or "").replace("chatcmpl-", "resp-"),
        "object": "response",
        "status": "completed",
        "model": payload.get("model"),
        "output_text": text,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }
    return jsonify(shaped), status


@app.route("/v1/models", methods=["GET"])
@app.route("/api/tags", methods=["GET"])
def models():
    items = [
        {"id": mid, "object": "model", "owned_by": "kieai-proxy", "name": mid}
        for mid in MODEL_IDS
    ]
    if request.path == "/api/tags":
        return jsonify({"models": items})
    return jsonify({"object": "list", "data": items})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "proxy": "kieai-proxy-v8-family"})


if __name__ == "__main__":
    print(f"Kie.ai family proxy v8 on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
