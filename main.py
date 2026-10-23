"""Authenticated ALRYN AI provider gateway for Render.

Provider credentials live only in Render environment variables. This service
never returns secret values and requires a shared bearer secret for all /v1 routes.
"""
from __future__ import annotations

import hmac
import os
import time
import logging
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from providers import ProviderFailure, ProviderKey, load_provider_keys, generate_text, search_web

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("alryn.gateway")
app = FastAPI(title="ALRYN AI Backend", version="1.0.0", docs_url=None, redoc_url=None)
STARTED_AT = time.time()
MAX_PROMPT_CHARS = int(os.getenv("ALRYN_BACKEND_MAX_PROMPT_CHARS", "12000"))
REQUEST_TIMEOUT = int(os.getenv("ALRYN_BACKEND_REQUEST_TIMEOUT_SECONDS", "45"))
MAX_HISTORY_MESSAGES = int(os.getenv("ALRYN_BACKEND_MAX_HISTORY_MESSAGES", "12"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("ALRYN_BACKEND_REQUESTS_PER_MINUTE", "90"))
_recent_requests: list[float] = []


class ChatRequest(BaseModel):
    user_id: int = Field(gt=0)
    prompt: str = Field(min_length=1, max_length=12000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=24)
    preferred_scope: str | None = Field(default=None, max_length=160)


class SearchRequest(BaseModel):
    user_id: int = Field(gt=0)
    query: str = Field(min_length=1, max_length=1000)
    preferred_scope: str | None = Field(default=None, max_length=160)


def _require_auth(authorization: str | None) -> None:
    expected = os.getenv("ALRYN_BACKEND_SHARED_SECRET", "").strip()
    if len(expected) < 32:
        raise HTTPException(status_code=503, detail="Backend authentication is not configured")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _rate_limit() -> None:
    now = time.monotonic()
    while _recent_requests and _recent_requests[0] < now - 60:
        _recent_requests.pop(0)
    if len(_recent_requests) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Backend request rate limit reached")
    _recent_requests.append(now)



async def _run_chat_route(keys: list[ProviderKey], prompt: str, history: list[dict[str, str]], timeout: int):
    """Try compatible keys, stopping sibling-key retries on a shared-scope rate limit."""
    failures: list[str] = []
    for key in keys:
        try:
            answer = await generate_text(key, prompt, history, timeout)
            return answer, key, failures
        except ProviderFailure as exc:
            failures.append(f"{key.provider}:{exc.category}")
            log.warning("Provider route %s failed (%s)", key.label, exc.category)
            if exc.category == "rate_limited":
                break
    return None, None, failures


def _public_key(key: ProviderKey) -> dict[str, str]:
    return key.public_status()


def _keys(capability: str, preferred_scope: str | None = None) -> list[ProviderKey]:
    keys = [k for k in load_provider_keys() if capability in k.capabilities]
    if preferred_scope:
        keys = [k for k in keys if k.quota_scope == preferred_scope]
    return keys


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "ALRYN AI Backend", "status": "ok", "version": app.version}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "alryn-ai-backend", "uptime_seconds": int(time.time() - STARTED_AT)}


@app.get("/ready")
def ready() -> dict[str, str]:
    if len(os.getenv("ALRYN_BACKEND_SHARED_SECRET", "").strip()) < 32:
        raise HTTPException(status_code=503, detail="Backend shared secret is missing or too short")
    return {"status": "ready"}


@app.get("/v1/providers")
def providers(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    return {"providers": [_public_key(k) for k in load_provider_keys()]}


@app.post("/v1/chat")
async def chat(payload: ChatRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
    _require_auth(authorization)
    _rate_limit()
    if len(payload.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail="Prompt exceeds configured limit")
    # Only accept the two roles the app itself creates; no client-supplied system prompt.
    history = []
    for item in payload.history[-MAX_HISTORY_MESSAGES:]:
        role, content = item.get("role"), item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        history.append({"role": role, "content": content[:MAX_PROMPT_CHARS]})
    keys = _keys("chat", payload.preferred_scope)
    if not keys:
        raise HTTPException(status_code=503, detail="No configured chat route for the requested quota scope")
    answer, key, failures = await _run_chat_route(keys, payload.prompt, history, REQUEST_TIMEOUT)
    if key is not None and answer:
        return {"answer": answer, "provider": key.provider, "label": key.label,
                "account": key.account_label, "quota_scope": key.quota_scope, "model": key.model}
    raise HTTPException(status_code=503, detail="No provider route succeeded: " + ", ".join(failures[:4]))


@app.post("/v1/search")
async def search(payload: SearchRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    _rate_limit()
    keys = _keys("search", payload.preferred_scope)
    if not keys:
        raise HTTPException(status_code=503, detail="No configured search route for the requested quota scope")
    failures: list[str] = []
    for key in keys:
        try:
            results = await search_web(key, payload.query, REQUEST_TIMEOUT)
            return {"results": results, "provider": key.provider, "label": key.label,
                    "account": key.account_label, "quota_scope": key.quota_scope}
        except ProviderFailure as exc:
            failures.append(f"{key.provider}:{exc.category}")
            log.warning("Search route %s failed (%s)", key.label, exc.category)
            if exc.category == "rate_limited":
                break
    raise HTTPException(status_code=503, detail="No search route succeeded: " + ", ".join(failures[:4]))


@app.get("/internal/config-check")
def config_check(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    keys = load_provider_keys()
    return {"service": "alryn-ai-backend", "shared_secret_configured": True,
            "provider_entries_configured": len(keys),
            "chat_entries": sum("chat" in k.capabilities for k in keys),
            "search_entries": sum("search" in k.capabilities for k in keys),
            "note": "Presence only; secret values are never returned."}
