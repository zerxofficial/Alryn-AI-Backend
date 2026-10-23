from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
import httpx


@dataclass(frozen=True)
class ProviderKey:
    provider: str
    label: str
    secret: str
    account_label: str
    quota_scope: str
    model: str
    style: str
    base_url: str
    capabilities: frozenset[str]
    extra: dict[str, str]

    def public_status(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "label": self.label,
            "account": self.account_label,
            "quota_scope": self.quota_scope,
            "model": self.model or "not-configured",
            "capabilities": ",".join(sorted(self.capabilities)) or "none",
        }


# These are provisional defaults. Model availability and account access must be tested.
PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "gemini": {"prefix": "GEMINI", "count": 5, "style": "gemini", "base": "https://generativelanguage.googleapis.com/v1beta/models", "model_env": "GEMINI_MODEL", "model_default": "gemini-2.0-flash", "caps": {"chat", "vision"}},
    "groq": {"prefix": "GROQ", "count": 5, "style": "openai", "base": "https://api.groq.com/openai/v1/chat/completions", "model_env": "GROQ_MODEL", "model_default": "llama-3.3-70b-versatile", "caps": {"chat", "code"}},
    "openrouter": {"prefix": "OPENROUTER", "count": 5, "style": "openai", "base": "https://openrouter.ai/api/v1/chat/completions", "model_env": "OPENROUTER_MODEL", "model_default": "", "caps": {"chat", "code", "vision"}},
    "mistral": {"prefix": "MISTRAL", "count": 5, "style": "openai", "base": "https://api.mistral.ai/v1/chat/completions", "model_env": "MISTRAL_MODEL", "model_default": "mistral-small-latest", "caps": {"chat", "code"}},
    "cloudflare": {"prefix": "CLOUDFLARE", "count": 2, "style": "cloudflare_openai", "base": "https://api.cloudflare.com/client/v4/accounts", "model_env": "CLOUDFLARE_MODEL", "model_default": "@cf/meta/llama-3.1-8b-instruct", "caps": {"chat"}},
    "huggingface": {"prefix": "HUGGINGFACE", "count": 5, "style": "openai", "base": "https://router.huggingface.co/v1/chat/completions", "model_env": "HUGGINGFACE_MODEL", "model_default": "", "caps": {"chat", "code"}},
    "sambanova": {"prefix": "SAMBANOVA", "count": 5, "style": "openai", "base": "https://api.sambanova.ai/v1/chat/completions", "model_env": "SAMBANOVA_MODEL", "model_default": "Meta-Llama-3.3-70B-Instruct", "caps": {"chat", "code"}},
    "nvidia": {"prefix": "NVIDIA", "count": 2, "style": "openai", "base": "https://integrate.api.nvidia.com/v1/chat/completions", "model_env": "NVIDIA_MODEL", "model_default": "meta/llama-3.3-70b-instruct", "caps": {"chat", "code"}},
    "tavily": {"prefix": "TAVILY", "count": 5, "style": "search_tavily", "base": "https://api.tavily.com/search", "model_env": "", "model_default": "", "caps": {"search"}},
    "exa": {"prefix": "EXA", "count": 5, "style": "search_exa", "base": "https://api.exa.ai/search", "model_env": "", "model_default": "", "caps": {"search"}},
}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def load_provider_keys() -> list[ProviderKey]:
    entries: list[ProviderKey] = []
    for provider, spec in PROVIDER_SPECS.items():
        prefix = spec["prefix"]
        model = _env(spec["model_env"], spec["model_default"]) if spec["model_env"] else ""
        for number in range(1, spec["count"] + 1):
            label = f"{prefix}_{number:02d}"
            secret_env = f"{prefix}_API_KEY_{number:02d}"
            secret = _env(secret_env)
            if not secret:
                continue
            account = _env(f"ALRYN_{label}_ACCOUNT", f"unknown-account:{provider}")
            # Conservative default: unknown keys from the same provider share a scope until mapped.
            scope = _env(f"ALRYN_{label}_QUOTA_SCOPE", f"unknown:{provider}")
            # Optional per-key model override supports accounts with different model entitlements.
            key_model = _env(f"ALRYN_{label}_MODEL", model)
            extra: dict[str, str] = {}
            if provider == "cloudflare":
                extra["account_id"] = _env("CLOUDFLARE_ACCOUNT_ID")
            entries.append(ProviderKey(
                provider=provider, label=label, secret=secret, account_label=account,
                quota_scope=scope, model=key_model, style=spec["style"], base_url=spec["base"],
                capabilities=frozenset(spec["caps"]), extra=extra,
            ))
    return entries


class ProviderFailure(Exception):
    def __init__(self, category: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.category = category
        self.retryable = retryable


def _classify_status(status: int, body: str) -> ProviderFailure:
    lowered = body.lower()
    if status in (401, 403):
        category = "invalid_credentials_or_permission"
        retryable = False
    elif status == 429:
        category = "rate_limited"
        retryable = True
    elif status in (408, 425) or status >= 500:
        category = "timeout_or_provider_outage"
        retryable = True
    elif "credit" in lowered or "billing" in lowered or "quota" in lowered:
        category = "billing_or_quota_exhausted"
        retryable = False
    elif status == 400:
        category = "unsupported_request_or_model"
        retryable = False
    else:
        category = "provider_error"
        retryable = status >= 500
    return ProviderFailure(category, f"Provider request failed (HTTP {status}; {category}).", retryable)


async def generate_text(key: ProviderKey, prompt: str, history: list[dict[str, str]], timeout_seconds: int) -> str:
    if not key.model and key.style in {"openai", "gemini", "cloudflare_openai"}:
        raise ProviderFailure("model_not_configured", f"{key.provider} has a key but no model is configured.", False)
    messages = history + [{"role": "user", "content": prompt}]
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if key.style == "gemini":
                contents = []
                for item in messages:
                    role = "model" if item["role"] == "assistant" else "user"
                    contents.append({"role": role, "parts": [{"text": item["content"]}]})
                url = f"{key.base_url}/{key.model}:generateContent"
                response = await client.post(url, headers={"x-goog-api-key": key.secret}, json={"contents": contents})
                if response.status_code >= 400:
                    raise _classify_status(response.status_code, response.text)
                data = response.json()
                try:
                    text = "".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"])
                except (KeyError, IndexError, TypeError) as exc:
                    raise ProviderFailure("malformed_provider_output", "Provider returned an unexpected response.", False) from exc
            elif key.style == "cloudflare_openai":
                account_id = key.extra.get("account_id", "")
                if not account_id:
                    raise ProviderFailure("account_id_missing", "Cloudflare requires CLOUDFLARE_ACCOUNT_ID.", False)
                url = f"{key.base_url}/{account_id}/ai/v1/chat/completions"
                response = await client.post(url, headers={"Authorization": f"Bearer {key.secret}"},
                                             json={"model": key.model, "messages": messages})
                if response.status_code >= 400:
                    raise _classify_status(response.status_code, response.text)
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                if isinstance(text, list):
                    text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
            elif key.style == "openai":
                if key.provider == "openrouter":
                    headers = {"Authorization": f"Bearer {key.secret}", "X-Title": "ALRYN AI"}
                else:
                    headers = {"Authorization": f"Bearer {key.secret}"}
                response = await client.post(key.base_url, headers=headers, json={
                    "model": key.model,
                    "messages": messages,
                    "temperature": 0.7,
                })
                if response.status_code >= 400:
                    raise _classify_status(response.status_code, response.text)
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                if isinstance(text, list):
                    text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
            else:
                raise ProviderFailure("capability_not_implemented", f"{key.provider} is registered but does not provide chat in this build.", False)
        except ProviderFailure:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderFailure("timeout", "Provider request timed out.", True) from exc
        except httpx.RequestError as exc:
            raise ProviderFailure("network_error", "Could not reach the provider endpoint.", True) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderFailure("malformed_provider_output", "Provider returned an unexpected response.", False) from exc
    if not isinstance(text, str) or not text.strip():
        raise ProviderFailure("empty_provider_output", "Provider returned an empty response.", True)
    return text.strip()


async def search_web(key: ProviderKey, query: str, timeout_seconds: int) -> list[dict[str, str]]:
    """Execute a bounded Tavily or Exa search and return normalized title/URL/snippet records."""
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if key.style == "search_tavily":
                response = await client.post(key.base_url, json={
                    "api_key": key.secret,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": False,
                })
                if response.status_code >= 400:
                    raise _classify_status(response.status_code, response.text)
                raw = response.json().get("results", [])
                results = [{"title": str(item.get("title", "Untitled")),
                            "url": str(item.get("url", "")),
                            "snippet": str(item.get("content", ""))[:700]}
                           for item in raw if isinstance(item, dict) and item.get("url")]
            elif key.style == "search_exa":
                response = await client.post(key.base_url,
                    headers={"x-api-key": key.secret, "Content-Type": "application/json"},
                    json={"query": query, "type": "auto", "numResults": 5,
                          "contents": {"text": {"maxCharacters": 900}}})
                if response.status_code >= 400:
                    raise _classify_status(response.status_code, response.text)
                raw = response.json().get("results", [])
                results = [{"title": str(item.get("title") or item.get("url", "Untitled")),
                            "url": str(item.get("url", "")),
                            "snippet": str(item.get("text", ""))[:700]}
                           for item in raw if isinstance(item, dict) and item.get("url")]
            else:
                raise ProviderFailure("capability_not_implemented", "This key is not a supported search provider.", False)
        except ProviderFailure:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderFailure("timeout", "Search provider request timed out.", True) from exc
        except httpx.RequestError as exc:
            raise ProviderFailure("network_error", "Could not reach the search provider.", True) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderFailure("malformed_provider_output", "Search provider returned an unexpected response.", False) from exc
    return results[:5]
