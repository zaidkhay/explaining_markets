"""Small OpenRouter structured-output client used by V3 reasoners.

The client intentionally uses Chat Completions rather than OpenRouter's beta
Responses API. Requests are strict JSON-schema, require providers that support
all requested parameters, and default to providers that deny non-transient
data collection. A process-wide call budget prevents historical backfills from
accidentally exhausting a free OpenRouter allowance.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import httpx

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_LOCK = threading.Lock()
_CALLS_BY_KEY: dict[str, int] = {}


def openrouter_api_key() -> str | None:
    """Return the configured OpenRouter key, accepting both common spellings."""
    return os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def openrouter_model() -> str:
    """Pinned/selected model. ``openrouter/free`` is useful for zero-cost smoke tests."""
    return os.getenv("OPEN_ROUTER_MODEL") or os.getenv("OPENROUTER_MODEL") or "openrouter/free"


def _reserve_call(api_key: str, max_calls: int) -> int:
    with _LOCK:
        used = _CALLS_BY_KEY.get(api_key, 0)
        if used >= max_calls:
            raise RuntimeError(f"OpenRouter process call budget exhausted ({max_calls})")
        used += 1
        _CALLS_BY_KEY[api_key] = used
        return used


def reset_openrouter_budget_for_tests() -> None:
    with _LOCK:
        _CALLS_BY_KEY.clear()


def structured_json(
    *,
    schema_name: str,
    schema: dict[str, Any],
    system_prompt: str,
    user_payload: dict[str, Any],
    model: str | None = None,
    timeout: float = 20.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return one strict JSON object from OpenRouter.

    The caller supplies only pre-cutoff evidence. The helper never performs web
    search or tool calls and therefore cannot add post-cutoff information.
    """
    api_key = openrouter_api_key()
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_API_KEY is not configured")

    max_calls = max(1, int(os.getenv("OPEN_ROUTER_MAX_CALLS", "25")))
    _reserve_call(api_key, max_calls)

    body = {
        "model": model or openrouter_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {
            "require_parameters": True,
            "data_collection": os.getenv("OPEN_ROUTER_DATA_COLLECTION", "deny"),
            "allow_fallbacks": True,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "Explaining Markets V3",
    }

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.post(_OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            http.close()

    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter returned a non-object response")
    if payload.get("error"):
        raise RuntimeError(f"OpenRouter error: {payload['error']}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter response did not contain text content")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise RuntimeError("OpenRouter structured response was not a JSON object")
    return data
