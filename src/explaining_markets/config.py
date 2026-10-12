"""Configuration read from the environment.

In deployment these come from your local ``.env`` file, which Modal loads at
deploy time (see ``.env.example`` and ``modal_app.py``).

Required:
  EM_API_KEY         your submission's API key; sent as the X-API-Key header
  EM_WEBHOOK_SECRET  your signing secret (whsec_...); verifies incoming webhooks

Additional submissions (optional, slots 2..5, served at /s/<n>):
  EM_API_KEY_S<n>         that submission's API key
  EM_WEBHOOK_SECRET_S<n>  that submission's signing secret

Optional:
  EM_API_BASE_URL   API base URL (default: production)
  OPENAI_API_KEY     when set, predict.py makes real LLM calls; otherwise it
                     falls back to a 0.5 baseline so the round-trip still works
  OPENAI_MODEL       model name for predict.py (default: gpt-5.4-nano)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.explainingmarkets.ai/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"


@dataclass(frozen=True)
class Config:
    api_key: str
    webhook_secret: str
    api_base_url: str

    @classmethod
    def from_env(cls, slot: int | None = None) -> "Config":
        """Load and validate required config. Raises if a required var is missing.

        ``slot=None`` is the original submission (EM_API_KEY / EM_WEBHOOK_SECRET).
        ``slot=n`` (2..5) is an additional submission served by the same
        deployment, configured by EM_API_KEY_S<n> / EM_WEBHOOK_SECRET_S<n>.
        """
        suffix = _slot_suffix(slot)
        return cls(
            api_key=_require(f"EM_API_KEY{suffix}"),
            webhook_secret=_require(f"EM_WEBHOOK_SECRET{suffix}"),
            api_base_url=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
        )


EXTRA_SLOTS = (2, 3, 4, 5)


def _slot_suffix(slot: int | None) -> str:
    if slot is None:
        return ""
    if slot not in EXTRA_SLOTS:
        raise ValueError(f"unknown submission slot {slot!r}; expected one of {EXTRA_SLOTS}")
    return f"_S{slot}"


def configured_slots() -> list[int]:
    """Extra slots whose API key and webhook secret are both set."""
    return [
        slot
        for slot in EXTRA_SLOTS
        if os.environ.get(f"EM_API_KEY_S{slot}") and os.environ.get(f"EM_WEBHOOK_SECRET_S{slot}")
    ]


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file "
            f"(copy .env.example to .env), then re-deploy. See the README."
        )
    return value


def openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
