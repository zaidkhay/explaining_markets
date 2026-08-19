"""Environment-driven construction of bounded live V3 providers."""
from __future__ import annotations

import os

from explaining_markets.providers.news_provider import AlphaVantageNewsProvider
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.reasoning.news_reasoner import NewsReasoner
from explaining_markets.reasoning.openrouter_client import openrouter_api_key
from explaining_markets.v3_providers import NullV3Providers, V3ProviderBundle


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def default_provider_bundle_from_env(*, production_safe: bool = True) -> V3ProviderBundle:
    """Create live-capable providers without requiring every credential.

    Production defaults are deliberately bounded: Alpha Vantage news calls use
    a short timeout and at most one peer query, and OpenRouter is disabled
    unless ``V3_LIVE_USE_OPENROUTER=1`` is explicitly set.  The quantitative
    model therefore keeps working when an LLM is rate-limited or slow.

    Current earnings/guidance/metadata/peer implementations remain fail-closed
    until a vendor with auditable ``available_at`` timestamps is configured.
    The existing SQLite cache is loaded separately by ``live_v3_context``.
    """
    null = NullV3Providers()
    news_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("NEWS_API_KEY")
    if news_key:
        if production_safe:
            timeout = float(os.getenv("V3_LIVE_NEWS_TIMEOUT_SECONDS", "5"))
            news = AlphaVantageNewsProvider(
                news_key,
                timeout_seconds=max(1.0, min(timeout, 10.0)),
                limit=50,
                max_peer_queries=1,
            )
        else:
            news = AlphaVantageNewsProvider(news_key)
    else:
        news = null

    use_openrouter = bool(openrouter_api_key()) and _truthy("V3_LIVE_USE_OPENROUTER", "0")
    return V3ProviderBundle(
        earnings=null,
        guidance=null,
        prices=null,
        metadata=null,
        peers=null,
        news=news,
        article_reasoner=NewsReasoner(use_openrouter=use_openrouter),
        event_reasoner=EventReasoner(use_openrouter=use_openrouter),
    )
