"""Environment-driven construction of bounded live V3 providers."""
from __future__ import annotations

import os

from explaining_markets.providers.news_provider import AlphaVantageNewsProvider
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.reasoning.news_reasoner import NewsReasoner
from explaining_markets.v3_providers import NullV3Providers, V3ProviderBundle


def default_provider_bundle_from_env() -> V3ProviderBundle:
    """Create live-capable providers without requiring every credential.

    Current earnings/guidance/metadata/peer implementations remain fail-closed
    until a vendor with auditable ``available_at`` timestamps is configured.
    The existing SQLite cache is loaded separately by ``live_v3_context``.
    """
    null = NullV3Providers()
    news_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("NEWS_API_KEY")
    news = AlphaVantageNewsProvider(news_key) if news_key else null
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    return V3ProviderBundle(
        earnings=null,
        guidance=null,
        prices=null,
        metadata=null,
        peers=null,
        news=news,
        article_reasoner=NewsReasoner(use_openai=use_openai),
        event_reasoner=EventReasoner(use_openai=use_openai),
    )
