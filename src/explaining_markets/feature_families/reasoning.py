"""Numeric event-reasoning features consumed by the tabular V3 model."""
from __future__ import annotations

from explaining_markets.reasoning.schemas import EventReasoning

REASONING_FEATURE_NAMES = (
    "reasoning_earnings_quality",
    "reasoning_revenue_quality",
    "reasoning_guidance_quality",
    "reasoning_expectations_gap",
    "reasoning_priced_in_score",
    "reasoning_company_news_signal",
    "reasoning_peer_signal",
    "reasoning_sector_signal",
    "reasoning_historical_analogy_signal",
    "reasoning_contradiction_score",
    "reasoning_overall_event_signal",
    "reasoning_materiality",
    "reasoning_confidence",
    "has_reasoning",
)


def reasoning_features(reasoning: EventReasoning | None) -> dict[str, float]:
    if reasoning is None:
        return {name: 0.0 for name in REASONING_FEATURE_NAMES}
    return {
        "reasoning_earnings_quality": reasoning.earnings_quality,
        "reasoning_revenue_quality": reasoning.revenue_quality,
        "reasoning_guidance_quality": reasoning.guidance_quality,
        "reasoning_expectations_gap": reasoning.expectations_gap,
        "reasoning_priced_in_score": reasoning.priced_in_score,
        "reasoning_company_news_signal": reasoning.company_news_signal,
        "reasoning_peer_signal": reasoning.peer_signal,
        "reasoning_sector_signal": reasoning.sector_signal,
        "reasoning_historical_analogy_signal": reasoning.historical_analogy_signal,
        "reasoning_contradiction_score": reasoning.contradiction_score,
        "reasoning_overall_event_signal": reasoning.overall_event_signal,
        "reasoning_materiality": reasoning.materiality,
        "reasoning_confidence": reasoning.confidence,
        "has_reasoning": 1.0,
    }
