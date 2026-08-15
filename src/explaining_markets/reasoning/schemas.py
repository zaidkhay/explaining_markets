"""Persistable reasoning schemas. No hidden chain-of-thought is stored."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


def _bounded(name: str, value: float, lower: float, upper: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}], got {value}")
    return value


@dataclass(frozen=True)
class ReasonedNewsItem:
    headline: str
    published_at: datetime
    available_at: datetime
    source: str
    source_id: str | None
    url: str | None
    entities: tuple[str, ...]
    relation: str
    topic: str
    sentiment: float
    expected_direction: float
    materiality: float
    relevance: float
    novelty: float
    source_quality: float
    confidence: float
    concise_rationale: str

    def __post_init__(self) -> None:
        for name in ("sentiment", "expected_direction"):
            object.__setattr__(self, name, _bounded(name, getattr(self, name), -1.0, 1.0))
        for name in ("materiality", "relevance", "novelty", "source_quality", "confidence"):
            object.__setattr__(self, name, _bounded(name, getattr(self, name), 0.0, 1.0))
        if self.available_at < self.published_at:
            raise ValueError("reasoned news available_at cannot predate published_at")
        if len(self.concise_rationale) > 500:
            raise ValueError("concise_rationale must be <= 500 characters")


@dataclass(frozen=True)
class EventReasoning:
    cutoff: datetime
    earnings_quality: float
    revenue_quality: float
    guidance_quality: float
    expectations_gap: float
    priced_in_score: float
    company_news_signal: float
    peer_signal: float
    sector_signal: float
    historical_analogy_signal: float
    contradiction_score: float
    overall_event_signal: float
    materiality: float
    confidence: float
    concise_rationale: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "earnings_quality", "revenue_quality", "guidance_quality", "expectations_gap",
            "priced_in_score", "company_news_signal", "peer_signal", "sector_signal",
            "historical_analogy_signal", "overall_event_signal",
        ):
            object.__setattr__(self, name, _bounded(name, getattr(self, name), -1.0, 1.0))
        for name in ("contradiction_score", "materiality", "confidence"):
            object.__setattr__(self, name, _bounded(name, getattr(self, name), 0.0, 1.0))
        if len(self.concise_rationale) > 800:
            raise ValueError("concise_rationale must be <= 800 characters")
