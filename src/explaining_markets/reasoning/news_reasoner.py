"""Structured article reasoning with a deterministic fallback and optional OpenAI JSON-schema mode."""
from __future__ import annotations

import json
import os
import re
from typing import Iterable

from explaining_markets.news_ranking import RankedNewsRecord
from explaining_markets.reasoning.schemas import ReasonedNewsItem

_TOPICS = {
    "guidance": ("guidance", "outlook", "forecast"),
    "demand": ("demand", "orders", "bookings"),
    "pricing": ("pricing", "price increase", "price cut"),
    "supply_constraint": ("supply", "shortage", "constraint"),
    "customer_win": ("customer win", "contract win", "selected by", "awarded"),
    "customer_loss": ("customer loss", "lost customer", "termination"),
    "product_launch": ("launch", "new product", "release"),
    "regulatory": ("regulator", "regulatory", "ftc", "doj", "sec "),
    "FDA": ("fda", "clinical", "drug approval"),
    "litigation": ("lawsuit", "litigation", "court", "ruling"),
    "M&A": ("acquisition", "acquire", "merger", "takeover"),
    "capital_raise": ("offering", "capital raise", "secondary", "convertible"),
    "layoffs": ("layoff", "workforce reduction", "job cuts"),
    "management_change": ("ceo", "cfo", "chief executive", "management change"),
    "margin_pressure": ("margin pressure", "compression", "cost pressure"),
    "credit_quality": ("credit quality", "delinquency", "default", "charge-off"),
    "commodity_exposure": ("commodity", "oil price", "gas price", "copper", "gold"),
    "industry_demand": ("industry demand", "sector demand", "market demand"),
    "analyst_action": ("upgrade", "downgrade", "price target", "analyst"),
}
_POSITIVE = ("beat", "raise", "raised", "approval", "win", "growth", "strong", "record", "upgrade", "expansion")
_NEGATIVE = ("miss", "cut", "lowered", "weak", "decline", "lawsuit", "layoff", "downgrade", "shortage", "recall", "investigation")

_ARTICLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "topic": {"type": "string"},
        "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
        "expected_direction": {"type": "number", "minimum": -1, "maximum": 1},
        "materiality": {"type": "number", "minimum": 0, "maximum": 1},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "concise_rationale": {"type": "string", "maxLength": 500},
    },
    "required": ["topic", "sentiment", "expected_direction", "materiality", "relevance", "confidence", "concise_rationale"],
}


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _topic(text: str) -> str:
    lowered = text.lower()
    for topic, terms in _TOPICS.items():
        if any(term in lowered for term in terms):
            return topic
    return "other"


def _lexical_direction(text: str) -> float:
    lowered = text.lower()
    positive = sum(bool(re.search(rf"\b{re.escape(word)}\b", lowered)) for word in _POSITIVE)
    negative = sum(bool(re.search(rf"\b{re.escape(word)}\b", lowered)) for word in _NEGATIVE)
    if positive == negative == 0:
        return 0.0
    return _clip((positive - negative) / max(positive + negative, 1))


class NewsReasoner:
    """Reason about already-ranked articles; never fetches additional information."""

    def __init__(self, *, use_openai: bool | None = None, model: str | None = None) -> None:
        self.use_openai = bool(os.getenv("OPENAI_API_KEY")) if use_openai is None else bool(use_openai)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def reason(self, ranked: RankedNewsRecord, *, relation: str) -> ReasonedNewsItem:
        if self.use_openai:
            try:
                return self._openai_reason(ranked, relation=relation)
            except Exception:
                pass
        return self._deterministic_reason(ranked, relation=relation)

    def reason_many(self, ranked: Iterable[RankedNewsRecord], *, relation: str) -> tuple[ReasonedNewsItem, ...]:
        return tuple(self.reason(item, relation=relation) for item in ranked)

    def _deterministic_reason(self, ranked: RankedNewsRecord, *, relation: str) -> ReasonedNewsItem:
        record = ranked.record
        text = f"{record.headline}. {record.summary or ''}"
        lexical = _lexical_direction(text)
        vendor = record.sentiment if record.sentiment is not None else lexical
        sentiment = _clip(0.65 * float(vendor) + 0.35 * lexical)
        expected = _clip(0.75 * sentiment + 0.25 * lexical)
        topic = _topic(text)
        confidence = max(0.20, min(0.95, 0.35 + 0.30 * ranked.source_quality + 0.20 * ranked.relevance + 0.15 * ranked.materiality))
        rationale = f"{topic}: {record.headline[:320]}"
        return ReasonedNewsItem(
            headline=record.headline,
            published_at=record.published_at,
            available_at=record.available_at,
            source=record.source,
            source_id=record.source_id,
            url=record.url,
            entities=record.entities,
            relation=relation,
            topic=topic,
            sentiment=sentiment,
            expected_direction=expected,
            materiality=ranked.materiality,
            relevance=ranked.relevance,
            novelty=ranked.novelty,
            source_quality=ranked.source_quality,
            confidence=confidence,
            concise_rationale=rationale,
        )

    def _openai_reason(self, ranked: RankedNewsRecord, *, relation: str) -> ReasonedNewsItem:
        from openai import OpenAI

        record = ranked.record
        packet = {
            "relation": relation,
            "headline": record.headline,
            "summary": record.summary,
            "source": record.source,
            "entities": list(record.entities),
            "published_at": record.published_at.isoformat(),
            "ranking": {
                "relevance": ranked.relevance,
                "materiality": ranked.materiality,
                "novelty": ranked.novelty,
                "source_quality": ranked.source_quality,
            },
        }
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You extract auditable market-event features from ONE pre-cutoff news item. Do not predict a competition percentile. Return only the requested structured scores and a short evidence-based rationale; do not provide hidden chain-of-thought."},
                {"role": "user", "content": json.dumps(packet, sort_keys=True)},
            ],
            text={"format": {"type": "json_schema", "name": "reasoned_news_item", "schema": _ARTICLE_SCHEMA, "strict": True}},
        )
        data = json.loads(response.output_text)
        return ReasonedNewsItem(
            headline=record.headline,
            published_at=record.published_at,
            available_at=record.available_at,
            source=record.source,
            source_id=record.source_id,
            url=record.url,
            entities=record.entities,
            relation=relation,
            topic=str(data["topic"]),
            sentiment=float(data["sentiment"]),
            expected_direction=float(data["expected_direction"]),
            materiality=float(data["materiality"]),
            relevance=float(data["relevance"]),
            novelty=ranked.novelty,
            source_quality=ranked.source_quality,
            confidence=float(data["confidence"]),
            concise_rationale=str(data["concise_rationale"])[:500],
        )
