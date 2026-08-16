"""Event-level structured reasoning over a curated pre-cutoff numeric/news packet."""
from __future__ import annotations

from statistics import mean

from explaining_markets.reasoning.openrouter_client import openrouter_api_key, openrouter_model, structured_json
from explaining_markets.reasoning.schemas import EventReasoning, ReasonedNewsItem

_EVENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "earnings_quality": {"type": "number", "minimum": -1, "maximum": 1},
        "revenue_quality": {"type": "number", "minimum": -1, "maximum": 1},
        "guidance_quality": {"type": "number", "minimum": -1, "maximum": 1},
        "expectations_gap": {"type": "number", "minimum": -1, "maximum": 1},
        "priced_in_score": {"type": "number", "minimum": -1, "maximum": 1},
        "company_news_signal": {"type": "number", "minimum": -1, "maximum": 1},
        "peer_signal": {"type": "number", "minimum": -1, "maximum": 1},
        "sector_signal": {"type": "number", "minimum": -1, "maximum": 1},
        "historical_analogy_signal": {"type": "number", "minimum": -1, "maximum": 1},
        "contradiction_score": {"type": "number", "minimum": 0, "maximum": 1},
        "overall_event_signal": {"type": "number", "minimum": -1, "maximum": 1},
        "materiality": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "concise_rationale": {"type": "string", "maxLength": 800},
    },
    "required": [
        "earnings_quality", "revenue_quality", "guidance_quality", "expectations_gap",
        "priced_in_score", "company_news_signal", "peer_signal", "sector_signal",
        "historical_analogy_signal", "contradiction_score", "overall_event_signal",
        "materiality", "confidence", "concise_rationale",
    ],
}


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _news_signal(items: tuple[ReasonedNewsItem, ...]) -> float:
    if not items:
        return 0.0
    weights = [max(0.05, item.materiality * item.relevance * item.confidence) for item in items]
    return _clip(sum(item.expected_direction * w for item, w in zip(items, weights, strict=True)) / sum(weights))


def _source_ids(*groups: tuple[ReasonedNewsItem, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for item in group:
            if item.source_id:
                values.append(item.source_id)
    return tuple(dict.fromkeys(values))


class EventReasoner:
    """Produces auditable features, never the final competition percentile."""

    def __init__(
        self,
        *,
        use_openrouter: bool | None = None,
        use_openai: bool | None = None,
        model: str | None = None,
    ) -> None:
        requested = use_openrouter if use_openrouter is not None else use_openai
        self.use_openrouter = bool(openrouter_api_key()) if requested is None else bool(requested)
        self.model = model or openrouter_model()

    def reason(
        self,
        *,
        values: dict[str, float],
        cutoff,
        company_news: tuple[ReasonedNewsItem, ...] = (),
        peer_news: tuple[ReasonedNewsItem, ...] = (),
        sector_news: tuple[ReasonedNewsItem, ...] = (),
    ) -> EventReasoning:
        deterministic = self._deterministic(values=values, cutoff=cutoff, company_news=company_news, peer_news=peer_news, sector_news=sector_news)
        if self.use_openrouter:
            try:
                return self._openrouter_reason(deterministic, values=values, cutoff=cutoff, company_news=company_news, peer_news=peer_news, sector_news=sector_news)
            except Exception:
                pass
        return deterministic

    def _deterministic(self, *, values, cutoff, company_news, peer_news, sector_news) -> EventReasoning:
        has_eps = bool(values.get("has_eps_surprise", 0.0))
        has_rev = bool(values.get("has_revenue_surprise", 0.0))
        has_guidance = bool(values.get("has_numeric_guidance", 0.0) or values.get("guidance_direction", 0.0))
        earnings = _clip(values.get("eps_surprise_percent", 0.0) * 8.0) if has_eps else 0.0
        revenue = _clip(values.get("revenue_surprise_percent", 0.0) * 10.0) if has_rev else 0.0
        guidance = _clip(values.get("guidance_surprise_percent", 0.0) * 8.0)
        if values.get("numeric_guidance_raised", 0.0) or values.get("guidance_raised", 0.0):
            guidance = _clip(guidance + 0.35)
        if values.get("numeric_guidance_lowered", 0.0) or values.get("guidance_lowered", 0.0):
            guidance = _clip(guidance - 0.35)
        if not has_guidance:
            guidance = 0.0

        headline = mean([x for x, present in ((earnings, has_eps), (revenue, has_rev)) if present]) if (has_eps or has_rev) else 0.0
        expectations_gap = _clip(0.45 * earnings + 0.25 * revenue + 0.30 * guidance)
        runup = values.get("return_20d", 0.0)
        relative_runup = values.get("stock_minus_sector_20d", 0.0)
        priced_in = _clip(-2.2 * runup - 1.4 * relative_runup)

        company_signal = _news_signal(company_news)
        peer_news_signal = _news_signal(peer_news)
        sector_news_signal = _news_signal(sector_news)
        peer_structured = _clip(
            2.5 * values.get("peer_abnormal_return_1d", 0.0)
            + 1.5 * values.get("recent_peer_eps_surprise_mean", 0.0)
            + 1.0 * values.get("recent_peer_revenue_surprise_mean", 0.0)
        )
        peer_signal = _clip(0.55 * peer_structured + 0.45 * peer_news_signal)
        sector_structured = _clip(3.0 * values.get("sector_return_5d", 0.0))
        sector_signal = _clip(0.35 * sector_structured + 0.65 * sector_news_signal)
        historical = _clip(values.get("similar_earnings_event_mean_reaction", 0.0) / 0.08)

        if has_guidance and (has_eps or has_rev):
            contradiction = min(1.0, abs(headline - guidance) / 1.6) if headline * guidance < 0 else min(0.35, abs(headline - guidance) / 3.0)
        else:
            contradiction = 0.0

        overall = _clip(
            0.20 * earnings + 0.13 * revenue + 0.24 * guidance + 0.10 * company_signal
            + 0.09 * peer_signal + 0.05 * sector_signal + 0.07 * historical
            + 0.07 * expectations_gap + 0.05 * priced_in
        )
        components = [abs(x) for x in (earnings, revenue, guidance, company_signal, peer_signal, sector_signal, historical)]
        materiality = max(components, default=0.0)
        available_blocks = sum((has_eps, has_rev, has_guidance, bool(company_news), bool(peer_news), bool(sector_news), bool(values.get("has_company_earnings_history", 0.0))))
        news_conf = mean([item.confidence for group in (company_news, peer_news, sector_news) for item in group]) if any((company_news, peer_news, sector_news)) else 0.0
        confidence = max(0.15, min(0.95, 0.20 + 0.08 * available_blocks + 0.20 * news_conf))

        rationale = (
            f"EPS={earnings:+.2f}, revenue={revenue:+.2f}, guidance={guidance:+.2f}; "
            f"priced-in={priced_in:+.2f}, company-news={company_signal:+.2f}, peers={peer_signal:+.2f}, "
            f"sector={sector_signal:+.2f}, contradiction={contradiction:.2f}."
        )
        return EventReasoning(
            cutoff=cutoff,
            earnings_quality=earnings,
            revenue_quality=revenue,
            guidance_quality=guidance,
            expectations_gap=expectations_gap,
            priced_in_score=priced_in,
            company_news_signal=company_signal,
            peer_signal=peer_signal,
            sector_signal=sector_signal,
            historical_analogy_signal=historical,
            contradiction_score=contradiction,
            overall_event_signal=overall,
            materiality=materiality,
            confidence=confidence,
            concise_rationale=rationale,
            source_ids=_source_ids(company_news, peer_news, sector_news),
        )

    def _openrouter_reason(self, base: EventReasoning, *, values, cutoff, company_news, peer_news, sector_news) -> EventReasoning:
        allowed_names = (
            "eps_surprise_percent", "revenue_surprise_percent", "guidance_surprise_percent",
            "guidance_above_consensus", "guidance_below_consensus", "guidance_direction",
            "return_20d", "stock_minus_sector_20d", "peer_abnormal_return_1d",
            "recent_peer_eps_surprise_mean", "recent_peer_revenue_surprise_mean",
            "sector_return_5d", "similar_earnings_event_mean_reaction",
        )
        packet = {
            "cutoff": cutoff.isoformat(),
            "numeric_context": {name: values.get(name, 0.0) for name in allowed_names},
            "company_news": [{"headline": x.headline, "topic": x.topic, "direction": x.expected_direction, "materiality": x.materiality, "rationale": x.concise_rationale} for x in company_news],
            "peer_news": [{"headline": x.headline, "topic": x.topic, "direction": x.expected_direction, "materiality": x.materiality, "rationale": x.concise_rationale} for x in peer_news],
            "sector_news": [{"headline": x.headline, "topic": x.topic, "direction": x.expected_direction, "materiality": x.materiality, "rationale": x.concise_rationale} for x in sector_news],
            "deterministic_reference": {
                "earnings_quality": base.earnings_quality,
                "revenue_quality": base.revenue_quality,
                "guidance_quality": base.guidance_quality,
                "priced_in_score": base.priced_in_score,
                "overall_event_signal": base.overall_event_signal,
            },
        }
        data = structured_json(
            schema_name="event_reasoning",
            schema=_EVENT_SCHEMA,
            model=self.model,
            system_prompt=(
                "You are a structured feature extractor for an event-study model. Use only the supplied "
                "pre-cutoff packet. Evaluate earnings, revenue, guidance, contradictions, priced-in context, "
                "company/peer/sector news, and historical analogy. Do not output a stock price or competition "
                "percentile. Do not reveal hidden chain-of-thought; provide only bounded scores and a concise rationale."
            ),
            user_payload=packet,
        )
        return EventReasoning(
            cutoff=cutoff,
            earnings_quality=float(data["earnings_quality"]),
            revenue_quality=float(data["revenue_quality"]),
            guidance_quality=float(data["guidance_quality"]),
            expectations_gap=float(data["expectations_gap"]),
            priced_in_score=float(data["priced_in_score"]),
            company_news_signal=float(data["company_news_signal"]),
            peer_signal=float(data["peer_signal"]),
            sector_signal=float(data["sector_signal"]),
            historical_analogy_signal=float(data["historical_analogy_signal"]),
            contradiction_score=float(data["contradiction_score"]),
            overall_event_signal=float(data["overall_event_signal"]),
            materiality=float(data["materiality"]),
            confidence=float(data["confidence"]),
            concise_rationale=str(data["concise_rationale"])[:800],
            source_ids=_source_ids(company_news, peer_news, sector_news),
        )
