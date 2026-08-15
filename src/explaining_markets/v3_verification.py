"""End-to-end V3 verification helpers.

This module deliberately separates three questions:

1. Does the V3 feature/reasoning/inference machinery produce materially
   different outputs across economically different scenarios?
2. Do live news/OpenAI inputs populate point-in-time-safe V3 contexts?
3. Is a *promoted production V3 artifact* actually selected?

Synthetic diagnostic scores are never submitted to the competition and are
never presented as trained model performance. They only prove that the V3
pipeline and pure-Python inference path are capable of producing non-collapsed
outputs when informative inputs differ.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev

from dotenv import load_dotenv

from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3, build_feature_vector_v3, family_availability
from explaining_markets.live_v3_context import build_live_v3_context, feed_diagnostics
from explaining_markets.model_v3 import MultiSignalV3Model
from explaining_markets.news_ranking import rank_news
from explaining_markets.point_in_time_audit_v3 import PointInTimeViolation, audit_context
from explaining_markets.providers.live_context import default_provider_bundle_from_env
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.reasoning.news_reasoner import NewsReasoner
from explaining_markets.v3_records import EarningsRecord, GuidanceRecord, NewsRecord, PriceRecord, V3Context


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    eps_actual: float | None
    eps_consensus: float | None
    revenue_actual: float | None
    revenue_consensus: float | None
    guidance_eps_mid: float | None
    guidance_eps_consensus: float | None
    guidance_direction: str | None
    company_headline: str | None
    company_sentiment: float | None
    peer_headline: str | None = None
    peer_sentiment: float | None = None
    sector_headline: str | None = None
    sector_sentiment: float | None = None
    return_20d_target: float = 0.0


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    score: float
    overall_event_signal: float
    eps_surprise_percent: float
    revenue_surprise_percent: float
    guidance_surprise_percent: float
    priced_in_score: float
    company_news_signal: float
    peer_signal: float
    sector_signal: float
    contradiction_score: float
    records_checked: int


SYNTHETIC_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "strong_double_beat_raise", 1.30, 1.00, 13.0, 10.0, 1.45, 1.00, "raised",
        "Company beats earnings, raises outlook and reports record demand", 0.85,
        "Peers see strong demand and raise forecasts", 0.70,
        "Technology demand remains strong across the sector", 0.50,
        -0.05,
    ),
    ScenarioSpec(
        "strong_double_miss_cut", 0.70, 1.00, 8.0, 10.0, 0.65, 1.00, "lowered",
        "Company misses earnings, cuts outlook and warns of weak demand", -0.90,
        "Peers warn of slowing demand and margin pressure", -0.65,
        "Sector outlook weakens as demand contracts", -0.55,
        0.18,
    ),
    ScenarioSpec(
        "beat_but_guidance_cut", 1.20, 1.00, 10.8, 10.0, 0.82, 1.00, "cut",
        "Company beats the quarter but lowers full-year guidance", -0.25,
        return_20d_target=0.12,
    ),
    ScenarioSpec(
        "miss_but_raise_after_selloff", 0.90, 1.00, 9.7, 10.0, 1.25, 1.00, "raised",
        "Company misses slightly but raises outlook on improving demand", 0.45,
        return_20d_target=-0.22,
    ),
    ScenarioSpec(
        "neutral_inline", 1.00, 1.00, 10.0, 10.0, 1.00, 1.00, "reaffirmed",
        "Company reports results broadly in line with expectations", 0.0,
        return_20d_target=0.0,
    ),
    ScenarioSpec(
        "positive_news_only", None, None, None, None, None, None, None,
        "Company wins a major customer contract and expands production", 0.90,
        "Peers report healthy industry demand", 0.45,
        "Sector demand trends improve", 0.35,
    ),
    ScenarioSpec(
        "negative_news_only", None, None, None, None, None, None, None,
        "Company faces recall, investigation and large customer loss", -0.95,
        "Peers report weaker orders", -0.45,
        "Sector demand deteriorates", -0.35,
    ),
    ScenarioSpec(
        "good_fundamentals_priced_in", 1.22, 1.00, 11.2, 10.0, 1.18, 1.00, "raised",
        "Company beats and raises after a large pre-earnings rally", 0.45,
        return_20d_target=0.35,
    ),
    ScenarioSpec(
        "bad_fundamentals_after_selloff", 0.82, 1.00, 9.0, 10.0, 0.80, 1.00, "lowered",
        "Company misses and cuts after a sharp pre-earnings selloff", -0.55,
        return_20d_target=-0.35,
    ),
    ScenarioSpec(
        "contradictory_signals", 1.18, 1.00, 10.7, 10.0, 0.78, 1.00, "lowered",
        "Company posts an earnings beat while management cuts guidance", -0.10,
        "Peers report strong earnings and demand", 0.65,
        "Sector backdrop remains supportive", 0.45,
        0.08,
    ),
)


def _ts(cutoff: datetime, days: int = 1) -> datetime:
    return cutoff - timedelta(days=days)


def _price_series(ticker: str, cutoff: datetime, target_20d: float, *, sessions: int = 1261) -> tuple[PriceRecord, ...]:
    if target_20d <= -0.95:
        raise ValueError("target return is too negative")
    daily = (1.0 + target_20d) ** (1.0 / 20.0) - 1.0
    start = cutoff - timedelta(days=sessions + 5)
    price = 100.0
    rows: list[PriceRecord] = []
    for i in range(sessions):
        price *= 1.0 + daily
        stamp = start + timedelta(days=i)
        rows.append(PriceRecord(
            value_timestamp=stamp,
            available_at=stamp,
            retrieved_at=cutoff,
            source="synthetic_verification",
            ticker=ticker,
            close=price,
            volume=1_000_000.0 + (i % 17) * 10_000.0,
        ))
    return tuple(rows)


def _history(ticker: str, cutoff: datetime) -> tuple[EarningsRecord, ...]:
    rows: list[EarningsRecord] = []
    patterns = ((1.08, 1.00, 10.4, 10.0, 0.035), (0.94, 1.00, 9.7, 10.0, -0.025), (1.03, 1.00, 10.1, 10.0, 0.015), (0.98, 1.00, 9.9, 10.0, -0.008))
    for i in range(12):
        eps, eps_c, rev, rev_c, reaction = patterns[i % len(patterns)]
        stamp = cutoff - timedelta(days=90 * (i + 1))
        rows.append(EarningsRecord(
            value_timestamp=stamp,
            available_at=stamp,
            retrieved_at=cutoff,
            source="synthetic_verification",
            ticker=ticker,
            reported_eps=eps,
            consensus_eps=eps_c,
            reported_revenue=rev,
            consensus_revenue=rev_c,
            abnormal_return=reaction,
            event_id=f"hist-{i}",
        ))
    return tuple(rows)


def _news(ticker: str, cutoff: datetime, headline: str | None, sentiment: float | None, *, source_id: str) -> tuple[NewsRecord, ...]:
    if not headline:
        return ()
    published = cutoff - timedelta(hours=6)
    return (NewsRecord(
        value_timestamp=published,
        available_at=published,
        retrieved_at=cutoff,
        source="Reuters",
        headline=headline,
        published_at=published,
        entities=(ticker,),
        url=f"https://example.test/{source_id}",
        source_id=source_id,
        sentiment=sentiment,
        material=True,
        topic="earnings",
        summary=headline,
        vendor_relevance=0.95,
    ),)


def _base_context(spec: ScenarioSpec, *, cutoff: datetime) -> V3Context:
    ticker = "TEST"
    earnings = None
    if spec.eps_actual is not None or spec.revenue_actual is not None:
        earnings = EarningsRecord(
            value_timestamp=_ts(cutoff),
            available_at=_ts(cutoff),
            retrieved_at=cutoff,
            source="synthetic_verification",
            ticker=ticker,
            reported_eps=spec.eps_actual,
            consensus_eps=spec.eps_consensus,
            reported_revenue=spec.revenue_actual,
            consensus_revenue=spec.revenue_consensus,
            event_id=spec.name,
        )
    guidance = None
    if spec.guidance_eps_mid is not None:
        guidance = GuidanceRecord(
            value_timestamp=_ts(cutoff),
            available_at=_ts(cutoff),
            retrieved_at=cutoff,
            source="synthetic_verification",
            ticker=ticker,
            eps_low=spec.guidance_eps_mid,
            eps_high=spec.guidance_eps_mid,
            eps_consensus=spec.guidance_eps_consensus,
            direction=spec.guidance_direction,
        )
    company_news = _news(ticker, cutoff, spec.company_headline, spec.company_sentiment, source_id=f"company-{spec.name}")
    peer_news = _news("PEER", cutoff, spec.peer_headline, spec.peer_sentiment, source_id=f"peer-{spec.name}")
    sector_news = _news("SECTOR", cutoff, spec.sector_headline, spec.sector_sentiment, source_id=f"sector-{spec.name}")
    stock = _price_series(ticker, cutoff, spec.return_20d_target)
    market = _price_series("SPY", cutoff, 0.01)
    sector = _price_series("XLK", cutoff, 0.015)
    return V3Context(
        ticker=ticker,
        cutoff=cutoff,
        earnings=earnings,
        guidance=guidance,
        company_history=_history(ticker, cutoff),
        stock_prices=stock,
        market_prices=market,
        sector_prices=sector,
        company_news=company_news,
        peer_news=peer_news,
        sector_news=sector_news,
    )


def _reason_context(context: V3Context) -> V3Context:
    preliminary = build_feature_vector_v3(disclosure=[], context=context)
    article_reasoner = NewsReasoner(use_openai=False)
    company_ranked = rank_news(context.company_news, context.cutoff, targets={context.ticker}, top_n=10)
    peer_targets = {entity for row in context.peer_news for entity in row.entities} or {"PEER"}
    sector_targets = {context.ticker, *peer_targets}
    peer_ranked = rank_news(context.peer_news, context.cutoff, targets=peer_targets, top_n=10)
    sector_ranked = rank_news(context.sector_news, context.cutoff, targets=sector_targets, top_n=10)
    company_reasoned = article_reasoner.reason_many(company_ranked, relation="company")
    peer_reasoned = article_reasoner.reason_many(peer_ranked, relation="peer")
    sector_reasoned = article_reasoner.reason_many(sector_ranked, relation="sector")
    event_reasoning = EventReasoner(use_openai=False).reason(
        values=preliminary.values,
        cutoff=context.cutoff,
        company_news=company_reasoned,
        peer_news=peer_reasoned,
        sector_news=sector_reasoned,
    )
    return replace(
        context,
        reasoned_company_news=company_reasoned,
        reasoned_peer_news=peer_reasoned,
        reasoned_sector_news=sector_reasoned,
        event_reasoning=event_reasoning,
    )


def _diagnostic_artifact(path: Path) -> None:
    coefficients = {name: 0.0 for name in MODEL_FEATURE_NAMES_V3}
    coefficients.update({
        "eps_surprise_percent": 0.16,
        "revenue_surprise_percent": 0.10,
        "guidance_surprise_percent": 0.14,
        "reasoning_overall_event_signal": 0.32,
        "reasoning_company_news_signal": 0.07,
        "reasoning_peer_signal": 0.04,
        "reasoning_sector_signal": 0.03,
        "reasoning_priced_in_score": 0.04,
        "reasoning_contradiction_score": -0.04,
    })
    artifact = {
        "model_version": "v3_verification_diagnostic",
        "feature_names": list(MODEL_FEATURE_NAMES_V3),
        "means": [0.0] * len(MODEL_FEATURE_NAMES_V3),
        "standard_deviations": [1.0] * len(MODEL_FEATURE_NAMES_V3),
        "coefficients": [coefficients[name] for name in MODEL_FEATURE_NAMES_V3],
        "intercept": 0.50,
        "clip_bounds": [0.05, 0.95],
        "promoted": False,
        "structured_only": False,
        "training_metadata": {"diagnostic_only": True},
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")


def run_synthetic_suite() -> list[ScenarioResult]:
    cutoff = datetime(2026, 8, 1, 20, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as temp:
        artifact_path = Path(temp) / "diagnostic_v3.json"
        _diagnostic_artifact(artifact_path)
        model = MultiSignalV3Model(artifact_path)
        results: list[ScenarioResult] = []
        for spec in SYNTHETIC_SCENARIOS:
            context = _reason_context(_base_context(spec, cutoff=cutoff))
            audit = audit_context(context)
            vector = build_feature_vector_v3(disclosure=[], context=context)
            score = model.predict_vector(vector)
            r = context.event_reasoning
            results.append(ScenarioResult(
                name=spec.name,
                score=score,
                overall_event_signal=float(r.overall_event_signal),
                eps_surprise_percent=vector.values["eps_surprise_percent"],
                revenue_surprise_percent=vector.values["revenue_surprise_percent"],
                guidance_surprise_percent=vector.values["guidance_surprise_percent"],
                priced_in_score=float(r.priced_in_score),
                company_news_signal=float(r.company_news_signal),
                peer_signal=float(r.peer_signal),
                sector_signal=float(r.sector_signal),
                contradiction_score=float(r.contradiction_score),
                records_checked=audit.records_checked,
            ))
    _assert_synthetic_suite(results)
    _assert_cutoff_rejection(cutoff)
    return results


def _assert_synthetic_suite(results: list[ScenarioResult]) -> None:
    by_name = {row.name: row for row in results}
    scores = [row.score for row in results]
    if max(scores) - min(scores) < 0.25:
        raise AssertionError(f"V3 diagnostic score spread collapsed: {min(scores):.4f}..{max(scores):.4f}")
    if pstdev(scores) < 0.07:
        raise AssertionError(f"V3 diagnostic score std too small: {pstdev(scores):.4f}")
    if sum(0.48 <= score <= 0.50 for score in scores) >= len(scores) // 2:
        raise AssertionError("too many V3 diagnostic scores collapsed around 0.49")
    if not by_name["strong_double_beat_raise"].score > by_name["neutral_inline"].score > by_name["strong_double_miss_cut"].score:
        raise AssertionError("fundamental scenario ordering failed")
    if not by_name["positive_news_only"].score > by_name["negative_news_only"].score:
        raise AssertionError("news polarity did not affect V3 diagnostic score")
    if not by_name["miss_but_raise_after_selloff"].score > by_name["beat_but_guidance_cut"].score:
        raise AssertionError("guidance/priced-in interaction did not affect V3 diagnostic score as expected")
    if by_name["beat_but_guidance_cut"].contradiction_score <= 0:
        raise AssertionError("beat + guidance cut should produce contradiction")


def _assert_cutoff_rejection(cutoff: datetime) -> None:
    future = cutoff + timedelta(minutes=1)
    bad = V3Context(
        ticker="TEST",
        cutoff=cutoff,
        company_news=(NewsRecord(
            value_timestamp=future,
            available_at=future,
            retrieved_at=future,
            source="synthetic_verification",
            headline="Future headline",
            published_at=future,
            entities=("TEST",),
            source_id="future",
        ),),
    )
    try:
        audit_context(bad)
    except PointInTimeViolation:
        return
    raise AssertionError("point-in-time audit failed to reject post-cutoff news")


def verify_openai_structured_output() -> dict[str, object]:
    """Make one real structured-output reasoning call; never fetch external facts."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"configured": False, "ok": False, "detail": "OPENAI_API_KEY missing"}
    cutoff = datetime.now(timezone.utc)
    row = _news("AAPL", cutoff, "Apple raises outlook on strong demand", 0.8, source_id="openai-smoke")[0]
    ranked = rank_news((row,), cutoff, targets={"AAPL"}, top_n=1)[0]
    reasoner = NewsReasoner(use_openai=True)
    try:
        item = reasoner._openai_reason(ranked, relation="company")
    except Exception as exc:
        return {"configured": True, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {
        "configured": True,
        "ok": True,
        "topic": item.topic,
        "expected_direction": item.expected_direction,
        "materiality": item.materiality,
        "confidence": item.confidence,
    }


def verify_live_ticker(ticker: str, *, sector: str | None = None, sector_ticker: str | None = None, peers: tuple[str, ...] = ()) -> dict[str, object]:
    load_dotenv()
    ticker = ticker.upper()
    cutoff = datetime.now(timezone.utc)
    providers = default_provider_bundle_from_env()
    event = {
        "event_id": f"v3-verify-{ticker}-{cutoff:%Y%m%dT%H%M%S}",
        "sector": sector,
        "sector_ticker": sector_ticker,
        "peer_tickers": list(peers),
        "disclosure": [],
    }
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit = audit_context(context)
    vector = build_feature_vector_v3(disclosure=[], context=context)
    diag = feed_diagnostics(context)
    reasoning = context.event_reasoning

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "diagnostic_v3.json"
        _diagnostic_artifact(path)
        diagnostic_score = MultiSignalV3Model(path).predict_vector(vector)

    try:
        from explaining_markets.model import get_default_model
        production_model = get_default_model()
        production_model_name = getattr(production_model, "model_version", production_model.__class__.__name__)
        production_v3_selected = isinstance(production_model, MultiSignalV3Model) and bool(production_model.promoted)
    except Exception as exc:
        production_model_name = f"ERROR:{type(exc).__name__}"
        production_v3_selected = False

    return {
        "ticker": ticker,
        "cutoff": cutoff.isoformat(),
        "records_checked": audit.records_checked,
        "company_news_count": diag["company_news_count"],
        "peer_news_count": diag["peer_news_count"],
        "sector_news_count": diag["sector_news_count"],
        "reasoned_news_count": diag["reasoned_news_count"],
        "price_rows": len(context.stock_prices),
        "history_rows": len(context.company_history),
        "earnings_available": context.earnings is not None,
        "guidance_available": context.guidance is not None,
        "family_availability": family_availability(vector),
        "reasoning_available": reasoning is not None,
        "overall_event_signal": None if reasoning is None else reasoning.overall_event_signal,
        "reasoning_confidence": None if reasoning is None else reasoning.confidence,
        "diagnostic_v3_score_not_submitted": diagnostic_score,
        "production_model": production_model_name,
        "production_v3_selected": production_v3_selected,
        "provider_receipts": diag["provider_receipts"],
    }


def summarize_scores(results: list[ScenarioResult]) -> dict[str, float]:
    scores = [row.score for row in results]
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": mean(scores),
        "std": pstdev(scores),
        "spread": max(scores) - min(scores),
        "fraction_048_052": sum(0.48 <= x <= 0.52 for x in scores) / len(scores),
    }
