from datetime import datetime, timedelta, timezone

import pytest

from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.live_v3_context import build_live_v3_context
from explaining_markets.news_ranking import deduplicate_news, rank_news
from explaining_markets.point_in_time_audit_v3 import PointInTimeViolation, audit_context
from explaining_markets.providers.news_provider import AlphaVantageNewsProvider
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.reasoning.news_reasoner import NewsReasoner
from explaining_markets.reasoning.schemas import ReasonedNewsItem
from explaining_markets.v3_providers import V3ProviderBundle
from explaining_markets.v3_records import NewsRecord, V3Context

UTC = timezone.utc
CUTOFF = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)


def _news(headline, *, hours=-1, entities=("AAPL",), url=None, source_id=None, sentiment=0.2, material=True):
    published = CUTOFF + timedelta(hours=hours)
    return NewsRecord(
        value_timestamp=published,
        available_at=published,
        retrieved_at=CUTOFF + timedelta(minutes=5),
        source="Reuters",
        headline=headline,
        published_at=published,
        entities=entities,
        url=url,
        source_id=source_id,
        sentiment=sentiment,
        material=material,
        summary=headline,
        vendor_relevance=0.95,
    )


def _values(**overrides):
    values = {
        "has_eps_surprise": 1.0,
        "has_revenue_surprise": 1.0,
        "has_numeric_guidance": 1.0,
        "eps_surprise_percent": 0.09,
        "revenue_surprise_percent": 0.02,
        "guidance_surprise_percent": 0.05,
        "numeric_guidance_raised": 1.0,
        "numeric_guidance_lowered": 0.0,
        "guidance_raised": 0.0,
        "guidance_lowered": 0.0,
        "guidance_direction": 1.0,
        "return_20d": 0.02,
        "stock_minus_sector_20d": 0.0,
        "peer_abnormal_return_1d": 0.0,
        "recent_peer_eps_surprise_mean": 0.0,
        "recent_peer_revenue_surprise_mean": 0.0,
        "sector_return_5d": 0.0,
        "similar_earnings_event_mean_reaction": 0.02,
        "has_company_earnings_history": 1.0,
    }
    values.update(overrides)
    return values


def _reasoned(direction: float, *, relation="company", sid="n1"):
    return ReasonedNewsItem(
        headline="Material demand update",
        published_at=CUTOFF - timedelta(hours=2),
        available_at=CUTOFF - timedelta(hours=2),
        source="Reuters",
        source_id=sid,
        url="https://example.com/item",
        entities=("AAPL",),
        relation=relation,
        topic="demand",
        sentiment=direction,
        expected_direction=direction,
        materiality=0.9,
        relevance=0.95,
        novelty=0.9,
        source_quality=0.95,
        confidence=0.9,
        concise_rationale="Material pre-cutoff demand update.",
    )


def test_future_news_excluded_and_duplicates_removed():
    first = _news("Apple raises guidance", url="https://example.com/a?utm=x", source_id="1")
    duplicate = _news("Apple raises guidance", hours=-0.5, url="https://example.com/a", source_id="2")
    future = _news("Future article", hours=1, url="https://example.com/future", source_id="3")
    rows = deduplicate_news((first, duplicate, future), CUTOFF)
    assert len(rows) == 1
    assert rows[0].headline == "Apple raises guidance"


def test_irrelevant_company_article_filtered_before_reasoning():
    relevant = _news("Apple demand improves", entities=("AAPL",), source_id="a")
    irrelevant = _news("Microsoft launches product", entities=("MSFT",), source_id="b")
    ranked = rank_news((relevant, irrelevant), CUTOFF, targets={"AAPL"}, require_target=True)
    assert [item.record.source_id for item in ranked] == ["a"]


def test_structured_news_reasoner_is_bounded_without_openai():
    ranked = rank_news((_news("Apple raises outlook after strong demand", sentiment=0.7),), CUTOFF, targets={"AAPL"})
    item = NewsReasoner(use_openai=False).reason(ranked[0], relation="company")
    assert -1.0 <= item.expected_direction <= 1.0
    assert 0.0 <= item.materiality <= 1.0
    assert item.relation == "company"
    assert item.concise_rationale


def test_scenario_double_beat_raise_is_positive():
    result = EventReasoner(use_openai=False).reason(values=_values(), cutoff=CUTOFF)
    assert result.earnings_quality > 0
    assert result.revenue_quality > 0
    assert result.guidance_quality > 0
    assert result.overall_event_signal > 0


def test_scenario_beat_guidance_cut_has_high_contradiction_and_is_weaker():
    reasoner = EventReasoner(use_openai=False)
    good = reasoner.reason(values=_values(), cutoff=CUTOFF)
    cut = reasoner.reason(
        values=_values(
            guidance_surprise_percent=-0.061,
            numeric_guidance_raised=0.0,
            numeric_guidance_lowered=1.0,
            guidance_direction=-1.0,
            return_20d=0.19,
            stock_minus_sector_20d=0.11,
        ),
        cutoff=CUTOFF,
    )
    assert cut.earnings_quality > 0
    assert cut.guidance_quality < 0
    assert cut.priced_in_score < 0
    assert cut.contradiction_score > 0.5
    assert cut.overall_event_signal < good.overall_event_signal


def test_scenario_double_miss_cut_is_negative():
    result = EventReasoner(use_openai=False).reason(
        values=_values(
            eps_surprise_percent=-0.10,
            revenue_surprise_percent=-0.05,
            guidance_surprise_percent=-0.08,
            numeric_guidance_raised=0.0,
            numeric_guidance_lowered=1.0,
            guidance_direction=-1.0,
        ),
        cutoff=CUTOFF,
    )
    assert result.earnings_quality < 0
    assert result.revenue_quality < 0
    assert result.guidance_quality < 0
    assert result.overall_event_signal < 0


def test_small_miss_raise_after_selloff_less_negative_than_double_miss_cut():
    reasoner = EventReasoner(use_openai=False)
    severe = reasoner.reason(
        values=_values(eps_surprise_percent=-0.10, revenue_surprise_percent=-0.05, guidance_surprise_percent=-0.08, numeric_guidance_raised=0.0, numeric_guidance_lowered=1.0),
        cutoff=CUTOFF,
    )
    mild = reasoner.reason(
        values=_values(eps_surprise_percent=-0.02, revenue_surprise_percent=-0.01, guidance_surprise_percent=0.04, numeric_guidance_raised=1.0, numeric_guidance_lowered=0.0, return_20d=-0.20, stock_minus_sector_20d=-0.10),
        cutoff=CUTOFF,
    )
    assert mild.overall_event_signal > severe.overall_event_signal


def test_opposite_material_news_changes_reasoning_and_feature_vector():
    reasoner = EventReasoner(use_openai=False)
    positive = reasoner.reason(values=_values(), cutoff=CUTOFF, company_news=(_reasoned(0.9, sid="pos"),))
    negative = reasoner.reason(values=_values(), cutoff=CUTOFF, company_news=(_reasoned(-0.9, sid="neg"),))
    assert positive.company_news_signal > negative.company_news_signal
    assert positive.overall_event_signal > negative.overall_event_signal

    pos_vector = build_feature_vector_v3(disclosure=[], context=V3Context(ticker="AAPL", cutoff=CUTOFF, event_reasoning=positive))
    neg_vector = build_feature_vector_v3(disclosure=[], context=V3Context(ticker="AAPL", cutoff=CUTOFF, event_reasoning=negative))
    assert pos_vector.values["reasoning_company_news_signal"] != neg_vector.values["reasoning_company_news_signal"]
    assert pos_vector.vector() != neg_vector.vector()


def test_point_in_time_audit_rejects_future_news():
    context = V3Context(ticker="AAPL", cutoff=CUTOFF, company_news=(_news("Future", hours=1),))
    with pytest.raises(PointInTimeViolation):
        audit_context(context)


def test_no_current_car1_feature_leakage():
    assert not any("car1" in name.lower() for name in MODEL_FEATURE_NAMES_V3)


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "feed": [
                {
                    "title": "Apple raises outlook",
                    "time_published": "20260813T190000",
                    "source": "Reuters",
                    "url": "https://example.com/a",
                    "summary": "Demand improved and guidance was raised.",
                    "overall_sentiment_score": "0.4",
                    "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.98", "ticker_sentiment_score": "0.5"}],
                    "topics": [{"topic": "earnings", "relevance_score": "0.9"}],
                },
                {
                    "title": "Post cutoff",
                    "time_published": "20260813T210000",
                    "source": "Reuters",
                    "url": "https://example.com/future",
                    "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.9"}],
                },
            ]
        }


class _FakeClient:
    def get(self, *args, **kwargs):
        return _FakeResponse()


def test_alpha_vantage_provider_normalizes_and_filters_cutoff():
    provider = AlphaVantageNewsProvider("test-key", client=_FakeClient())
    rows = provider.company_news("AAPL", CUTOFF)
    assert len(rows) == 1
    assert rows[0].published_at == rows[0].available_at
    assert rows[0].entities == ("AAPL",)
    assert provider.receipts[-1]["status"] == "ok"


class _BrokenProvider:
    receipts = ()

    def current(self, *args, **kwargs):
        raise RuntimeError("unavailable")

    def history(self, *args, **kwargs):
        raise RuntimeError("unavailable")

    def metadata(self, *args, **kwargs):
        raise RuntimeError("unavailable")

    def peers(self, *args, **kwargs):
        raise RuntimeError("unavailable")

    def company_news(self, *args, **kwargs):
        raise RuntimeError("unavailable")

    def peer_news(self, *args, **kwargs):
        raise RuntimeError("unavailable")

    def sector_news(self, *args, **kwargs):
        raise RuntimeError("unavailable")


def test_provider_failure_does_not_break_live_context(monkeypatch, tmp_path):
    monkeypatch.setenv("V3_HISTORY_CACHE_PATH", str(tmp_path / "missing.sqlite"))
    broken = _BrokenProvider()
    bundle = V3ProviderBundle(
        earnings=broken,
        guidance=broken,
        prices=broken,
        metadata=broken,
        peers=broken,
        news=broken,
        article_reasoner=NewsReasoner(use_openai=False),
        event_reasoner=EventReasoner(use_openai=False),
    )
    context = build_live_v3_context(ticker="AAPL", event={"event_id": "x", "disclosure": []}, cutoff=CUTOFF, providers=bundle)
    assert context.event_reasoning is not None
    assert context.company_news == ()
    assert audit_context(context).violations == 0
