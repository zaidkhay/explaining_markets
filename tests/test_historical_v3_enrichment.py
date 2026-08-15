from __future__ import annotations

import json
from datetime import datetime, timezone

from explaining_markets.historical import HistoricalEvent
from explaining_markets.historical_v3_enrichment import (
    DiskJsonCache,
    LocalDailyPriceStore,
    _match_earnings,
    _normalize_broad_news,
)


def _event() -> HistoricalEvent:
    return HistoricalEvent(
        event_id="evt-1",
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        event_datetime="2026-01-29T21:00:00+00:00",
        disclosure=["Revenue increased and management raised guidance."],
        car1=0.03,
        earnings_surprise=0.10,
        quarter="2026Q1",
    )


def test_match_earnings_uses_nearby_reported_date_and_event_cutoff():
    cutoff = datetime(2026, 1, 29, 21, tzinfo=timezone.utc)
    payload = {
        "quarterlyEarnings": [
            {
                "reportedDate": "2026-01-29",
                "reportedEPS": "2.40",
                "estimatedEPS": "2.20",
            },
            {
                "reportedDate": "2025-10-30",
                "reportedEPS": "1.85",
                "estimatedEPS": "1.80",
            },
        ]
    }
    record = _match_earnings(payload, _event(), cutoff)
    assert record is not None
    assert record.reported_eps == 2.40
    assert record.consensus_eps == 2.20
    assert record.available_at == cutoff
    assert record.event_id == "evt-1"


def test_match_earnings_fails_closed_when_no_nearby_report():
    cutoff = datetime(2026, 1, 29, 21, tzinfo=timezone.utc)
    payload = {
        "quarterlyEarnings": [
            {"reportedDate": "2025-10-30", "reportedEPS": "1.85", "estimatedEPS": "1.80"}
        ]
    }
    assert _match_earnings(payload, _event(), cutoff) is None


def test_disk_json_cache_round_trip(tmp_path):
    cache = DiskJsonCache(tmp_path)
    payload = {"quarterlyEarnings": [{"reportedDate": "2026-01-29"}]}
    assert cache.get("earnings", "AAPL") is None
    cache.put("earnings", "AAPL", payload)
    assert cache.get("earnings", "AAPL") == payload


def test_local_daily_price_store_loads_only_explicit_adjusted_input(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "ticker,date,close,volume,source\n"
        "AAPL,2026-01-27,100,1000,test_adjusted\n"
        "AAPL,2026-01-28,102,1100,test_adjusted\n"
        "MSFT,2026-01-28,200,900,test_adjusted\n",
        encoding="utf-8",
    )
    store = LocalDailyPriceStore(path)
    aapl = store.prices("AAPL")
    assert len(aapl) == 2
    assert aapl[-1].close == 102.0
    assert aapl[-1].source == "test_adjusted"
    assert len(store.prices("MSFT")) == 1


def test_historical_news_normalization_rejects_future_articles():
    cutoff = datetime(2026, 1, 29, 21, tzinfo=timezone.utc)
    payload = {
        "feed": [
            {
                "title": "Apple raises outlook after earnings beat",
                "time_published": "20260129T200000",
                "source": "Reuters",
                "url": "https://example.com/a",
                "summary": "Apple raised its outlook.",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "ticker_sentiment_score": "0.8", "relevance_score": "0.9"}
                ],
                "topics": [{"topic": "Earnings", "relevance_score": "0.95"}],
            },
            {
                "title": "Future Apple headline",
                "time_published": "20260129T220000",
                "source": "Reuters",
                "url": "https://example.com/b",
                "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "1.0"}],
            },
        ]
    }
    rows = _normalize_broad_news(payload, cutoff=cutoff)
    assert len(rows) == 1
    assert rows[0].headline.startswith("Apple raises")
    assert rows[0].available_at <= cutoff
