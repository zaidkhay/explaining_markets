from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

import explaining_markets.historical_v3_enrichment as enrichment
from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.historical import HistoricalEvent
from explaining_markets.historical_v3_enrichment import (
    AlphaHistoricalClient,
    DiskJsonCache,
    LocalDailyPriceStore,
    _match_earnings,
    _normalize_broad_news,
    enrich_training_rows,
)
from explaining_markets.v3_records import V3Context
from explaining_markets.v3_training import V3TrainingRow
from explaining_markets.v3_training_data import load_training_rows, write_training_rows


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


def test_alpha_timeout_is_bounded_and_consumes_attempt_budget(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow vendor", request=request)

    transport_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AlphaHistoricalClient(
        "test-key",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=2,
        timeout=0.5,
        retries=1,
        retry_backoff=0.0,
        min_request_interval=0.0,
        client=transport_client,
        progress=lambda _message: None,
    )
    with pytest.raises(RuntimeError, match="ReadTimeout"):
        client.earnings_payload("AAPL")
    assert client.api_calls == 2
    assert client.failures == 2
    transport_client.close()


def test_alpha_cache_hit_uses_zero_network_attempts(tmp_path):
    cache = DiskJsonCache(tmp_path)
    payload = {"quarterlyEarnings": [{"reportedDate": "2026-01-29"}]}
    cache.put("earnings", "AAPL", payload)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("network should not be used on cache hit")

    transport_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AlphaHistoricalClient(
        "test-key",
        cache=cache,
        max_api_calls=1,
        client=transport_client,
        progress=lambda _message: None,
    )
    assert client.earnings_payload("AAPL") == payload
    assert client.api_calls == 0
    assert client.cache_hits == 1
    transport_client.close()


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


def test_enrichment_turns_on_eps_prices_and_reasoning_without_network(tmp_path, monkeypatch):
    event = _event()
    cutoff = datetime(2026, 1, 29, 21, tzinfo=timezone.utc)
    seed_vector = build_feature_vector_v3(
        disclosure=list(event.disclosure),
        context=V3Context(ticker="AAPL", cutoff=cutoff),
    )
    seed = V3TrainingRow(
        event_id=event.event_id,
        ticker=event.ticker,
        quarter=event.quarter or "2026Q1",
        target_percentile=0.5,
        values=seed_vector.values,
        surprise_percentile=0.5,
        leakage_violations=0,
    )
    rows_path = tmp_path / "seed.jsonl.gz"
    output_path = tmp_path / "enriched.jsonl.gz"
    write_training_rows([seed], rows_path)

    cache_dir = tmp_path / "cache"
    DiskJsonCache(cache_dir).put(
        "earnings",
        "AAPL",
        {
            "quarterlyEarnings": [
                {
                    "reportedDate": "2026-01-29",
                    "reportedEPS": "2.40",
                    "estimatedEPS": "2.20",
                }
            ]
        },
    )

    price_path = tmp_path / "prices.csv"
    start = datetime(2020, 1, 1)
    lines = ["ticker,date,close,volume,source"]
    for i in range(1262):
        day = (start + timedelta(days=i)).date().isoformat()
        lines.append(f"AAPL,{day},{100.0 + i * 0.01},1000,test_adjusted")
    price_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(enrichment, "load_historical_events", lambda _source: [event])
    report = enrich_training_rows(
        rows_path=rows_path,
        historical_dir=tmp_path,
        output_path=output_path,
        alpha_api_key="test-key",
        cache_dir=cache_dir,
        max_api_calls=0,
        price_csv=price_path,
        include_historical_news=False,
        reasoning_mode="deterministic",
        progress_every=0,
    )
    enriched = load_training_rows(output_path)
    values = enriched[0].values
    assert report.alpha_api_calls == 0
    assert values["has_eps_surprise"] == 1.0
    assert values["has_5y_price_history"] == 1.0
    assert values["has_reasoning"] == 1.0
    assert values["eps_surprise_percent"] > 0.0
    assert report.family_coverage["eps"] == 1.0
    assert report.family_coverage["price_5y"] == 1.0
    assert report.family_coverage["reasoning"] == 1.0
