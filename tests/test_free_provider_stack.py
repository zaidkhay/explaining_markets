from __future__ import annotations

from datetime import datetime, timezone

import httpx

from explaining_markets.historical import HistoricalEvent
from explaining_markets.historical_v3_enrichment import DiskJsonCache
from explaining_markets.historical_v3_enrichment_free import (
    _prices_available_by_cutoff,
    enrich_training_rows_free,
)
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.providers.free_historical import (
    FinnhubHistoricalClient,
    TiingoHistoricalClient,
    finnhub_earnings_record,
    finnhub_news_records,
    tiingo_price_records,
)
from explaining_markets.reasoning.openrouter_client import reset_openrouter_budget_for_tests, structured_json
from explaining_markets.v3_records import PriceRecord, V3Context


def test_free_enrichment_entrypoint_is_importable():
    assert callable(enrich_training_rows_free)


def test_tiingo_adjusted_prices_and_cache(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Authorization"] == "Token test-tiingo"
        return httpx.Response(
            200,
            json=[
                {"date": "2026-01-27T00:00:00.000Z", "close": 50.0, "adjClose": 100.0, "volume": 100, "adjVolume": 50},
                {"date": "2026-01-28T00:00:00.000Z", "close": 51.0, "adjClose": 102.0, "volume": 110, "adjVolume": 55},
            ],
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cache = DiskJsonCache(tmp_path)
    client = TiingoHistoricalClient("test-tiingo", cache=cache, max_api_calls=1, client=http, progress=lambda _: None)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    payload = client.prices_payload("AAPL", start=start, end=end)
    payload2 = client.prices_payload("AAPL", start=start, end=end)
    rows = tiingo_price_records(payload, "AAPL", retrieved_at=end)
    assert payload2 == payload
    assert calls == 1
    assert client.api_calls == 1
    assert client.cache_hits == 1
    assert rows[-1].close == 102.0
    assert rows[-1].volume == 55.0
    assert rows[-1].source == "tiingo_eod_adjusted"
    assert rows[-1].available_at > rows[-1].value_timestamp
    http.close()


def test_tiingo_bulk_cache_is_trimmed_to_each_event_cutoff():
    cutoff = datetime(2026, 1, 29, 21, 0, tzinfo=timezone.utc)
    rows = (
        PriceRecord(
            value_timestamp=datetime(2026, 1, 28, 21, 0, tzinfo=timezone.utc),
            available_at=datetime(2026, 1, 29, 0, 0, tzinfo=timezone.utc),
            retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            source="tiingo_eod_adjusted",
            ticker="AAPL",
            close=100.0,
        ),
        PriceRecord(
            value_timestamp=datetime(2026, 1, 29, 21, 0, tzinfo=timezone.utc),
            available_at=datetime(2026, 1, 30, 0, 0, tzinfo=timezone.utc),
            retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            source="tiingo_eod_adjusted",
            ticker="AAPL",
            close=101.0,
        ),
        PriceRecord(
            value_timestamp=datetime(2026, 2, 2, 21, 0, tzinfo=timezone.utc),
            available_at=datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc),
            retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            source="tiingo_eod_adjusted",
            ticker="AAPL",
            close=110.0,
        ),
    )
    eligible = _prices_available_by_cutoff(rows, cutoff)
    assert [row.close for row in eligible] == [100.0]
    context = V3Context(ticker="AAPL", cutoff=cutoff, stock_prices=eligible)
    assert audit_context(context).violations == 0


def test_finnhub_eps_matches_preceding_fiscal_period():
    event = HistoricalEvent(
        event_id="evt",
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        event_datetime="2026-01-29T21:00:00+00:00",
        quarter="2026Q1",
    )
    payload = [
        {"actual": 2.4, "estimate": 2.2, "period": "2025-12-31", "quarter": 1, "year": 2026},
        {"actual": 1.8, "estimate": 1.7, "period": "2025-09-30", "quarter": 4, "year": 2025},
    ]
    cutoff = datetime(2026, 1, 29, 21, tzinfo=timezone.utc)
    record = finnhub_earnings_record(payload, event, cutoff)
    assert record is not None
    assert record.reported_eps == 2.4
    assert record.consensus_eps == 2.2
    assert record.available_at == cutoff
    assert record.source == "finnhub_historical_eps_surprise"


def test_finnhub_eps_fails_closed_when_period_is_too_old():
    event = HistoricalEvent(
        event_id="evt",
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        event_datetime="2026-08-01T21:00:00+00:00",
        quarter="2026Q3",
    )
    payload = [{"actual": 2.4, "estimate": 2.2, "period": "2025-12-31"}]
    assert finnhub_earnings_record(payload, event, datetime(2026, 8, 1, 21, tzinfo=timezone.utc)) is None


def test_finnhub_client_and_news_normalization(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["X-Finnhub-Token"] == "test-finnhub"
        if request.url.path.endswith("/stock/earnings"):
            return httpx.Response(200, json=[{"actual": 1.2, "estimate": 1.1, "period": "2026-06-30"}])
        return httpx.Response(
            200,
            json=[{
                "category": "company",
                "datetime": 1760000000,
                "headline": "Apple raises guidance",
                "id": 123,
                "related": "AAPL,MSFT",
                "source": "Reuters",
                "summary": "Guidance increased.",
                "url": "https://example.test/a",
            }],
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cache = DiskJsonCache(tmp_path)
    client = FinnhubHistoricalClient("test-finnhub", cache=cache, max_api_calls=2, client=http, progress=lambda _: None)
    assert len(client.earnings_payload("AAPL")) == 1
    raw = client.company_news_payload(
        "AAPL",
        start=datetime(2025, 10, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    rows = finnhub_news_records(raw, "AAPL", retrieved_at=datetime.now(timezone.utc))
    assert calls == 2
    assert len(rows) == 1
    assert rows[0].headline == "Apple raises guidance"
    assert "AAPL" in rows[0].entities
    assert rows[0].source_id == "123"
    http.close()


def test_openrouter_strict_structured_output(monkeypatch):
    reset_openrouter_budget_for_tests()
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-openrouter")
    monkeypatch.setenv("OPEN_ROUTER_MODEL", "openrouter/free")
    monkeypatch.setenv("OPEN_ROUTER_MAX_CALLS", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-openrouter"
        body = __import__("json").loads(request.content)
        assert body["model"] == "openrouter/free"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["provider"]["require_parameters"] is True
        assert body["provider"]["data_collection"] == "deny"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"direction":0.8,"confidence":0.9}'}}
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    result = structured_json(
        schema_name="smoke",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "direction": {"type": "number"},
                "confidence": {"type": "number"},
            },
            "required": ["direction", "confidence"],
        },
        system_prompt="Return structured scores only.",
        user_payload={"headline": "Synthetic beat and raise"},
        client=http,
    )
    assert result == {"direction": 0.8, "confidence": 0.9}
    http.close()
