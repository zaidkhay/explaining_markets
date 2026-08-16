from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from explaining_markets.historical_v3_enrichment import DiskJsonCache
from explaining_markets.providers.free_historical import (
    TwelveDataHistoricalClient,
    twelve_data_price_records,
)


def test_twelve_data_adjusted_prices_and_cache(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/time_series"
        params = dict(request.url.params)
        assert params["symbol"] == "AAPL"
        assert params["interval"] == "1day"
        assert params["adjust"] == "all"
        assert params["order"] == "ASC"
        assert params["apikey"] == "test-twelve"
        return httpx.Response(
            200,
            json={
                "meta": {"symbol": "AAPL", "interval": "1day"},
                "values": [
                    {"datetime": "2026-01-27", "close": "100.0", "volume": "1000"},
                    {"datetime": "2026-01-28", "close": "102.0", "volume": "1100"},
                ],
                "status": "ok",
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cache = DiskJsonCache(tmp_path)
    client = TwelveDataHistoricalClient(
        "test-twelve",
        cache=cache,
        max_api_calls=1,
        client=http,
        progress=lambda _: None,
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    first = client.prices_payload("AAPL", start=start, end=end)
    second = client.prices_payload("AAPL", start=start, end=end)
    rows = twelve_data_price_records(first, "AAPL", retrieved_at=end)

    assert second == first
    assert calls == 1
    assert client.api_calls == 1
    assert client.cache_hits == 1
    assert len(rows) == 2
    assert rows[-1].close == 102.0
    assert rows[-1].volume == 1100.0
    assert rows[-1].source == "twelve_data_eod_adjust_all"
    assert rows[-1].available_at > rows[-1].value_timestamp
    http.close()


def test_twelve_data_symbol_error_does_not_trip_run_wide_circuit(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        symbol = request.url.params["symbol"]
        if symbol == "BAD":
            return httpx.Response(
                200,
                json={"code": 400, "message": "symbol is not available", "status": "error"},
            )
        return httpx.Response(
            200,
            json={
                "meta": {"symbol": symbol},
                "values": [{"datetime": "2026-01-27", "close": "100.0"}],
                "status": "ok",
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = TwelveDataHistoricalClient(
        "test-twelve",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=2,
        client=http,
        progress=lambda _: None,
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="symbol is not available"):
        client.prices_payload("BAD", start=start, end=end)

    assert client.unavailable_reason is None
    assert len(client.prices_payload("AAPL", start=start, end=end)) == 1
    assert calls == 2
    http.close()
