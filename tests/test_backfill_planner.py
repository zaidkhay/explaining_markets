"""Phase A backfill engine: prioritization, coverage skip, retries, accounting."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from explaining_markets.backfill_planner import (
    MIN_USABLE_SESSIONS,
    PriceCacheIndex,
    PriceFetcher,
    TickerDemand,
    build_ticker_demand,
    count_rows_by_ticker,
    plan_price_backfill,
    prioritize,
    run_price_backfill,
)
from explaining_markets.historical import HistoricalEvent
from explaining_markets.historical_v3_enrichment import DiskJsonCache
from explaining_markets.providers.free_historical import (
    ProviderUnavailable,
    TwelveDataHistoricalClient,
)
from explaining_markets.providers.retry_policy import (
    RetryPolicy,
    TransientProviderError,
    UnsupportedSymbolError,
    classify_provider_message,
    classify_status,
    is_transient_exception,
    looks_unsupported_symbol,
    parse_retry_after,
)
from explaining_markets.providers.unsupported_cache import (
    UnsupportedSymbolCache,
    default_unsupported_path,
    sanitize_provider_message,
)
from explaining_markets.v3_training import V3TrainingRow

UTC = timezone.utc


def _row(event_id: str, ticker: str, quarter: str = "2026Q1") -> V3TrainingRow:
    return V3TrainingRow(
        event_id=event_id, ticker=ticker, quarter=quarter, target_percentile=0.5, values={}
    )


def _event(event_id: str, ticker: str, dt: str) -> HistoricalEvent:
    return HistoricalEvent(
        event_id=event_id,
        ticker=ticker,
        event_type="EARNINGS_RELEASE",
        event_datetime=dt,
        disclosure=[],
        car1=0.01,
        quarter="2026Q1",
    )


def _no_sleep_policy(max_attempts: int = 3) -> RetryPolicy:
    """Deterministic, instant retry policy for tests."""
    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay=0.01,
        max_delay=0.02,
        jitter_ratio=0.0,
        sleep=lambda _seconds: None,
        jitter=lambda: 0.0,
    )


def _twelve(*args, **kwargs) -> TwelveDataHistoricalClient:
    """Twelve Data client with production pacing disabled for fast tests."""
    kwargs.setdefault("min_request_interval", 0.0)
    return TwelveDataHistoricalClient(*args, **kwargs)


# ===================== 1. FREQUENCY PRIORITIZATION =========================


def test_count_rows_by_ticker_upper_cases() -> None:
    counts = count_rows_by_ticker(["aapl", "AAPL", "msft"])
    assert counts["AAPL"] == 2
    assert counts["MSFT"] == 1


def test_priority_orders_by_row_frequency_descending() -> None:
    rows = [_row("e1", "LOW")] + [_row(f"h{i}", "HIGH") for i in range(5)] + [_row(f"m{i}", "MID") for i in range(3)]
    events = [_event("e1", "LOW", "2026-01-05T12:00:00Z")]
    events += [_event(f"h{i}", "HIGH", "2026-01-05T12:00:00Z") for i in range(5)]
    events += [_event(f"m{i}", "MID", "2026-01-05T12:00:00Z") for i in range(3)]
    demand = build_ticker_demand(rows, events)
    assert [d.ticker for d in demand] == ["HIGH", "MID", "LOW"]
    assert [d.row_count for d in demand] == [5, 3, 1]


def test_priority_tie_breaks_on_ticker_ascending_deterministically() -> None:
    demands = [
        TickerDemand("ZZZ", 3, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        TickerDemand("AAA", 3, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        TickerDemand("MMM", 3, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    assert [d.ticker for d in prioritize(demands)] == ["AAA", "MMM", "ZZZ"]
    # Deterministic across repeated calls / input orders.
    assert prioritize(demands) == prioritize(list(reversed(demands)))


def test_demand_is_not_file_order() -> None:
    """A late-appearing high-frequency ticker must outrank an early single row."""
    rows = [_row("first", "ONCE")] + [_row(f"late{i}", "MANY") for i in range(4)]
    events = [_event("first", "ONCE", "2026-01-02T12:00:00Z")]
    events += [_event(f"late{i}", "MANY", "2026-01-03T12:00:00Z") for i in range(4)]
    assert build_ticker_demand(rows, events)[0].ticker == "MANY"


def test_demand_computes_fetch_span_from_cutoff_bounds() -> None:
    rows = [_row("a", "AAPL"), _row("b", "AAPL")]
    events = [
        _event("a", "AAPL", "2025-10-01T12:00:00Z"),
        _event("b", "AAPL", "2026-05-01T12:00:00Z"),
    ]
    demand = build_ticker_demand(rows, events)[0]
    assert demand.first_cutoff == datetime(2025, 10, 1, 12, tzinfo=UTC)
    assert demand.last_cutoff == datetime(2026, 5, 1, 12, tzinfo=UTC)
    assert demand.fetch_start < demand.first_cutoff - timedelta(days=5 * 365)
    assert demand.fetch_end == demand.last_cutoff


def test_demand_skips_tickers_without_resolvable_cutoff() -> None:
    rows = [_row("ghost", "GHOST")]
    assert build_ticker_demand(rows, []) == ()


def test_projected_rows_covered_by_next_n_symbols() -> None:
    rows = [_row(f"h{i}", "HIGH") for i in range(5)] + [_row("m", "MID")]
    events = [_event(f"h{i}", "HIGH", "2026-01-05T12:00:00Z") for i in range(5)]
    events += [_event("m", "MID", "2026-01-05T12:00:00Z")]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        plan = plan_price_backfill(rows, events, cache=DiskJsonCache(tmp))
    assert plan.projected_rows_covered_by_next(1) == 5
    assert plan.projected_rows_covered_by_next(2) == 6
    assert plan.projected_rows_covered_by_next(99) == 6


# ===================== 2. TIMEOUT RETRIES ==================================


def test_transient_exception_classification() -> None:
    assert is_transient_exception(httpx.ConnectTimeout("x"))
    assert is_transient_exception(httpx.ReadTimeout("x"))
    assert is_transient_exception(httpx.ConnectError("x"))
    assert is_transient_exception(httpx.RemoteProtocolError("x"))
    assert not is_transient_exception(ValueError("x"))


def test_status_classification() -> None:
    assert classify_status(429) == "rate_limit"
    for code in (408, 500, 502, 503, 504):
        assert classify_status(code) == "transient"
    for code in (400, 401, 403, 404):
        assert classify_status(code) == "permanent"


def test_retry_policy_backoff_is_exponential_bounded_and_jittered() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter_ratio=0.0, jitter=lambda: 0.0)
    assert policy.delay_for(1) == pytest.approx(1.0)
    assert policy.delay_for(2) == pytest.approx(2.0)
    assert policy.delay_for(3) == pytest.approx(4.0)
    assert policy.delay_for(9) == pytest.approx(8.0)  # capped
    jittered = RetryPolicy(base_delay=4.0, max_delay=8.0, jitter_ratio=0.5, jitter=lambda: 1.0)
    assert jittered.delay_for(1) == pytest.approx(4.0)
    floor = RetryPolicy(base_delay=4.0, max_delay=8.0, jitter_ratio=0.5, jitter=lambda: 0.0)
    assert floor.delay_for(1) == pytest.approx(2.0)


def test_retry_after_parsing() -> None:
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after("-5") is None


def test_timeout_retry_eventually_succeeds(tmp_path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectTimeout("boom")
        return httpx.Response(
            200, json={"values": [{"datetime": "2026-01-27", "close": "100.0"}], "status": "ok"}
        )

    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=5,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=_no_sleep_policy(3),
    )
    values = client.prices_payload(
        "AAPL", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC)
    )
    assert len(values) == 1
    assert attempts == 3
    assert client.stats.successful_symbols == 1
    assert client.stats.retries_performed == 2
    assert client.stats.timeout_failures == 2
    assert client.stats.request_attempts == 3
    assert client.stats.symbols_requested == 1
    client.stats.check_invariants()
    client.close()


def test_retry_exhaustion_raises_transient_and_does_not_mark_unsupported(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("always")

    unsupported = UnsupportedSymbolCache(tmp_path / "u.json", provider="twelve_data")
    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=_no_sleep_policy(3),
        unsupported_cache=unsupported,
    )
    with pytest.raises(TransientProviderError):
        client.prices_payload(
            "AAPL", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC)
        )
    # CRITICAL: a timeout must never blacklist the symbol.
    assert "AAPL" not in unsupported
    assert client.stats.transient_failures == 1
    assert client.stats.permanent_failures == 0
    assert client.stats.request_attempts == 3
    client.stats.check_invariants()
    client.close()


def test_http_500_is_retried(tmp_path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "5"}]})

    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=4,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=_no_sleep_policy(3),
    )
    assert len(client.prices_payload("X", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))) == 1
    assert attempts == 2
    assert client.stats.retries_performed == 1
    client.close()


def test_429_honors_retry_after_then_succeeds(tmp_path) -> None:
    attempts = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"message": "slow down"})
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "9"}]})

    policy = RetryPolicy(
        max_attempts=3, base_delay=99.0, jitter_ratio=0.0,
        sleep=waits.append, jitter=lambda: 0.0,
    )
    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=4,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=policy,
    )
    client.prices_payload("X", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))
    # Retry-After (3s) beat the 99s exponential backoff.
    assert waits == [3.0]
    assert client.stats.rate_limit_failures == 1
    client.close()


def test_persistent_429_trips_run_wide_circuit(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "rate limit"})

    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=_no_sleep_policy(2),
    )
    with pytest.raises(ProviderUnavailable):
        client.prices_payload("X", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))
    assert client.unavailable_reason is not None
    client.close()


def test_budget_is_respected_despite_retries(tmp_path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("always")

    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=2,  # smaller than max_attempts=5
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=_no_sleep_policy(5),
    )
    with pytest.raises(TransientProviderError):
        client.prices_payload("X", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))
    assert attempts == 2  # never exceeded the budget
    assert client.stats.request_attempts == 2
    assert client.stats.request_attempts <= client.stats.max_api_calls
    client.stats.check_invariants()
    client.close()


# ===================== 3. UNSUPPORTED-SYMBOL CACHE =========================


def test_unsupported_message_classification() -> None:
    assert looks_unsupported_symbol("**symbol** not found: ZZZZ")
    assert looks_unsupported_symbol("symbol is not available")
    assert looks_unsupported_symbol("no data is available for this symbol")
    assert not looks_unsupported_symbol("connection reset by peer")
    assert classify_provider_message("upgrade your plan to access this symbol") == "entitlement"
    assert classify_provider_message("**symbol** not found") == "unsupported_symbol"
    assert classify_provider_message("internal error") is None


def test_permanent_symbol_error_is_recorded_with_metadata(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"code": 400, "message": "**symbol** not found: BAD", "status": "error"}
        )

    path = tmp_path / "twelve_data_unsupported.json"
    unsupported = UnsupportedSymbolCache(path, provider="twelve_data")
    client = _twelve(
        "k",
        cache=DiskJsonCache(tmp_path),
        max_api_calls=5,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None,
        retry_policy=_no_sleep_policy(3),
        unsupported_cache=unsupported,
    )
    with pytest.raises(UnsupportedSymbolError):
        client.prices_payload("BAD", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))

    entry = unsupported.entry("BAD")
    assert entry is not None
    assert entry.ticker == "BAD"
    assert entry.provider == "twelve_data"
    assert entry.reason == "unsupported_symbol"
    assert entry.status_code == 400
    assert "not found" in entry.provider_message
    assert entry.attempt_count == 1
    assert entry.first_seen_at and entry.last_seen_at
    # Permanent errors are NOT retried.
    assert client.stats.request_attempts == 1
    assert client.stats.permanent_failures == 1
    client.close()


def test_unsupported_cache_persists_across_runs_and_costs_zero_calls(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"code": 400, "message": "symbol not found", "status": "error"})

    path = tmp_path / "twelve_data_unsupported.json"
    cache = DiskJsonCache(tmp_path)

    first = UnsupportedSymbolCache(path, provider="twelve_data")
    client = _twelve(
        "k", cache=cache, max_api_calls=5,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(3), unsupported_cache=first,
    )
    with pytest.raises(UnsupportedSymbolError):
        client.prices_payload("BAD", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))
    client.close()
    assert first.save() is True
    assert path.exists()
    assert calls == 1

    # Fresh process: cache is loaded from disk and the symbol is skipped for free.
    second = UnsupportedSymbolCache(path, provider="twelve_data")
    assert "BAD" in second
    assert second.should_skip("BAD")
    messages: list[str] = []
    client2 = _twelve(
        "k", cache=DiskJsonCache(tmp_path / "fresh"), max_api_calls=5,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=messages.append, retry_policy=_no_sleep_policy(3), unsupported_cache=second,
    )
    with pytest.raises(UnsupportedSymbolError):
        client2.prices_payload("BAD", start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2026, 2, 1, tzinfo=UTC))
    assert calls == 1  # zero additional network calls
    assert client2.stats.request_attempts == 0
    assert client2.stats.symbols_skipped_unsupported == 1
    assert any("skip unsupported ticker=BAD" in m for m in messages)
    client2.close()


def test_retry_unsupported_flag_reattempts(tmp_path) -> None:
    path = tmp_path / "u.json"
    cache = UnsupportedSymbolCache(path, provider="twelve_data")
    cache.record("BAD", reason="unsupported_symbol", status_code=400, provider_message="not found")
    cache.save()

    normal = UnsupportedSymbolCache(path, provider="twelve_data")
    assert normal.should_skip("BAD") is True

    retrying = UnsupportedSymbolCache(path, provider="twelve_data", retry_unsupported=True)
    assert retrying.should_skip("BAD") is False
    assert "BAD" in retrying  # history is retained, not discarded


def test_unsupported_cache_clear_and_attempt_count(tmp_path) -> None:
    cache = UnsupportedSymbolCache(tmp_path / "u.json", provider="twelve_data")
    cache.record("A", provider_message="not found")
    cache.record("A", provider_message="not found")
    assert cache.entry("A").attempt_count == 2
    assert cache.entry("A").first_seen_at <= cache.entry("A").last_seen_at
    cache.record("B", provider_message="not found")
    assert cache.clear(["A"]) == 1
    assert "A" not in cache and "B" in cache
    assert cache.clear() == 1
    assert len(cache) == 0


def test_unsupported_cache_never_stores_credentials(tmp_path) -> None:
    assert "secret" not in sanitize_provider_message("failed for apikey=secret123")
    assert "<redacted>" in sanitize_provider_message("failed for apikey=secret123")
    assert "<redacted>" in sanitize_provider_message("Authorization: Bearer abc.def")
    cache = UnsupportedSymbolCache(tmp_path / "u.json", provider="twelve_data")
    cache.record("A", provider_message="bad request apikey=supersecret")
    cache.save()
    assert "supersecret" not in (tmp_path / "u.json").read_text()


def test_corrupt_unsupported_cache_does_not_abort(tmp_path) -> None:
    path = tmp_path / "u.json"
    path.write_text("{not valid json", encoding="utf-8")
    cache = UnsupportedSymbolCache(path, provider="twelve_data")
    assert len(cache) == 0  # rebuilt rather than raising
    cache.record("A", provider_message="not found")
    assert cache.save() is True
    assert json.loads(path.read_text())["symbols"]["A"]["ticker"] == "A"


def test_default_unsupported_path_layout(tmp_path) -> None:
    assert default_unsupported_path(tmp_path, "twelve_data").name == "twelve_data_unsupported.json"


# ===================== 4. ALREADY-COVERED SKIP =============================


def _cached_twelve_payload(sessions: int, start_date: datetime) -> dict:
    return {
        "values": [
            {
                "datetime": (start_date + timedelta(days=i)).date().isoformat(),
                "close": str(100.0 + i),
                "volume": "1000",
            }
            for i in range(sessions)
        ],
        "status": "ok",
    }


def test_covered_ticker_is_excluded_from_the_queue(tmp_path) -> None:
    rows = [_row("a", "COVERED"), _row("b", "MISSING")]
    events = [
        _event("a", "COVERED", "2026-01-05T12:00:00Z"),
        _event("b", "MISSING", "2026-01-05T12:00:00Z"),
    ]
    cache = DiskJsonCache(tmp_path)
    demand = {d.ticker: d for d in build_ticker_demand(rows, events)}["COVERED"]
    cache.put(
        "twelve_data_prices",
        demand.cache_key("twelve_data_prices"),
        _cached_twelve_payload(200, datetime(2021, 1, 1, tzinfo=UTC)),
    )
    plan = plan_price_backfill(rows, events, cache=cache)
    assert [d.ticker for d in plan.queue] == ["MISSING"]
    assert plan.tickers_already_covered == 1
    assert plan.total_rows_already_covered == 1
    assert plan.total_rows_needing_prices == 1


def test_insufficient_cached_series_is_not_treated_as_covered(tmp_path) -> None:
    rows = [_row("a", "SHORT")]
    events = [_event("a", "SHORT", "2026-01-05T12:00:00Z")]
    cache = DiskJsonCache(tmp_path)
    demand = build_ticker_demand(rows, events)[0]
    cache.put(
        "twelve_data_prices",
        demand.cache_key("twelve_data_prices"),
        _cached_twelve_payload(MIN_USABLE_SESSIONS - 1, datetime(2021, 1, 1, tzinfo=UTC)),
    )
    plan = plan_price_backfill(rows, events, cache=cache)
    assert [d.ticker for d in plan.queue] == ["SHORT"]
    assert plan.tickers_already_covered == 0


def test_empty_and_corrupt_cache_entries_are_not_covered(tmp_path) -> None:
    rows = [_row("a", "EMPTY"), _row("b", "CORRUPT")]
    events = [
        _event("a", "EMPTY", "2026-01-05T12:00:00Z"),
        _event("b", "CORRUPT", "2026-01-05T12:00:00Z"),
    ]
    cache = DiskJsonCache(tmp_path)
    by_ticker = {d.ticker: d for d in build_ticker_demand(rows, events)}
    cache.put("twelve_data_prices", by_ticker["EMPTY"].cache_key("twelve_data_prices"), {"values": []})
    corrupt_path = cache.path("twelve_data_prices", by_ticker["CORRUPT"].cache_key("twelve_data_prices"))
    corrupt_path.write_text("{broken", encoding="utf-8")

    index = PriceCacheIndex(cache)
    assert index.status(by_ticker["EMPTY"]).covered is False
    assert index.status(by_ticker["CORRUPT"]).covered is False
    plan = plan_price_backfill(rows, events, cache=cache)
    assert {d.ticker for d in plan.queue} == {"EMPTY", "CORRUPT"}


def test_coverage_detects_provider_independent_cache(tmp_path) -> None:
    """Tiingo-cached history must satisfy coverage without a Twelve Data call."""
    rows = [_row("a", "TIINGO")]
    events = [_event("a", "TIINGO", "2026-01-05T12:00:00Z")]
    cache = DiskJsonCache(tmp_path)
    demand = build_ticker_demand(rows, events)[0]
    payload = [
        {"date": (datetime(2021, 1, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat(),
         "adjClose": 100.0 + i, "volume": 1000}
        for i in range(120)
    ]
    cache.put("tiingo_prices", demand.cache_key("tiingo_prices"), payload)
    plan = plan_price_backfill(rows, events, cache=cache)
    assert plan.queue == ()
    assert plan.covered[0].provider == "tiingo_prices"


def test_unsupported_tickers_are_dropped_from_the_queue(tmp_path) -> None:
    rows = [_row("a", "GOOD"), _row("b", "BANNED")]
    events = [
        _event("a", "GOOD", "2026-01-05T12:00:00Z"),
        _event("b", "BANNED", "2026-01-05T12:00:00Z"),
    ]
    unsupported = UnsupportedSymbolCache(tmp_path / "u.json", provider="twelve_data")
    unsupported.record("BANNED", provider_message="not found")
    plan = plan_price_backfill(
        rows, events, cache=DiskJsonCache(tmp_path), unsupported_cache=unsupported
    )
    assert [d.ticker for d in plan.queue] == ["GOOD"]
    assert plan.tickers_skipped_unsupported == 1
    assert plan.rows_blocked_by_unsupported == 1
    assert plan.skipped_unsupported == ("BANNED",)


def test_5y_coverage_is_reported_separately_from_scheduling(tmp_path) -> None:
    """A short-but-usable series is 'covered' for scheduling, not for 5y features."""
    rows = [_row("a", "SHORTISH")]
    events = [_event("a", "SHORTISH", "2026-01-05T12:00:00Z")]
    cache = DiskJsonCache(tmp_path)
    demand = build_ticker_demand(rows, events)[0]
    cache.put(
        "twelve_data_prices",
        demand.cache_key("twelve_data_prices"),
        _cached_twelve_payload(100, datetime(2021, 1, 1, tzinfo=UTC)),
    )
    plan = plan_price_backfill(rows, events, cache=cache)
    assert plan.tickers_already_covered == 1
    assert plan.tickers_with_5y_history == 0
    assert plan.rows_with_5y_history == 0


# ===================== 5. PROVIDER STATISTICS ==============================


def test_stats_invariants_hold_for_mixed_outcomes(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        if symbol == "BAD":
            return httpx.Response(200, json={"code": 400, "message": "symbol not found", "status": "error"})
        if symbol == "SLOW":
            raise httpx.ReadTimeout("nope")
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "1"}]})

    unsupported = UnsupportedSymbolCache(tmp_path / "u.json", provider="twelve_data")
    client = _twelve(
        "k", cache=DiskJsonCache(tmp_path), max_api_calls=20,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(2), unsupported_cache=unsupported,
    )
    start, end = datetime(2021, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    client.prices_payload("GOOD", start=start, end=end)
    with pytest.raises(UnsupportedSymbolError):
        client.prices_payload("BAD", start=start, end=end)
    with pytest.raises(TransientProviderError):
        client.prices_payload("SLOW", start=start, end=end)

    s = client.stats
    assert s.symbols_requested == 3
    assert s.successful_symbols == 1
    assert s.permanent_failures == 1
    assert s.transient_failures == 1
    # requested == success + transient + permanent
    s.check_invariants()
    # attempts: GOOD 1 + BAD 1 + SLOW 2 = 4
    assert s.request_attempts == 4
    assert s.budget_remaining == 16
    client.close()


def test_cache_hits_do_not_consume_budget(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "1"}]})

    client = _twelve(
        "k", cache=DiskJsonCache(tmp_path), max_api_calls=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(3),
    )
    start, end = datetime(2021, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    client.prices_payload("X", start=start, end=end)
    client.prices_payload("X", start=start, end=end)  # served from cache
    assert client.stats.request_attempts == 1
    assert client.stats.cache_hits == 1
    assert client.stats.symbols_requested == 1
    client.stats.check_invariants()
    client.close()


def test_budget_exhaustion_does_not_count_as_a_requested_symbol(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "1"}]})

    from explaining_markets.providers.free_historical import ProviderBudgetExhausted

    client = _twelve(
        "k", cache=DiskJsonCache(tmp_path), max_api_calls=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(3),
    )
    start, end = datetime(2021, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    client.prices_payload("A", start=start, end=end)
    with pytest.raises(ProviderBudgetExhausted):
        client.prices_payload("B", start=start, end=end)
    assert client.stats.symbols_requested == 1
    assert client.stats.budget_exhausted_events == 1
    client.stats.check_invariants()
    client.close()


# ===================== PREFETCH EXECUTION ==================================


def test_run_price_backfill_fetches_in_priority_order(tmp_path) -> None:
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(request.url.params["symbol"])
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "1"}]})

    rows = [_row(f"h{i}", "HIGH") for i in range(3)] + [_row("l", "LOW")]
    events = [_event(f"h{i}", "HIGH", "2026-01-05T12:00:00Z") for i in range(3)]
    events += [_event("l", "LOW", "2026-01-05T12:00:00Z")]
    plan = plan_price_backfill(rows, events, cache=DiskJsonCache(tmp_path))
    client = _twelve(
        "k", cache=DiskJsonCache(tmp_path), max_api_calls=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(2),
    )
    outcome = run_price_backfill(
        plan,
        [PriceFetcher("twelve_data", client, "twelve_data_prices")],
        progress=lambda _: None,
    )
    assert fetched == ["HIGH", "LOW"]
    assert outcome.rows_unlocked == 4
    assert set(outcome.successful_tickers) == {"HIGH", "LOW"}
    client.close()


def test_run_price_backfill_stops_when_budget_exhausted(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "1"}]})

    rows = [_row(f"r{i}", f"T{i}") for i in range(5)]
    events = [_event(f"r{i}", f"T{i}", "2026-01-05T12:00:00Z") for i in range(5)]
    plan = plan_price_backfill(rows, events, cache=DiskJsonCache(tmp_path))
    client = _twelve(
        "k", cache=DiskJsonCache(tmp_path), max_api_calls=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(1),
    )
    outcome = run_price_backfill(
        plan, [PriceFetcher("twelve_data", client, "twelve_data_prices")], progress=lambda _: None
    )
    assert len(outcome.successful_tickers) == 2
    assert outcome.stopped_reason == "all price providers exhausted or unavailable"
    client.stats.check_invariants()
    client.close()


def test_run_price_backfill_records_unsupported_and_continues(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["symbol"] == "BAD":
            return httpx.Response(200, json={"code": 400, "message": "symbol not found", "status": "error"})
        return httpx.Response(200, json={"values": [{"datetime": "2026-01-27", "close": "1"}]})

    rows = [_row("a", "BAD"), _row("b", "GOOD")]
    events = [_event("a", "BAD", "2026-01-05T12:00:00Z"), _event("b", "GOOD", "2026-01-05T12:00:00Z")]
    unsupported = UnsupportedSymbolCache(tmp_path / "u.json", provider="twelve_data")
    plan = plan_price_backfill(rows, events, cache=DiskJsonCache(tmp_path), unsupported_cache=unsupported)
    client = _twelve(
        "k", cache=DiskJsonCache(tmp_path), max_api_calls=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        progress=lambda _: None, retry_policy=_no_sleep_policy(2), unsupported_cache=unsupported,
    )
    outcome = run_price_backfill(
        plan, [PriceFetcher("twelve_data", client, "twelve_data_prices")],
        unsupported_cache=unsupported, progress=lambda _: None,
    )
    assert outcome.successful_tickers == ("GOOD",)
    assert outcome.unsupported_tickers == ("BAD",)
    assert "BAD" in unsupported
    assert (tmp_path / "u.json").exists()  # persisted for the next run
    client.close()
