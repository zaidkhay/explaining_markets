"""Provider records, SQLite cache, and point-in-time filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from explaining_markets.data_providers import (
    CompanyHistoryCache,
    EarningsRecord,
    InMemoryProvider,
    PriceBar,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 15, 21, 0, tzinfo=UTC)


def _bar(days_before: int, close: float = 100.0, *, available_lag_days: int = 0) -> PriceBar:
    ts = T0 - timedelta(days=days_before)
    return PriceBar(
        ticker="AAPL",
        value_timestamp=ts,
        adjusted_close=close,
        source="fixture",
        available_at=ts + timedelta(days=available_lag_days),
        retrieved_at=T0,
    )


def _earnings(days_before: int, *, surprise=0.01, reaction=0.02, reaction_lag_days=2) -> EarningsRecord:
    ts = T0 - timedelta(days=days_before)
    return EarningsRecord(
        ticker="AAPL",
        event_timestamp=ts,
        source="fixture",
        available_at=ts,
        retrieved_at=T0,
        eps_surprise=surprise,
        abnormal_return=reaction,
        next_session_return=reaction,
        benchmark_next_session_return=0.0,
        benchmark="SPY",
        reaction_available_at=ts + timedelta(days=reaction_lag_days) if reaction_lag_days else None,
    )


# ----- record validation ----------------------------------------------------


def test_price_bar_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PriceBar(
            ticker="AAPL",
            value_timestamp=datetime(2026, 1, 1),  # naive
            adjusted_close=100.0,
            source="fixture",
            available_at=T0,
            retrieved_at=T0,
        )


def test_price_bar_rejects_non_positive_close() -> None:
    with pytest.raises(ValueError, match="positive"):
        _bar(1, close=0.0)


def test_price_bar_usable_at_is_strict() -> None:
    bar = _bar(0)  # available exactly at T0
    assert not bar.usable_at(T0)  # equal timestamp is NOT strictly before
    assert bar.usable_at(T0 + timedelta(seconds=1))


def test_late_adjusted_bar_is_excluded_before_its_revision_time() -> None:
    # A back-adjusted close revised 5 days after the session must not be
    # usable at a cutoff between the session and the revision.
    bar = _bar(10, available_lag_days=5)
    assert not bar.usable_at(T0 - timedelta(days=7))  # before revision
    assert bar.usable_at(T0)  # after revision


def test_earnings_reaction_fails_closed_without_reaction_timestamp() -> None:
    record = _earnings(30, reaction_lag_days=0)  # reaction present, availability unknown
    assert record.figures_usable_at(T0)
    assert not record.reaction_usable_at(T0)  # unknown availability -> unusable


def test_earnings_reaction_usable_only_after_reaction_available_at() -> None:
    record = _earnings(30, reaction_lag_days=2)
    assert not record.reaction_usable_at(record.event_timestamp + timedelta(days=1))
    assert record.reaction_usable_at(record.event_timestamp + timedelta(days=3))


# ----- in-memory provider -----------------------------------------------------


def test_in_memory_provider_filters_future_prices() -> None:
    provider = InMemoryProvider(prices=[_bar(10), _bar(5), _bar(-5)])  # one in the future
    bars = provider.daily_prices_before("AAPL", T0)
    assert len(bars) == 2
    assert all(b.available_at < T0 for b in bars)


def test_in_memory_provider_filters_future_earnings() -> None:
    provider = InMemoryProvider(earnings=[_earnings(90), _earnings(-10)])
    records = provider.earnings_before("AAPL", T0)
    assert len(records) == 1
    assert records[0].available_at < T0


# ----- SQLite cache -----------------------------------------------------------


def test_cache_round_trip_preserves_provenance(tmp_path) -> None:
    cache = CompanyHistoryCache(tmp_path / "cache.sqlite")
    cache.upsert_prices([_bar(3), _bar(2), _bar(1)])
    cache.upsert_earnings([_earnings(90)])

    bars = cache.daily_prices_before("AAPL", T0)
    assert [b.adjusted_close for b in bars] == [100.0, 100.0, 100.0]
    assert all(b.source == "fixture" for b in bars)
    assert bars == sorted(bars, key=lambda b: b.value_timestamp)  # ascending

    records = cache.earnings_before("AAPL", T0)
    assert len(records) == 1
    assert records[0].benchmark == "SPY"
    assert records[0].reaction_available_at is not None
    cache.close()


def test_cache_excludes_records_at_or_after_cutoff(tmp_path) -> None:
    cache = CompanyHistoryCache(tmp_path / "cache.sqlite")
    cache.upsert_prices([_bar(5), _bar(0), _bar(-5)])  # at-cutoff and future rows
    bars = cache.daily_prices_before("AAPL", T0)
    assert len(bars) == 1
    assert bars[0].value_timestamp == T0 - timedelta(days=5)
    cache.close()


def test_cache_cutoff_is_mandatory(tmp_path) -> None:
    cache = CompanyHistoryCache(tmp_path / "cache.sqlite")
    with pytest.raises((TypeError, ValueError)):
        cache.daily_prices_before("AAPL", None)  # type: ignore[arg-type]
    cache.close()


def test_cache_isolates_tickers(tmp_path) -> None:
    cache = CompanyHistoryCache(tmp_path / "cache.sqlite")
    cache.upsert_prices([_bar(5)])
    assert cache.daily_prices_before("MSFT", T0) == []
    cache.close()
