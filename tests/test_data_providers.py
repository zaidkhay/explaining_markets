"""Point-in-time record validation and live SQLite-cache behavior."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from explaining_markets.data_providers import CompanyHistoryCache, EarningsRecord, PriceBar

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


def _earnings(days_before: int, *, reaction=0.02, reaction_lag_days=2) -> EarningsRecord:
    ts = T0 - timedelta(days=days_before)
    return EarningsRecord(
        ticker="AAPL",
        event_timestamp=ts,
        source="fixture",
        available_at=ts,
        retrieved_at=T0,
        abnormal_return=reaction,
        next_session_return=reaction,
        benchmark_next_session_return=0.0,
        benchmark="SPY",
        reaction_available_at=ts + timedelta(days=reaction_lag_days) if reaction_lag_days else None,
    )


def test_records_fail_closed_on_time_and_value_validation() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PriceBar(
            ticker="AAPL",
            value_timestamp=datetime(2026, 1, 1),
            adjusted_close=100.0,
            source="fixture",
            available_at=T0,
            retrieved_at=T0,
        )
    with pytest.raises(ValueError, match="positive"):
        _bar(1, close=0.0)

    bar = _bar(0)
    assert not bar.usable_at(T0)
    assert bar.usable_at(T0 + timedelta(seconds=1))

    record = _earnings(30, reaction_lag_days=0)
    assert record.figures_usable_at(T0)
    assert not record.reaction_usable_at(T0)


def test_cache_round_trip_and_cutoff_filtering(tmp_path) -> None:
    cache = CompanyHistoryCache(tmp_path / "cache.sqlite")
    cache.upsert_prices([_bar(5), _bar(0), _bar(-5)])
    cache.upsert_earnings([_earnings(90)])

    bars = cache.daily_prices_before("AAPL", T0)
    assert len(bars) == 1
    assert bars[0].value_timestamp == T0 - timedelta(days=5)
    assert bars[0].source == "fixture"

    records = cache.earnings_before("AAPL", T0)
    assert len(records) == 1
    assert records[0].benchmark == "SPY"
    assert records[0].reaction_available_at is not None

    assert cache.daily_prices_before("MSFT", T0) == []
    cache.close()
