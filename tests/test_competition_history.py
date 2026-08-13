"""Archive-sourced history: availability lag, walk-forward, live snapshot."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from explaining_markets.competition_history import (
    AVAILABILITY_LAG_DAYS,
    CUTOFF_GUARD_DAYS,
    SnapshotCompanyHistoryProvider,
    build_snapshot,
    competition_feature_values,
    eligible_prior_archive_events,
    outcome_available_at,
    parse_event_datetime,
    to_history_event,
    walk_forward_history,
    write_snapshot,
)
from explaining_markets.historical import HistoricalEvent, load_historical_events

UTC = timezone.utc


def _event(event_id: str, ticker: str, dt: str | None, *, car1=None, surprise=None) -> HistoricalEvent:
    return HistoricalEvent(
        event_id=event_id,
        ticker=ticker,
        event_type="EARNINGS_RELEASE",
        event_datetime=dt,
        disclosure=[],
        car1=car1,
        earnings_surprise=surprise,
        quarter=None,
    )


# ----- parse_event_datetime -----------------------------------------------------


def test_parse_event_datetime_accepts_z_suffix() -> None:
    event = _event("e1", "AAPL", "2026-01-01T10:00:00Z")
    parsed = parse_event_datetime(event)
    assert parsed == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def test_parse_event_datetime_fails_closed_on_naive_or_invalid() -> None:
    assert parse_event_datetime(_event("e1", "AAPL", "2026-01-01T10:00:00")) is None  # naive
    assert parse_event_datetime(_event("e2", "AAPL", "not-a-date")) is None
    assert parse_event_datetime(_event("e3", "AAPL", None)) is None


# ----- availability rule ----------------------------------------------------------


def test_outcome_available_at_applies_conservative_lag() -> None:
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    assert outcome_available_at(dt) == dt + timedelta(days=AVAILABILITY_LAG_DAYS)


def test_eligible_excludes_recent_prior_events_inside_the_lag() -> None:
    # Prior event 5 days earlier: outcome NOT yet conservatively knowable.
    target = _event("t", "AAPL", "2026-03-10T00:00:00+00:00")
    recent = _event("recent", "AAPL", "2026-03-05T00:00:00+00:00", car1=0.05)
    old = _event("old", "AAPL", "2025-12-01T00:00:00+00:00", car1=0.01)
    timeline = [old, recent, target]
    eligible = eligible_prior_archive_events(timeline, target)
    assert [e.event_id for e in eligible] == ["old"]


def test_eligible_boundary_is_strict_including_the_guard() -> None:
    # Prior event exactly LAG+GUARD days earlier is still NOT eligible
    # (strict inequality), one second older is.
    target_dt = datetime(2026, 3, 10, tzinfo=UTC)
    boundary = target_dt - timedelta(days=AVAILABILITY_LAG_DAYS + CUTOFF_GUARD_DAYS)
    target = _event("t", "AAPL", target_dt.isoformat())
    at_boundary = _event("b", "AAPL", boundary.isoformat(), car1=0.05)
    just_older = _event("o", "AAPL", (boundary - timedelta(seconds=1)).isoformat(), car1=0.05)
    eligible = eligible_prior_archive_events([at_boundary, just_older, target], target)
    assert [e.event_id for e in eligible] == ["o"]


def test_eligible_excludes_the_target_itself_and_unparseable() -> None:
    target = _event("t", "AAPL", "2026-03-10T00:00:00+00:00")
    undated = _event("u", "AAPL", None, car1=0.9)
    eligible = eligible_prior_archive_events([target, undated], target)
    assert eligible == []


# ----- adapters / aggregates ---------------------------------------------------------


def test_to_history_event_maps_car1_to_abnormal_return() -> None:
    he = to_history_event(_event("e", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.07, surprise=0.02))
    assert he is not None
    assert he.abnormal_return == 0.07
    assert he.eps_surprise == 0.02
    assert he.source == "competition_archive"


def test_competition_feature_values_last_and_mean() -> None:
    events = [
        to_history_event(_event("e1", "AAPL", "2025-10-01T00:00:00+00:00", car1=0.02)),
        to_history_event(_event("e2", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.06)),
    ]
    values = competition_feature_values(events)
    assert values["prior_competition_event_count"] == 2.0
    assert values["mean_prior_competition_car1"] == pytest.approx(0.04)
    assert values["last_prior_competition_car1"] == pytest.approx(0.06)
    assert values["has_competition_history"] == 1.0


def test_competition_feature_values_empty() -> None:
    values = competition_feature_values([])
    assert values["prior_competition_event_count"] == 0.0
    assert values["mean_prior_competition_car1"] is None
    assert values["has_competition_history"] == 0.0


# ----- walk-forward ---------------------------------------------------------------


def test_walk_forward_history_never_uses_future_or_cross_ticker() -> None:
    events = [
        _event("a1", "AAPL", "2025-10-01T00:00:00+00:00", car1=0.01, surprise=0.01),
        _event("a2", "AAPL", "2026-01-05T00:00:00+00:00", car1=0.02, surprise=-0.01),
        _event("a3", "AAPL", "2026-04-10T00:00:00+00:00", car1=0.03),
        _event("m1", "MSFT", "2025-11-01T00:00:00+00:00", car1=0.99, surprise=0.99),
    ]
    history = walk_forward_history(events)
    first = history["a1:AAPL"]
    assert first.values["prior_earnings_count"] == 0.0  # nothing before the first event
    third = history["a3:AAPL"]
    assert third.values["prior_earnings_count"] == 2.0
    ids = {e.source_event_id for e in third.source_events}
    assert ids == {"a1", "a2"}  # never m1 (cross-ticker), never a3 (self)


# ----- snapshot / live provider ------------------------------------------------------


def test_snapshot_round_trip_provider(tmp_path) -> None:
    events = [
        _event("a1", "AAPL", "2025-10-01T00:00:00+00:00", car1=0.01, surprise=0.02),
        _event("a2", "AAPL", "2026-01-05T00:00:00+00:00", car1=-0.02, surprise=-0.01),
    ]
    path = write_snapshot(events, tmp_path / "snapshot.json")
    provider = SnapshotCompanyHistoryProvider(path)

    live_cutoff = datetime(2026, 9, 1, tzinfo=UTC)
    features = provider.history_before("AAPL", live_cutoff)
    assert features.values["prior_earnings_count"] == 2.0
    assert features.values["mean_prior_earnings_abnormal_return"] == pytest.approx(-0.005)

    unknown = provider.history_before("ZZZZ", live_cutoff)
    assert unknown.values["prior_earnings_count"] == 0.0


def test_snapshot_provider_respects_cutoff_lag(tmp_path) -> None:
    events = [_event("a1", "AAPL", "2026-01-05T00:00:00+00:00", car1=0.01)]
    path = write_snapshot(events, tmp_path / "snapshot.json")
    provider = SnapshotCompanyHistoryProvider(path)
    # A cutoff only 3 days after the event: outcome not conservatively knowable.
    early = provider.history_before("AAPL", datetime(2026, 1, 8, tzinfo=UTC))
    assert early.values["prior_earnings_count"] == 0.0
    late = provider.history_before("AAPL", datetime(2026, 2, 1, tzinfo=UTC))
    assert late.values["prior_earnings_count"] == 1.0


# ----- real archive integration --------------------------------------------------------


def test_walk_forward_on_real_archive_is_leakage_free() -> None:
    events = load_historical_events()
    if not events:
        import pytest as _pytest

        _pytest.skip("data/historical/ is empty in this environment")
    history = walk_forward_history(events)
    checked = 0
    for key, features in history.items():
        for source in features.source_events:
            checked += 1
            assert outcome_available_at(source.event_timestamp) < features.cutoff
    assert checked > 0
