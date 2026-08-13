"""Leakage audit: point-in-time integrity across the V2 archive sweep."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from explaining_markets.competition_history import (
    AVAILABILITY_LAG_DAYS,
    outcome_available_at,
    parse_event_datetime,
    walk_forward_history,
)
from explaining_markets.historical import HistoricalEvent, load_historical_events
from explaining_markets.leakage_audit import (
    LeakageError,
    audit_current_surprise_neutrality,
    audit_feature_names,
    audit_history_row,
)

UTC = timezone.utc


def test_outcome_available_at_is_strictly_after_event() -> None:
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    assert outcome_available_at(dt) > dt
    assert (outcome_available_at(dt) - dt).days == AVAILABILITY_LAG_DAYS


def test_audit_feature_names_passes_on_clean_spec() -> None:
    # Should not raise — MODEL_FEATURE_NAMES_V2 has no forbidden names.
    audit_feature_names()


def test_audit_feature_names_detects_forbidden_names() -> None:
    with pytest.raises(LeakageError, match="forbidden"):
        audit_feature_names(("fls_ratio", "car1", "earnings_surprise"))


def test_audit_history_row_rejects_self_as_source() -> None:
    from explaining_markets.company_history import HistoricalEarningsEvent

    focal_dt = datetime(2026, 3, 10, tzinfo=UTC)
    source = HistoricalEarningsEvent(
        event_timestamp=focal_dt - timedelta(days=100),
        source="competition_archive",
        source_event_id="SELF",
    )
    from explaining_markets.company_history import CompanyHistoryFeatures

    history = CompanyHistoryFeatures(
        ticker="AAPL", cutoff=focal_dt, values={}, source_events=(source,)
    )
    with pytest.raises(LeakageError, match="its own record"):
        audit_history_row(
            focal_event_id="SELF",
            focal_ticker="AAPL",
            focal_event_datetime=focal_dt,
            history=history,
        )


def test_audit_history_row_rejects_late_outcome() -> None:
    from explaining_markets.company_history import CompanyHistoryFeatures, HistoricalEarningsEvent

    focal_dt = datetime(2026, 3, 10, tzinfo=UTC)
    # Source 5 days before focal: outcome (+7d) is NOT yet conservatively knowable.
    source = HistoricalEarningsEvent(
        event_timestamp=focal_dt - timedelta(days=5),
        source="competition_archive",
        source_event_id="early",
    )
    history = CompanyHistoryFeatures(
        ticker="AAPL", cutoff=focal_dt, values={}, source_events=(source,)
    )
    with pytest.raises(LeakageError, match="violates outcome_available_at"):
        audit_history_row(
            focal_event_id="focal",
            focal_ticker="AAPL",
            focal_event_datetime=focal_dt,
            history=history,
        )


def test_audit_current_surprise_neutrality_passes_on_zeros() -> None:
    audit_current_surprise_neutrality({"current_eps_surprise": 0.0})


def test_audit_current_surprise_neutrality_rejects_populated_surprise() -> None:
    with pytest.raises(LeakageError, match="populated"):
        audit_current_surprise_neutrality({"current_eps_surprise": 0.05})


def test_walk_forward_on_real_archive_passes_audit() -> None:
    events = load_historical_events()
    if not events:
        import pytest as _pytest

        _pytest.skip("data/historical/ is empty in this environment")
    history = walk_forward_history(events)
    checked = 0
    for key, features in history.items():
        event_id, ticker = key.split(":", 1)
        # Find the focal event to get its datetime.
        focal = next((e for e in events if e.event_id == event_id), None)
        if focal is None:
            continue
        focal_dt = parse_event_datetime(focal)
        if focal_dt is None:
            continue
        checked += audit_history_row(
            focal_event_id=event_id,
            focal_ticker=ticker,
            focal_event_datetime=focal_dt,
            history=features,
        )
    assert checked > 0
