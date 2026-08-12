"""Feature store: walk-forward correctness, leakage assertions, provenance."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from explaining_markets.feature_store import (
    DEFAULT_WINDOW,
    HistoricalFeatures,
    ProvenanceRecord,
    assert_feature_is_leakage_free,
    assert_no_target_leakage,
    build_feature_store,
    build_ticker_timelines,
    compute_historical_features,
    eligible_prior_events,
)
from explaining_markets.historical import HistoricalEvent, load_historical_events


def _event(event_id: str, ticker: str, dt: str, *, car1=None, surprise=None) -> HistoricalEvent:
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


# ----- build_ticker_timelines -------------------------------------------


def test_build_ticker_timelines_groups_and_sorts() -> None:
    events = [
        _event("e3", "AAPL", "2026-03-01T00:00:00+00:00"),
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00"),
        _event("e2", "MSFT", "2026-02-01T00:00:00+00:00"),
    ]
    timelines = build_ticker_timelines(events)
    assert set(timelines) == {"AAPL", "MSFT"}
    assert [e.event_id for e in timelines["AAPL"]] == ["e1", "e3"]  # chronological
    assert [e.event_id for e in timelines["MSFT"]] == ["e2"]


def test_build_ticker_timelines_handles_missing_datetime() -> None:
    events = [
        _event("e1", "AAPL", None),
        _event("e2", "AAPL", "2026-01-01T00:00:00+00:00"),
    ]
    timelines = build_ticker_timelines(events)
    # Missing-datetime events sort to the epoch fallback, never crash.
    assert len(timelines["AAPL"]) == 2


# ----- eligible_prior_events ---------------------------------------------


def test_eligible_prior_events_excludes_target_and_later_events() -> None:
    e1 = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.01)
    e2 = _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=0.02)
    target = _event("e3", "AAPL", "2026-03-01T00:00:00+00:00")
    timeline = [e1, e2, target]

    prior = eligible_prior_events(timeline, target)
    assert [e.event_id for e in prior] == ["e1", "e2"]


def test_eligible_prior_events_excludes_equal_timestamp() -> None:
    # A same-timestamp "prior" event is NOT strictly before - excluded.
    e1 = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    target = _event("e2", "AAPL", "2026-01-01T00:00:00+00:00")
    assert eligible_prior_events([e1, target], target) == []


def test_eligible_prior_events_never_crosses_a_different_timeline() -> None:
    # Even if a timeline erroneously contains another ticker's event, only
    # timestamp ordering is checked here - ticker isolation is guaranteed by
    # build_ticker_timelines grouping, not by this function alone. This test
    # documents that eligible_prior_events is timestamp-only and relies on
    # its caller for ticker isolation.
    other = _event("other", "MSFT", "2026-01-01T00:00:00+00:00", car1=0.5)
    target = _event("e2", "AAPL", "2026-02-01T00:00:00+00:00")
    prior = eligible_prior_events([other, target], target)
    assert [e.event_id for e in prior] == ["other"]  # timestamp ordering alone matched


def test_eligible_prior_events_returns_empty_for_unparseable_target() -> None:
    e1 = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    target = _event("e2", "AAPL", None)
    assert eligible_prior_events([e1, target], target) == []


def test_eligible_prior_events_skips_unparseable_candidates() -> None:
    e1 = _event("e1", "AAPL", None)  # no timestamp - excluded, not guessed
    e2 = _event("e2", "AAPL", "2026-01-01T00:00:00+00:00")
    target = _event("e3", "AAPL", "2026-02-01T00:00:00+00:00")
    prior = eligible_prior_events([e1, e2, target], target)
    assert [e.event_id for e in prior] == ["e2"]


# ----- assert_no_target_leakage -------------------------------------------


def test_assert_no_target_leakage_passes_for_valid_prior_events() -> None:
    e1 = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    target = _event("e2", "AAPL", "2026-02-01T00:00:00+00:00")
    assert_no_target_leakage(target, [e1])  # must not raise


def test_assert_no_target_leakage_raises_when_source_is_target() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="is the target event itself"):
        assert_no_target_leakage(target, [target])


def test_assert_no_target_leakage_raises_when_source_is_later() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    later = _event("e2", "AAPL", "2026-06-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="not strictly earlier"):
        assert_no_target_leakage(target, [later])


def test_assert_no_target_leakage_raises_when_source_is_equal_timestamp() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    same_time = _event("e2", "AAPL", "2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="not strictly earlier"):
        assert_no_target_leakage(target, [same_time])


def test_assert_no_target_leakage_raises_when_target_has_no_timestamp() -> None:
    target = _event("e1", "AAPL", None)
    with pytest.raises(ValueError, match="no parseable event_datetime"):
        assert_no_target_leakage(target, [])


def test_assert_no_target_leakage_raises_when_source_has_no_timestamp() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    undated = _event("e2", "AAPL", None)
    with pytest.raises(ValueError, match="not strictly earlier"):
        assert_no_target_leakage(target, [undated])


# ----- compute_historical_features: minimum-history rules -----------------


def test_first_event_in_a_timeline_has_no_history() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    features = compute_historical_features(target, [target])
    assert features.previous_car1 is None
    assert features.rolling_mean_car1 is None
    assert features.rolling_car1_volatility is None
    assert features.previous_earnings_surprise is None
    assert features.rolling_mean_surprise is None
    assert features.number_of_previous_positive_surprises == 0
    assert features.historical_reaction_asymmetry is None
    assert features.number_of_prior_earnings_events == 0
    assert features.provenance == {}


def test_previous_car1_uses_most_recent_qualifying_prior_event() -> None:
    e1 = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.01)
    e2 = _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=0.02)
    target = _event("e3", "AAPL", "2026-03-01T00:00:00+00:00")
    features = compute_historical_features(target, [e1, e2, target])
    assert features.previous_car1 == 0.02  # e2, not e1
    (record,) = features.provenance["previous_car1"]
    assert record.source_event_id == "e2"
    assert record.target_event_id == "e3"


def test_previous_car1_skips_prior_events_without_realized_car1() -> None:
    e1 = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.01)
    e2 = _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=None)  # unrealized
    target = _event("e3", "AAPL", "2026-03-01T00:00:00+00:00")
    features = compute_historical_features(target, [e1, e2, target])
    assert features.previous_car1 == 0.01  # e2 skipped, falls back to e1


def test_rolling_mean_and_volatility_use_trailing_window_only() -> None:
    events = [
        _event(f"e{i}", "AAPL", f"2026-0{i}-01T00:00:00+00:00", car1=float(i))
        for i in range(1, 6)  # car1 = 1.0 .. 5.0
    ]
    target = _event("target", "AAPL", "2026-06-01T00:00:00+00:00")
    features = compute_historical_features(target, [*events, target], window=3)
    # Trailing window of 3 -> car1 values [3.0, 4.0, 5.0]
    assert features.rolling_mean_car1 == pytest.approx(4.0)
    assert len(features.provenance["rolling_mean_car1"]) == 3
    assert {r.source_event_id for r in features.provenance["rolling_mean_car1"]} == {"e3", "e4", "e5"}


def test_rolling_car1_volatility_requires_at_least_two_observations() -> None:
    only_one = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.05)
    target = _event("e2", "AAPL", "2026-02-01T00:00:00+00:00")
    features = compute_historical_features(target, [only_one, target])
    assert features.rolling_car1_volatility is None  # n=1, undefined


def test_number_of_previous_positive_surprises_counts_within_window() -> None:
    events = [
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", surprise=0.01),
        _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", surprise=-0.02),
        _event("e3", "AAPL", "2026-03-01T00:00:00+00:00", surprise=0.03),
    ]
    target = _event("target", "AAPL", "2026-04-01T00:00:00+00:00")
    features = compute_historical_features(target, [*events, target], window=DEFAULT_WINDOW)
    assert features.number_of_previous_positive_surprises == 2  # e1, e3


def test_historical_reaction_asymmetry_needs_both_signs() -> None:
    only_positive = [
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.05),
        _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=0.03),
    ]
    target = _event("target", "AAPL", "2026-03-01T00:00:00+00:00")
    features = compute_historical_features(target, [*only_positive, target])
    assert features.historical_reaction_asymmetry is None  # no negative observation


def test_historical_reaction_asymmetry_computes_when_both_signs_present() -> None:
    events = [
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.10),
        _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=-0.04),
        _event("e3", "AAPL", "2026-03-01T00:00:00+00:00", car1=-0.06),
    ]
    target = _event("target", "AAPL", "2026-04-01T00:00:00+00:00")
    features = compute_historical_features(target, [*events, target])
    # mean(positive) = 0.10; mean(negative) = -0.05; asymmetry = 0.10 - 0.05 = 0.05
    assert features.historical_reaction_asymmetry == pytest.approx(0.05)
    # Uses the FULL history, not a window - all 3 prior events contribute.
    assert len(features.provenance["historical_reaction_asymmetry"]) == 3


def test_number_of_prior_earnings_events_counts_existence_regardless_of_outcome() -> None:
    events = [
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.01),
        _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=None),  # no realized outcome
    ]
    target = _event("target", "AAPL", "2026-03-01T00:00:00+00:00")
    features = compute_historical_features(target, [*events, target])
    assert features.number_of_prior_earnings_events == 2  # counts both, unlike previous_car1


# ----- provenance invariants -----------------------------------------------


def test_every_provenance_record_is_strictly_earlier_than_target() -> None:
    events = [
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.01, surprise=0.02),
        _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=-0.02, surprise=-0.01),
    ]
    target = _event("target", "AAPL", "2026-03-01T00:00:00+00:00")
    features = compute_historical_features(target, [*events, target])

    target_dt = datetime.fromisoformat(target.event_datetime)
    all_records: list[ProvenanceRecord] = [r for recs in features.provenance.values() for r in recs]
    assert all_records  # sanity: this scenario does produce provenance
    for record in all_records:
        assert record.source_event_id != target.event_id
        source_dt = datetime.fromisoformat(record.source_event_datetime)
        assert source_dt < target_dt
        assert record.target_event_id == target.event_id


def test_feature_values_contain_no_forbidden_keys() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    features = compute_historical_features(target, [target])
    assert_feature_is_leakage_free(features)  # must not raise
    values = features.feature_values()
    for forbidden in ("car1", "earnings_surprise", "event_returns", "surprise", "y", "predicted_percentile"):
        assert forbidden not in values


def test_as_dict_includes_ticker_and_feature_values_only() -> None:
    target = _event("e1", "AAPL", "2026-01-01T00:00:00+00:00")
    features = compute_historical_features(target, [target])
    d = features.as_dict()
    assert d["ticker"] == "AAPL"
    assert "provenance" not in d
    assert "target_event_id" not in d


# ----- build_feature_store: batch, cross-ticker isolation ------------------


def test_build_feature_store_returns_one_row_per_input_event() -> None:
    events = [
        _event("e1", "AAPL", "2026-01-01T00:00:00+00:00", car1=0.01),
        _event("e2", "AAPL", "2026-02-01T00:00:00+00:00", car1=0.02),
        _event("e3", "MSFT", "2026-01-15T00:00:00+00:00", car1=-0.01),
    ]
    store = build_feature_store(events)
    assert len(store) == 3
    assert {row.target_event_id for row in store} == {"e1", "e2", "e3"}


def test_build_feature_store_never_leaks_across_tickers() -> None:
    # MSFT's only prior event is chronologically before AAPL's target, but on
    # a DIFFERENT ticker - it must never appear in AAPL's provenance.
    msft = _event("msft1", "MSFT", "2026-01-01T00:00:00+00:00", car1=0.99)
    aapl_target = _event("aapl1", "AAPL", "2026-02-01T00:00:00+00:00")
    store = build_feature_store([msft, aapl_target])

    aapl_row = next(r for r in store if r.target_event_id == "aapl1")
    assert aapl_row.number_of_prior_earnings_events == 0
    assert aapl_row.previous_car1 is None
    all_sources = [r for recs in aapl_row.provenance.values() for r in recs]
    assert all(r.source_ticker == "AAPL" for r in all_sources)


def test_build_feature_store_spans_quarters_for_the_same_ticker() -> None:
    # A target in a later quarter must be able to see an earlier quarter's
    # realized outcome for the same ticker - this is the whole point.
    q4 = HistoricalEvent(
        event_id="q4", ticker="AAPL", event_type="EARNINGS_RELEASE",
        event_datetime="2025-10-09T10:00:00+00:00", car1=0.05, quarter="2025Q4",
    )
    q1 = HistoricalEvent(
        event_id="q1", ticker="AAPL", event_type="EARNINGS_RELEASE",
        event_datetime="2026-01-15T10:00:00+00:00", quarter="2026Q1",
    )
    store = build_feature_store([q4, q1])
    q1_row = next(r for r in store if r.target_event_id == "q1")
    assert q1_row.previous_car1 == 0.05
    (record,) = q1_row.provenance["previous_car1"]
    assert record.source_event_id == "q4"


# ----- Integration: the real, downloaded historical archive ----------------


def test_build_feature_store_on_real_archive_has_no_leakage() -> None:
    events = load_historical_events()
    if not events:
        pytest.skip("data/historical/ is empty in this environment - no real archive to sweep")

    store = build_feature_store(events)
    assert len(store) == len(events)

    violations = 0
    checked_provenance = 0
    for row in store:
        target_dt = datetime.fromisoformat(row.target_event_datetime)
        for records in row.provenance.values():
            for record in records:
                checked_provenance += 1
                if record.source_event_id == row.target_event_id:
                    violations += 1
                    continue
                source_dt = datetime.fromisoformat(record.source_event_datetime)
                if not source_dt < target_dt:
                    violations += 1
                if record.source_ticker != row.ticker:
                    violations += 1

    assert checked_provenance > 0  # sanity: the real archive does produce provenance
    assert violations == 0
