"""Backtesting: within-quarter percentile ranking, leakage guards, temporal split."""

from __future__ import annotations

import pytest

from explaining_markets.backtest import (
    build_training_rows,
    percentile_ranks,
    run_backtest,
    temporal_split,
)
from explaining_markets.features import FeatureVector
from explaining_markets.historical import HistoricalEvent
from explaining_markets.model import BaselineModel, HeuristicFactModel


def _event(
    event_id: str,
    ticker: str,
    quarter: str,
    *,
    car1=None,
    surprise=None,
    disclosure=None,
) -> HistoricalEvent:
    return HistoricalEvent(
        event_id=event_id,
        ticker=ticker,
        event_type="EARNINGS_RELEASE",
        event_datetime="2025-07-31T21:00:00Z",
        disclosure=disclosure or [],
        car1=car1,
        earnings_surprise=surprise,
        quarter=quarter,
    )


# ----- percentile_ranks -------------------------------------------------


def test_percentile_ranks_basic() -> None:
    assert percentile_ranks([]) == []
    assert percentile_ranks([42.0]) == [0.5]
    assert percentile_ranks([30.0, 10.0, 20.0]) == [1.0, 0.0, 0.5]


def test_percentile_ranks_ties_share_average() -> None:
    assert percentile_ranks([1.0, 1.0, 2.0]) == [0.25, 0.25, 1.0]


# ----- run_backtest: correctness, quarter isolation, missing labels ------


def test_run_backtest_skips_events_without_car1() -> None:
    events = [
        _event("e1", "AAPL", "2025Q3", car1=0.05),
        _event("e2", "MSFT", "2025Q3", car1=None),  # no label -> excluded
    ]
    result = run_backtest(events, BaselineModel())
    assert result.n_obs == 1
    assert result.rows[0].event_id == "e1"


def test_run_backtest_ranks_within_quarter_only() -> None:
    # Two quarters, each internally ranked - the same car1 value's rank must
    # NOT depend on the other quarter's events.
    events = [
        _event("e1", "AAPL", "2025Q3", car1=0.0),
        _event("e2", "MSFT", "2025Q3", car1=1.0),
        _event("e3", "GOOGL", "2025Q4", car1=-5.0),
        _event("e4", "TSLA", "2025Q4", car1=5.0),
    ]
    result = run_backtest(events, BaselineModel())
    realized_by_id = {r.event_id: r.realized_percentile for r in result.rows}
    # Within each quarter, the lower car1 -> 0.0, the higher -> 1.0 - regardless
    # of the other quarter's absolute values.
    assert realized_by_id["e1"] == 0.0
    assert realized_by_id["e2"] == 1.0
    assert realized_by_id["e3"] == 0.0
    assert realized_by_id["e4"] == 1.0


def test_run_backtest_constant_predictions_have_no_correlation() -> None:
    events = [
        _event("e1", "AAPL", "2025Q3", car1=-0.1),
        _event("e2", "MSFT", "2025Q3", car1=0.0),
        _event("e3", "GOOGL", "2025Q3", car1=0.1),
    ]
    result = run_backtest(events, BaselineModel())
    # A constant-0.5 model has zero variance in its predictions -> correlation
    # is undefined (None), matching the competition's own degenerate-fit guard.
    assert result.correlation is None


def test_run_backtest_heuristic_model_correlates_with_sentiment_aligned_labels() -> None:
    # Construct disclosures whose sentiment direction matches the realized
    # car1 direction, so a correctly-wired heuristic model should score
    # positively correlated predictions.
    events = [
        _event("e1", "AAPL", "2025Q3", car1=-0.10, disclosure=["Guidance cut. Demand missed."]),
        _event("e2", "MSFT", "2025Q3", car1=0.0, disclosure=["Results in line."]),
        _event("e3", "GOOGL", "2025Q3", car1=0.10, disclosure=["Revenue beat. Guidance raised."]),
    ]
    result = run_backtest(events, HeuristicFactModel())
    assert result.n_obs == 3
    assert result.correlation is not None
    assert result.correlation > 0


def test_run_backtest_never_passes_realized_fields_to_the_model() -> None:
    seen_features: list[FeatureVector] = []

    class SpyModel:
        def predict_percentile(self, features: FeatureVector) -> float:
            seen_features.append(features)
            return 0.5

        def fit(self, training_rows):  # noqa: ANN001
            return None

    events = [_event("e1", "AAPL", "2025Q3", car1=0.05, surprise=0.02)]
    run_backtest(events, SpyModel())

    assert len(seen_features) == 1
    d = seen_features[0].as_dict()
    assert "car1" not in d
    assert "earnings_surprise" not in d
    assert "surprise" not in d


def test_run_backtest_empty_input_returns_empty_result() -> None:
    result = run_backtest([], BaselineModel())
    assert result.n_obs == 0
    assert result.correlation is None
    assert result.mean_abs_error is None
    assert result.beats_surprise_benchmark is None


def test_run_backtest_surprise_benchmark_is_populated_when_available() -> None:
    events = [
        _event("e1", "AAPL", "2025Q3", car1=-0.1, surprise=-0.05),
        _event("e2", "MSFT", "2025Q3", car1=0.0, surprise=0.0),
        _event("e3", "GOOGL", "2025Q3", car1=0.1, surprise=0.05),
    ]
    result = run_backtest(events, HeuristicFactModel())
    assert all(r.surprise_percentile is not None for r in result.rows)


# ----- temporal_split: chronological, never random ------------------------


def test_temporal_split_holds_out_the_most_recent_quarter() -> None:
    events = [
        _event("e1", "AAPL", "2025Q1", car1=0.01),
        _event("e2", "MSFT", "2025Q2", car1=0.02),
        _event("e3", "GOOGL", "2025Q3", car1=0.03),
    ]
    train, test = temporal_split(events, holdout_quarters=1)
    assert {e.quarter for e in train} == {"2025Q1", "2025Q2"}
    assert {e.quarter for e in test} == {"2025Q3"}


def test_temporal_split_excludes_events_with_no_quarter() -> None:
    events = [
        _event("e1", "AAPL", "2025Q1", car1=0.01),
        HistoricalEvent(event_id="e2", ticker="MSFT", event_type="EARNINGS_RELEASE", quarter=None),
    ]
    train, test = temporal_split(events, holdout_quarters=0)
    all_ids = {e.event_id for e in train} | {e.event_id for e in test}
    assert "e2" not in all_ids


def test_temporal_split_with_too_few_quarters_returns_everything_as_train() -> None:
    events = [_event("e1", "AAPL", "2025Q1", car1=0.01)]
    train, test = temporal_split(events, holdout_quarters=1)
    assert len(train) == 1
    assert test == []


# ----- build_training_rows -------------------------------------------------


def test_build_training_rows_pairs_features_with_realized_percentile() -> None:
    events = [
        _event("e1", "AAPL", "2025Q3", car1=0.0),
        _event("e2", "MSFT", "2025Q3", car1=1.0),
    ]
    rows = build_training_rows(events)
    assert len(rows) == 2
    labels = sorted(y for _features, y in rows)
    assert labels == [0.0, 1.0]
    for features, _y in rows:
        assert isinstance(features, FeatureVector)


def test_build_training_rows_train_test_never_overlap_quarters() -> None:
    events = [
        _event("e1", "AAPL", "2025Q1", car1=0.01),
        _event("e2", "MSFT", "2025Q2", car1=0.02),
        _event("e3", "GOOGL", "2025Q3", car1=0.03),
    ]
    train, test = temporal_split(events, holdout_quarters=1)
    train_rows = build_training_rows(train)
    # The held-out quarter's event must never contribute a training row.
    assert len(train_rows) == 2
    assert not any(f.ticker == "GOOGL" for f, _y in train_rows)
