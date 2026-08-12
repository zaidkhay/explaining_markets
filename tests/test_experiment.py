"""Experiment layer: model selection, leakage guards, chronological split.

These tests cover the research-only orchestration in ``experiment.py`` —
specifically the bugs and invariants called out in
``HISTORICAL_MODEL_EXPERIMENT.md``:

* model selection must always return a usable hyperparameter (never
  ``alpha=None``), even when validation correlation is undefined for every
  candidate (the cold-start case that originally crashed the pipeline);
* every built feature matrix must be free of the forbidden realized fields
  (``car1``, ``earnings_surprise``, ``surprise``, ``y``,
  ``predicted_percentile``);
* the chronological split must keep ``2026Q2`` rows out of TRAIN and
  VALIDATION, and TRAIN+VALIDATION rows out of the LOCKED HOLDOUT;
* permutation ablation must be deterministic for a fixed seed and must
  return a non-negative baseline correlation when the model has signal.

The tests use small synthetic datasets (no network, no archive files) so
they run quickly and deterministically anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from explaining_markets.experiment import (
    DISCLOSURE_NUMERIC_FIELDS,
    FEATURE_FAMILIES,
    HISTORICAL_ALL_FIELDS,
    HISTORICAL_IMPUTED_FIELDS,
    QUARTER_HOLDOUT,
    QUARTER_TRAIN,
    QUARTER_VALIDATION,
    ExperimentRow,
    MatrixBuilder,
    RidgeQuantModel,
    RandomForestQuantModel,
    assert_matrix_is_leakage_free,
    build_experiment_rows,
    compute_metrics,
    permutation_ablation,
    run_experiment,
)
from explaining_markets.features import FORBIDDEN_KEYS
from explaining_markets.historical import HistoricalEvent, load_historical_events


# ----------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------


def _event(
    event_id: str,
    ticker: str,
    quarter: str,
    *,
    car1: float | None = 0.0,
    surprise: float | None = 0.0,
    disclosure: list[str] | None = None,
    event_datetime: str = "2025-07-31T21:00:00Z",
) -> HistoricalEvent:
    return HistoricalEvent(
        event_id=event_id,
        ticker=ticker,
        event_type="EARNINGS_RELEASE",
        event_datetime=event_datetime,
        disclosure=disclosure or ["Revenue beat. Guidance raised."],
        car1=car1,
        earnings_surprise=surprise,
        quarter=quarter,
    )


def _row(
    event_id: str,
    quarter: str,
    *,
    y: float = 0.5,
    disclosure: dict | None = None,
    historical: dict | None = None,
) -> ExperimentRow:
    base_disclosure = {
        "ticker": "AAPL",
        "event_type": "EARNINGS_RELEASE",
        "n_facts": 1,
        "text_length": 10,
        "positive_hits": 1,
        "negative_hits": 0,
        "net_sentiment": 1,
        "has_guidance_mention": True,
    }
    if disclosure:
        base_disclosure.update(disclosure)
    base_historical = {name: None for name in HISTORICAL_ALL_FIELDS}
    base_historical["number_of_previous_positive_surprises"] = 0
    base_historical["number_of_prior_earnings_events"] = 0
    if historical:
        base_historical.update(historical)
    return ExperimentRow(
        event_id=event_id,
        ticker="AAPL",
        quarter=quarter,
        disclosure=base_disclosure,
        historical=base_historical,
        y=y,
        surprise_percentile=0.5,
    )


def _synthetic_events_for_run_experiment() -> list[HistoricalEvent]:
    """Three quarters, two tickers, sentiment-aligned labels.

    Enough rows per quarter that within-quarter percentile ranking produces
    non-degenerate targets, and enough history that the historical feature
    store has something to work with for the later quarters.
    """
    events: list[HistoricalEvent] = []
    # 2025Q4: 4 events per ticker, sentiment aligned with car1.
    for i, (sent, car1) in enumerate(
        [("weak demand missed", -0.10), ("in line", 0.0), ("beat raised", 0.05), ("strong beat", 0.10)]
    ):
        events.append(_event(f"q4a{i}", "AAPL", QUARTER_TRAIN, car1=car1, disclosure=[sent]))
        events.append(_event(f"q4b{i}", "MSFT", QUARTER_TRAIN, car1=car1, disclosure=[sent]))
    # 2026Q1: same shape, later datetime so 2025Q4 events become history.
    for i, (sent, car1) in enumerate(
        [("weak demand missed", -0.08), ("in line", 0.01), ("beat raised", 0.06), ("strong beat", 0.12)]
    ):
        events.append(
            _event(
                f"q1a{i}",
                "AAPL",
                QUARTER_VALIDATION,
                car1=car1,
                disclosure=[sent],
                event_datetime="2026-02-15T21:00:00Z",
            )
        )
        events.append(
            _event(
                f"q1b{i}",
                "MSFT",
                QUARTER_VALIDATION,
                car1=car1,
                disclosure=[sent],
                event_datetime="2026-02-15T21:00:00Z",
            )
        )
    # 2026Q2 (LOCKED HOLDOUT): same shape again, even later datetime.
    for i, (sent, car1) in enumerate(
        [("weak demand missed", -0.09), ("in line", 0.02), ("beat raised", 0.07), ("strong beat", 0.13)]
    ):
        events.append(
            _event(
                f"q2a{i}",
                "AAPL",
                QUARTER_HOLDOUT,
                car1=car1,
                disclosure=[sent],
                event_datetime="2026-05-15T21:00:00Z",
            )
        )
        events.append(
            _event(
                f"q2b{i}",
                "MSFT",
                QUARTER_HOLDOUT,
                car1=car1,
                disclosure=[sent],
                event_datetime="2026-05-15T21:00:00Z",
            )
        )
    return events


# ----------------------------------------------------------------------
# assert_matrix_is_leakage_free
# ----------------------------------------------------------------------


def test_assert_matrix_is_leakage_free_passes_for_safe_names() -> None:
    assert_matrix_is_leakage_free(list(DISCLOSURE_NUMERIC_FIELDS))
    assert_matrix_is_leakage_free(list(HISTORICAL_ALL_FIELDS))


def test_assert_matrix_is_leakage_free_raises_for_forbidden_names() -> None:
    # Every key in FORBIDDEN_KEYS must trip the guard.
    for forbidden in FORBIDDEN_KEYS:
        with pytest.raises(ValueError, match="leaked realized field"):
            assert_matrix_is_leakage_free([forbidden])


def test_assert_matrix_is_leakage_free_reports_all_offenders() -> None:
    with pytest.raises(ValueError) as excinfo:
        assert_matrix_is_leakage_free(["car1", "earnings_surprise", "n_facts"])
    msg = str(excinfo.value)
    assert "car1" in msg
    assert "earnings_surprise" in msg
    assert "n_facts" not in msg  # safe name is not reported


# ----------------------------------------------------------------------
# MatrixBuilder: imputation, missing-indicators, leakage-free names
# ----------------------------------------------------------------------


def test_matrix_builder_feature_names_include_missing_indicators() -> None:
    builder = MatrixBuilder(include_disclosure=True, include_historical=True)
    names = builder.feature_names()
    assert "n_facts" in names  # disclosure
    assert "previous_car1" in names  # historical
    for name in HISTORICAL_IMPUTED_FIELDS:
        assert f"{name}_was_missing" in names


def test_matrix_builder_fit_then_transform_uses_train_medians_only() -> None:
    # TRAIN has a non-null previous_car1; VALIDATION has all-null. The
    # VALIDATION imputed value must equal the TRAIN median, NOT something
    # computed from VALIDATION itself.
    train = [
        _row("t1", QUARTER_TRAIN, y=0.4, historical={"previous_car1": 0.1}),
        _row("t2", QUARTER_TRAIN, y=0.6, historical={"previous_car1": 0.3}),
    ]
    val = [_row("v1", QUARTER_VALIDATION, y=0.5, historical={"previous_car1": None})]
    builder = MatrixBuilder(include_disclosure=False, include_historical=True).fit(train)
    X_val = builder.transform(val)
    idx = builder.feature_names().index("previous_car1")
    assert X_val[0, idx] == pytest.approx(0.2)  # median(0.1, 0.3)
    missing_idx = builder.feature_names().index("previous_car1_was_missing")
    assert X_val[0, missing_idx] == 1.0


def test_matrix_builder_transform_requires_fit() -> None:
    builder = MatrixBuilder()
    with pytest.raises(RuntimeError, match="called before fit"):
        builder.transform([_row("x", QUARTER_TRAIN)])


def test_matrix_builder_empty_train_falls_back_to_zero_imputation() -> None:
    # TRAIN has zero coverage for every imputed field (cold-start quarter).
    # The fallback must be 0.0, never None, never NaN, and never derived
    # from validation/holdout.
    train = [_row("t1", QUARTER_TRAIN, y=0.5)]  # all historical fields None
    val = [_row("v1", QUARTER_VALIDATION, y=0.5)]
    builder = MatrixBuilder(include_disclosure=False, include_historical=True).fit(train)
    X_val = builder.transform(val)
    for name in HISTORICAL_IMPUTED_FIELDS:
        idx = builder.feature_names().index(name)
        assert X_val[0, idx] == 0.0
        missing_idx = builder.feature_names().index(f"{name}_was_missing")
        assert X_val[0, missing_idx] == 1.0


def test_matrix_builder_names_are_leakage_free_for_all_combinations() -> None:
    for include_disclosure in (True, False):
        for include_historical in (True, False):
            builder = MatrixBuilder(
                include_disclosure=include_disclosure,
                include_historical=include_historical,
            )
            # Must not raise.
            assert_matrix_is_leakage_free(builder.feature_names())


# ----------------------------------------------------------------------
# compute_metrics / _pearson: degenerate-input behavior
# ----------------------------------------------------------------------


def test_compute_metrics_empty_returns_none() -> None:
    m = compute_metrics([], [])
    assert m.n_obs == 0
    assert m.correlation is None
    assert m.mean_abs_error is None


def test_compute_metrics_constant_predictions_have_undefined_correlation() -> None:
    # This is the exact degenerate case that originally caused
    # `alpha=None`: a model that emits constant predictions has zero
    # variance, so Pearson is undefined.
    m = compute_metrics([0.5, 0.5, 0.5], [0.1, 0.5, 0.9])
    assert m.correlation is None
    assert m.mean_abs_error is not None


def test_compute_metrics_aligned_predictions_correlate_positively() -> None:
    m = compute_metrics([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
    assert m.correlation is not None
    assert m.correlation > 0.9


# ----------------------------------------------------------------------
# Model selection: the regression test for the alpha=None bug
# ----------------------------------------------------------------------


def test_ridge_quant_model_predicts_in_unit_interval() -> None:
    # A model fit on a constant target should still produce in-range
    # predictions (Ridge may drift outside [0,1] without the clip).
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = [0.5, 0.5, 0.5, 0.5]
    model = RidgeQuantModel(alpha=1.0).fit(X, y)
    preds = model.predict(np.array([[100.0], [-100.0]]))
    assert preds.shape == (2,)
    assert float(preds.min()) >= 0.0
    assert float(preds.max()) <= 1.0


def test_random_forest_quant_model_predicts_in_unit_interval() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = [0.1, 0.4, 0.6, 0.9]
    model = RandomForestQuantModel(max_depth=2).fit(X, y)
    preds = model.predict(np.array([[100.0], [-100.0]]))
    assert preds.shape == (2,)
    assert float(preds.min()) >= 0.0
    assert float(preds.max()) <= 1.0


def test_run_experiment_never_returns_none_hyperparameters() -> None:
    # Regression for the original crash: when validation correlation is
    # undefined for every Ridge/forest candidate (which happens on the real
    # archive because 2025Q4 is a cold-start quarter), the selection loop
    # must still pick a valid float alpha / int max_depth rather than
    # leaving it as None and crashing the final refit.
    events = _synthetic_events_for_run_experiment()
    report = run_experiment(events)
    selected = report.selected_hyperparameters
    for key in ("ridge_alpha_historical", "ridge_alpha_combined"):
        assert key in selected
        assert selected[key] is not None
        assert isinstance(selected[key], (int, float))
        assert float(selected[key]) > 0.0
    for key in ("forest_max_depth_historical", "forest_max_depth_combined"):
        assert key in selected
        assert selected[key] is not None
        assert int(selected[key]) >= 1


def test_run_experiment_preserves_chronological_split() -> None:
    events = _synthetic_events_for_run_experiment()
    rows = build_experiment_rows(events)
    by_quarter: dict[str, set[str]] = {}
    for row in rows:
        by_quarter.setdefault(row.quarter, set()).add(row.event_id)
    train_ids = by_quarter[QUARTER_TRAIN]
    val_ids = by_quarter[QUARTER_VALIDATION]
    holdout_ids = by_quarter[QUARTER_HOLDOUT]
    # No overlap between any two splits.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert val_ids.isdisjoint(holdout_ids)


def test_run_experiment_holdout_is_populated() -> None:
    events = _synthetic_events_for_run_experiment()
    report = run_experiment(events)
    # The locked holdout must produce metrics for every model family
    # (constant, heuristic, ridge, forest, surprise).
    assert QUARTER_HOLDOUT in report.per_quarter
    holdout = report.per_quarter[QUARTER_HOLDOUT]
    expected = {
        "constant_0.5",
        "disclosure_heuristic",
        "historical_ridge",
        "historical_forest",
        "combined_ridge",
        "combined_forest",
        "surprise_benchmark",
    }
    assert expected.issubset(holdout.keys())
    for name, metrics in holdout.items():
        assert metrics.n_obs > 0


def test_run_experiment_ablation_covers_every_family() -> None:
    events = _synthetic_events_for_run_experiment()
    report = run_experiment(events)
    # Every family in FEATURE_FAMILIES must appear in both ablation dicts.
    for family in FEATURE_FAMILIES:
        assert family in report.ablation_validation
        assert family in report.ablation_holdout


# ----------------------------------------------------------------------
# permutation_ablation: determinism + baseline correlation
# ----------------------------------------------------------------------


def test_permutation_ablation_is_deterministic_for_fixed_seed() -> None:
    rows = [
        _row(
            f"r{i}",
            QUARTER_TRAIN,
            y=float(i) / 10.0,
            historical={"previous_car1": float(i) / 10.0, "number_of_prior_earnings_events": i},
        )
        for i in range(10)
    ]
    builder = MatrixBuilder(include_disclosure=False, include_historical=True).fit(rows)
    X = builder.transform(rows)
    y = [r.y for r in rows]
    model = RidgeQuantModel(alpha=1.0).fit(X, y)
    families = {"historical_car1": ("previous_car1",)}
    first = permutation_ablation(model=model, X=X, y=y, feature_names=builder.feature_names(), families=families, seed=7)
    second = permutation_ablation(model=model, X=X, y=y, feature_names=builder.feature_names(), families=families, seed=7)
    assert first == second


def test_permutation_ablation_shuffling_a_predictive_family_drops_correlation() -> None:
    # previous_car1 is perfectly aligned with y; shuffling it must not
    # improve correlation, and typically drops it noticeably.
    rows = [
        _row(f"r{i}", QUARTER_TRAIN, y=float(i) / 10.0, historical={"previous_car1": float(i) / 10.0})
        for i in range(20)
    ]
    builder = MatrixBuilder(include_disclosure=False, include_historical=True).fit(rows)
    X = builder.transform(rows)
    y = [r.y for r in rows]
    model = RidgeQuantModel(alpha=1.0).fit(X, y)
    families = {"historical_car1": ("previous_car1",)}
    drops = permutation_ablation(model=model, X=X, y=y, feature_names=builder.feature_names(), families=families, seed=0)
    assert "historical_car1" in drops
    # Shuffling a perfectly-aligned column cannot help.
    assert drops["historical_car1"] >= -1e-9


# ----------------------------------------------------------------------
# Real-archive integration: leakage sweep + chronological split
# ----------------------------------------------------------------------


def test_run_experiment_on_real_archive_has_no_leakage_and_clean_split() -> None:
    events = load_historical_events()
    if not events:
        pytest.skip("data/historical/ is empty in this environment - no real archive to sweep")

    rows = build_experiment_rows(events)
    # 1. No realized field appears in any feature name produced by any
    #    MatrixBuilder configuration.
    for include_disclosure in (True, False):
        for include_historical in (True, False):
            builder = MatrixBuilder(
                include_disclosure=include_disclosure,
                include_historical=include_historical,
            )
            assert_matrix_is_leakage_free(builder.feature_names())

    # 2. The chronological split is strict: each event_id appears in
    #    exactly one quarter bucket.
    seen: dict[str, str] = {}
    for row in rows:
        prev = seen.get(row.event_id)
        assert prev is None or prev == row.quarter
        seen[row.event_id] = row.quarter

    # 3. The locked holdout (2026Q2) is non-empty and disjoint from
    #    TRAIN/VALIDATION.
    by_quarter: dict[str, set[str]] = {}
    for row in rows:
        by_quarter.setdefault(row.quarter, set()).add(row.event_id)
    assert QUARTER_HOLDOUT in by_quarter
    assert by_quarter[QUARTER_HOLDOUT]
    assert by_quarter[QUARTER_HOLDOUT].isdisjoint(by_quarter.get(QUARTER_TRAIN, set()))
    assert by_quarter[QUARTER_HOLDOUT].isdisjoint(by_quarter.get(QUARTER_VALIDATION, set()))


def test_run_experiment_on_real_archive_completes_without_crash() -> None:
    # End-to-end smoke test against the real archive. This is the test that
    # originally surfaced the `alpha=None` crash; keeping it as a regression
    # guard. Skipped (not failed) when the archive is absent.
    events = load_historical_events()
    if not events:
        pytest.skip("data/historical/ is empty in this environment - no real archive to sweep")
    report = run_experiment(events)
    # The holdout must have produced metrics for every model family.
    holdout = report.per_quarter[QUARTER_HOLDOUT]
    assert holdout["constant_0.5"].n_obs > 0
    assert holdout["combined_ridge"].n_obs > 0
    # And every selected hyperparameter must be a usable value (not None).
    for key, value in report.selected_hyperparameters.items():
        assert value is not None
