"""Tests for V3-lite chronological training and ablation evaluation."""
from __future__ import annotations

import json
import math

import pytest

import explaining_markets.v3_lite_training as v3lt
from explaining_markets.v3_lite_training import (
    CLIP_BOUNDS,
    coverage_buckets,
    evaluate_v3_lite,
    fit_predict,
    format_summary,
    metric_block,
    write_report,
)
from explaining_markets.v3_training import (
    HONEST_HOLDOUT_QUARTER,
    LEGACY_HOLDOUT_QUARTER,
    TRAIN_QUARTER,
    VALIDATION_QUARTER,
    V3TrainingRow,
)


def _make_row(event_id, ticker, quarter, target, values, surprise=None, leakage=0):
    return V3TrainingRow(
        event_id=event_id, ticker=ticker, quarter=quarter,
        target_percentile=target, values=values,
        surprise_percentile=surprise, leakage_violations=leakage,
    )


SYNTHETIC_ABLATIONS = {
    "v1_fls_only": ("f1", "f2"),
    "fls_plus_reasoning": ("f1", "f2", "has_reasoning"),
    "fls_plus_availability": ("f1", "f2", "has_reasoning", "has_company_earnings_history", "has_5y_price_history"),
    "full_v3_available": ("f1", "f2", "has_reasoning", "has_company_earnings_history", "has_5y_price_history"),
}


@pytest.fixture
def synthetic_ablations(monkeypatch):
    """Replace production ABLATIONS with synthetic feature sets for testing."""
    monkeypatch.setattr(v3lt, "ABLATIONS", SYNTHETIC_ABLATIONS)
    monkeypatch.setattr(v3lt, "MODEL_FEATURE_NAMES", ("f1", "f2"))
    monkeypatch.setattr(v3lt, "AVAILABILITY_NAMES", ("has_reasoning", "has_company_earnings_history", "has_5y_price_history"))


def _synthetic_rows(n_train=60, n_val=40, n_legacy=30, feature_names=("f1", "f2")):
    """Build deterministic synthetic rows with signal in f1."""
    rows = []
    for i in range(n_train):
        f1 = (i % 10) / 10.0
        f2 = (i % 5) / 5.0
        target = 0.3 + 0.4 * f1 + 0.01 * (i % 7)
        vals = {"f1": f1, "f2": f2, "has_reasoning": 1.0 if i % 3 == 0 else 0.0,
                "has_company_earnings_history": 1.0 if i % 2 == 0 else 0.0,
                "has_5y_price_history": 1.0 if i % 4 == 0 else 0.0}
        rows.append(_make_row(f"tr_{i}", f"T{i}", TRAIN_QUARTER, target, vals, surprise=f1))
    for i in range(n_val):
        f1 = (i % 10) / 10.0
        f2 = (i % 5) / 5.0
        target = 0.3 + 0.4 * f1 + 0.01 * (i % 7)
        vals = {"f1": f1, "f2": f2, "has_reasoning": 1.0 if i % 3 == 0 else 0.0,
                "has_company_earnings_history": 1.0 if i % 2 == 0 else 0.0,
                "has_5y_price_history": 1.0 if i % 4 == 0 else 0.0}
        rows.append(_make_row(f"va_{i}", f"V{i}", VALIDATION_QUARTER, target, vals, surprise=f1))
    for i in range(n_legacy):
        f1 = (i % 10) / 10.0
        f2 = (i % 5) / 5.0
        target = 0.3 + 0.4 * f1 + 0.01 * (i % 7)
        vals = {"f1": f1, "f2": f2, "has_reasoning": 1.0 if i % 3 == 0 else 0.0,
                "has_company_earnings_history": 1.0 if i % 2 == 0 else 0.0,
                "has_5y_price_history": 1.0 if i % 4 == 0 else 0.0}
        rows.append(_make_row(f"le_{i}", f"L{i}", LEGACY_HOLDOUT_QUARTER, target, vals, surprise=f1))
    return rows


def test_fit_predict_uses_train_only():
    rows = _synthetic_rows()
    train = [r for r in rows if r.quarter == TRAIN_QUARTER]
    val = [r for r in rows if r.quarter == VALIDATION_QUARTER]
    fit = fit_predict(train, val, ("f1", "f2"), "ridge", {"alpha": 10.0})
    assert len(fit.predictions) == len(val)
    assert all(0.05 <= p <= 0.95 for p in fit.predictions)
    assert fit.feature_names == ("f1", "f2")


def test_fit_predict_clips_to_bounds():
    rows = _synthetic_rows()
    train = [r for r in rows if r.quarter == TRAIN_QUARTER]
    val = [r for r in rows if r.quarter == VALIDATION_QUARTER]
    fit = fit_predict(train, val, ("f1", "f2"), "ridge", {"alpha": 0.001})
    assert all(CLIP_BOUNDS[0] <= p <= CLIP_BOUNDS[1] for p in fit.predictions)


def test_metric_block_computes_correlations():
    rows = _synthetic_rows(n_val=20)
    val = [r for r in rows if r.quarter == VALIDATION_QUARTER]
    preds = [r.target_percentile for r in val]  # perfect prediction
    block = metric_block(preds, val)
    assert block["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert block["spearman"] == pytest.approx(1.0, abs=1e-6)
    assert block["mae"] == pytest.approx(0.0, abs=1e-6)


def test_metric_block_constant_predictions():
    rows = _synthetic_rows(n_val=10)
    val = [r for r in rows if r.quarter == VALIDATION_QUARTER]
    block = metric_block([0.5] * len(val), val)
    assert block["pearson"] is None
    assert block["spearman"] is None


def test_coverage_buckets_partition_rows():
    rows = _synthetic_rows(n_val=30)
    val = [r for r in rows if r.quarter == VALIDATION_QUARTER]
    buckets = coverage_buckets(val)
    total = sum(len(v) for v in buckets.values())
    assert total == len(val)
    # Every index appears exactly once
    all_indices = sorted(idx for indices in buckets.values() for idx in indices)
    assert all_indices == list(range(len(val)))


def test_evaluate_v3_lite_runs_on_synthetic(synthetic_ablations):
    rows = _synthetic_rows()
    results, calibrator, fit = evaluate_v3_lite(rows, include_nonlinear=False)
    assert results["n_train"] == 60
    assert results["n_validation"] == 40
    assert results["n_legacy_evaluation"] == 30
    assert results["n_honest_holdout"] == 0
    assert results["honest_holdout_available"] is False
    assert results["legacy_holdout_is_pristine"] is False
    assert "selected_ablation" in results
    assert "calibration" in results
    assert results["calibration"]["preserves_ranking"] in (True, False)


def test_evaluate_v3_lite_rejects_leakage_rows(synthetic_ablations):
    rows = _synthetic_rows()
    rows[0] = _make_row(rows[0].event_id, rows[0].ticker, TRAIN_QUARTER,
                        rows[0].target_percentile, rows[0].values, leakage=1)
    with pytest.raises(ValueError, match="audit violations"):
        evaluate_v3_lite(rows, include_nonlinear=False)


def test_evaluate_v3_lite_requires_train_and_validation():
    rows = [r for r in _synthetic_rows() if r.quarter != TRAIN_QUARTER]
    with pytest.raises(RuntimeError, match="requires 2025Q4 train"):
        evaluate_v3_lite(rows, include_nonlinear=False)


def test_calibration_preserves_spearman_on_synthetic(synthetic_ablations):
    rows = _synthetic_rows()
    results, calibrator, fit = evaluate_v3_lite(rows, include_nonlinear=False)
    raw_spear = results["calibration"]["validation_raw"]["spearman"]
    cal_spear = results["calibration"]["validation_calibrated"]["spearman"]
    if raw_spear is not None and cal_spear is not None:
        # Monotonic calibration must preserve ranking (up to clamp tolerance)
        assert abs(cal_spear - raw_spear) < 1e-4


def test_write_report_creates_json(tmp_path, synthetic_ablations):
    rows = _synthetic_rows()
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    path = write_report(results, tmp_path / "report.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["n_train"] == 60
    assert "ablations" in payload


def test_format_summary_contains_key_sections(synthetic_ablations):
    rows = _synthetic_rows()
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    summary = format_summary(results)
    assert "V3-LITE CHRONOLOGICAL EVALUATION" in summary
    assert "selected ablation" in summary
    assert "calibration" in summary.lower()


def test_legacy_evaluation_includes_v1_comparison(synthetic_ablations):
    rows = _synthetic_rows()
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    assert "legacy_evaluation" in results
    legacy = results["legacy_evaluation"]
    assert "v1_raw" in legacy
    assert "selected_raw" in legacy
    assert "selected_calibrated" in legacy


def test_ablations_include_v1_baseline():
    assert "v1_fls_only" in v3lt.ABLATIONS
    assert "full_v3_available" in v3lt.ABLATIONS


def test_honest_holdout_declared_but_unavailable(synthetic_ablations):
    rows = _synthetic_rows()
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    assert results["honest_holdout_quarter"] == HONEST_HOLDOUT_QUARTER
    assert results["honest_holdout_available"] is False


# ---------------------------------------------------------------------------
# Promotion gate tests
# ---------------------------------------------------------------------------

from explaining_markets.v3_lite_training import (
    DEFAULT_V3_LITE_ARTIFACT,
    V3_LITE_PROMOTION_GATE,
    evaluate_promotion_gate,
    serialize_v3_lite_artifact,
)


def test_promotion_gate_fails_without_honest_holdout(synthetic_ablations):
    rows = _synthetic_rows()
    results, cal, fit = evaluate_v3_lite(rows, include_nonlinear=False)
    promotion = evaluate_promotion_gate(results, tests_passing=True, latency_ok=True)
    assert promotion["promoted"] is False
    assert promotion["checks"]["honest_holdout_available"] is False


def test_promotion_gate_fails_without_tests_passing(synthetic_ablations):
    rows = _synthetic_rows()
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    promotion = evaluate_promotion_gate(results, tests_passing=False)
    assert promotion["promoted"] is False
    assert promotion["checks"]["tests_passing"] is False


def test_promotion_gate_records_evidence(synthetic_ablations):
    rows = _synthetic_rows()
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    promotion = evaluate_promotion_gate(results)
    assert "evidence" in promotion
    assert "validation_pearson_gain_over_v1" in promotion["evidence"]
    assert "honest_holdout_available" in promotion["evidence"]
    assert "calibration_preserves_ranking" in promotion["evidence"]


def test_promotion_gate_has_predeclared_thresholds():
    assert V3_LITE_PROMOTION_GATE["min_validation_pearson_gain_over_v1"] == 0.01
    assert V3_LITE_PROMOTION_GATE["max_honest_holdout_pearson_regression"] == 0.01
    assert V3_LITE_PROMOTION_GATE["require_honest_holdout_available"] is True


def test_serialize_refuses_when_not_promoted(synthetic_ablations, tmp_path):
    rows = _synthetic_rows()
    results, cal, fit = evaluate_v3_lite(rows, include_nonlinear=False)
    promotion = evaluate_promotion_gate(results)
    assert promotion["promoted"] is False
    with pytest.raises(RuntimeError, match="refusing to serialize"):
        serialize_v3_lite_artifact(rows, results, cal, fit, promotion, tmp_path / "artifact.json")
