"""Tests for the historical percentile calibration layer."""
from __future__ import annotations

import json
import math

import pytest

from explaining_markets.calibration import (
    CALIBRATION_METHOD,
    CALIBRATION_VERSION,
    PercentileCalibrator,
    is_monotonic,
    spearman,
)


def test_fit_requires_predictions():
    with pytest.raises(ValueError, match="zero predictions"):
        PercentileCalibrator.fit([], source="test")


def test_fit_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        PercentileCalibrator.fit([0.1, float("nan"), 0.3], source="test")


def test_knots_are_ascending():
    cal = PercentileCalibrator.fit([0.3, 0.1, 0.2], source="test")
    assert cal.knots == (0.1, 0.2, 0.3)


def test_calibrate_is_monotonic():
    cal = PercentileCalibrator.fit([0.1, 0.2, 0.3, 0.4, 0.5], source="test")
    assert is_monotonic(cal)
    grid = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    outputs = [cal.calibrate(x) for x in grid]
    assert all(b >= a - 1e-12 for a, b in zip(outputs, outputs[1:]))


def test_calibrate_preserves_spearman():
    """Monotonic transform must not change rank correlation."""
    raw = [0.30, 0.45, 0.50, 0.55, 0.62, 0.71, 0.38, 0.49]
    target = [0.20, 0.55, 0.60, 0.40, 0.80, 0.90, 0.30, 0.50]
    cal = PercentileCalibrator.fit(raw, source="test")
    calibrated = cal.calibrate_many(raw)
    assert spearman(raw, target) == pytest.approx(spearman(calibrated, target), abs=1e-9)


def test_calibrate_midrank_tie_handling():
    """Tied raw scores get the mid-rank CDF position.

    For knots [0.1, 0.2, 0.2, 0.2, 0.5] and score 0.2:
      below = 1 (just 0.1), equal = 3
      mid-rank CDF = (1 + 0.5*3) / 5 = 2.5/5 = 0.5
    This is the standard mid-point between left-continuous and right-continuous CDFs.
    """
    cal = PercentileCalibrator.fit([0.1, 0.2, 0.2, 0.2, 0.5], source="test")
    assert cal.raw_percentile(0.2) == pytest.approx(0.5)


def test_calibrate_below_min_is_zero():
    cal = PercentileCalibrator.fit([0.2, 0.3, 0.4], source="test")
    assert cal.raw_percentile(0.1) == 0.0


def test_calibrate_above_max_is_one():
    cal = PercentileCalibrator.fit([0.2, 0.3, 0.4], source="test")
    assert cal.raw_percentile(0.5) == 1.0


def test_calibrate_clamps_to_bounds():
    cal = PercentileCalibrator.fit([0.2, 0.3, 0.4], source="test", bounds=(0.05, 0.95))
    assert cal.calibrate(0.1) == 0.05
    assert cal.calibrate(0.5) == 0.95


def test_calibrate_output_in_bounds():
    cal = PercentileCalibrator.fit([0.2, 0.3, 0.4], source="test", bounds=(0.01, 0.99))
    for x in [-1.0, 0.0, 0.25, 0.35, 0.45, 1.0, 100.0]:
        result = cal.calibrate(x)
        assert 0.01 <= result <= 0.99


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError, match="invalid calibration bounds"):
        PercentileCalibrator(knots=(0.1,), bounds=(0.5, 0.5))


def test_empty_knots_rejected():
    with pytest.raises(ValueError, match="at least one"):
        PercentileCalibrator(knots=())


def test_non_ascending_knots_rejected():
    with pytest.raises(ValueError, match="ascending"):
        PercentileCalibrator(knots=(0.3, 0.1, 0.2))


def test_serialization_roundtrip():
    cal = PercentileCalibrator.fit(
        [0.15, 0.25, 0.35, 0.45, 0.55], source="validation_2026Q1", bounds=(0.02, 0.98)
    )
    payload = cal.as_dict()
    assert payload["method"] == CALIBRATION_METHOD
    assert payload["version"] == CALIBRATION_VERSION
    assert payload["source"] == "validation_2026Q1"
    assert payload["n_fitted"] == 5
    assert payload["n_knots"] == 5
    restored = PercentileCalibrator.from_dict(payload)
    assert restored.knots == cal.knots
    assert restored.bounds == cal.bounds
    assert restored.source == cal.source
    assert restored.n_fitted == cal.n_fitted
    # Calibration values must match exactly
    for x in [0.1, 0.3, 0.5, 0.7]:
        assert restored.calibrate(x) == cal.calibrate(x)


def test_thinning_preserves_extremes():
    """When knots exceed max_knots, thinning keeps min and max."""
    values = [i / 10000.0 for i in range(10000)]
    cal = PercentileCalibrator.fit(values, source="test", max_knots=100)
    assert cal.n_fitted == 10000
    assert len(cal.knots) == 100
    assert cal.knots[0] == pytest.approx(0.0)
    assert cal.knots[-1] == pytest.approx(0.9999)


def test_source_provenance_preserved():
    cal = PercentileCalibrator.fit([0.3, 0.5], source="2026Q1 validation, model on 2025Q4")
    assert "2026Q1" in cal.source
    assert "2025Q4" in cal.source


def test_spearman_handles_ties():
    a = [1.0, 2.0, 2.0, 3.0]
    b = [1.0, 2.0, 2.0, 3.0]
    assert spearman(a, b) == pytest.approx(1.0)


def test_spearman_constant_returns_none():
    assert spearman([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]) is None


def test_spearman_length_mismatch_returns_none():
    assert spearman([0.1, 0.2], [0.1, 0.2, 0.3]) is None
