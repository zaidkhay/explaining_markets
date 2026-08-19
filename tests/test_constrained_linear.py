import numpy as np

from explaining_markets.constrained_linear import (
    DEFAULT_SIGN_CONSTRAINTS,
    fit_sign_constrained_ridge,
)
from explaining_markets.v3_training import V3TrainingRow


def _row(i: int, surprise: float, target: float) -> V3TrainingRow:
    return V3TrainingRow(
        event_id=f"e{i}",
        ticker="TEST",
        quarter="2025Q4",
        target_percentile=target,
        values={"eps_surprise_percent": surprise},
    )


def test_eps_surprise_coefficient_cannot_flip_negative():
    # Deliberately give the fitter data that would prefer a negative slope.
    rows = [
        _row(0, -0.20, 0.8),
        _row(1, 0.00, 0.5),
        _row(2, 0.20, 0.2),
    ]
    fit = fit_sign_constrained_ridge(
        rows,
        rows,
        ("eps_surprise_percent",),
        alpha=1.0,
    )
    assert fit.kind == "constrained_ridge"
    assert fit.coefficients[0] >= -1e-12
    assert np.all(np.isfinite(fit.predictions))


def test_semantic_sign_constraint_map_covers_core_result_direction():
    assert DEFAULT_SIGN_CONSTRAINTS["eps_surprise_percent"] == 1
    assert DEFAULT_SIGN_CONSTRAINTS["revenue_surprise_percent"] == 1
    assert DEFAULT_SIGN_CONSTRAINTS["is_eps_beat"] == 1
    assert DEFAULT_SIGN_CONSTRAINTS["is_eps_miss"] == -1
    assert DEFAULT_SIGN_CONSTRAINTS["is_revenue_beat"] == 1
    assert DEFAULT_SIGN_CONSTRAINTS["is_revenue_miss"] == -1
