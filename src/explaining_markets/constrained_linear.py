"""Research-only sign-constrained linear fitting for emergency V3-lite candidates.

The production runtime remains pure Python and only consumes serialized
coefficients.  This module is used while building an operator candidate to
prevent semantically obvious realized-result relationships from flipping sign
because of sparse historical coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear

from explaining_markets.v3_lite_training import CLIP_BOUNDS
from explaining_markets.v3_training import V3TrainingRow

# +1 => coefficient must be non-negative; -1 => non-positive.
# Only features with an unambiguous economic direction are constrained.
DEFAULT_SIGN_CONSTRAINTS: dict[str, int] = {
    "eps_surprise_percent": 1,
    "eps_surprise_signed": 1,
    "eps_surprise_zscore_company": 1,
    "eps_surprise_percentile_company": 1,
    "is_eps_beat": 1,
    "is_eps_miss": -1,
    "is_large_eps_beat": 1,
    "is_large_eps_miss": -1,
    "revenue_surprise_percent": 1,
    "revenue_surprise_zscore_company": 1,
    "revenue_surprise_percentile_company": 1,
    "is_revenue_beat": 1,
    "is_revenue_miss": -1,
    "eps_beat_and_revenue_beat": 1,
    "eps_miss_and_revenue_miss": -1,
    "reasoning_earnings_quality": 1,
    "reasoning_revenue_quality": 1,
    "reasoning_expectations_gap": 1,
    "reasoning_overall_event_signal": 1,
}


@dataclass(frozen=True)
class ConstrainedRidgeFit:
    kind: str
    params: dict
    predictions: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    coefficients: np.ndarray
    intercept: float
    feature_names: tuple[str, ...]


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds <= 1e-12] = 1.0
    return means, stds


def _coefficient_bounds(
    feature_names: Sequence[str],
    constraints: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(feature_names), -np.inf, dtype=float)
    upper = np.full(len(feature_names), np.inf, dtype=float)
    for index, name in enumerate(feature_names):
        sign = int(constraints.get(name, 0))
        if sign > 0:
            lower[index] = 0.0
        elif sign < 0:
            upper[index] = 0.0
    return lower, upper


def _fit_parameters(
    rows: Sequence[V3TrainingRow],
    feature_names: Sequence[str],
    *,
    alpha: float,
    constraints: Mapping[str, int] = DEFAULT_SIGN_CONSTRAINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not rows:
        raise ValueError("constrained ridge requires at least one training row")
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("constrained ridge alpha must be finite and non-negative")

    names = tuple(feature_names)
    X = np.asarray([row.x(names) for row in rows], dtype=float)
    y = np.asarray([row.target_percentile for row in rows], dtype=float)
    means, stds = _standardize(X)
    Z = (X - means) / stds

    # Since every standardized column has zero training mean, the optimal
    # unpenalized intercept is exactly mean(y), even under coefficient bounds.
    intercept = float(y.mean())
    centered = y - intercept

    ridge = math.sqrt(float(alpha)) * np.eye(len(names), dtype=float)
    design = np.vstack([Z, ridge])
    target = np.concatenate([centered, np.zeros(len(names), dtype=float)])
    lower, upper = _coefficient_bounds(names, constraints)
    solved = lsq_linear(
        design,
        target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not solved.success:
        raise RuntimeError(f"sign-constrained ridge failed to converge: {solved.message}")
    coefficients = np.asarray(solved.x, dtype=float)
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("sign-constrained ridge produced non-finite coefficients")
    return means, stds, coefficients, intercept


def fit_sign_constrained_ridge(
    train: Sequence[V3TrainingRow],
    evaluate: Sequence[V3TrainingRow],
    feature_names: Sequence[str],
    *,
    alpha: float,
    constraints: Mapping[str, int] = DEFAULT_SIGN_CONSTRAINTS,
) -> ConstrainedRidgeFit:
    names = tuple(feature_names)
    means, stds, coefficients, intercept = _fit_parameters(
        train,
        names,
        alpha=alpha,
        constraints=constraints,
    )
    X_eval = np.asarray([row.x(names) for row in evaluate], dtype=float)
    Z_eval = (X_eval - means) / stds
    predictions = np.clip(intercept + Z_eval @ coefficients, *CLIP_BOUNDS)
    return ConstrainedRidgeFit(
        kind="constrained_ridge",
        params={"alpha": float(alpha)},
        predictions=np.asarray(predictions, dtype=float),
        means=means,
        stds=stds,
        coefficients=coefficients,
        intercept=intercept,
        feature_names=names,
    )


def fit_sign_constrained_parameters(
    rows: Sequence[V3TrainingRow],
    feature_names: Sequence[str],
    *,
    alpha: float,
    constraints: Mapping[str, int] = DEFAULT_SIGN_CONSTRAINTS,
) -> dict[str, object]:
    means, stds, coefficients, intercept = _fit_parameters(
        rows,
        feature_names,
        alpha=alpha,
        constraints=constraints,
    )
    return {
        "means": [float(x) for x in means],
        "standard_deviations": [float(x) for x in stds],
        "coefficients": [float(x) for x in coefficients],
        "intercept": float(intercept),
        "sign_constraints": {
            name: int(constraints[name])
            for name in feature_names
            if name in constraints
        },
    }
