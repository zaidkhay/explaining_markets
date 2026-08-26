"""Pure-Python port of the frozen Explaining Markets scoring core.

Source of truth:
https://github.com/explaining-markets/examples/blob/501bd31/src/examples/scoring.py

For a complete-prediction sample the competition fits

    y = alpha + beta_prediction * prediction + beta_surprise * surprise_pct

and ranks submissions by the incremental R^2 over the surprise-only fit.
The contest common-sample variant mean-imputes missing predictions first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DEGENERATE_SD = 1e-6


def percentile_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    denom = n - 1
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = ((i + j) / 2.0) / denom
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def no_spread(sum_sq_dev: float, n: int) -> bool:
    return sum_sq_dev <= n * DEGENERATE_SD**2


@dataclass(frozen=True)
class OLSFit:
    n: int
    alpha: float
    beta: float
    r_squared: float
    mse: float
    beta_surprise: float | None = None


def ols_fit(points: list[tuple[float, float]]) -> OLSFit | None:
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    s_xx = sum((p[0] - mean_x) ** 2 for p in points)
    s_yy = sum((p[1] - mean_y) ** 2 for p in points)
    s_xy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    if no_spread(s_xx, n):
        return None
    beta = s_xy / s_xx
    alpha = mean_y - beta * mean_x
    ss_res = sum((p[1] - (alpha + beta * p[0])) ** 2 for p in points)
    mse = ss_res / n
    r_squared = 0.0 if s_yy == 0.0 else 1.0 - ss_res / s_yy
    return OLSFit(n=n, alpha=alpha, beta=beta, r_squared=r_squared, mse=mse)


def ols_fit2(points: list[tuple[float, float, float]]) -> OLSFit | None:
    n = len(points)
    if n < 3:
        return None
    mean_x1 = sum(p[0] for p in points) / n
    mean_x2 = sum(p[1] for p in points) / n
    mean_y = sum(p[2] for p in points) / n
    s11 = sum((p[0] - mean_x1) ** 2 for p in points)
    s22 = sum((p[1] - mean_x2) ** 2 for p in points)
    s12 = sum((p[0] - mean_x1) * (p[1] - mean_x2) for p in points)
    s1y = sum((p[0] - mean_x1) * (p[2] - mean_y) for p in points)
    s2y = sum((p[1] - mean_x2) * (p[2] - mean_y) for p in points)
    s_yy = sum((p[2] - mean_y) ** 2 for p in points)
    if no_spread(s22, n):
        return None
    if no_spread(s11, n):
        beta2 = s2y / s22
        alpha = mean_y - beta2 * mean_x2
        ss_res = sum((p[2] - (alpha + beta2 * p[1])) ** 2 for p in points)
        return OLSFit(
            n=n,
            alpha=alpha,
            beta=0.0,
            r_squared=0.0 if s_yy == 0.0 else 1.0 - ss_res / s_yy,
            mse=ss_res / n,
            beta_surprise=beta2,
        )
    det = s11 * s22 - s12 * s12
    if det == 0.0:
        return None
    beta1 = (s22 * s1y - s12 * s2y) / det
    beta2 = (s11 * s2y - s12 * s1y) / det
    alpha = mean_y - beta1 * mean_x1 - beta2 * mean_x2
    ss_res = sum((p[2] - (alpha + beta1 * p[0] + beta2 * p[1])) ** 2 for p in points)
    mse = ss_res / n
    r_squared = 0.0 if s_yy == 0.0 else 1.0 - ss_res / s_yy
    return OLSFit(n=n, alpha=alpha, beta=beta1, r_squared=r_squared, mse=mse, beta_surprise=beta2)


def score_complete_predictions(
    predictions: Sequence[float],
    realized_percentiles: Sequence[float],
    surprise_percentiles: Sequence[float],
) -> dict[str, float | int | None]:
    """Official no-missing score family for a common complete sample."""
    if not (len(predictions) == len(realized_percentiles) == len(surprise_percentiles)):
        raise ValueError("prediction, realized, and surprise lengths differ")
    points = [
        (float(p), float(s), float(y))
        for p, s, y in zip(predictions, surprise_percentiles, realized_percentiles, strict=True)
    ]
    fit = ols_fit2(points)
    surprise_fit = ols_fit([(float(s), float(y)) for s, y in zip(surprise_percentiles, realized_percentiles, strict=True)]) if fit else None
    return {
        "n": len(points),
        "r_squared_surprise": None if surprise_fit is None else surprise_fit.r_squared,
        "r_squared": None if fit is None else fit.r_squared,
        "delta_r_squared": None if fit is None or surprise_fit is None else fit.r_squared - surprise_fit.r_squared,
        "beta": None if fit is None else fit.beta,
        "beta_surprise": None if fit is None else fit.beta_surprise,
        "alpha": None if fit is None else fit.alpha,
        "mse": None if fit is None else fit.mse,
    }


def score_with_missing_predictions(
    predictions: Sequence[float | None],
    realized_percentiles: Sequence[float],
    surprise_percentiles: Sequence[float],
) -> dict[str, float | int | None]:
    """Official contest common-sample mean-imputation score family."""
    if not (len(predictions) == len(realized_percentiles) == len(surprise_percentiles)):
        raise ValueError("prediction, realized, and surprise lengths differ")
    present = [float(p) for p in predictions if p is not None]
    if not present:
        return {
            "n_obs": 0,
            "imputed_mean": None,
            "imputed_event_count": len(predictions),
            "r_squared_surprise_imputed": None,
            "r_squared_imputed": None,
            "delta_r_squared_imputed": None,
            "beta_imputed": None,
            "beta_surprise_imputed": None,
            "alpha_imputed": None,
            "mse_imputed": None,
        }
    imputed_mean = sum(present) / len(present)
    filled = [imputed_mean if p is None else float(p) for p in predictions]
    block = score_complete_predictions(filled, realized_percentiles, surprise_percentiles)
    return {
        "n_obs": len(present),
        "imputed_mean": imputed_mean,
        "imputed_event_count": len(predictions) - len(present),
        "r_squared_surprise_imputed": block["r_squared_surprise"],
        "r_squared_imputed": block["r_squared"],
        "delta_r_squared_imputed": block["delta_r_squared"],
        "beta_imputed": block["beta"],
        "beta_surprise_imputed": block["beta_surprise"],
        "alpha_imputed": block["alpha"],
        "mse_imputed": block["mse"],
    }
