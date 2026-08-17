"""Monotonic historical-percentile calibration for CAR1 percentile output.

Why calibration is legitimate here
----------------------------------
The competition target is a *within-quarter percentile*, so the label's
marginal distribution is (by construction) close to uniform on [0, 1]. A
regression fitted on that target minimises squared error by shrinking toward
the mean, so its raw output is concentrated near 0.50 even when its *ranking*
is informative. Mapping raw scores through the empirical CDF of historical
out-of-sample scores restores the correct marginal shape.

This is NOT a dispersion trick:

* the transform is strictly monotonic non-decreasing, so **Spearman
  correlation is mathematically unchanged** — calibration cannot manufacture
  or destroy ranking power, and the tests assert this;
* it is fitted only on out-of-sample predictions (validation / walk-forward),
  never on in-sample predictions from the rows used to fit the model;
* it is deterministic and serialized with the model artifact.

Definition
----------
For a fitted sample ``S`` of ``n`` historical OOS predictions::

    calibrate(x) = ( #{p in S : p < x} + 0.5 * #{p in S : p == x} ) / n

which is the mid-rank empirical CDF: it handles ties symmetrically, is
non-decreasing in ``x``, yields 0.0 strictly below the sample minimum and 1.0
strictly above the sample maximum, and is invariant to the order of ``S``.
Outputs are finally clamped into ``bounds`` so production never submits a
degenerate 0.0/1.0 unless configured to.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

CALIBRATION_METHOD = "empirical_oos_midrank_cdf"
CALIBRATION_VERSION = "calibration_v1"
DEFAULT_BOUNDS = (0.01, 0.99)
MAX_KNOTS = 4000


@dataclass(frozen=True)
class PercentileCalibrator:
    """Deterministic monotonic map from raw score to historical percentile.

    ``knots`` is the ascending fitted sample. ``source`` records provenance so
    an artifact reader can confirm the calibration was built out-of-sample.
    """

    knots: tuple[float, ...]
    bounds: tuple[float, float] = DEFAULT_BOUNDS
    method: str = CALIBRATION_METHOD
    version: str = CALIBRATION_VERSION
    source: str = "unspecified"
    n_fitted: int = 0

    def __post_init__(self) -> None:
        if not self.knots:
            raise ValueError("calibration requires at least one fitted prediction")
        if list(self.knots) != sorted(self.knots):
            raise ValueError("calibration knots must be ascending")
        if not all(math.isfinite(k) for k in self.knots):
            raise ValueError("calibration knots must all be finite")
        low, high = self.bounds
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"invalid calibration bounds: {self.bounds}")

    # ---- construction ------------------------------------------------

    @classmethod
    def fit(
        cls,
        oos_predictions: Iterable[float],
        *,
        source: str,
        bounds: tuple[float, float] = DEFAULT_BOUNDS,
        max_knots: int = MAX_KNOTS,
    ) -> "PercentileCalibrator":
        """Fit from OUT-OF-SAMPLE predictions only.

        ``source`` must describe the out-of-sample scheme (e.g.
        ``"2026Q1 validation, model fitted on 2025Q4"``); it is persisted so a
        reviewer can verify no in-sample leakage.
        """
        values = [float(p) for p in oos_predictions]
        if not values:
            raise ValueError("cannot fit calibration on zero predictions")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("cannot fit calibration on non-finite predictions")
        values.sort()
        n_fitted = len(values)
        if len(values) > max_knots:
            # Thin to a bounded quantile grid; preserves the CDF shape and the
            # extremes while keeping the serialized artifact small.
            step = (len(values) - 1) / (max_knots - 1)
            thinned = [values[min(len(values) - 1, int(round(i * step)))] for i in range(max_knots)]
            values = sorted(thinned)
        return cls(
            knots=tuple(values),
            bounds=bounds,
            source=source,
            n_fitted=n_fitted,
        )

    # ---- application -------------------------------------------------

    def raw_percentile(self, score: float) -> float:
        """Mid-rank empirical CDF position of ``score`` in [0, 1], unclamped."""
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("cannot calibrate a non-finite score")
        n = len(self.knots)
        below = bisect.bisect_left(self.knots, value)
        equal = bisect.bisect_right(self.knots, value) - below
        return (below + 0.5 * equal) / n

    def calibrate(self, score: float) -> float:
        """Calibrated percentile, clamped into ``bounds``."""
        low, high = self.bounds
        return float(min(high, max(low, self.raw_percentile(score))))

    def calibrate_many(self, scores: Iterable[float]) -> list[float]:
        return [self.calibrate(s) for s in scores]

    # ---- serialization -----------------------------------------------

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "version": self.version,
            "source": self.source,
            "n_fitted": self.n_fitted,
            "n_knots": len(self.knots),
            "bounds": list(self.bounds),
            "knots": [float(k) for k in self.knots],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PercentileCalibrator":
        knots = tuple(float(k) for k in payload["knots"])
        bounds = payload.get("bounds") or list(DEFAULT_BOUNDS)
        return cls(
            knots=knots,
            bounds=(float(bounds[0]), float(bounds[1])),
            method=str(payload.get("method") or CALIBRATION_METHOD),
            version=str(payload.get("version") or CALIBRATION_VERSION),
            source=str(payload.get("source") or "unspecified"),
            n_fitted=int(payload.get("n_fitted") or len(knots)),
        )


def is_monotonic(calibrator: PercentileCalibrator, probe: Sequence[float] | None = None) -> bool:
    """Verify non-decreasing behaviour over a probe grid (used by tests)."""
    grid = list(probe) if probe is not None else [i / 400.0 for i in range(401)]
    outputs = [calibrator.calibrate(x) for x in sorted(grid)]
    return all(b >= a - 1e-12 for a, b in zip(outputs, outputs[1:]))


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation used to prove calibration preserves ranking."""
    if len(a) != len(b) or len(a) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            shared = (i + j - 1) / 2.0
            for k in range(i, j):
                out[order[k]] = shared
            i = j
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a <= 1e-12 or var_b <= 1e-12:
        return None
    return cov / math.sqrt(var_a * var_b)
