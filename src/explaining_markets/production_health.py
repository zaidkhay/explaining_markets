"""Rolling live prediction diagnostics. These metrics never alter predictions."""
from __future__ import annotations

from collections import deque
from statistics import pstdev


class RollingPredictionHealth:
    def __init__(self, max_events: int = 100) -> None:
        self.predictions = deque(maxlen=max_events)
        self.availability = deque(maxlen=max_events)

    def add(self, prediction: float, family_availability: dict[str, float] | None = None) -> dict:
        self.predictions.append(float(prediction))
        self.availability.append(dict(family_availability or {}))
        return self.summary()

    def summary(self) -> dict:
        values = list(self.predictions)
        n = len(values)
        family_names = sorted({name for row in self.availability for name in row})
        availability_rates = {
            name: sum(float(row.get(name, 0.0)) > 0 for row in self.availability) / len(self.availability)
            for name in family_names
        } if self.availability else {}
        return {
            "n": n,
            "prediction_std": pstdev(values) if n > 1 else 0.0,
            "fraction_near_0_5": sum(0.48 <= x <= 0.52 for x in values) / n if n else 0.0,
            "unique_predictions_after_rounding": len({round(x, 4) for x in values}),
            "feature_family_availability_rates": availability_rates,
        }

    @staticmethod
    def is_low_dispersion(summary: dict, *, std_threshold: float = 0.01, near_fraction: float = 0.8) -> bool:
        return summary["prediction_std"] < std_threshold and summary["fraction_near_0_5"] >= near_fraction
