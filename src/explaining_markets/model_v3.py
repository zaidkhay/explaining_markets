"""Pure-Python inference for a validated multi-signal V3 artifact."""
from __future__ import annotations

import json
import math
from pathlib import Path

from explaining_markets.features_v3 import FeatureVectorV3, MODEL_FEATURE_NAMES_V3

DEFAULT_V3_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "multi_signal_v3.json"


class MultiSignalV3Model:
    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_V3_ARTIFACT_PATH
        raw = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.model_version = str(raw["model_version"])
        self.feature_names = tuple(str(x) for x in raw["feature_names"])
        self.means = tuple(float(x) for x in raw["means"])
        self.standard_deviations = tuple(float(x) for x in raw["standard_deviations"])
        self.coefficients = tuple(float(x) for x in raw["coefficients"])
        self.intercept = float(raw["intercept"])
        self.promoted = bool(raw.get("promoted", False))
        self.structured_only = bool(raw.get("structured_only", False))
        bounds = raw.get("clip_bounds", [0.05, 0.95])
        self.clip_lower, self.clip_upper = float(bounds[0]), float(bounds[1])
        self.training_metadata = dict(raw.get("training_metadata") or {})
        self._validate()

    def _validate(self) -> None:
        if self.feature_names != MODEL_FEATURE_NAMES_V3:
            raise ValueError("V3 artifact feature order does not match MODEL_FEATURE_NAMES_V3")
        n = len(self.feature_names)
        if not (len(self.means) == len(self.standard_deviations) == len(self.coefficients) == n):
            raise ValueError("V3 artifact vector lengths do not match")
        numbers = (*self.means, *self.standard_deviations, *self.coefficients, self.intercept)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("V3 artifact contains non-finite parameters")
        if any(sd <= 0 for sd in self.standard_deviations):
            raise ValueError("V3 artifact standard deviations must be positive")
        if not (0 <= self.clip_lower < self.clip_upper <= 1):
            raise ValueError("V3 artifact clip bounds are invalid")

    def predict_vector(self, vector: FeatureVectorV3) -> float:
        raw = vector.vector(self.feature_names)
        prediction = self.intercept + sum(
            coef * (value - mean) / sd
            for coef, value, mean, sd in zip(
                self.coefficients, raw, self.means, self.standard_deviations, strict=True
            )
        )
        if not math.isfinite(prediction):
            raise ValueError("V3 model produced a non-finite prediction")
        return float(max(self.clip_lower, min(self.clip_upper, prediction)))
