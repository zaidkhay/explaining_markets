"""Production percentile models and fail-safe default-model selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from explaining_markets.features import FeatureVector
from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES,
    ForwardLookingFeatures,
    extract_forward_looking_features,
)

DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "fls_ridge_v1.json"
DEFAULT_V2_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "fls_company_history_ridge_v2.json"


@runtime_checkable
class PercentileModel(Protocol):
    def predict_percentile(self, features: FeatureVector) -> float: ...
    def fit(self, rows) -> "PercentileModel": ...


class BaselineModel:
    def fit(self, rows):
        return self

    def predict_percentile(self, features: FeatureVector) -> float:
        return 0.5


class HeuristicFactModel:
    """Existing transparent disclosure heuristic, retained as production fallback."""

    _STEP = 0.08
    _LOWER = 0.10
    _UPPER = 0.90

    def fit(self, rows):
        return self

    def predict_percentile(self, features: FeatureVector) -> float:
        raw = 0.5 + self._STEP * features.net_sentiment
        return max(self._LOWER, min(self._UPPER, raw))


class ForwardLookingRidgeModel:
    """Pure-Python inference from the serialized paper-informed Ridge artifact."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
        raw = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.model_version = str(raw["model_version"])
        self.feature_names = tuple(str(x) for x in raw["feature_names"])
        self.means = tuple(float(x) for x in raw["means"])
        self.standard_deviations = tuple(float(x) for x in raw["standard_deviations"])
        self.coefficients = tuple(float(x) for x in raw["coefficients"])
        self.intercept = float(raw["intercept"])
        bounds = raw.get("clip_bounds", [0.05, 0.95])
        self.clip_lower, self.clip_upper = float(bounds[0]), float(bounds[1])
        self.alpha = float(raw["selected_alpha"])
        self.training_metadata = dict(raw.get("training_metadata") or {})
        self._validate()

    def _validate(self) -> None:
        if self.feature_names != MODEL_FEATURE_NAMES:
            raise ValueError("FLS artifact feature order does not match production extractor")
        n = len(self.feature_names)
        if not (len(self.means) == len(self.standard_deviations) == len(self.coefficients) == n):
            raise ValueError("FLS artifact vector lengths do not match")
        numbers = (*self.means, *self.standard_deviations, *self.coefficients, self.intercept)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("FLS artifact contains non-finite parameters")
        if any(sd <= 0.0 for sd in self.standard_deviations):
            raise ValueError("FLS artifact standard deviations must be positive")
        if not (0.0 <= self.clip_lower < self.clip_upper <= 1.0):
            raise ValueError("FLS artifact clip bounds are invalid")

    def standardize(self, features: ForwardLookingFeatures) -> list[float]:
        raw = features.vector(self.feature_names)
        return [
            (value - mean) / sd
            for value, mean, sd in zip(raw, self.means, self.standard_deviations, strict=True)
        ]

    def predict_features(self, features: ForwardLookingFeatures) -> float:
        standardized = self.standardize(features)
        prediction = self.intercept + sum(
            coef * value for coef, value in zip(self.coefficients, standardized, strict=True)
        )
        if not math.isfinite(prediction):
            raise ValueError("FLS Ridge produced a non-finite prediction")
        return float(max(self.clip_lower, min(self.clip_upper, prediction)))

    def predict_disclosure(self, disclosure: list[str]) -> float:
        return self.predict_features(extract_forward_looking_features(disclosure))

    def predict_with_features(self, disclosure: list[str]) -> tuple[float, ForwardLookingFeatures]:
        features = extract_forward_looking_features(disclosure)
        return self.predict_features(features), features


class CompanyHistoryRidgeModel:
    """Offline/evaluation V2 model retained for reproducibility, not live promotion."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        from explaining_markets.features_v2 import MODEL_FEATURE_NAMES_V2

        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_V2_ARTIFACT_PATH
        raw = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.model_version = str(raw["model_version"])
        self.feature_names = tuple(str(x) for x in raw["feature_names"])
        self.means = tuple(float(x) for x in raw["means"])
        self.standard_deviations = tuple(float(x) for x in raw["standard_deviations"])
        self.coefficients = tuple(float(x) for x in raw["coefficients"])
        self.intercept = float(raw["intercept"])
        bounds = raw.get("clip_bounds", [0.05, 0.95])
        self.clip_lower, self.clip_upper = float(bounds[0]), float(bounds[1])
        self.alpha = float(raw["selected_alpha"])
        self.promoted = bool(raw.get("promoted", False))
        self.training_metadata = dict(raw.get("training_metadata") or {})
        self._expected_names = MODEL_FEATURE_NAMES_V2
        self._validate()

    def _validate(self) -> None:
        if self.feature_names != self._expected_names:
            raise ValueError("V2 artifact feature order does not match production extractor")
        n = len(self.feature_names)
        if not (len(self.means) == len(self.standard_deviations) == len(self.coefficients) == n):
            raise ValueError("V2 artifact vector lengths do not match")
        numbers = (*self.means, *self.standard_deviations, *self.coefficients, self.intercept)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("V2 artifact contains non-finite parameters")
        if any(sd <= 0.0 for sd in self.standard_deviations):
            raise ValueError("V2 artifact standard deviations must be positive")
        if not (0.0 <= self.clip_lower < self.clip_upper <= 1.0):
            raise ValueError("V2 artifact clip bounds are invalid")

    def predict_vector(self, vector: "object") -> float:
        raw = vector.vector(self.feature_names)
        prediction = self.intercept + sum(
            coef * (value - mean) / sd
            for coef, value, mean, sd in zip(
                self.coefficients, raw, self.means, self.standard_deviations, strict=True
            )
        )
        if not math.isfinite(prediction):
            raise ValueError("V2 Ridge produced a non-finite prediction")
        return float(max(self.clip_lower, min(self.clip_upper, prediction)))

    def predict(self, *, disclosure: list[str], history) -> tuple[float, "object"]:
        from explaining_markets.features_v2 import build_feature_vector_v2

        vector = build_feature_vector_v2(
            fls=extract_forward_looking_features(disclosure), history=history
        )
        return self.predict_vector(vector), vector


def get_default_model():
    """Production chain: promoted V3 -> fls_ridge_v1 -> heuristic -> baseline.

    V2 remains available for historical experiments but never displaces V1.
    V3 can displace V1 only when an artifact exists, validates against the
    frozen V3 feature order, and records ``promoted: true`` from its predeclared
    chronological evaluation gate.
    """
    try:
        from explaining_markets.model_v3 import MultiSignalV3Model

        v3 = MultiSignalV3Model()
        if v3.promoted:
            return v3
        print("[MODEL] V3 artifact present but not promoted; using fls_ridge_v1")
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[MODEL] V3 artifact unavailable/invalid; using fls_ridge_v1: {type(exc).__name__}")

    try:
        return ForwardLookingRidgeModel()
    except Exception as exc:
        print(f"[MODEL] fls_ridge unavailable; using heuristic fallback: {type(exc).__name__}")
        try:
            return HeuristicFactModel()
        except Exception:
            return BaselineModel()
