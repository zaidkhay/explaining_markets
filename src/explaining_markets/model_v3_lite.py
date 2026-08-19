"""Pure-Python runtime for an explicitly operator-selected V3-lite candidate.

This is intentionally separate from the normal promotion path.  The artifact
must remain marked ``promoted: false`` and ``operator_override: true`` so we do
not rewrite history or pretend the untouched-holdout gate passed.  Production
may select this candidate only through the explicit model-selection logic in
``model.get_default_model``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from explaining_markets.calibration import PercentileCalibrator
from explaining_markets.features_v3 import FeatureVectorV3, MODEL_FEATURE_NAMES_V3
from explaining_markets.model_v3 import MultiSignalV3Model

DEFAULT_V3_LITE_CANDIDATE_PATH = (
    Path(__file__).with_name("artifacts") / "v3_lite_candidate.json"
)


class V3LiteCandidateModel(MultiSignalV3Model):
    """Linear V3-lite model with OOS empirical-percentile calibration."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_V3_LITE_CANDIDATE_PATH
        raw = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.model_version = str(raw["model_version"])
        self.feature_spec_version = str(raw.get("feature_spec_version") or "")
        self.feature_names = tuple(str(x) for x in raw["feature_names"])
        self.means = tuple(float(x) for x in raw["means"])
        self.standard_deviations = tuple(float(x) for x in raw["standard_deviations"])
        self.coefficients = tuple(float(x) for x in raw["coefficients"])
        self.intercept = float(raw["intercept"])
        bounds = raw.get("clip_bounds", [0.05, 0.95])
        self.clip_lower, self.clip_upper = float(bounds[0]), float(bounds[1])
        self.promoted = bool(raw.get("promoted", False))
        self.operator_override = bool(raw.get("operator_override", False))
        self.production_candidate = bool(raw.get("production_candidate", False))
        self.structured_only = bool(raw.get("structured_only", False))
        self.ablation = str(raw.get("ablation") or "unknown")
        self.training_metadata = dict(raw.get("training_metadata") or {})
        self.calibrator = PercentileCalibrator.from_dict(dict(raw["calibration"]))
        self.last_raw_prediction: float | None = None
        self.last_calibrated_prediction: float | None = None
        self._validate_candidate()

    def _validate_candidate(self) -> None:
        if self.promoted:
            raise ValueError("operator V3-lite candidate must not claim promoted=true")
        if not self.operator_override or not self.production_candidate:
            raise ValueError("V3-lite candidate lacks explicit operator-override metadata")
        if self.feature_spec_version != "v3_lite_v1":
            raise ValueError("unsupported V3-lite candidate feature spec")
        if not self.feature_names:
            raise ValueError("V3-lite candidate has no features")
        if any(name not in MODEL_FEATURE_NAMES_V3 for name in self.feature_names):
            raise ValueError("V3-lite candidate contains unknown feature names")
        positions = [MODEL_FEATURE_NAMES_V3.index(name) for name in self.feature_names]
        if positions != sorted(positions):
            raise ValueError("V3-lite candidate feature order does not follow V3 feature order")
        n = len(self.feature_names)
        if not (len(self.means) == len(self.standard_deviations) == len(self.coefficients) == n):
            raise ValueError("V3-lite candidate vector lengths do not match")
        numbers = (*self.means, *self.standard_deviations, *self.coefficients, self.intercept)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("V3-lite candidate contains non-finite parameters")
        if any(sd <= 0.0 for sd in self.standard_deviations):
            raise ValueError("V3-lite candidate standard deviations must be positive")
        if not (0.0 <= self.clip_lower < self.clip_upper <= 1.0):
            raise ValueError("V3-lite candidate clip bounds are invalid")
        source = self.calibrator.source.lower()
        if "validation" not in source or "2025q4" not in source:
            raise ValueError("V3-lite calibration provenance is not clearly out-of-sample")

    def predict_raw_vector(self, vector: FeatureVectorV3) -> float:
        values = vector.vector(self.feature_names)
        prediction = self.intercept + sum(
            coef * (value - mean) / sd
            for coef, value, mean, sd in zip(
                self.coefficients,
                values,
                self.means,
                self.standard_deviations,
                strict=True,
            )
        )
        if not math.isfinite(prediction):
            raise ValueError("V3-lite candidate produced a non-finite prediction")
        return float(max(self.clip_lower, min(self.clip_upper, prediction)))

    def predict_vector(self, vector: FeatureVectorV3) -> float:
        raw = self.predict_raw_vector(vector)
        calibrated = self.calibrator.calibrate(raw)
        self.last_raw_prediction = raw
        self.last_calibrated_prediction = calibrated
        print(
            "[V3_LITE_MODEL] "
            f"model={self.model_version} ablation={self.ablation} "
            f"raw={raw:.4f} submitted={calibrated:.4f} "
            f"calibration={self.calibrator.version} operator_override=1"
        )
        return calibrated
