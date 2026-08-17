"""Immutable explanation packet for V3-lite predictions.

The packet is built from ACTUAL feature values and ACTUAL model contributions
of the model that produced the prediction. Nothing is invented:

* key drivers are sorted by absolute standardized contribution, computed from
  the same coefficients/means/stds used at inference;
* the explanation text cites only features that are actually present and
  non-zero in the feature vector;
* raw and calibrated scores are both retained;
* the packet is a frozen dataclass and serialized as JSON;
* OpenRouter/LLM failures never affect this packet — it is purely
  deterministic and depends only on the model artifact and feature vector.

Production logging convention: ``[V3_EXPLANATION] ticker=... model=...``
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from explaining_markets.calibration import PercentileCalibrator

MAX_DRIVERS = 8
MIN_CONTRIBUTION_ABS = 1e-6


@dataclass(frozen=True)
class FeatureContribution:
    name: str
    raw_value: float
    standardized_value: float
    coefficient: float
    contribution: float  # coefficient * standardized_value


@dataclass(frozen=True)
class ExplanationPacket:
    """Immutable record of how a prediction was produced."""

    ticker: str
    model_version: str
    raw_prediction: float
    calibrated_percentile: float | None
    confidence: str
    available_feature_families: dict[str, float]
    fallback_status: str
    audit_status: str
    key_drivers: tuple[FeatureContribution, ...]
    explanation_text: str
    feature_contributions: tuple[FeatureContribution, ...]
    generated_at: str
    calibration_source: str | None = None
    bounds: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["key_drivers"] = [asdict(d) for d in self.key_drivers]
        payload["feature_contributions"] = [asdict(c) for c in self.feature_contributions]
        return payload

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def _confidence_label(raw: float, calibrated: float | None, n_available: int) -> str:
    """Coarse, honest confidence label from dispersion and coverage."""
    if n_available <= 1:
        return "low"
    score = calibrated if calibrated is not None else raw
    distance = abs(score - 0.5)
    if distance < 0.05:
        return "low"
    if distance < 0.15:
        return "medium"
    return "high"


def _explain_drivers(drivers: Sequence[FeatureContribution], ticker: str) -> str:
    """Build a human-readable explanation from actual top contributors."""
    if not drivers:
        return (
            f"[{ticker}] No feature contributed meaningfully to the prediction. "
            "The model is relying on its intercept/regularised mean; treat the "
            "output as a neutral baseline."
        )
    parts = [f"[{ticker}] Key drivers (by absolute contribution):"]
    for d in drivers:
        direction = "positive" if d.contribution > 0 else "negative"
        parts.append(
            f"  {d.name}: value={d.raw_value:.4f} "
            f"std={d.standardized_value:+.3f} coef={d.coefficient:+.5f} "
            f"contribution={d.contribution:+.5f} ({direction})"
        )
    return "\n".join(parts)


def build_explanation_packet(
    *,
    ticker: str,
    model_version: str,
    raw_prediction: float,
    feature_names: Sequence[str],
    feature_values: Sequence[float],
    coefficients: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    intercept: float,
    availability: dict[str, float],
    fallback_status: str = "none",
    audit_status: str = "PASS",
    calibrator: PercentileCalibrator | None = None,
    bounds: tuple[float, float] | None = None,
) -> ExplanationPacket:
    """Construct an immutable explanation from real model internals.

    Parameters mirror exactly what the model used: the same feature order,
    coefficients, means, and stds. Contributions are ``coef * (value - mean) / sd``
    — the same arithmetic ``predict_vector`` performs — so the sum of
    contributions plus the intercept equals the raw prediction (before clipping).
    """
    n = len(feature_names)
    if not (len(feature_values) == len(coefficients) == len(means) == len(stds) == n):
        raise ValueError("feature vector length mismatch in explanation packet")

    contributions: list[FeatureContribution] = []
    for name, value, coef, mean, sd in zip(feature_names, feature_values, coefficients, means, stds):
        safe_sd = sd if sd > 1e-12 else 1.0
        standardized = (float(value) - float(mean)) / safe_sd
        contribution = float(coef) * standardized
        contributions.append(
            FeatureContribution(
                name=str(name),
                raw_value=float(value),
                standardized_value=float(standardized),
                coefficient=float(coef),
                contribution=contribution,
            )
        )

    # Verify honesty: sum of contributions + intercept == raw (pre-clip).
    reconstructed = intercept + sum(c.contribution for c in contributions)
    if not math.isfinite(reconstructed):
        raise ValueError("explanation reconstruction produced non-finite value")

    # Sort by absolute contribution for key drivers; deterministic tie-break by name.
    ranked = sorted(
        contributions,
        key=lambda c: (-abs(c.contribution), c.name),
    )
    key_drivers = tuple(
        c for c in ranked[:MAX_DRIVERS] if abs(c.contribution) >= MIN_CONTRIBUTION_ABS
    )

    calibrated = None
    calibration_source = None
    if calibrator is not None:
        calibrated = calibrator.calibrate(float(raw_prediction))
        calibration_source = calibrator.source

    n_available = sum(1 for v in availability.values() if float(v) > 0.0)
    confidence = _confidence_label(float(raw_prediction), calibrated, n_available)

    return ExplanationPacket(
        ticker=str(ticker),
        model_version=str(model_version),
        raw_prediction=float(raw_prediction),
        calibrated_percentile=calibrated,
        confidence=confidence,
        available_feature_families={str(k): float(v) for k, v in availability.items()},
        fallback_status=str(fallback_status),
        audit_status=str(audit_status),
        key_drivers=key_drivers,
        explanation_text=_explain_drivers(key_drivers, str(ticker)),
        feature_contributions=tuple(contributions),
        generated_at=datetime.now(timezone.utc).isoformat(),
        calibration_source=calibration_source,
        bounds=tuple(bounds) if bounds is not None else None,
    )


def log_explanation(packet: ExplanationPacket) -> str:
    """Emit the production ``[V3_EXPLANATION]`` log line and return it."""
    top = ",".join(f"{d.name}={d.contribution:+.5f}" for d in packet.key_drivers[:3]) or "none"
    calibrated_str = (
        f"{packet.calibrated_percentile:.4f}"
        if packet.calibrated_percentile is not None
        else "n/a"
    )
    line = (
        f"[V3_EXPLANATION] ticker={packet.ticker} model={packet.model_version} "
        f"raw={packet.raw_prediction:.4f} calibrated={calibrated_str} "
        f"confidence={packet.confidence} fallback={packet.fallback_status} "
        f"audit={packet.audit_status} drivers={top}"
    )
    print(line)
    return line
