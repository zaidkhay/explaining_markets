"""Quantitative diagnosis of prediction-dispersion collapse.

Live V1 predictions cluster in 0.49-0.50. This module measures *why*, without
touching the predictions themselves. Nothing here injects variance; it only
reports the mechanism so a fix can target the real cause.

Two candidate causes are separated:

1. INPUT STARVATION — the disclosure text yields few/zero active FLS features,
   so the standardized feature vector sits at the training mean and the Ridge
   returns its intercept (~the mean target, i.e. ~0.50). Symptoms: low
   ``nonzero_feature_count``, small ``standardized_norm``.
2. MODEL SHRINKAGE — features vary, but heavy L2 regularisation shrinks the
   coefficients so far that the linear response barely moves. Symptoms:
   healthy ``standardized_norm`` with tiny ``contribution_absolute_sum``.

``diagnose_rows`` reports the distribution statistics the task requires
(fraction in 0.48-0.52, prediction std, unique rounded predictions, ...) for
any (model, rows) pair, so V1 and any candidate can be compared on identical
data.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import pstdev
from typing import Iterable, Sequence

from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES,
    extract_forward_looking_features,
)

NEAR_HALF_TIGHT = (0.48, 0.52)
NEAR_HALF_WIDE = (0.45, 0.55)


@dataclass(frozen=True)
class EventDiagnostic:
    """Per-event feature-supply and response diagnosis."""

    event_id: str
    ticker: str
    disclosure_fact_count: int
    disclosure_empty: bool
    nonzero_feature_count: int
    raw_norm: float
    standardized_norm: float
    contribution_absolute_sum: float
    prediction: float
    information_url_ok: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DispersionReport:
    """Distribution statistics for a set of predictions."""

    n: int
    prediction_std: float
    minimum_prediction: float
    maximum_prediction: float
    p05: float
    p25: float
    median: float
    p75: float
    p95: float
    fraction_between_048_052: float
    fraction_between_045_055: float
    unique_predictions_4dp: int
    mean_prediction: float

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def collapsed(self) -> bool:
        """True when the output is effectively a constant."""
        return self.prediction_std < 0.01 or self.fraction_between_048_052 > 0.90


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of zero values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[int(position)])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def dispersion_report(predictions: Iterable[float]) -> DispersionReport:
    """Distribution diagnostics for predicted percentiles."""
    values = [float(p) for p in predictions]
    if not values:
        raise ValueError("cannot build a dispersion report for zero predictions")
    ordered = sorted(values)
    return DispersionReport(
        n=len(values),
        prediction_std=float(pstdev(values)) if len(values) >= 2 else 0.0,
        minimum_prediction=ordered[0],
        maximum_prediction=ordered[-1],
        p05=_quantile(ordered, 0.05),
        p25=_quantile(ordered, 0.25),
        median=_quantile(ordered, 0.50),
        p75=_quantile(ordered, 0.75),
        p95=_quantile(ordered, 0.95),
        fraction_between_048_052=sum(
            1 for v in values if NEAR_HALF_TIGHT[0] <= v <= NEAR_HALF_TIGHT[1]
        ) / len(values),
        fraction_between_045_055=sum(
            1 for v in values if NEAR_HALF_WIDE[0] <= v <= NEAR_HALF_WIDE[1]
        ) / len(values),
        unique_predictions_4dp=len({round(v, 4) for v in values}),
        mean_prediction=float(sum(values) / len(values)),
    )


def diagnose_fls_event(
    model,
    *,
    event_id: str,
    ticker: str,
    disclosure: Sequence[str],
    information_url_ok: bool = True,
) -> EventDiagnostic:
    """Diagnose one event through a linear FLS model (V1-shaped).

    ``model`` must expose ``feature_names``/``means``/``standard_deviations``/
    ``coefficients``/``intercept`` and ``predict_features``.
    """
    facts = [str(f) for f in disclosure]
    features = extract_forward_looking_features(facts)
    raw = [float(features.values[name]) for name in model.feature_names]
    standardized = [
        (value - mean) / sd
        for value, mean, sd in zip(raw, model.means, model.standard_deviations, strict=True)
    ]
    contributions = [
        coef * z for coef, z in zip(model.coefficients, standardized, strict=True)
    ]
    return EventDiagnostic(
        event_id=event_id,
        ticker=ticker,
        disclosure_fact_count=len(facts),
        disclosure_empty=not any(f.strip() for f in facts),
        nonzero_feature_count=sum(1 for value in raw if abs(value) > 1e-12),
        raw_norm=math.sqrt(sum(value * value for value in raw)),
        standardized_norm=math.sqrt(sum(z * z for z in standardized)),
        contribution_absolute_sum=sum(abs(c) for c in contributions),
        prediction=float(model.predict_features(features)),
        information_url_ok=information_url_ok,
    )


@dataclass(frozen=True)
class CollapseDiagnosis:
    """Aggregate diagnosis across many events."""

    dispersion: DispersionReport
    events: tuple[EventDiagnostic, ...] = field(default=())
    mean_disclosure_fact_count: float = 0.0
    fraction_empty_disclosure: float = 0.0
    mean_nonzero_feature_count: float = 0.0
    mean_standardized_norm: float = 0.0
    mean_contribution_absolute_sum: float = 0.0
    clip_lower_hits: int = 0
    clip_upper_hits: int = 0

    @property
    def primary_cause(self) -> str:
        """Best-supported explanation for the observed dispersion."""
        if not self.dispersion.collapsed:
            return "not_collapsed"
        if self.fraction_empty_disclosure > 0.5:
            return "input_starvation_empty_disclosure"
        if self.mean_standardized_norm < 1.0:
            return "input_starvation_features_at_training_mean"
        if self.mean_contribution_absolute_sum < 0.05:
            return "model_shrinkage_coefficients_too_small"
        return "collapsed_cause_undetermined"

    def as_dict(self) -> dict:
        return {
            "dispersion": self.dispersion.as_dict(),
            "collapsed": self.dispersion.collapsed,
            "primary_cause": self.primary_cause,
            "mean_disclosure_fact_count": self.mean_disclosure_fact_count,
            "fraction_empty_disclosure": self.fraction_empty_disclosure,
            "mean_nonzero_feature_count": self.mean_nonzero_feature_count,
            "mean_standardized_norm": self.mean_standardized_norm,
            "mean_contribution_absolute_sum": self.mean_contribution_absolute_sum,
            "clip_lower_hits": self.clip_lower_hits,
            "clip_upper_hits": self.clip_upper_hits,
            "n_features_total": len(MODEL_FEATURE_NAMES),
        }


def diagnose_events(model, events: Sequence, *, keep_events: int = 0) -> CollapseDiagnosis:
    """Diagnose a set of archive events through a linear FLS model."""
    diagnostics = [
        diagnose_fls_event(
            model,
            event_id=event.event_id,
            ticker=event.ticker,
            disclosure=list(event.disclosure),
        )
        for event in events
    ]
    if not diagnostics:
        raise ValueError("cannot diagnose zero events")
    n = len(diagnostics)
    lower = getattr(model, "clip_lower", 0.0)
    upper = getattr(model, "clip_upper", 1.0)
    return CollapseDiagnosis(
        dispersion=dispersion_report([d.prediction for d in diagnostics]),
        events=tuple(diagnostics[:keep_events]) if keep_events else (),
        mean_disclosure_fact_count=sum(d.disclosure_fact_count for d in diagnostics) / n,
        fraction_empty_disclosure=sum(1 for d in diagnostics if d.disclosure_empty) / n,
        mean_nonzero_feature_count=sum(d.nonzero_feature_count for d in diagnostics) / n,
        mean_standardized_norm=sum(d.standardized_norm for d in diagnostics) / n,
        mean_contribution_absolute_sum=sum(d.contribution_absolute_sum for d in diagnostics) / n,
        clip_lower_hits=sum(1 for d in diagnostics if abs(d.prediction - lower) < 1e-9),
        clip_upper_hits=sum(1 for d in diagnostics if abs(d.prediction - upper) < 1e-9),
    )


def format_diagnosis(diagnosis: CollapseDiagnosis, *, label: str = "fls_ridge_v1") -> str:
    d = diagnosis.dispersion
    return "\n".join(
        [
            f"=== DISPERSION DIAGNOSIS: {label} ===",
            f"n:                          {d.n}",
            f"prediction std:             {d.prediction_std:.6f}",
            f"min / max:                  {d.minimum_prediction:.4f} / {d.maximum_prediction:.4f}",
            f"p05 / p25 / med / p75 / p95: {d.p05:.4f} / {d.p25:.4f} / {d.median:.4f} / "
            f"{d.p75:.4f} / {d.p95:.4f}",
            f"fraction in [0.48, 0.52]:   {d.fraction_between_048_052:.3f}",
            f"fraction in [0.45, 0.55]:   {d.fraction_between_045_055:.3f}",
            f"unique predictions (4dp):   {d.unique_predictions_4dp}",
            f"collapsed:                  {d.collapsed}",
            "",
            f"mean disclosure facts:      {diagnosis.mean_disclosure_fact_count:.2f}",
            f"fraction empty disclosure:  {diagnosis.fraction_empty_disclosure:.3f}",
            f"mean nonzero features:      {diagnosis.mean_nonzero_feature_count:.2f}"
            f" / {len(MODEL_FEATURE_NAMES)}",
            f"mean standardized norm:     {diagnosis.mean_standardized_norm:.4f}",
            f"mean |contributions| sum:   {diagnosis.mean_contribution_absolute_sum:.6f}",
            f"clip hits (low / high):     {diagnosis.clip_lower_hits} / {diagnosis.clip_upper_hits}",
            f"primary cause:              {diagnosis.primary_cause}",
        ]
    )
