"""Production-realism gate for operator-selected V3-lite candidates.

This gate is deliberately separate from historical validation. Historical
Spearman decides whether a candidate has ranking evidence; this module checks
an operational invariant exposed by live competition traffic: realized
negative/neutral/positive disclosure facts must survive parsing and produce
strictly ordered, meaningfully separated submitted percentiles even when the
legacy FLS block is zero.

Because the production calibration is an empirical percentile transform, a
fixed adjacent-percentile floor is arbitrary and depends on validation sample
size. By default we therefore require each adjacent scenario to be separated
by at least ``min_adjacent_rank_steps`` historical OOS calibration ranks.
An explicit percentile-gap override remains available for diagnostics.

The scenarios are a safety constraint, not a training target or optimization
objective. Candidates still have to beat V1 on chronological validation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.v3_records import V3Context


REALIZED_SCENARIOS: dict[str, list[str]] = {
    "negative": [
        "Revenue missed consensus by 8%.",
        "EPS missed consensus by 12%.",
    ],
    "neutral": [
        "Revenue was in line with consensus.",
        "EPS matched consensus.",
    ],
    "positive": [
        "Revenue beat consensus by 8%.",
        "EPS beat consensus by 12%.",
    ],
}


@dataclass(frozen=True)
class LiveGateScenario:
    label: str
    eps_surprise: float
    revenue_surprise: float
    earnings_quality: float
    revenue_quality: float
    raw: float
    submitted: float
    fls_nonzero: int
    has_eps_surprise: float
    has_revenue_surprise: float


@dataclass(frozen=True)
class V3LiteLiveGateResult:
    scenarios: tuple[LiveGateScenario, ...]
    parsed_ok: bool
    zero_fls: bool
    ordered: bool
    submitted_spread: float
    negative_neutral_gap: float
    neutral_positive_gap: float
    minimum_adjacent_gap_required: float
    adjacent_rank_steps_required: int
    calibration_n_fitted: int
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "parsed_ok": self.parsed_ok,
            "zero_fls": self.zero_fls,
            "ordered": self.ordered,
            "submitted_spread": self.submitted_spread,
            "negative_neutral_gap": self.negative_neutral_gap,
            "neutral_positive_gap": self.neutral_positive_gap,
            "minimum_adjacent_gap_required": self.minimum_adjacent_gap_required,
            "adjacent_rank_steps_required": self.adjacent_rank_steps_required,
            "calibration_n_fitted": self.calibration_n_fitted,
            "passed": self.passed,
            "scenarios": [scenario.__dict__ for scenario in self.scenarios],
        }


def _adjacent_gap_floor(
    model,
    *,
    explicit_gap: float | None,
    min_adjacent_rank_steps: int,
) -> tuple[float, int]:
    if explicit_gap is not None:
        if explicit_gap < 0:
            raise ValueError("min_adjacent_submitted_gap must be non-negative")
        n_fitted = int(getattr(getattr(model, "calibrator", None), "n_fitted", 0) or 0)
        return float(explicit_gap), n_fitted

    if min_adjacent_rank_steps < 1:
        raise ValueError("min_adjacent_rank_steps must be >= 1")

    calibrator = getattr(model, "calibrator", None)
    n_fitted = int(getattr(calibrator, "n_fitted", 0) or 0)
    if n_fitted > 0:
        # A five-rank gap means multiple historical OOS predictions lie between
        # adjacent live scenarios. This scales naturally with calibration size.
        return float(min_adjacent_rank_steps / n_fitted), n_fitted

    # Test/dummy calibrators may not expose fitted-sample provenance. Keep a
    # small deterministic fallback rather than silently disabling separation.
    return 0.005, 0


def evaluate_v3_lite_live_gate(
    model,
    *,
    min_submitted_spread: float = 0.05,
    min_adjacent_submitted_gap: float | None = None,
    min_adjacent_rank_steps: int = 5,
) -> V3LiteLiveGateResult:
    """Run the realized-disclosure invariant through the exact runtime model."""
    cutoff = datetime.now(timezone.utc)
    reasoner = EventReasoner(use_openrouter=False)
    scenarios: list[LiveGateScenario] = []

    for label in ("negative", "neutral", "positive"):
        disclosure = REALIZED_SCENARIOS[label]
        base = V3Context(ticker="TEST", cutoff=cutoff)
        before = build_feature_vector_v3(disclosure=disclosure, context=base)
        reasoning = reasoner.reason(values=before.values, cutoff=cutoff)
        final = replace(base, event_reasoning=reasoning)
        vector = build_feature_vector_v3(disclosure=disclosure, context=final)
        raw = model.predict_raw_vector(vector)
        submitted = model.calibrator.calibrate(raw)
        scenarios.append(
            LiveGateScenario(
                label=label,
                eps_surprise=float(vector.values["eps_surprise_percent"]),
                revenue_surprise=float(vector.values["revenue_surprise_percent"]),
                earnings_quality=float(reasoning.earnings_quality),
                revenue_quality=float(reasoning.revenue_quality),
                raw=float(raw),
                submitted=float(submitted),
                fls_nonzero=sum(abs(x) > 1e-12 for x in vector.fls.values.values()),
                has_eps_surprise=float(vector.values["has_eps_surprise"]),
                has_revenue_surprise=float(vector.values["has_revenue_surprise"]),
            )
        )

    parsed_ok = all(
        scenario.has_eps_surprise == 1.0 and scenario.has_revenue_surprise == 1.0
        for scenario in scenarios
    )
    zero_fls = all(scenario.fls_nonzero == 0 for scenario in scenarios)
    ordered = (
        scenarios[0].raw < scenarios[1].raw < scenarios[2].raw
        and scenarios[0].submitted < scenarios[1].submitted < scenarios[2].submitted
    )
    negative_neutral_gap = scenarios[1].submitted - scenarios[0].submitted
    neutral_positive_gap = scenarios[2].submitted - scenarios[1].submitted
    spread = scenarios[2].submitted - scenarios[0].submitted
    adjacent_floor, n_fitted = _adjacent_gap_floor(
        model,
        explicit_gap=min_adjacent_submitted_gap,
        min_adjacent_rank_steps=min_adjacent_rank_steps,
    )
    adjacent_ok = (
        negative_neutral_gap >= adjacent_floor
        and neutral_positive_gap >= adjacent_floor
    )
    passed = (
        parsed_ok
        and zero_fls
        and ordered
        and spread > min_submitted_spread
        and adjacent_ok
    )
    return V3LiteLiveGateResult(
        scenarios=tuple(scenarios),
        parsed_ok=parsed_ok,
        zero_fls=zero_fls,
        ordered=ordered,
        submitted_spread=float(spread),
        negative_neutral_gap=float(negative_neutral_gap),
        neutral_positive_gap=float(neutral_positive_gap),
        minimum_adjacent_gap_required=float(adjacent_floor),
        adjacent_rank_steps_required=int(min_adjacent_rank_steps),
        calibration_n_fitted=int(n_fitted),
        passed=passed,
    )
