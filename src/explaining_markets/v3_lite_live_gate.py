"""Production-realism gate for operator-selected V3-lite candidates.

This gate is deliberately separate from historical validation.  Historical
Spearman decides whether a candidate has ranking evidence; this module only
checks an operational invariant exposed by live competition traffic: realized
negative/neutral/positive disclosure facts must survive parsing and produce
ordered, meaningfully separated scores even when the legacy FLS block is zero.

The scenarios are a safety constraint, not a training target or optimization
objective.  Candidates still have to beat V1 on chronological validation.
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
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "parsed_ok": self.parsed_ok,
            "zero_fls": self.zero_fls,
            "ordered": self.ordered,
            "submitted_spread": self.submitted_spread,
            "passed": self.passed,
            "scenarios": [scenario.__dict__ for scenario in self.scenarios],
        }


def evaluate_v3_lite_live_gate(model, *, min_submitted_spread: float = 0.05) -> V3LiteLiveGateResult:
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
        and scenarios[0].submitted <= scenarios[1].submitted <= scenarios[2].submitted
    )
    spread = scenarios[2].submitted - scenarios[0].submitted
    passed = parsed_ok and zero_fls and ordered and spread > min_submitted_spread
    return V3LiteLiveGateResult(
        scenarios=tuple(scenarios),
        parsed_ok=parsed_ok,
        zero_fls=zero_fls,
        ordered=ordered,
        submitted_spread=float(spread),
        passed=passed,
    )
