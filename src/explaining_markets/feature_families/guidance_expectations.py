"""Management-guidance features with point-in-time expectation comparisons."""
from __future__ import annotations

from explaining_markets.v3_records import GuidanceRecord

GUIDANCE_FEATURE_NAMES = (
    "revenue_guidance_low", "revenue_guidance_high", "revenue_guidance_mid",
    "eps_guidance_low", "eps_guidance_high", "eps_guidance_mid", "ebitda_guidance",
    "margin_guidance", "revenue_guidance_vs_consensus", "eps_guidance_vs_consensus",
    "guidance_surprise_percent", "guidance_above_consensus", "guidance_below_consensus",
    "guidance_inline", "numeric_guidance_raised", "numeric_guidance_lowered",
    "guidance_reaffirmed", "has_numeric_guidance", "has_guidance_consensus",
)


def _mid(low: float | None, high: float | None) -> float | None:
    if low is None and high is None:
        return None
    if low is None:
        return float(high)
    if high is None:
        return float(low)
    return (float(low) + float(high)) / 2.0


def _relative(value: float | None, consensus: float | None) -> float | None:
    if value is None or consensus is None:
        return None
    return (value - consensus) / max(abs(consensus), 1e-6)


def guidance_expectation_features(record: GuidanceRecord | None, cutoff) -> dict[str, float]:
    out = {name: 0.0 for name in GUIDANCE_FEATURE_NAMES}
    if record is None or not record.eligible(cutoff):
        return out
    rev_mid = _mid(record.revenue_low, record.revenue_high)
    eps_mid = _mid(record.eps_low, record.eps_high)
    rev_vs = _relative(rev_mid, record.revenue_consensus)
    eps_vs = _relative(eps_mid, record.eps_consensus)
    comparisons = [x for x in (rev_vs, eps_vs) if x is not None]
    combined = sum(comparisons) / len(comparisons) if comparisons else 0.0
    has_numeric = any(x is not None for x in (rev_mid, eps_mid, record.ebitda, record.margin))
    has_consensus = bool(comparisons)
    tol = 0.0025
    direction = (record.direction or "").strip().lower()
    out.update({
        "revenue_guidance_low": float(record.revenue_low or 0.0),
        "revenue_guidance_high": float(record.revenue_high or 0.0),
        "revenue_guidance_mid": float(rev_mid or 0.0),
        "eps_guidance_low": float(record.eps_low or 0.0),
        "eps_guidance_high": float(record.eps_high or 0.0),
        "eps_guidance_mid": float(eps_mid or 0.0),
        "ebitda_guidance": float(record.ebitda or 0.0),
        "margin_guidance": float(record.margin or 0.0),
        "revenue_guidance_vs_consensus": float(rev_vs or 0.0),
        "eps_guidance_vs_consensus": float(eps_vs or 0.0),
        "guidance_surprise_percent": float(combined),
        "guidance_above_consensus": float(has_consensus and combined > tol),
        "guidance_below_consensus": float(has_consensus and combined < -tol),
        "guidance_inline": float(has_consensus and abs(combined) <= tol),
        "numeric_guidance_raised": float(direction == "raised"),
        "numeric_guidance_lowered": float(direction in {"lowered", "cut"}),
        "guidance_reaffirmed": float(direction in {"reaffirmed", "maintained"}),
        "has_numeric_guidance": float(has_numeric),
        "has_guidance_consensus": float(has_consensus),
    })
    return out
