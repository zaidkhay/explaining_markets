"""Revenue result-versus-consensus features for V3."""
from __future__ import annotations

from statistics import mean, pstdev

from explaining_markets.v3_records import EarningsRecord

REVENUE_SURPRISE_FEATURE_NAMES = (
    "reported_revenue", "consensus_revenue", "revenue_surprise_absolute",
    "revenue_surprise_percent", "revenue_surprise_zscore_company",
    "revenue_surprise_percentile_company", "is_revenue_beat", "is_revenue_miss",
    "has_revenue_surprise", "eps_beat_and_revenue_beat", "eps_beat_revenue_miss",
    "eps_miss_revenue_beat", "eps_miss_and_revenue_miss",
)


def revenue_surprise_ratio(reported: float, consensus: float) -> float:
    return (reported - consensus) / max(abs(consensus), 1.0)


def revenue_surprise_features(current: EarningsRecord | None, history: tuple[EarningsRecord, ...], cutoff) -> dict[str, float]:
    values = {name: 0.0 for name in REVENUE_SURPRISE_FEATURE_NAMES}
    if current is None or not current.eligible(cutoff):
        return values
    if current.reported_revenue is None or current.consensus_revenue is None:
        return values
    reported = float(current.reported_revenue)
    consensus = float(current.consensus_revenue)
    absolute = reported - consensus
    ratio = revenue_surprise_ratio(reported, consensus)
    prior = [revenue_surprise_ratio(float(r.reported_revenue), float(r.consensus_revenue)) for r in history if r.eligible(cutoff) and r.reported_revenue is not None and r.consensus_revenue is not None]
    zscore = 0.0
    if len(prior) >= 2:
        sd = pstdev(prior)
        if sd > 1e-12:
            zscore = (ratio - mean(prior)) / sd
    percentile = 0.5 if not prior else (sum(x < ratio for x in prior) + 0.5 * sum(x == ratio for x in prior)) / len(prior)
    eps_delta = None
    if current.reported_eps is not None and current.consensus_eps is not None:
        eps_delta = float(current.reported_eps) - float(current.consensus_eps)
    values.update({
        "reported_revenue": reported,
        "consensus_revenue": consensus,
        "revenue_surprise_absolute": absolute,
        "revenue_surprise_percent": ratio,
        "revenue_surprise_zscore_company": zscore,
        "revenue_surprise_percentile_company": percentile,
        "is_revenue_beat": float(absolute > 0),
        "is_revenue_miss": float(absolute < 0),
        "has_revenue_surprise": 1.0,
        "eps_beat_and_revenue_beat": float(eps_delta is not None and eps_delta > 0 and absolute > 0),
        "eps_beat_revenue_miss": float(eps_delta is not None and eps_delta > 0 and absolute < 0),
        "eps_miss_revenue_beat": float(eps_delta is not None and eps_delta < 0 and absolute > 0),
        "eps_miss_and_revenue_miss": float(eps_delta is not None and eps_delta < 0 and absolute < 0),
    })
    return values
