"""Point-in-time-safe EPS surprise features."""
from __future__ import annotations

import math
from statistics import mean, pstdev

from explaining_markets.v3_records import EarningsRecord

EPS_DENOM_FLOOR = 0.05
LARGE_SURPRISE_PCT = 0.10

EARNINGS_SURPRISE_FEATURE_NAMES = (
    "reported_eps", "consensus_eps", "eps_surprise_absolute", "eps_surprise_percent",
    "eps_surprise_signed", "eps_surprise_abs", "eps_surprise_zscore_company",
    "eps_surprise_percentile_company", "is_eps_beat", "is_eps_miss",
    "is_large_eps_beat", "is_large_eps_miss", "has_eps_surprise",
)


def _safe_pct(reported: float, consensus: float) -> float:
    denom = max(abs(consensus), EPS_DENOM_FLOOR)
    return (reported - consensus) / denom


def _percentile(value: float, history: list[float]) -> float:
    if not history:
        return 0.5
    less = sum(x < value for x in history)
    equal = sum(x == value for x in history)
    return (less + 0.5 * equal) / len(history)


def earnings_surprise_features(
    current: EarningsRecord | None,
    history: tuple[EarningsRecord, ...],
    cutoff,
) -> dict[str, float]:
    out = {name: 0.0 for name in EARNINGS_SURPRISE_FEATURE_NAMES}
    if current is None or not current.eligible(cutoff):
        return out
    if current.reported_eps is None or current.consensus_eps is None:
        return out

    reported = float(current.reported_eps)
    consensus = float(current.consensus_eps)
    absolute = reported - consensus
    pct = _safe_pct(reported, consensus)
    historical = []
    for row in history:
        if not row.eligible(cutoff) or row.reported_eps is None or row.consensus_eps is None:
            continue
        historical.append(_safe_pct(float(row.reported_eps), float(row.consensus_eps)))
    z = 0.0
    if len(historical) >= 2:
        sd = pstdev(historical)
        if sd > 1e-12:
            z = (pct - mean(historical)) / sd
    out.update({
        "reported_eps": reported,
        "consensus_eps": consensus,
        "eps_surprise_absolute": absolute,
        "eps_surprise_percent": pct,
        "eps_surprise_signed": math.copysign(abs(pct), absolute) if absolute else 0.0,
        "eps_surprise_abs": abs(pct),
        "eps_surprise_zscore_company": z,
        "eps_surprise_percentile_company": _percentile(pct, historical),
        "is_eps_beat": float(absolute > 0),
        "is_eps_miss": float(absolute < 0),
        "is_large_eps_beat": float(pct >= LARGE_SURPRISE_PCT),
        "is_large_eps_miss": float(pct <= -LARGE_SURPRISE_PCT),
        "has_eps_surprise": 1.0,
    })
    return out
