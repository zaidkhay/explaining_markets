"""Five-year company earnings-history features for V3.

Only records whose ``available_at`` is no later than the focal cutoff are used.
Sparse conditional reaction means may be shrunk toward externally supplied
industry/global priors; when no prior is supplied, the raw eligible mean is
used and the sample count remains explicit.
"""
from __future__ import annotations

from statistics import mean, median, pstdev

from explaining_markets.v3_records import EarningsRecord

COMPANY_HISTORY_V3_FEATURE_NAMES = (
    "prior_earnings_count",
    "mean_prior_earnings_abnormal_return",
    "median_prior_earnings_abnormal_return",
    "std_prior_earnings_abnormal_return",
    "positive_prior_earnings_rate",
    "negative_prior_earnings_rate",
    "mean_prior_eps_surprise",
    "std_prior_eps_surprise",
    "mean_prior_revenue_surprise",
    "std_prior_revenue_surprise",
    "mean_reaction_after_eps_beat",
    "mean_reaction_after_eps_miss",
    "mean_reaction_after_revenue_beat",
    "mean_reaction_after_revenue_miss",
    "mean_reaction_after_double_beat",
    "mean_reaction_after_double_miss",
    "similar_eps_surprise_mean_reaction",
    "similar_revenue_surprise_mean_reaction",
    "similar_earnings_event_mean_reaction",
    "similar_event_count",
    "has_company_earnings_history",
)


def _eps_surprise(row: EarningsRecord) -> float | None:
    if row.reported_eps is None or row.consensus_eps is None:
        return None
    return (float(row.reported_eps) - float(row.consensus_eps)) / max(abs(float(row.consensus_eps)), 0.05)


def _revenue_surprise(row: EarningsRecord) -> float | None:
    if row.reported_revenue is None or row.consensus_revenue is None:
        return None
    return (float(row.reported_revenue) - float(row.consensus_revenue)) / max(abs(float(row.consensus_revenue)), 1.0)


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _shrunk(values: list[float], prior: float | None, prior_strength: float) -> float:
    if not values:
        return float(prior or 0.0)
    if prior is None or prior_strength <= 0:
        return mean(values)
    return (sum(values) + prior_strength * float(prior)) / (len(values) + prior_strength)


def company_history_features_v3(
    history: tuple[EarningsRecord, ...],
    cutoff,
    *,
    current_eps_surprise: float | None = None,
    current_revenue_surprise: float | None = None,
    shrinkage_priors: dict[str, float] | None = None,
    prior_strength: float = 5.0,
    similar_k: int = 5,
) -> dict[str, float]:
    rows = [r for r in history if r.eligible(cutoff) and r.value_timestamp < cutoff]
    rows.sort(key=lambda r: r.value_timestamp)
    reactions = [float(r.abnormal_return) for r in rows if r.abnormal_return is not None]
    eps_pairs = [(r, _eps_surprise(r)) for r in rows]
    rev_pairs = [(r, _revenue_surprise(r)) for r in rows]
    eps_values = [float(v) for _, v in eps_pairs if v is not None]
    rev_values = [float(v) for _, v in rev_pairs if v is not None]
    priors = shrinkage_priors or {}

    def reactions_where(predicate) -> list[float]:
        return [float(r.abnormal_return) for r in rows if r.abnormal_return is not None and predicate(r)]

    eps_beat = reactions_where(lambda r: (_eps_surprise(r) or 0.0) > 0)
    eps_miss = reactions_where(lambda r: (_eps_surprise(r) or 0.0) < 0)
    rev_beat = reactions_where(lambda r: (_revenue_surprise(r) or 0.0) > 0)
    rev_miss = reactions_where(lambda r: (_revenue_surprise(r) or 0.0) < 0)
    double_beat = reactions_where(lambda r: (_eps_surprise(r) or 0.0) > 0 and (_revenue_surprise(r) or 0.0) > 0)
    double_miss = reactions_where(lambda r: (_eps_surprise(r) or 0.0) < 0 and (_revenue_surprise(r) or 0.0) < 0)

    def similar(target: float | None, extractor) -> list[float]:
        if target is None:
            return []
        candidates = []
        for r in rows:
            value = extractor(r)
            if value is None or r.abnormal_return is None:
                continue
            candidates.append((abs(float(value) - target), float(r.abnormal_return)))
        candidates.sort(key=lambda x: x[0])
        return [reaction for _, reaction in candidates[:similar_k]]

    similar_eps = similar(current_eps_surprise, _eps_surprise)
    similar_rev = similar(current_revenue_surprise, _revenue_surprise)
    both = []
    if current_eps_surprise is not None and current_revenue_surprise is not None:
        candidates = []
        for r in rows:
            e, v = _eps_surprise(r), _revenue_surprise(r)
            if e is None or v is None or r.abnormal_return is None:
                continue
            distance = ((float(e) - current_eps_surprise) ** 2 + (float(v) - current_revenue_surprise) ** 2) ** 0.5
            candidates.append((distance, float(r.abnormal_return)))
        candidates.sort(key=lambda x: x[0])
        both = [reaction for _, reaction in candidates[:similar_k]]

    return {
        "prior_earnings_count": float(len(rows)),
        "mean_prior_earnings_abnormal_return": _avg(reactions),
        "median_prior_earnings_abnormal_return": median(reactions) if reactions else 0.0,
        "std_prior_earnings_abnormal_return": pstdev(reactions) if len(reactions) >= 2 else 0.0,
        "positive_prior_earnings_rate": sum(x > 0 for x in reactions) / len(reactions) if reactions else 0.0,
        "negative_prior_earnings_rate": sum(x < 0 for x in reactions) / len(reactions) if reactions else 0.0,
        "mean_prior_eps_surprise": _avg(eps_values),
        "std_prior_eps_surprise": pstdev(eps_values) if len(eps_values) >= 2 else 0.0,
        "mean_prior_revenue_surprise": _avg(rev_values),
        "std_prior_revenue_surprise": pstdev(rev_values) if len(rev_values) >= 2 else 0.0,
        "mean_reaction_after_eps_beat": _shrunk(eps_beat, priors.get("eps_beat"), prior_strength),
        "mean_reaction_after_eps_miss": _shrunk(eps_miss, priors.get("eps_miss"), prior_strength),
        "mean_reaction_after_revenue_beat": _shrunk(rev_beat, priors.get("revenue_beat"), prior_strength),
        "mean_reaction_after_revenue_miss": _shrunk(rev_miss, priors.get("revenue_miss"), prior_strength),
        "mean_reaction_after_double_beat": _shrunk(double_beat, priors.get("double_beat"), prior_strength),
        "mean_reaction_after_double_miss": _shrunk(double_miss, priors.get("double_miss"), prior_strength),
        "similar_eps_surprise_mean_reaction": _avg(similar_eps),
        "similar_revenue_surprise_mean_reaction": _avg(similar_rev),
        "similar_earnings_event_mean_reaction": _avg(both),
        "similar_event_count": float(len(both) if both else max(len(similar_eps), len(similar_rev))),
        "has_company_earnings_history": float(bool(rows)),
    }
