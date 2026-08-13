"""Point-in-time-safe company-history features (prices, earnings reactions).

This is the feature layer behind ``fls_company_history_ridge_v2``. It answers,
strictly from information available before a mandatory cutoff:

* How has this company historically reacted to earnings?
* How volatile is the stock; how has it performed over 3m..5y?
* How unusual is the current surprise vs the company's own history?
* What happened after historically similar surprises?

Cutoff rules (documented once, enforced everywhere)
----------------------------------------------------

The focal event's cutoff ``T`` is the prediction knowledge cutoff. Every
source observation must satisfy ``available_at < T`` (strict). Additional
conservatism:

* PRICES — a daily close is available at the close itself. The "final
  eligible pre-event close" is the last session close with
  ``available_at < T``; for a before-market (BMO) report that is the prior
  day's close, for an after-market (AMC) report it is the same day's close,
  for an intraday report it is the previous close (the in-progress session
  has not closed). Weekends/holidays need no special-casing: no close exists
  on those days, so the last eligible close is simply the most recent actual
  session. The focal event's own post-event reaction can never enter: it
  happens at/after ``T``.
* EARNINGS REACTIONS — the reaction window is the FIRST full trading session
  after the report (BMO: the report day's own session; AMC/intraday: the next
  session), matching the competition's CAR1 convention.
  ``abnormal_return = next_session_return - benchmark_next_session_return``
  against one broad benchmark used consistently for all rows.
  A prior event's reaction is usable only when its
  ``reaction_available_at < T``; if unknown, it is EXCLUDED (fail closed).
* COMPETITION ARCHIVE — prior competition outcomes (CAR1/surprise) get a
  conservative availability rule (see ``competition_history.py``): outcome
  fields are treated as available only ``AVAILABILITY_LAG_DAYS`` after the
  prior event, plus a 1-day guard before ``T``, because the archive does not
  record the exact moment each outcome became public.

Imputation policy
-----------------

Missing history is NEVER silently treated as an economic zero: every family
carries an explicit availability indicator (``has_*`` / counts), and the
model matrix imputes missing values with the neutral 0.0 AFTER recording the
indicator. Standardization means a 0.0 with indicator 0 is distinguishable
from a real 0.0 with indicator 1.

Current-event surprise
----------------------

``current_eps_surprise*`` features have full plumbing here but are populated
ONLY when a legal, point-in-time source proves availability before ``T``.
Per ``docs/PREDICTION_TIME_INFORMATION_AUDIT.md``, no such live source exists
today (the webhook payload and information_url disclosure carry no
reported-vs-estimate figures), so these features are neutral-with-indicator-0
in BOTH training and live — keeping train/serve distributions identical.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence, runtime_checkable

# Trading-session approximations for calendar windows (NYSE ~252 sessions/yr).
SESSIONS_3M = 63
SESSIONS_6M = 126
SESSIONS_1Y = 252
SESSIONS_3Y = 756
SESSIONS_5Y = 1260

SIMILAR_K = 3  # nearest prior surprises; chosen a priori, tunable on validation only
RECENCY_HALF_LIFE_DAYS = 365.0  # exponential decay half-life; validation-tunable only
MIN_SURPRISE_HISTORY_FOR_ZSCORE = 3
ZSCORE_CLIP = 5.0

PRICE_FEATURE_NAMES: tuple[str, ...] = (
    "return_3m", "return_6m", "return_1y", "return_3y", "return_5y",
    "volatility_3m", "volatility_1y",
    "max_drawdown_1y", "max_drawdown_5y",
)
EARNINGS_FEATURE_NAMES: tuple[str, ...] = (
    "prior_earnings_count",
    "mean_prior_earnings_abnormal_return",
    "median_prior_earnings_abnormal_return",
    "std_prior_earnings_abnormal_return",
    "positive_prior_earnings_rate",
)
SURPRISE_FEATURE_NAMES: tuple[str, ...] = (
    "mean_prior_eps_surprise",
    "std_prior_eps_surprise",
    "positive_eps_surprise_rate",
    "negative_eps_surprise_rate",
    "mean_reaction_after_positive_surprise",
    "mean_reaction_after_negative_surprise",
)
CURRENT_VS_HISTORY_FEATURE_NAMES: tuple[str, ...] = (
    "current_eps_surprise",
    "current_eps_surprise_zscore",
    "current_eps_surprise_percentile_company",
)
SIMILAR_FEATURE_NAMES: tuple[str, ...] = (
    "similar_surprise_mean_reaction",
    "similar_surprise_median_reaction",
    "similar_surprise_count",
)
RECENCY_FEATURE_NAMES: tuple[str, ...] = (
    "recency_weighted_earnings_reaction",
)
AVAILABILITY_FEATURE_NAMES: tuple[str, ...] = (
    "has_1y_price_history",
    "has_5y_price_history",
    "has_eps_surprise_history",
    "has_similar_surprise_history",
)

COMPANY_HISTORY_FEATURE_NAMES: tuple[str, ...] = (
    PRICE_FEATURE_NAMES
    + EARNINGS_FEATURE_NAMES
    + SURPRISE_FEATURE_NAMES
    + CURRENT_VS_HISTORY_FEATURE_NAMES
    + SIMILAR_FEATURE_NAMES
    + RECENCY_FEATURE_NAMES
    + AVAILABILITY_FEATURE_NAMES
)


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class HistoricalPriceStats:
    """Price-derived history for one ticker as of a cutoff. None = unavailable."""

    n_sessions: int
    return_3m: float | None = None
    return_6m: float | None = None
    return_1y: float | None = None
    return_3y: float | None = None
    return_5y: float | None = None
    volatility_3m: float | None = None
    volatility_1y: float | None = None
    max_drawdown_1y: float | None = None
    max_drawdown_5y: float | None = None


@dataclass(frozen=True)
class HistoricalEarningsEvent:
    """Normalized view of one PRIOR earnings event, already cutoff-filtered.

    Constructed only by adapters that have verified availability before the
    focal cutoff (see ``eligible`` factories in ``competition_history.py`` and
    the ``EarningsRecord.reaction_usable_at`` rule). ``abnormal_return`` is the
    CAR1-like next-session market-adjusted reaction; for sealed competition
    events it IS the historical CAR1.
    """

    event_timestamp: datetime
    eps_surprise: float | None = None
    abnormal_return: float | None = None
    source: str = "unknown"
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        _require_aware(self.event_timestamp, "event_timestamp")


@dataclass(frozen=True)
class CompanyHistoryFeatures:
    """The full company-history feature block for one (event, ticker)."""

    ticker: str
    cutoff: datetime
    values: dict[str, float | None]
    # Provenance: which prior events fed the earnings-history features.
    source_events: tuple[HistoricalEarningsEvent, ...] = ()

    def vector_value(self, name: str) -> float | None:
        return self.values[name]

    def as_dict(self) -> dict[str, float | None]:
        return dict(self.values)


@runtime_checkable
class CompanyHistoryProvider(Protocol):
    """The single interface the model layer depends on. Cutoff is mandatory."""

    def history_before(self, ticker: str, cutoff: datetime) -> CompanyHistoryFeatures: ...


# ----------------------------------------------------------------------
# Price features
# ----------------------------------------------------------------------


def compute_price_stats(closes: Sequence[float]) -> HistoricalPriceStats:
    """Compute price features from ascending, cutoff-filtered adjusted closes.

    ``closes[-1]`` is the final eligible pre-event close. Each ``return_X``
    needs the full window (e.g. 252 sessions for 1y) or is None — a partial
    window is not silently shortened. Volatility is the stdev of daily simple
    returns over the window (needs >= 2 returns). Drawdown is the max
    peak-to-trough decline within the window, reported as a positive number.
    """

    n = len(closes)
    if n < 2:
        return HistoricalPriceStats(n_sessions=n)

    def window_return(sessions: int) -> float | None:
        if n <= sessions:
            return None
        past, last = closes[-1 - sessions], closes[-1]
        return None if past <= 0 else (last / past) - 1.0

    def window_volatility(sessions: int) -> float | None:
        if n <= sessions:
            return None
        window = closes[-1 - sessions:]
        rets = [window[i] / window[i - 1] - 1.0 for i in range(1, len(window)) if window[i - 1] > 0]
        return statistics.pstdev(rets) if len(rets) >= 2 else None

    def window_drawdown(sessions: int) -> float | None:
        if n <= sessions:
            return None
        window = closes[-1 - sessions:]
        peak, worst = window[0], 0.0
        for price in window:
            peak = max(peak, price)
            if peak > 0:
                worst = max(worst, 1.0 - price / peak)
        return worst

    return HistoricalPriceStats(
        n_sessions=n,
        return_3m=window_return(SESSIONS_3M),
        return_6m=window_return(SESSIONS_6M),
        return_1y=window_return(SESSIONS_1Y),
        return_3y=window_return(SESSIONS_3Y),
        return_5y=window_return(SESSIONS_5Y),
        volatility_3m=window_volatility(SESSIONS_3M),
        volatility_1y=window_volatility(SESSIONS_1Y),
        max_drawdown_1y=window_drawdown(SESSIONS_1Y),
        max_drawdown_5y=window_drawdown(SESSIONS_5Y),
    )


# ----------------------------------------------------------------------
# Earnings-reaction / surprise-history features
# ----------------------------------------------------------------------


def compute_company_history_features(
    *,
    ticker: str,
    cutoff: datetime,
    prior_events: Sequence[HistoricalEarningsEvent],
    price_stats: HistoricalPriceStats | None = None,
    current_eps_surprise: float | None = None,
    current_surprise_available: bool = False,
    similar_k: int = SIMILAR_K,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> CompanyHistoryFeatures:
    """Assemble the full company-history feature block.

    ``prior_events`` must ALREADY be cutoff-eligible (callers/adapters own the
    availability rule); this function still fail-closes on any event whose
    timestamp is not strictly before ``cutoff``.

    ``current_eps_surprise`` may be passed only with
    ``current_surprise_available=True``, i.e. when the caller has verified a
    legal point-in-time source with ``data_available_at <= cutoff``.
    """
    cutoff = _require_aware(cutoff, "cutoff")
    for ev in prior_events:
        if ev.event_timestamp >= cutoff:
            raise ValueError(
                f"prior event {ev.source_event_id or ev.event_timestamp} is not strictly "
                f"before the cutoff {cutoff.isoformat()} — refusing to build features"
            )
    if current_eps_surprise is not None and not current_surprise_available:
        raise ValueError(
            "current_eps_surprise was supplied without current_surprise_available=True; "
            "a verified point-in-time source is required"
        )

    events = sorted(prior_events, key=lambda e: e.event_timestamp)
    reactions = [e.abnormal_return for e in events if e.abnormal_return is not None]
    surprises = [e.eps_surprise for e in events if e.eps_surprise is not None]
    reaction_pairs = [
        (e.eps_surprise, e.abnormal_return)
        for e in events
        if e.eps_surprise is not None and e.abnormal_return is not None
    ]

    values: dict[str, float | None] = {name: None for name in COMPANY_HISTORY_FEATURE_NAMES}

    # ---- price family -----------------------------------------------------
    stats = price_stats or HistoricalPriceStats(n_sessions=0)
    for name in PRICE_FEATURE_NAMES:
        values[name] = getattr(stats, name)
    values["has_1y_price_history"] = 1.0 if stats.return_1y is not None else 0.0
    values["has_5y_price_history"] = 1.0 if stats.return_5y is not None else 0.0

    # ---- earnings reaction family ------------------------------------------
    values["prior_earnings_count"] = float(len(events))
    if reactions:
        values["mean_prior_earnings_abnormal_return"] = statistics.fmean(reactions)
        values["median_prior_earnings_abnormal_return"] = statistics.median(reactions)
        values["positive_prior_earnings_rate"] = sum(1 for r in reactions if r > 0) / len(reactions)
    if len(reactions) >= 2:
        values["std_prior_earnings_abnormal_return"] = statistics.stdev(reactions)

    # ---- surprise history family --------------------------------------------
    values["has_eps_surprise_history"] = 1.0 if surprises else 0.0
    if surprises:
        values["mean_prior_eps_surprise"] = statistics.fmean(surprises)
        values["positive_eps_surprise_rate"] = sum(1 for s in surprises if s > 0) / len(surprises)
        values["negative_eps_surprise_rate"] = sum(1 for s in surprises if s < 0) / len(surprises)
    if len(surprises) >= 2:
        values["std_prior_eps_surprise"] = statistics.stdev(surprises)
    positive_reactions = [r for s, r in reaction_pairs if s > 0]
    negative_reactions = [r for s, r in reaction_pairs if s < 0]
    if positive_reactions:
        values["mean_reaction_after_positive_surprise"] = statistics.fmean(positive_reactions)
    if negative_reactions:
        values["mean_reaction_after_negative_surprise"] = statistics.fmean(negative_reactions)

    # ---- current surprise vs company history ---------------------------------
    if current_surprise_available and current_eps_surprise is not None:
        values["current_eps_surprise"] = float(current_eps_surprise)
        if (
            len(surprises) >= MIN_SURPRISE_HISTORY_FOR_ZSCORE
            and values["std_prior_eps_surprise"] is not None
            and values["std_prior_eps_surprise"] > 1e-12
        ):
            z = (current_eps_surprise - values["mean_prior_eps_surprise"]) / values[
                "std_prior_eps_surprise"
            ]
            values["current_eps_surprise_zscore"] = max(-ZSCORE_CLIP, min(ZSCORE_CLIP, z))
        if surprises:
            below = sum(1 for s in surprises if s < current_eps_surprise)
            equal = sum(1 for s in surprises if s == current_eps_surprise)
            values["current_eps_surprise_percentile_company"] = (below + 0.5 * equal) / len(surprises)

    # ---- similar historical surprises (kNN on |surprise diff|) ----------------
    has_similar = 0.0
    if current_surprise_available and current_eps_surprise is not None and reaction_pairs:
        ranked = sorted(reaction_pairs, key=lambda sr: abs(current_eps_surprise - sr[0]))
        nearest = ranked[: max(1, similar_k)]
        neighbor_reactions = [r for _, r in nearest]
        values["similar_surprise_mean_reaction"] = statistics.fmean(neighbor_reactions)
        values["similar_surprise_median_reaction"] = statistics.median(neighbor_reactions)
        values["similar_surprise_count"] = float(len(nearest))
        has_similar = 1.0
    values["has_similar_surprise_history"] = has_similar

    # ---- recency-weighted reaction --------------------------------------------
    weighted = [
        (e.abnormal_return, _recency_weight(e.event_timestamp, cutoff, recency_half_life_days))
        for e in events
        if e.abnormal_return is not None
    ]
    if weighted:
        total = sum(w for _, w in weighted)
        if total > 0:
            values["recency_weighted_earnings_reaction"] = (
                sum(r * w for r, w in weighted) / total
            )

    for name, value in values.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"non-finite company-history feature {name}={value}")

    return CompanyHistoryFeatures(
        ticker=ticker, cutoff=cutoff, values=values, source_events=tuple(events)
    )


def empty_company_history(ticker: str, cutoff: datetime) -> CompanyHistoryFeatures:
    """The no-history block: all values None except zeroed counts/indicators."""
    return compute_company_history_features(ticker=ticker, cutoff=cutoff, prior_events=[])


def _recency_weight(event_ts: datetime, cutoff: datetime, half_life_days: float) -> float:
    age_days = max(0.0, (cutoff - event_ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / max(half_life_days, 1e-9))
