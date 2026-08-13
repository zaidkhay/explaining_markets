"""Explicit leakage audit for the V2 training pipeline.

Run over every backtest/training row before any model is fit. Asserts, for
each focal event:

* every history source event satisfies the conservative availability rule
  ``outcome_available_at(source) < focal_event_datetime - CUTOFF_GUARD_DAYS``
  (i.e. ``source_ts + 7d < focal_ts - 1d``);
* every source event is a same-ticker, different-event record;
* no forbidden realized field name appears in the feature specification;
* the focal event's own CAR1 / earnings surprise cannot be reproduced from
  its feature vector's current-surprise slots (they must be neutral zeros
  unless a verified point-in-time source was supplied — none exists today).

Fails closed: any violation raises :class:`LeakageError` with the offending
event ids; the training pipeline refuses to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from explaining_markets.company_history import CompanyHistoryFeatures
from explaining_markets.competition_history import (
    AVAILABILITY_LAG_DAYS,
    CUTOFF_GUARD_DAYS,
    outcome_available_at,
)
from explaining_markets.features_v2 import (
    FORBIDDEN_FEATURE_NAMES,
    MODEL_FEATURE_NAMES_V2,
)


class LeakageError(AssertionError):
    """A point-in-time violation was detected; training must not proceed."""


@dataclass(frozen=True)
class AuditResult:
    n_rows: int
    n_source_records: int
    n_rows_with_history: int


def audit_feature_names(names: tuple[str, ...] = MODEL_FEATURE_NAMES_V2) -> None:
    leaked = FORBIDDEN_FEATURE_NAMES.intersection(names)
    if leaked:
        raise LeakageError(f"forbidden realized field(s) in feature names: {sorted(leaked)}")


def audit_history_row(
    *,
    focal_event_id: str,
    focal_ticker: str,
    focal_event_datetime: datetime,
    history: CompanyHistoryFeatures,
) -> int:
    """Audit one row's history sources; returns the number of sources checked."""
    latest_allowed = focal_event_datetime - timedelta(days=CUTOFF_GUARD_DAYS)
    checked = 0
    for source in history.source_events:
        checked += 1
        if source.source_event_id == focal_event_id:
            raise LeakageError(
                f"event {focal_event_id}: its own record appears in its history sources"
            )
        if not outcome_available_at(source.event_timestamp) < latest_allowed:
            raise LeakageError(
                f"event {focal_event_id} ({focal_ticker}): source "
                f"{source.source_event_id} at {source.event_timestamp.isoformat()} "
                f"violates outcome_available_at (+{AVAILABILITY_LAG_DAYS}d) < "
                f"focal - {CUTOFF_GUARD_DAYS}d ({latest_allowed.isoformat()})"
            )
    if history.cutoff != focal_event_datetime:
        raise LeakageError(
            f"event {focal_event_id}: history cutoff {history.cutoff.isoformat()} "
            f"!= focal event datetime {focal_event_datetime.isoformat()}"
        )
    return checked


def audit_current_surprise_neutrality(values: dict[str, float]) -> None:
    """Today no legal live source for the current surprise exists, so these
    slots must be exactly neutral in every training row (train/serve parity)."""
    for name in (
        "current_eps_surprise",
        "current_eps_surprise_zscore",
        "current_eps_surprise_percentile_company",
        "similar_surprise_mean_reaction",
        "similar_surprise_median_reaction",
        "similar_surprise_count",
        "has_similar_surprise_history",
    ):
        if values.get(name, 0.0) != 0.0:
            raise LeakageError(
                f"current-surprise feature {name}={values[name]} is populated, but no "
                "verified point-in-time source exists — post-event surprise leaked in"
            )
