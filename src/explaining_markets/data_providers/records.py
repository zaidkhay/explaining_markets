"""Provenance-carrying record types shared by every data provider.

These are deliberately vendor-neutral: a Polygon bar, an Alpha Vantage bar,
and a cached SQLite row all normalize into the same :class:`PriceBar` /
:class:`EarningsRecord`, so the feature layer never knows which vendor it is
talking to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _require_aware(value: datetime, name: str) -> datetime:
    """Reject naive datetimes — cutoff comparisons must never be ambiguous."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class PriceBar:
    """One daily adjusted price observation with full provenance.

    ``value_timestamp`` is the session close time (UTC). ``available_at`` is
    when this close was publicly knowable — for a normal daily close this is
    the close itself, but a vendor's late back-adjustment (split/dividend
    revision) must set it to the revision time instead. When a vendor cannot
    supply ``available_at``, the cache stores it equal to ``value_timestamp``
    only for plain closes; adjusted-history rewrites must be timestamped or
    excluded.
    """

    ticker: str
    value_timestamp: datetime  # session close, UTC
    adjusted_close: float
    source: str
    available_at: datetime
    retrieved_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.value_timestamp, "value_timestamp")
        _require_aware(self.available_at, "available_at")
        _require_aware(self.retrieved_at, "retrieved_at")
        if not (self.adjusted_close > 0.0):
            raise ValueError(f"adjusted_close must be positive, got {self.adjusted_close}")

    def usable_at(self, cutoff: datetime) -> bool:
        """True when this observation was publicly knowable strictly before ``cutoff``."""
        return self.available_at < _require_aware(cutoff, "cutoff")


@dataclass(frozen=True)
class EarningsRecord:
    """One historical earnings event for a company, with reaction fields.

    Reaction fields (``next_session_return`` / ``benchmark_next_session_return``
    / ``abnormal_return``) describe the FIRST full trading session after the
    report (see ``company_history.py`` for the exact window rules per report
    timing). They become knowable only after that session closes, so
    ``reaction_available_at`` is tracked separately from ``available_at``
    (which covers the reported figures themselves).

    ``abnormal_return = next_session_return - benchmark_next_session_return``
    against one consistent broad-market benchmark (see ``benchmark`` field).
    """

    ticker: str
    event_timestamp: datetime  # when the report was released, UTC
    source: str
    available_at: datetime  # when the reported figures were knowable
    retrieved_at: datetime

    eps_actual: float | None = None
    eps_estimate: float | None = None  # point-in-time consensus BEFORE the report
    eps_surprise: float | None = None
    eps_surprise_pct: float | None = None
    revenue_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_surprise: float | None = None

    next_session_return: float | None = None
    benchmark_next_session_return: float | None = None
    abnormal_return: float | None = None
    benchmark: str | None = None  # e.g. "SPY"; must be the same for all rows
    reaction_available_at: datetime | None = None

    competition_car1: float | None = None  # only when historically available

    def __post_init__(self) -> None:
        _require_aware(self.event_timestamp, "event_timestamp")
        _require_aware(self.available_at, "available_at")
        _require_aware(self.retrieved_at, "retrieved_at")
        if self.reaction_available_at is not None:
            _require_aware(self.reaction_available_at, "reaction_available_at")

    def figures_usable_at(self, cutoff: datetime) -> bool:
        """Reported EPS/revenue figures knowable strictly before ``cutoff``."""
        return self.available_at < _require_aware(cutoff, "cutoff")

    def reaction_usable_at(self, cutoff: datetime) -> bool:
        """Post-report reaction knowable strictly before ``cutoff``.

        Fails closed: when ``reaction_available_at`` is unknown but a reaction
        value is present, the reaction is treated as UNUSABLE — we refuse to
        guess when a return became knowable.
        """
        if self.abnormal_return is None and self.next_session_return is None:
            return False
        if self.reaction_available_at is None:
            return False
        return self.reaction_available_at < _require_aware(cutoff, "cutoff")
