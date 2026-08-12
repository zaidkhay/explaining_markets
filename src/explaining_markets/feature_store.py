"""Point-in-time-safe historical feature store.

Builds the `SAFE_IF_TIMESTAMPED` rolling/aggregate features specified in
``docs/PREDICTION_TIME_INFORMATION_AUDIT.md`` §5 (previous CAR1, rolling
mean/volatility, previous earnings surprise, rolling surprise, count of
positive surprises, historical reaction asymmetry, count of prior earnings
events) from the historical events loaded by
:mod:`explaining_markets.historical`.

The single governing rule, restated from the audit, is:

    For a target event E (ticker t, event_datetime T), a feature may only be
    built from events for the SAME ticker with event_datetime STRICTLY
    before T. Never the target event itself. Never an event at or after T.

This module enforces that rule at three independent points, on purpose,
rather than trusting any one of them alone:

1. Structurally — :func:`eligible_prior_events` is the only way source
   events reach a feature computation, and it filters on the rule above.
2. At runtime — :func:`assert_no_target_leakage` re-checks the same
   invariant on whatever was actually selected, immediately before use, so a
   future refactor of the selection logic cannot silently reintroduce
   leakage without a test catching it.
3. On the output — every :class:`HistoricalFeatures` instance carries a
   :class:`ProvenanceRecord` per feature, naming the exact source event(s)
   used and the timestamp rule they satisfied, so leakage is auditable after
   the fact, not just prevented at construction time.

Realized, post-event fields on the TARGET event itself (its own ``car1``,
``earnings_surprise``, ``event_returns``, ``baseline_predictions``) never
enter this module at all — every function here takes a single ``target``
event only to read its ``ticker``/``event_id``/``event_datetime``, never its
outcome fields. This is a design property, not just a convention: nothing in
this module has a code path that reads ``target.car1`` or
``target.earnings_surprise``.

Nothing in this module is wired into ``predict.py`` or ``model.py`` — it is
offline research infrastructure, matching the pattern already established by
``backtest.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, stdev

from explaining_markets.features import FORBIDDEN_KEYS
from explaining_markets.historical import HistoricalEvent

# Trailing window size (in qualifying prior events, not calendar time) used
# by every windowed feature below. A single, documented constant rather than
# a per-feature magic number, so all windowed features share one, auditable
# definition of "recent." `historical_reaction_asymmetry` deliberately does
# NOT use this window — see its docstring in `compute_historical_features`.
DEFAULT_WINDOW = 4


@dataclass(frozen=True)
class ProvenanceRecord:
    """One prior event's contribution to one feature of one target event.

    Exists so a feature's value is never just a number — it is always
    accompanied by exactly which source event(s) produced it and why they
    were safe to use, per docs/PREDICTION_TIME_INFORMATION_AUDIT.md.
    """

    feature_name: str
    source_event_id: str
    source_ticker: str
    source_event_datetime: str
    target_event_id: str
    target_event_datetime: str
    rule: str


@dataclass(frozen=True)
class HistoricalFeatures:
    """Point-in-time-safe rolling/aggregate features for one target event.

    Every field below is computed ONLY from events strictly earlier
    (``event_datetime`` < the target's) for the same ticker — see the module
    docstring. None of these fields, nor anything that fed into them, is
    ever the target event's own ``car1``/``earnings_surprise``/
    ``event_returns``/``baseline_predictions``.

    Fields are ``None`` (never zero-filled) when there is insufficient
    qualifying history — see ``compute_historical_features`` for the exact
    per-field minimum-history rule.
    """

    ticker: str
    target_event_id: str
    target_event_datetime: str

    previous_car1: float | None
    rolling_mean_car1: float | None
    rolling_car1_volatility: float | None
    previous_earnings_surprise: float | None
    rolling_mean_surprise: float | None
    number_of_previous_positive_surprises: int
    historical_reaction_asymmetry: float | None
    number_of_prior_earnings_events: int

    provenance: dict[str, tuple[ProvenanceRecord, ...]] = field(default_factory=dict)

    def feature_values(self) -> dict:
        """The 8 numeric feature values only — what a model would consume.

        Deliberately excludes ``provenance`` (not a feature; an audit trail)
        and the target's own identity/timestamp fields (not signal about the
        company, only bookkeeping about which row this is).
        """
        return {
            "previous_car1": self.previous_car1,
            "rolling_mean_car1": self.rolling_mean_car1,
            "rolling_car1_volatility": self.rolling_car1_volatility,
            "previous_earnings_surprise": self.previous_earnings_surprise,
            "rolling_mean_surprise": self.rolling_mean_surprise,
            "number_of_previous_positive_surprises": self.number_of_previous_positive_surprises,
            "historical_reaction_asymmetry": self.historical_reaction_asymmetry,
            "number_of_prior_earnings_events": self.number_of_prior_earnings_events,
        }

    def as_dict(self) -> dict:
        """``feature_values()`` plus the ticker, for joining/inspection."""
        return {"ticker": self.ticker, **self.feature_values()}


# ----------------------------------------------------------------------
# Walk-forward backbone
# ----------------------------------------------------------------------


def build_ticker_timelines(events: Iterable[HistoricalEvent]) -> dict[str, list[HistoricalEvent]]:
    """Group events by ticker, each timeline sorted chronologically.

    This is the walk-forward backbone: for any target event, its eligible
    history (`eligible_prior_events`) is always a strict, chronological
    prefix of its OWN ticker's timeline — never another ticker's, and never
    anything after the target.
    """
    by_ticker: dict[str, list[HistoricalEvent]] = {}
    for event in events:
        by_ticker.setdefault(event.ticker, []).append(event)
    for timeline in by_ticker.values():
        timeline.sort(key=lambda e: _parse_dt(e.event_datetime) or _EPOCH)
    return by_ticker


def eligible_prior_events(
    timeline: Sequence[HistoricalEvent], target: HistoricalEvent
) -> list[HistoricalEvent]:
    """Every event in ``timeline`` strictly before ``target`` (same ticker, own timeline).

    "Strictly before" is evaluated on parsed datetimes, not raw string
    comparison, so differing but equivalent ISO-8601 formatting never causes
    a false pass or fail. An event with no parseable ``event_datetime``
    (target or candidate) is excluded rather than guessed. The target event
    itself (matched by ``event_id``) is always excluded, even if some
    upstream bug caused it to appear twice in ``timeline``.
    """
    target_dt = _parse_dt(target.event_datetime)
    if target_dt is None:
        return []
    out = []
    for event in timeline:
        if event.event_id == target.event_id:
            continue
        event_dt = _parse_dt(event.event_datetime)
        if event_dt is not None and event_dt < target_dt:
            out.append(event)
    return out


def assert_no_target_leakage(
    target: HistoricalEvent, source_events: Iterable[HistoricalEvent]
) -> None:
    """Raise if any candidate source event is not strictly earlier than ``target``.

    This is the feature store's runtime safety net: it re-checks, at the
    point of use, the exact invariant :func:`eligible_prior_events` is
    supposed to already guarantee — so a future bug in the selection logic
    is caught here rather than silently producing a leaking feature. Two
    failure modes are checked: the source event IS the target (same
    ``event_id``), or the source event's ``event_datetime`` is not strictly
    before the target's (including the case where either timestamp fails to
    parse — treated as unsafe, never as "probably fine").
    """
    target_dt = _parse_dt(target.event_datetime)
    if target_dt is None:
        raise ValueError(
            f"target event {target.event_id!r} has no parseable event_datetime; "
            "cannot establish a walk-forward boundary"
        )
    for source in source_events:
        if source.event_id == target.event_id:
            raise ValueError(
                f"leakage: source event {source.event_id!r} is the target event itself"
            )
        source_dt = _parse_dt(source.event_datetime)
        if source_dt is None or not source_dt < target_dt:
            raise ValueError(
                f"leakage: source event {source.event_id!r} "
                f"(event_datetime={source.event_datetime!r}) is not strictly earlier than "
                f"target {target.event_id!r} (event_datetime={target.event_datetime!r})"
            )


# ----------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------


def compute_historical_features(
    target: HistoricalEvent,
    timeline: Sequence[HistoricalEvent],
    *,
    window: int = DEFAULT_WINDOW,
) -> HistoricalFeatures:
    """Build every rolling/aggregate feature for one target event.

    ``timeline`` should be that target's own ticker's full, chronologically
    sorted event list (see :func:`build_ticker_timelines`) — passing a
    different ticker's timeline is harmless (the target simply won't be
    found among earlier events, since ``eligible_prior_events`` matches on
    the target's own timestamp regardless of ticker), but is not the
    intended use; :func:`build_feature_store` always pairs a target with its
    own ticker's timeline.

    Per-field minimum-history rules (fields are ``None``/``0`` rather than
    guessed when unmet):

    * ``previous_car1`` / ``previous_earnings_surprise`` — need >= 1
      qualifying prior event (``car1`` or ``earnings_surprise`` respectively
      not ``None`` on that prior event).
    * ``rolling_mean_car1`` / ``rolling_mean_surprise`` /
      ``number_of_previous_positive_surprises`` — need >= 1 qualifying prior
      event within the trailing ``window``; use whatever is available up to
      ``window``, never fewer than 1 and never padded to reach ``window``.
    * ``rolling_car1_volatility`` — needs >= 2 qualifying prior events within
      the window (`statistics.stdev` is undefined for n < 2).
    * ``historical_reaction_asymmetry`` — needs at least one prior event with
      a positive ``car1`` AND at least one with a negative ``car1``, over
      the FULL available car1 history (deliberately unwindowed, matching the
      audit's literal formula in §5 — this is the one feature in this family
      that is not bounded by ``window``).
    * ``number_of_prior_earnings_events`` — counts ALL prior events for the
      ticker, regardless of whether their outcome (``car1``) is known to us;
      this mirrors the audit's finding that mere event *existence* has a
      different, better availability profile than outcome-bearing features.
    """
    prior = eligible_prior_events(timeline, target)
    assert_no_target_leakage(target, prior)

    car1_history = [e for e in prior if e.car1 is not None]
    surprise_history = [e for e in prior if e.earnings_surprise is not None]
    window_car1 = car1_history[-window:] if window > 0 else []
    window_surprise = surprise_history[-window:] if window > 0 else []

    provenance: dict[str, tuple[ProvenanceRecord, ...]] = {}

    previous_car1 = None
    if car1_history:
        last = car1_history[-1]
        previous_car1 = last.car1
        provenance["previous_car1"] = _provenance(
            "previous_car1", [last], target, "most recent prior event with car1 realized"
        )

    rolling_mean_car1 = mean(e.car1 for e in window_car1) if window_car1 else None
    if window_car1:
        provenance["rolling_mean_car1"] = _provenance(
            "rolling_mean_car1",
            window_car1,
            target,
            f"trailing window of up to {window} prior car1-realized events",
        )

    rolling_car1_volatility = None
    if len(window_car1) >= 2:
        rolling_car1_volatility = stdev(e.car1 for e in window_car1)
        provenance["rolling_car1_volatility"] = _provenance(
            "rolling_car1_volatility",
            window_car1,
            target,
            f"trailing window of up to {window} prior car1-realized events (n>=2 required)",
        )

    previous_earnings_surprise = None
    if surprise_history:
        last_surprise = surprise_history[-1]
        previous_earnings_surprise = last_surprise.earnings_surprise
        provenance["previous_earnings_surprise"] = _provenance(
            "previous_earnings_surprise",
            [last_surprise],
            target,
            "most recent prior event with earnings_surprise realized (surprise_status ok)",
        )

    rolling_mean_surprise = mean(e.earnings_surprise for e in window_surprise) if window_surprise else None
    if window_surprise:
        provenance["rolling_mean_surprise"] = _provenance(
            "rolling_mean_surprise",
            window_surprise,
            target,
            f"trailing window of up to {window} prior surprise-realized events",
        )

    number_of_previous_positive_surprises = sum(
        1 for e in window_surprise if e.earnings_surprise is not None and e.earnings_surprise > 0
    )
    if window_surprise:
        provenance["number_of_previous_positive_surprises"] = _provenance(
            "number_of_previous_positive_surprises",
            window_surprise,
            target,
            f"trailing window of up to {window} prior surprise-realized events; counted where surprise > 0",
        )

    positive = [e.car1 for e in car1_history if e.car1 is not None and e.car1 > 0]
    negative = [e.car1 for e in car1_history if e.car1 is not None and e.car1 < 0]
    historical_reaction_asymmetry = None
    if positive and negative:
        historical_reaction_asymmetry = mean(positive) - abs(mean(negative))
        provenance["historical_reaction_asymmetry"] = _provenance(
            "historical_reaction_asymmetry",
            car1_history,
            target,
            "full available car1-realized history (unwindowed, per audit formula)",
        )

    number_of_prior_earnings_events = len(prior)
    if prior:
        provenance["number_of_prior_earnings_events"] = _provenance(
            "number_of_prior_earnings_events",
            prior,
            target,
            "counts prior event existence only, regardless of realized outcome",
        )

    return HistoricalFeatures(
        ticker=target.ticker,
        target_event_id=target.event_id,
        target_event_datetime=target.event_datetime or "",
        previous_car1=previous_car1,
        rolling_mean_car1=rolling_mean_car1,
        rolling_car1_volatility=rolling_car1_volatility,
        previous_earnings_surprise=previous_earnings_surprise,
        rolling_mean_surprise=rolling_mean_surprise,
        number_of_previous_positive_surprises=number_of_previous_positive_surprises,
        historical_reaction_asymmetry=historical_reaction_asymmetry,
        number_of_prior_earnings_events=number_of_prior_earnings_events,
        provenance=provenance,
    )


def build_feature_store(
    events: Iterable[HistoricalEvent], *, window: int = DEFAULT_WINDOW
) -> list[HistoricalFeatures]:
    """Build point-in-time-safe historical features for every event in ``events``.

    Walk-forward by construction: events are grouped into per-ticker,
    chronologically sorted timelines (:func:`build_ticker_timelines`), and
    every event, acting as a target, only ever sees events strictly earlier
    in that SAME ticker's timeline (:func:`eligible_prior_events`, re-checked
    by :func:`assert_no_target_leakage`).

    Passing the full, multi-quarter historical dataset (e.g. all 6,287
    sealed-quarter events) rather than one quarter at a time is intentional
    and safe: it lets a target event in a later quarter see realized
    outcomes from earlier quarters — exactly the intended use of these
    features per docs/PREDICTION_TIME_INFORMATION_AUDIT.md §5. Quarter
    boundaries are irrelevant to this module; only per-ticker chronology
    matters.

    Returns one :class:`HistoricalFeatures` row per input event, in no
    particular overall order (each ticker's rows are internally
    chronological; use ``target_event_datetime`` to re-sort globally if
    needed).
    """
    timelines = build_ticker_timelines(events)
    out: list[HistoricalFeatures] = []
    for timeline in timelines.values():
        for target in timeline:
            out.append(compute_historical_features(target, timeline, window=window))
    return out


def assert_feature_is_leakage_free(features: HistoricalFeatures) -> None:
    """Raise if a built :class:`HistoricalFeatures`' values include a forbidden field.

    Belt-and-suspenders: ``HistoricalFeatures`` has no field named ``car1``
    etc. by construction, so this should never trigger — but it costs
    nothing to check, and it guards against a future field being added to
    the dataclass without updating this module's leakage discipline.
    """
    leaked = [k for k in features.feature_values() if k in FORBIDDEN_KEYS]
    if leaked:
        raise ValueError(f"leaked realized field(s) into historical features: {leaked}")


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _provenance(
    feature_name: str,
    sources: Sequence[HistoricalEvent],
    target: HistoricalEvent,
    rule_detail: str,
) -> tuple[ProvenanceRecord, ...]:
    return tuple(
        ProvenanceRecord(
            feature_name=feature_name,
            source_event_id=source.event_id,
            source_ticker=source.ticker,
            source_event_datetime=source.event_datetime or "",
            target_event_id=target.event_id,
            target_event_datetime=target.event_datetime or "",
            rule=(
                f"source_event_datetime({source.event_datetime}) < "
                f"target_event_datetime({target.event_datetime}); {rule_detail}"
            ),
        )
        for source in sources
    )
