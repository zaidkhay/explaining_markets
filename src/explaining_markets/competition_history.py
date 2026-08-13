"""Point-in-time company history sourced from the sealed competition archive.

The sealed archive (``data/historical/``) is the one legal, real source of
per-ticker earnings history available to this repository today: each prior
event carries a realized CAR1 (the competition's own next-session abnormal
return) and an EPS surprise. This module adapts those prior events into
:class:`~explaining_markets.company_history.HistoricalEarningsEvent` records
under a CONSERVATIVE availability rule, for two consumers:

* TRAINING (walk-forward): for each focal archive event, only same-ticker
  prior events whose outcomes were conservatively knowable before the focal
  event may contribute (:func:`eligible_prior_archive_events`).
* LIVE: any event received now occurs strictly after everything in the sealed
  archive, so a prebuilt per-ticker snapshot of the full archive is
  point-in-time valid for all live events. The snapshot
  (:func:`build_snapshot` / :class:`SnapshotCompanyHistoryProvider`) is a
  compact JSON shipped inside the package's ``artifacts/`` directory so the
  Modal image mounts it automatically.

Conservative availability rule
------------------------------

The archive records each outcome's value (CAR1, surprise) but NOT the moment
it became publicly knowable (``returns_computed_at`` is a batch job months
later and would be uselessly conservative; the true availability is ~2
trading sessions after the prior event). We therefore treat a prior event's
outcome as available only

    prior_event_datetime + AVAILABILITY_LAG_DAYS (7 calendar days)

and additionally require it to precede the focal event by at least
``CUTOFF_GUARD_DAYS`` (1 day) — covering the gap between the focal event's
``knowledge_cutoff`` and its ``event_datetime`` (the archive's
``knowledge_cutoff`` always precedes ``event_datetime``; see
``docs/PREDICTION_TIME_INFORMATION_AUDIT.md``). Quarterly spacing (~90 days)
means this costs essentially nothing while failing closed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from explaining_markets.company_history import (
    CompanyHistoryFeatures,
    HistoricalEarningsEvent,
    compute_company_history_features,
)
from explaining_markets.feature_store import build_ticker_timelines
from explaining_markets.historical import HistoricalEvent


def parse_event_datetime(event: HistoricalEvent) -> datetime | None:
    """Parse an archive event's datetime; fail closed on missing/naive/invalid.

    Same ISO handling as ``feature_store._parse_dt`` but additionally rejects
    timezone-naive values: a timestamp whose zone is ambiguous cannot prove
    availability ordering, so it is excluded rather than guessed.
    """
    raw = event.event_datetime
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return None
    return dt.astimezone(timezone.utc)

AVAILABILITY_LAG_DAYS = 7
CUTOFF_GUARD_DAYS = 1
SNAPSHOT_VERSION = "company_history_snapshot_v1"
DEFAULT_SNAPSHOT_PATH = Path(__file__).with_name("artifacts") / "company_history_snapshot_v1.json"

COMPETITION_FEATURE_NAMES: tuple[str, ...] = (
    "prior_competition_event_count",
    "mean_prior_competition_car1",
    "last_prior_competition_car1",
    "has_competition_history",
)


def outcome_available_at(prior_event_datetime: datetime) -> datetime:
    """Conservative moment a prior archive event's outcome became knowable."""
    return prior_event_datetime + timedelta(days=AVAILABILITY_LAG_DAYS)


def eligible_prior_archive_events(
    timeline: list[HistoricalEvent], target: HistoricalEvent
) -> list[HistoricalEvent]:
    """Same-ticker prior events whose OUTCOMES were conservatively knowable.

    Requires ``outcome_available_at(prior) < target_event_datetime - CUTOFF_GUARD_DAYS``.
    Unparseable timestamps fail closed (excluded / empty result).
    """
    target_dt = parse_event_datetime(target)
    if target_dt is None:
        return []
    latest_allowed = target_dt - timedelta(days=CUTOFF_GUARD_DAYS)
    out: list[HistoricalEvent] = []
    for candidate in timeline:
        if candidate.event_id == target.event_id:
            continue
        candidate_dt = parse_event_datetime(candidate)
        if candidate_dt is None:
            continue
        if outcome_available_at(candidate_dt) < latest_allowed:
            out.append(candidate)
    return out


def to_history_event(event: HistoricalEvent) -> HistoricalEarningsEvent | None:
    """Adapt one archive event into the vendor-neutral prior-event record."""
    dt = parse_event_datetime(event)
    if dt is None:
        return None
    return HistoricalEarningsEvent(
        event_timestamp=dt,
        eps_surprise=event.earnings_surprise,
        abnormal_return=event.car1,  # the competition's own CAR1
        source="competition_archive",
        source_event_id=event.event_id,
    )


def competition_feature_values(
    prior_events: list[HistoricalEarningsEvent],
) -> dict[str, float | None]:
    """The Part-8 competition-specific aggregate features."""
    car1s = [e.abnormal_return for e in prior_events if e.abnormal_return is not None]
    values: dict[str, float | None] = {
        "prior_competition_event_count": float(len(prior_events)),
        "mean_prior_competition_car1": None,
        "last_prior_competition_car1": None,
        "has_competition_history": 1.0 if car1s else 0.0,
    }
    if car1s:
        values["mean_prior_competition_car1"] = sum(car1s) / len(car1s)
        values["last_prior_competition_car1"] = car1s[-1]
    return values


def walk_forward_history(
    events: list[HistoricalEvent],
) -> dict[str, CompanyHistoryFeatures]:
    """TRAINING-side history: event_id -> CompanyHistoryFeatures, walk-forward.

    For each archive event, features are built exclusively from same-ticker
    prior events passing :func:`eligible_prior_archive_events`. Cross-ticker
    isolation comes from ``build_ticker_timelines`` grouping. Keys are
    ``(event_id, ticker)``-unique because archive rows are one per
    (event, ticker) already; the dict is keyed by ``f"{event_id}:{ticker}"``.
    """
    out: dict[str, CompanyHistoryFeatures] = {}
    timelines = build_ticker_timelines(events)
    for ticker, timeline in timelines.items():
        for target in timeline:
            target_dt = parse_event_datetime(target)
            if target_dt is None:
                continue
            prior = [
                he
                for e in eligible_prior_archive_events(timeline, target)
                if (he := to_history_event(e)) is not None
            ]
            out[f"{target.event_id}:{ticker}"] = compute_company_history_features(
                ticker=ticker, cutoff=target_dt, prior_events=prior
            )
    return out


# ----------------------------------------------------------------------
# Live snapshot: compact per-ticker archive history shipped with the package
# ----------------------------------------------------------------------


def build_snapshot(events: list[HistoricalEvent]) -> dict:
    """Serialize per-ticker (event_datetime, car1, surprise) triples.

    Valid for live use because every live event is strictly after the sealed
    archive; the live provider still applies the availability-lag rule against
    the live cutoff, so a hypothetical cutoff inside the archive period would
    be handled correctly too.
    """
    tickers: dict[str, list[dict]] = {}
    for ticker, timeline in sorted(build_ticker_timelines(events).items()):
        rows = []
        for event in timeline:
            dt = parse_event_datetime(event)
            if dt is None:
                continue
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_datetime": dt.astimezone(timezone.utc).isoformat(),
                    "car1": event.car1,
                    "earnings_surprise": event.earnings_surprise,
                }
            )
        if rows:
            tickers[ticker] = rows
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "source": "sealed Explaining Markets archive (data/historical/)",
        "availability_lag_days": AVAILABILITY_LAG_DAYS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(tickers),
        "n_rows": sum(len(v) for v in tickers.values()),
        "tickers": tickers,
    }


def write_snapshot(events: list[HistoricalEvent], path: str | Path = DEFAULT_SNAPSHOT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_snapshot(events), sort_keys=True) + "\n", encoding="utf-8")
    return path


class SnapshotCompanyHistoryProvider:
    """Live ``CompanyHistoryProvider`` reading the packaged snapshot JSON.

    Millisecond loads, no network. Applies the same conservative
    availability-lag rule per prior event against the mandatory cutoff.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_SNAPSHOT_PATH
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("snapshot_version") != SNAPSHOT_VERSION:
            raise ValueError(f"unexpected snapshot version {raw.get('snapshot_version')!r}")
        self.generated_at = str(raw.get("generated_at"))
        self._tickers: dict[str, list[dict]] = raw.get("tickers") or {}

    def history_before(self, ticker: str, cutoff: datetime) -> CompanyHistoryFeatures:
        rows = self._tickers.get(ticker) or []
        latest_allowed = cutoff - timedelta(days=CUTOFF_GUARD_DAYS)
        prior: list[HistoricalEarningsEvent] = []
        for row in rows:
            dt = datetime.fromisoformat(row["event_datetime"])
            if outcome_available_at(dt) < latest_allowed:
                prior.append(
                    HistoricalEarningsEvent(
                        event_timestamp=dt,
                        eps_surprise=row.get("earnings_surprise"),
                        abnormal_return=row.get("car1"),
                        source="competition_archive",
                        source_event_id=row.get("event_id"),
                    )
                )
        return compute_company_history_features(ticker=ticker, cutoff=cutoff, prior_events=prior)
