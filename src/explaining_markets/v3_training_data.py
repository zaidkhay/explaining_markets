"""Build and persist point-in-time-safe V3 training rows.

The competition archive contains both prediction-time disclosure and realized
post-event outcomes. This module keeps those roles separate:

* CAR1 is used only to construct the within-quarter target percentile.
* current-event earnings surprise is used only as an optional benchmark.
* V3 model inputs come from disclosure plus records that were already
  available before the focal event cutoff.

The archive-only builder is intentionally a *seed* dataset. It can populate
FLS and prior-company-reaction history without fabricating unavailable EPS,
revenue, guidance, price, peer, news, or reasoning history.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import pstdev
from typing import Iterable

from explaining_markets.backtest import percentile_ranks
from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.historical import HistoricalEvent, labeled_events, load_historical_events
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.v3_records import EarningsRecord, V3Context
from explaining_markets.v3_training import V3TrainingRow

DEFAULT_ROWS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "v3_training_rows.jsonl.gz"
DEFAULT_EVENT_TYPES = ("EARNINGS_RELEASE",)
PRIOR_REACTION_AVAILABILITY_LAG = timedelta(days=2)

FAMILY_AVAILABILITY_FIELDS = {
    "eps": "has_eps_surprise",
    "revenue": "has_revenue_surprise",
    "guidance": "has_numeric_guidance",
    "guidance_consensus": "has_guidance_consensus",
    "price_5y": "has_5y_price_history",
    "company_history": "has_company_earnings_history",
    "peers": "has_peer_data",
    "company_news": "has_company_news",
    "peer_news": "has_peer_news",
    "sector_news": "has_sector_news",
    "reasoning": "has_reasoning",
}


@dataclass(frozen=True)
class TrainingDataReport:
    rows: int
    quarter_counts: dict[str, int]
    family_coverage: dict[str, float]
    active_non_fls_features: int
    non_fls_features_with_variance: tuple[str, ...]
    target_min: float | None
    target_max: float | None
    target_std: float | None
    archive_seed_only: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _key(event: HistoricalEvent) -> tuple[str, str]:
    return event.event_id, event.ticker


def _quarter_percentiles(events: Iterable[HistoricalEvent], *, attr: str) -> dict[tuple[str, str], float]:
    groups: dict[str, list[HistoricalEvent]] = {}
    for event in events:
        if event.quarter and getattr(event, attr) is not None:
            groups.setdefault(event.quarter, []).append(event)
    out: dict[tuple[str, str], float] = {}
    for group in groups.values():
        ranks = percentile_ranks([float(getattr(event, attr)) for event in group])
        for event, rank in zip(group, ranks, strict=True):
            out[_key(event)] = float(rank)
    return out


def target_percentiles(events: Iterable[HistoricalEvent]) -> dict[tuple[str, str], float]:
    """Competition-style CAR1 percentile labels, independently by quarter."""
    return _quarter_percentiles(events, attr="car1")


def surprise_percentiles(events: Iterable[HistoricalEvent]) -> dict[tuple[str, str], float]:
    """Quarterly earnings-surprise percentile benchmark; never a V3 input."""
    return _quarter_percentiles(events, attr="earnings_surprise")


def _prior_company_history(
    target: HistoricalEvent,
    timeline: list[HistoricalEvent],
    *,
    retrieved_at: datetime,
) -> tuple[EarningsRecord, ...]:
    cutoff = _parse_dt(target.event_datetime)
    if cutoff is None:
        return ()
    rows: list[EarningsRecord] = []
    for prior in timeline:
        if prior.event_id == target.event_id or prior.event_type != "EARNINGS_RELEASE":
            continue
        prior_dt = _parse_dt(prior.event_datetime)
        if prior_dt is None or prior_dt >= cutoff or prior.car1 is None:
            continue
        available_at = prior_dt + PRIOR_REACTION_AVAILABILITY_LAG
        if available_at > cutoff:
            continue
        rows.append(
            EarningsRecord(
                value_timestamp=prior_dt,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source="competition_archive_prior_reaction",
                ticker=target.ticker,
                abnormal_return=float(prior.car1),
                event_id=prior.event_id,
            )
        )
    rows.sort(key=lambda row: row.value_timestamp)
    return tuple(rows)


def build_archive_seed_rows(
    events: list[HistoricalEvent] | None = None,
    *,
    source: str | Path | None = None,
    event_types: tuple[str, ...] = DEFAULT_EVENT_TYPES,
) -> list[V3TrainingRow]:
    """Build leakage-safe archive seed rows for V3 research.

    Labels are ranked against all labeled competition observations in the
    quarter before event-type filtering. Focal-event inputs use only disclosure
    plus prior same-ticker reactions that were available before the cutoff.
    """
    loaded = list(events) if events is not None else load_historical_events(source)
    labeled = labeled_events(loaded)
    targets = target_percentiles(labeled)
    surprise_benchmark = surprise_percentiles(labeled)

    by_ticker: dict[str, list[HistoricalEvent]] = {}
    for event in loaded:
        by_ticker.setdefault(event.ticker, []).append(event)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    for timeline in by_ticker.values():
        timeline.sort(key=lambda event: _parse_dt(event.event_datetime) or epoch)

    retrieved_at = datetime.now(timezone.utc)
    allowed = set(event_types)
    rows: list[V3TrainingRow] = []
    for event in labeled:
        if allowed and event.event_type not in allowed:
            continue
        if not event.quarter:
            continue
        cutoff = _parse_dt(event.event_datetime)
        target = targets.get(_key(event))
        if cutoff is None or target is None:
            continue
        context = V3Context(
            ticker=event.ticker,
            cutoff=cutoff,
            company_history=_prior_company_history(
                event,
                by_ticker.get(event.ticker, []),
                retrieved_at=retrieved_at,
            ),
            extras={"training_source": "competition_archive_seed"},
        )
        audit = audit_context(context)
        vector = build_feature_vector_v3(disclosure=list(event.disclosure), context=context)
        values = {name: float(vector.values[name]) for name in MODEL_FEATURE_NAMES_V3}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite V3 feature in archive row {event.event_id}/{event.ticker}")
        rows.append(
            V3TrainingRow(
                event_id=event.event_id,
                ticker=event.ticker,
                quarter=event.quarter,
                target_percentile=float(target),
                values=values,
                surprise_percentile=surprise_benchmark.get(_key(event)),
                leakage_violations=int(audit.violations),
            )
        )
    return rows


def training_data_report(rows: list[V3TrainingRow], *, archive_seed_only: bool = False) -> TrainingDataReport:
    quarter_counts: dict[str, int] = {}
    for row in rows:
        quarter_counts[row.quarter] = quarter_counts.get(row.quarter, 0) + 1

    family_coverage = {
        family: (
            sum(float(row.values.get(field, 0.0)) > 0.0 for row in rows) / len(rows)
            if rows else 0.0
        )
        for family, field in FAMILY_AVAILABILITY_FIELDS.items()
    }
    non_fls = [name for name in MODEL_FEATURE_NAMES_V3 if name not in MODEL_FEATURE_NAMES]
    variable = tuple(
        name
        for name in non_fls
        if len(rows) >= 2 and pstdev([float(row.values[name]) for row in rows]) > 1e-12
    )
    targets = [float(row.target_percentile) for row in rows]
    return TrainingDataReport(
        rows=len(rows),
        quarter_counts=dict(sorted(quarter_counts.items())),
        family_coverage=family_coverage,
        active_non_fls_features=len(variable),
        non_fls_features_with_variance=variable,
        target_min=min(targets) if targets else None,
        target_max=max(targets) if targets else None,
        target_std=pstdev(targets) if len(targets) >= 2 else None,
        archive_seed_only=bool(archive_seed_only),
    )


def validate_training_rows(rows: list[V3TrainingRow]) -> None:
    seen: set[tuple[str, str]] = set()
    expected = set(MODEL_FEATURE_NAMES_V3)
    for row in rows:
        key = (row.event_id, row.ticker)
        if key in seen:
            raise ValueError(f"duplicate V3 training row: {row.event_id}/{row.ticker}")
        seen.add(key)
        if not row.quarter:
            raise ValueError(f"row {row.event_id}/{row.ticker} has no quarter")
        if not 0.0 <= float(row.target_percentile) <= 1.0:
            raise ValueError(f"row {row.event_id}/{row.ticker} target is outside [0, 1]")
        if row.leakage_violations:
            raise ValueError(f"row {row.event_id}/{row.ticker} reports leakage violations")
        actual = set(row.values)
        if actual != expected:
            raise ValueError(
                f"row {row.event_id}/{row.ticker} feature schema mismatch; "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        if not all(math.isfinite(float(value)) for value in row.values.values()):
            raise ValueError(f"row {row.event_id}/{row.ticker} contains non-finite features")


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_training_rows(rows: list[V3TrainingRow], path: str | Path = DEFAULT_ROWS_PATH) -> Path:
    validate_training_rows(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(output, "w") as fh:
        for row in rows:
            # Preserve the frozen feature order inside ``values`` for easier
            # human inspection and reproducible diffs of generated rows.
            fh.write(json.dumps(asdict(row)) + "\n")
    return output


def load_training_rows(path: str | Path = DEFAULT_ROWS_PATH) -> list[V3TrainingRow]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    rows: list[V3TrainingRow] = []
    with _open_text(source, "r") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
                raw_values = raw["values"]
                # Rebuild in frozen production order regardless of how an
                # external JSON writer ordered object keys.
                values = {name: float(raw_values[name]) for name in MODEL_FEATURE_NAMES_V3}
                row = V3TrainingRow(
                    event_id=str(raw["event_id"]),
                    ticker=str(raw["ticker"]),
                    quarter=str(raw["quarter"]),
                    target_percentile=float(raw["target_percentile"]),
                    values=values,
                    surprise_percentile=(
                        None if raw.get("surprise_percentile") is None
                        else float(raw["surprise_percentile"])
                    ),
                    leakage_violations=int(raw.get("leakage_violations", 0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid V3 training row at {source}:{line_number}: {exc}") from exc
            rows.append(row)
    validate_training_rows(rows)
    return rows
