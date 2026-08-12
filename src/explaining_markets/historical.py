"""Historical event loading for offline research and backtesting.

Loads realized (historical) competition events from local gzip-JSONL archive
files, shaped like the competition's ``/archive`` endpoint (one JSON record
per line, with a ``disclosure`` block and, once realized/scored, an
``event_returns``/``metrics.earnings_surprise`` block).

This module is read-only offline research infrastructure: nothing here is
imported by ``modal_app.py``'s deployed image, and it never talks to the
network — it only reads whatever files you have already placed in
``data/historical/`` (see that directory's README for the expected layout
and how to populate it).

A :class:`HistoricalEvent` carries both the pre-event/event-time information
a live model could see (``disclosure``) and the post-event, realized fields
(``car1``, ``earnings_surprise``) needed to construct a training/evaluation
label. Treat the realized fields as strictly off-limits for feature
construction — see ``features.py`` and ``backtest.py`` for the enforced
separation; this module only stores them, it never feeds them to a model.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# markets/src/explaining_markets/historical.py -> markets/data/historical
DEFAULT_HISTORICAL_DIR = Path(__file__).resolve().parents[2] / "data" / "historical"


@dataclass(frozen=True)
class HistoricalEvent:
    """One realized ``(event, focal asset)`` observation for offline research.

    ``disclosure`` plus ``event_type``/``ticker`` are safe pre/at-event
    information. ``car1`` and ``earnings_surprise`` are realized, POST-EVENT
    fields that exist on this object only so ``backtest.py`` can construct a
    label and a benchmark — ``features.py`` must never read them.
    """

    event_id: str
    ticker: str
    event_type: str
    event_datetime: str | None = None
    disclosure: list[str] = field(default_factory=list)
    car1: float | None = None
    earnings_surprise: float | None = None
    quarter: str | None = None


def read_jsonl_gz(path: str | Path) -> Iterator[dict]:
    """Yield each record from a gzip- or plain-JSONL file, skipping blank lines."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_historical_events(source: str | Path | None = None) -> list[HistoricalEvent]:
    """Load every historical ``(event, ticker)`` observation found under ``source``.

    Defaults to :data:`DEFAULT_HISTORICAL_DIR` (``data/historical/``).
    Deliberately non-fatal: returns ``[]`` if the directory is missing, empty,
    or a file fails to parse — this is what lets ``predict.py`` and
    ``backtest.py`` degrade to a deterministic baseline rather than crash
    when no historical archive has been downloaded yet.
    """
    directory = Path(source) if source is not None else DEFAULT_HISTORICAL_DIR
    if not directory.exists() or not directory.is_dir():
        return []

    events: list[HistoricalEvent] = []
    for path in sorted(p for p in directory.glob("*.jsonl*") if p.is_file()):
        quarter_from_name = _quarter_from_filename(path)
        try:
            records = list(read_jsonl_gz(path))
        except (OSError, json.JSONDecodeError):
            continue
        for record in records:
            events.extend(_events_from_record(record, quarter_from_name))
    return events


def labeled_events(events: list[HistoricalEvent]) -> list[HistoricalEvent]:
    """Filter to observations that carry a realized ``car1`` (usable as a label)."""
    return [e for e in events if e.car1 is not None]


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _events_from_record(record: dict, quarter_from_name: str | None) -> list[HistoricalEvent]:
    event_id = record.get("event_id")
    if not event_id:
        return []
    event_type = record.get("event_type", "UNKNOWN")
    event_datetime = record.get("event_datetime")
    disclosure = _facts_from_record(record)
    surprise = _surprise_from_record(record)
    quarter = record.get("quarter") or quarter_from_name

    out: list[HistoricalEvent] = []
    for asset in record.get("focal_assets") or []:
        ticker = asset.get("identifier_value")
        if not ticker:
            continue
        out.append(
            HistoricalEvent(
                event_id=event_id,
                ticker=ticker,
                event_type=event_type,
                event_datetime=event_datetime,
                disclosure=disclosure,
                car1=_car1_for_ticker(record, ticker),
                earnings_surprise=surprise,
                quarter=quarter,
            )
        )
    return out


def _facts_from_record(record: dict) -> list[str]:
    """Pull the earnings-call facts out of a raw archive record, if present."""
    items = (record.get("disclosure") or {}).get("items") or []
    for item in items:
        if item.get("kind") == "facts":
            return [str(f) for f in (item.get("content") or [])]
    return []


def _car1_for_ticker(record: dict, ticker: str) -> float | None:
    leg: Any = (record.get("event_returns") or {}).get(ticker) or {}
    value = leg.get("car1") if isinstance(leg, dict) else None
    return float(value) if value is not None else None


def _surprise_from_record(record: dict) -> float | None:
    es = (record.get("metrics") or {}).get("earnings_surprise") or {}
    if es.get("surprise_status") != "ok":
        return None
    value = es.get("surprise")
    return float(value) if value is not None else None


def _quarter_from_filename(path: Path) -> str | None:
    """``EARNINGS_RELEASE_2025Q3.jsonl.gz`` -> ``"2025Q3"``."""
    stem = path.name.removesuffix(".jsonl.gz").removesuffix(".jsonl")
    _, _, quarter = stem.rpartition("_")
    return quarter or None
