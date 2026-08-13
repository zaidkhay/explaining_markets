"""SQLite-backed offline cache implementing both provider Protocols.

The cache is built OFFLINE (from a real vendor provider once credentials
exist) and read at live-prediction time in milliseconds. Nothing here talks
to a network.

Default location: ``data/company_history/company_history.sqlite`` (gitignored
— bulk history is never committed).

Schema
------
``prices``::

    ticker TEXT, value_timestamp TEXT (UTC ISO), adjusted_close REAL,
    source TEXT, available_at TEXT, retrieved_at TEXT
    PRIMARY KEY (ticker, value_timestamp, source)

``earnings``::

    ticker TEXT, event_timestamp TEXT (UTC ISO), source TEXT,
    available_at TEXT, retrieved_at TEXT,
    eps_actual REAL, eps_estimate REAL, eps_surprise REAL, eps_surprise_pct REAL,
    revenue_actual REAL, revenue_estimate REAL, revenue_surprise REAL,
    next_session_return REAL, benchmark_next_session_return REAL,
    abnormal_return REAL, benchmark TEXT, reaction_available_at TEXT,
    competition_car1 REAL
    PRIMARY KEY (ticker, event_timestamp, source)

All timestamps are stored as UTC ISO-8601 strings so lexicographic SQL
comparison equals chronological comparison. Point-in-time filtering happens
in SQL (``available_at < :cutoff``) AND is re-checked in Python via the
record types' own ``usable_at``/``figures_usable_at`` methods —
belt-and-suspenders, matching the rest of this repository.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from explaining_markets.data_providers.records import EarningsRecord, PriceBar

# markets/src/explaining_markets/data_providers/cache.py -> markets/data/company_history
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "company_history" / "company_history.sqlite"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    value_timestamp TEXT NOT NULL,
    adjusted_close REAL NOT NULL,
    source TEXT NOT NULL,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (ticker, value_timestamp, source)
);
CREATE INDEX IF NOT EXISTS idx_prices_lookup ON prices (ticker, available_at, value_timestamp);

CREATE TABLE IF NOT EXISTS earnings (
    ticker TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    eps_actual REAL,
    eps_estimate REAL,
    eps_surprise REAL,
    eps_surprise_pct REAL,
    revenue_actual REAL,
    revenue_estimate REAL,
    revenue_surprise REAL,
    next_session_return REAL,
    benchmark_next_session_return REAL,
    abnormal_return REAL,
    benchmark TEXT,
    reaction_available_at TEXT,
    competition_car1 REAL,
    PRIMARY KEY (ticker, event_timestamp, source)
);
CREATE INDEX IF NOT EXISTS idx_earnings_lookup ON earnings (ticker, available_at, event_timestamp);
"""


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class CompanyHistoryCache:
    """Read/write SQLite cache satisfying MarketDataProvider + EarningsDataProvider."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- write side (offline cache building only) ------------------------

    def upsert_prices(self, bars: list[PriceBar]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    b.ticker,
                    _iso(b.value_timestamp),
                    b.adjusted_close,
                    b.source,
                    _iso(b.available_at),
                    _iso(b.retrieved_at),
                )
                for b in bars
            ],
        )
        self._conn.commit()

    def upsert_earnings(self, records: list[EarningsRecord]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.ticker,
                    _iso(r.event_timestamp),
                    r.source,
                    _iso(r.available_at),
                    _iso(r.retrieved_at),
                    r.eps_actual,
                    r.eps_estimate,
                    r.eps_surprise,
                    r.eps_surprise_pct,
                    r.revenue_actual,
                    r.revenue_estimate,
                    r.revenue_surprise,
                    r.next_session_return,
                    r.benchmark_next_session_return,
                    r.abnormal_return,
                    r.benchmark,
                    _iso(r.reaction_available_at),
                    r.competition_car1,
                )
                for r in records
            ],
        )
        self._conn.commit()

    # ---- read side (live + offline; point-in-time filtered) ---------------

    def daily_prices_before(
        self, ticker: str, cutoff: datetime, *, max_days: int = 5 * 365
    ) -> list[PriceBar]:
        cutoff_iso = _iso(cutoff)
        if cutoff_iso is None:
            raise ValueError("cutoff is mandatory")
        rows = self._conn.execute(
            "SELECT ticker, value_timestamp, adjusted_close, source, available_at, retrieved_at"
            " FROM prices WHERE ticker = ? AND available_at < ?"
            " ORDER BY value_timestamp DESC LIMIT ?",
            (ticker, cutoff_iso, max_days),
        ).fetchall()
        bars = [
            PriceBar(
                ticker=row[0],
                value_timestamp=_dt(row[1]),
                adjusted_close=row[2],
                source=row[3],
                available_at=_dt(row[4]),
                retrieved_at=_dt(row[5]),
            )
            for row in reversed(rows)  # back to ascending order
        ]
        return [b for b in bars if b.usable_at(cutoff)]  # re-check in Python

    def earnings_before(
        self, ticker: str, cutoff: datetime, *, max_events: int = 40
    ) -> list[EarningsRecord]:
        cutoff_iso = _iso(cutoff)
        if cutoff_iso is None:
            raise ValueError("cutoff is mandatory")
        rows = self._conn.execute(
            "SELECT * FROM earnings WHERE ticker = ? AND available_at < ?"
            " ORDER BY event_timestamp DESC LIMIT ?",
            (ticker, cutoff_iso, max_events),
        ).fetchall()
        records = [
            EarningsRecord(
                ticker=row[0],
                event_timestamp=_dt(row[1]),
                source=row[2],
                available_at=_dt(row[3]),
                retrieved_at=_dt(row[4]),
                eps_actual=row[5],
                eps_estimate=row[6],
                eps_surprise=row[7],
                eps_surprise_pct=row[8],
                revenue_actual=row[9],
                revenue_estimate=row[10],
                revenue_surprise=row[11],
                next_session_return=row[12],
                benchmark_next_session_return=row[13],
                abnormal_return=row[14],
                benchmark=row[15],
                reaction_available_at=_dt(row[16]),
                competition_car1=row[17],
            )
            for row in reversed(rows)
        ]
        return [r for r in records if r.figures_usable_at(cutoff)]
