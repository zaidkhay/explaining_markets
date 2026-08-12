"""Local, leakage-guarded backtesting for percentile-prediction models.

Runs entirely offline against :mod:`explaining_markets.historical`'s loaded
events — no network calls. Reproduces the competition's own within-quarter,
cross-sectional percentile-rank target (min -> 0, max -> 1, ties share the
average rank, a single value -> 0.5) so a model's backtested performance is
comparable, in spirit, to how the live leaderboard would score it. This is a
self-contained reimplementation of that transform — it does not import the
separate ``examples`` research repo, since that repo is not part of the
deployed image.

Leakage discipline, enforced at every step:

* Features are built from ``event.disclosure`` / ``event_type`` / ``ticker``
  ONLY, via :func:`explaining_markets.features.extract_features`.
* :func:`explaining_markets.features.assert_no_leakage` runs on every
  features dict before it reaches a model.
* ``car1`` and ``earnings_surprise`` are read only to build the realized
  target/benchmark — never passed to ``model.predict_percentile``.
* Percentile ranks are computed independently **within each quarter's own
  labeled subset** — never across the whole historical set at once, and
  never mixing quarters.
* Train/test splitting is chronological (:func:`temporal_split`), never
  random, so a "test" quarter can never leak into "train".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from explaining_markets.features import FeatureVector, assert_no_leakage, extract_features
from explaining_markets.historical import HistoricalEvent, labeled_events
from explaining_markets.model import PercentileModel


def percentile_ranks(values: list[float]) -> list[float]:
    """Rank each value into ``[0, 1]`` within its own list; ties share the average rank.

    Same semantics as the competition's own scoring transform: min -> 0,
    max -> 1, a single value -> 0.5, empty -> ``[]``.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    denom = n - 1
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = ((i + j) / 2.0) / denom
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


@dataclass(frozen=True)
class BacktestRow:
    """One scored observation from a backtest run."""

    event_id: str
    ticker: str
    quarter: str | None
    predicted_percentile: float
    realized_percentile: float  # y, ranked from car1 within its quarter
    surprise_percentile: float | None  # naive benchmark, ranked from earnings_surprise


@dataclass(frozen=True)
class BacktestResult:
    """Summary + row-level detail from :func:`run_backtest`."""

    rows: list[BacktestRow]
    n_obs: int
    correlation: float | None  # Pearson r between predicted and realized percentile
    mean_abs_error: float | None
    beats_surprise_benchmark: bool | None


def temporal_split(
    events: list[HistoricalEvent], *, holdout_quarters: int = 1
) -> tuple[list[HistoricalEvent], list[HistoricalEvent]]:
    """Split chronologically by quarter — never randomly.

    Every quarter in the returned ``test`` set is strictly later (by lexical
    ``YYYYQn`` ordering) than every quarter in ``train``, so a model trained
    on ``train`` can never have seen anything from ``test``'s time period.
    Events with no ``quarter`` tag are excluded from both (there is no safe
    way to place them on a timeline).
    """
    quarters = sorted({e.quarter for e in events if e.quarter})
    if len(quarters) <= holdout_quarters:
        return [e for e in events if e.quarter], []
    holdout = set(quarters[-holdout_quarters:]) if holdout_quarters > 0 else set()
    train = [e for e in events if e.quarter and e.quarter not in holdout]
    test = [e for e in events if e.quarter and e.quarter in holdout]
    return train, test


def build_training_rows(events: list[HistoricalEvent]) -> list[tuple[FeatureVector, float]]:
    """``(features, realized_percentile)`` pairs suitable for ``model.fit``.

    Uses the same within-quarter percentile ranking as :func:`run_backtest`,
    computed only over ``events`` (so a caller who passes only the ``train``
    half of :func:`temporal_split` never leaks a held-out quarter's
    cross-section into the label).
    """
    rows: list[tuple[FeatureVector, float]] = []
    for quarter_events in _group_by_quarter(labeled_events(events)).values():
        y_values = percentile_ranks([e.car1 for e in quarter_events])  # type: ignore[misc]
        for event, y in zip(quarter_events, y_values, strict=True):
            features = extract_features(
                ticker=event.ticker, event_type=event.event_type, disclosure=event.disclosure
            )
            assert_no_leakage(features.as_dict())
            rows.append((features, y))
    return rows


def run_backtest(events: list[HistoricalEvent], model: PercentileModel) -> BacktestResult:
    """Evaluate ``model`` on every quarter's own realized cross-section.

    See the module docstring for the leakage guarantees this function
    enforces. Events without a realized ``car1`` are skipped (there is no
    label to score against, matching the competition's own "assets whose
    return leg carries no car1 cannot be scored" rule).
    """
    rows: list[BacktestRow] = []
    for quarter, quarter_events in _group_by_quarter(labeled_events(events)).items():
        car1_values = [e.car1 for e in quarter_events]
        y_values = percentile_ranks(car1_values)  # type: ignore[arg-type]

        surprise_idx = [
            i for i, e in enumerate(quarter_events) if e.earnings_surprise is not None
        ]
        surprise_ranks = percentile_ranks(
            [quarter_events[i].earnings_surprise for i in surprise_idx]  # type: ignore[misc]
        )
        surprise_pct_by_idx = dict(zip(surprise_idx, surprise_ranks, strict=True))

        for i, event in enumerate(quarter_events):
            features = extract_features(
                ticker=event.ticker, event_type=event.event_type, disclosure=event.disclosure
            )
            assert_no_leakage(features.as_dict())
            predicted = float(model.predict_percentile(features))
            rows.append(
                BacktestRow(
                    event_id=event.event_id,
                    ticker=event.ticker,
                    quarter=quarter,
                    predicted_percentile=predicted,
                    realized_percentile=y_values[i],
                    surprise_percentile=surprise_pct_by_idx.get(i),
                )
            )
    return _summarize(rows)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _group_by_quarter(events: list[HistoricalEvent]) -> dict[str, list[HistoricalEvent]]:
    groups: dict[str, list[HistoricalEvent]] = {}
    for event in events:
        key = event.quarter or "UNKNOWN"
        groups.setdefault(key, []).append(event)
    return groups


def _summarize(rows: list[BacktestRow]) -> BacktestResult:
    n = len(rows)
    if n == 0:
        return BacktestResult(
            rows=rows, n_obs=0, correlation=None, mean_abs_error=None, beats_surprise_benchmark=None
        )

    predicted = [r.predicted_percentile for r in rows]
    realized = [r.realized_percentile for r in rows]
    correlation = _pearson(predicted, realized)
    mean_abs_error = sum(abs(p - r) for p, r in zip(predicted, realized, strict=True)) / n

    beats_benchmark = None
    surprise_rows = [r for r in rows if r.surprise_percentile is not None]
    if len(surprise_rows) >= 2:
        model_corr = _pearson(
            [r.predicted_percentile for r in surprise_rows],
            [r.realized_percentile for r in surprise_rows],
        )
        surprise_corr = _pearson(
            [r.surprise_percentile for r in surprise_rows],  # type: ignore[misc]
            [r.realized_percentile for r in surprise_rows],
        )
        if model_corr is not None and surprise_corr is not None:
            beats_benchmark = abs(model_corr) > abs(surprise_corr)

    return BacktestResult(
        rows=rows,
        n_obs=n,
        correlation=correlation,
        mean_abs_error=mean_abs_error,
        beats_surprise_benchmark=beats_benchmark,
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    s_xx = sum((x - mean_x) ** 2 for x in xs)
    s_yy = sum((y - mean_y) ** 2 for y in ys)
    if s_xx == 0.0 or s_yy == 0.0:
        return None
    s_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return s_xy / math.sqrt(s_xx * s_yy)
