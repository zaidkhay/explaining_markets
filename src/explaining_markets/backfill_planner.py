"""Ticker-frequency-prioritized price backfill planning.

Why this exists
---------------
The archive contains ~6.3k training rows spread over ~2.6k unique tickers. The
previous enrichment loop walked rows in file order and discovered missing
tickers one row at a time, so a run with an 800-call daily budget spent its
calls on whatever happened to appear first — frequently single-row tickers —
and re-discovered the same missing symbol thousands of times.

This module inverts the loop:

    load training rows
        -> group rows by ticker (row demand)
        -> inspect the existing normalized price cache (already covered?)
        -> drop permanently unsupported symbols
        -> order the remainder by rows unlocked per API call
        -> fetch the highest-value symbols first

Ordering is deterministic: row frequency descending, then ticker ascending, so
two runs with the same inputs plan the same work.

"Covered" semantics
-------------------
A ticker counts as covered only when the cache holds a *usable* normalized
series for the exact span the enricher would request. An entry that is absent,
corrupt, empty, or shorter than ``min_usable_sessions`` is NOT covered, so
insufficient cache entries are re-fetched rather than silently accepted.

Feature-level coverage (does the series actually satisfy the 5-year window at
each focal cutoff?) is reported separately as a diagnostic, because a provider
returning a short history is a data-quality fact, not a scheduling bug.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from explaining_markets.historical import HistoricalEvent
from explaining_markets.providers.free_historical import (
    fmp_price_records,
    tiingo_price_records,
    twelve_data_price_records,
)
from explaining_markets.v3_records import PriceRecord

# The enricher requests just over five years before a ticker's first event so
# the 1260-session (5y) window can be satisfied at the earliest cutoff.
PRICE_LOOKBACK_DAYS = 5 * 366 + 45

# 1261 eligible sessions is what price_context.has_5y_price_history requires.
SESSIONS_FOR_5Y = 1261
# A series shorter than this is treated as a failed/corrupt fetch worth retrying.
MIN_USABLE_SESSIONS = 30

PRICE_NAMESPACES: tuple[str, ...] = ("tiingo_prices", "twelve_data_prices", "fmp_prices")


def _utc_date_key(value: datetime) -> str:
    return value.date().isoformat()


@dataclass(frozen=True)
class TickerDemand:
    """How much training-row value a single ticker represents."""

    ticker: str
    row_count: int
    first_cutoff: datetime
    last_cutoff: datetime

    @property
    def fetch_start(self) -> datetime:
        return self.first_cutoff - timedelta(days=PRICE_LOOKBACK_DAYS)

    @property
    def fetch_end(self) -> datetime:
        return self.last_cutoff

    def cache_key(self, namespace: str) -> str:
        """The exact cache key the enricher uses for this ticker/span."""
        base = f"{self.ticker}|{_utc_date_key(self.fetch_start)}|{_utc_date_key(self.fetch_end)}"
        return f"{base}|adjust=all" if namespace == "twelve_data_prices" else base

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "row_count": self.row_count,
            "first_cutoff": self.first_cutoff.isoformat(),
            "last_cutoff": self.last_cutoff.isoformat(),
            "fetch_start": self.fetch_start.isoformat(),
            "fetch_end": self.fetch_end.isoformat(),
        }


@dataclass(frozen=True)
class CoverageStatus:
    """Result of inspecting the cache for one ticker."""

    ticker: str
    covered: bool
    provider: str | None
    sessions: int
    satisfies_5y_at_first_cutoff: bool
    detail: str

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "covered": self.covered,
            "provider": self.provider,
            "sessions": self.sessions,
            "satisfies_5y_at_first_cutoff": self.satisfies_5y_at_first_cutoff,
            "detail": self.detail,
        }


def count_rows_by_ticker(tickers: Iterable[str]) -> Counter:
    """Row frequency per upper-cased ticker."""
    return Counter(str(t).upper() for t in tickers)


def build_ticker_demand(
    rows: Sequence,
    events: Sequence[HistoricalEvent] | None = None,
    *,
    cutoffs_by_key: dict[tuple[str, str], datetime] | None = None,
) -> tuple[TickerDemand, ...]:
    """Aggregate per-ticker row counts and cutoff bounds.

    ``rows`` are V3 training rows (``event_id``/``ticker``). Cutoffs come from
    ``cutoffs_by_key`` when supplied, else from the archive ``events``. Rows
    whose cutoff cannot be resolved still contribute to the row count via the
    ticker's other rows, but a ticker with no resolvable cutoff at all is
    skipped because no fetch span can be computed for it.
    """
    from explaining_markets.v3_training_data import _parse_dt

    resolved: dict[tuple[str, str], datetime] = dict(cutoffs_by_key or {})
    if not resolved and events:
        for event in events:
            parsed = _parse_dt(event.event_datetime)
            if parsed is not None:
                resolved[(event.event_id, event.ticker)] = parsed

    counts: Counter = Counter()
    bounds: dict[str, list[datetime]] = {}
    for row in rows:
        ticker = str(row.ticker).upper()
        counts[ticker] += 1
        cutoff = resolved.get((row.event_id, row.ticker))
        if cutoff is not None:
            bounds.setdefault(ticker, []).append(cutoff)

    out: list[TickerDemand] = []
    for ticker, count in counts.items():
        stamps = bounds.get(ticker)
        if not stamps:
            continue
        out.append(
            TickerDemand(
                ticker=ticker,
                row_count=int(count),
                first_cutoff=min(stamps),
                last_cutoff=max(stamps),
            )
        )
    return prioritize(out)


def prioritize(demands: Iterable[TickerDemand]) -> tuple[TickerDemand, ...]:
    """Deterministic priority: row count desc, then ticker asc."""
    return tuple(sorted(demands, key=lambda d: (-d.row_count, d.ticker)))


def _normalize_cached_prices(namespace: str, payload, ticker: str) -> tuple[PriceRecord, ...]:
    """Convert a cached provider payload into normalized price records."""
    retrieved_at = datetime.now(timezone.utc)
    if namespace == "tiingo_prices":
        return tiingo_price_records(payload, ticker, retrieved_at=retrieved_at) if isinstance(payload, list) else ()
    if namespace == "twelve_data_prices":
        values = payload.get("values") if isinstance(payload, dict) else None
        return twelve_data_price_records(values, ticker, retrieved_at=retrieved_at) if isinstance(values, list) else ()
    if namespace == "fmp_prices":
        return fmp_price_records(payload, ticker, retrieved_at=retrieved_at) if isinstance(payload, list) else ()
    return ()


class PriceCacheIndex:
    """Provider-independent view of what price history is already cached."""

    def __init__(
        self,
        cache,
        *,
        namespaces: tuple[str, ...] = PRICE_NAMESPACES,
        min_usable_sessions: int = MIN_USABLE_SESSIONS,
    ) -> None:
        self.cache = cache
        self.namespaces = namespaces
        self.min_usable_sessions = int(min_usable_sessions)

    def cached_prices(self, demand: TickerDemand) -> tuple[str | None, tuple[PriceRecord, ...]]:
        """Return the first usable cached series for this ticker/span."""
        best: tuple[str | None, tuple[PriceRecord, ...]] = (None, ())
        for namespace in self.namespaces:
            try:
                payload = self.cache.get(namespace, demand.cache_key(namespace))
            except (json.JSONDecodeError, OSError, ValueError):
                continue  # corrupt entry: treat as absent
            if payload is None:
                continue
            records = _normalize_cached_prices(namespace, payload, demand.ticker)
            if len(records) > len(best[1]):
                best = (namespace, records)
        return best

    def status(self, demand: TickerDemand) -> CoverageStatus:
        namespace, records = self.cached_prices(demand)
        sessions = len(records)
        if sessions == 0:
            return CoverageStatus(
                ticker=demand.ticker,
                covered=False,
                provider=None,
                sessions=0,
                satisfies_5y_at_first_cutoff=False,
                detail="no cached price series",
            )
        if sessions < self.min_usable_sessions:
            return CoverageStatus(
                ticker=demand.ticker,
                covered=False,
                provider=namespace,
                sessions=sessions,
                satisfies_5y_at_first_cutoff=False,
                detail=f"insufficient cached series ({sessions} < {self.min_usable_sessions} sessions)",
            )
        eligible_at_first = sum(1 for r in records if r.available_at <= demand.first_cutoff)
        return CoverageStatus(
            ticker=demand.ticker,
            covered=True,
            provider=namespace,
            sessions=sessions,
            satisfies_5y_at_first_cutoff=eligible_at_first >= SESSIONS_FOR_5Y,
            detail=f"cached {sessions} sessions via {namespace}",
        )


@dataclass(frozen=True)
class BackfillPlan:
    """Deterministic work plan plus the diagnostics the CLI reports."""

    total_rows: int
    unique_tickers: int
    tickers_already_covered: int
    tickers_needing_prices: int
    tickers_skipped_unsupported: int
    total_rows_already_covered: int
    total_rows_needing_prices: int
    rows_blocked_by_unsupported: int
    tickers_with_5y_history: int
    rows_with_5y_history: int
    queue: tuple[TickerDemand, ...] = ()
    covered: tuple[CoverageStatus, ...] = field(default=())
    skipped_unsupported: tuple[str, ...] = ()

    def projected_rows_covered_by_next(self, n: int) -> int:
        """Rows unlocked if the next ``n`` queued symbols all succeed."""
        return sum(d.row_count for d in self.queue[: max(0, int(n))])

    def next_symbols(self, n: int) -> tuple[TickerDemand, ...]:
        return self.queue[: max(0, int(n))]

    def as_dict(self, *, projections: Sequence[int] = (10, 50, 100, 250, 500, 800)) -> dict:
        return {
            "total_rows": self.total_rows,
            "unique_tickers": self.unique_tickers,
            "tickers_already_covered": self.tickers_already_covered,
            "tickers_needing_prices": self.tickers_needing_prices,
            "tickers_skipped_unsupported": self.tickers_skipped_unsupported,
            "total_rows_already_covered": self.total_rows_already_covered,
            "total_rows_needing_prices": self.total_rows_needing_prices,
            "rows_blocked_by_unsupported": self.rows_blocked_by_unsupported,
            "tickers_with_5y_history": self.tickers_with_5y_history,
            "rows_with_5y_history": self.rows_with_5y_history,
            "row_coverage_fraction": (
                self.total_rows_already_covered / self.total_rows if self.total_rows else 0.0
            ),
            "projected_rows_covered_by_next_N_symbols": {
                str(n): self.projected_rows_covered_by_next(n) for n in projections
            },
            "next_symbols": [d.as_dict() for d in self.next_symbols(25)],
            "skipped_unsupported_sample": list(self.skipped_unsupported[:25]),
        }


def plan_price_backfill(
    rows: Sequence,
    events: Sequence[HistoricalEvent],
    *,
    cache,
    unsupported_cache=None,
    min_usable_sessions: int = MIN_USABLE_SESSIONS,
    limit: int | None = None,
) -> BackfillPlan:
    """Build the prioritized fetch plan without making any network call."""
    demands = build_ticker_demand(rows, events)
    index = PriceCacheIndex(cache, min_usable_sessions=min_usable_sessions)

    covered: list[CoverageStatus] = []
    needing: list[TickerDemand] = []
    skipped: list[str] = []
    rows_covered = 0
    rows_needing = 0
    rows_blocked = 0
    tickers_5y = 0
    rows_5y = 0

    for demand in demands:
        status = index.status(demand)
        if status.covered:
            covered.append(status)
            rows_covered += demand.row_count
            if status.satisfies_5y_at_first_cutoff:
                tickers_5y += 1
                rows_5y += demand.row_count
            continue
        if unsupported_cache is not None and unsupported_cache.should_skip(demand.ticker):
            skipped.append(demand.ticker)
            rows_blocked += demand.row_count
            continue
        needing.append(demand)
        rows_needing += demand.row_count

    queue = prioritize(needing)
    if limit is not None:
        queue = queue[: max(0, int(limit))]

    return BackfillPlan(
        total_rows=len(rows),
        unique_tickers=len(demands),
        tickers_already_covered=len(covered),
        tickers_needing_prices=len(needing),
        tickers_skipped_unsupported=len(skipped),
        total_rows_already_covered=rows_covered,
        total_rows_needing_prices=rows_needing,
        rows_blocked_by_unsupported=rows_blocked,
        tickers_with_5y_history=tickers_5y,
        rows_with_5y_history=rows_5y,
        queue=queue,
        covered=tuple(covered),
        skipped_unsupported=tuple(sorted(skipped)),
    )


@dataclass(frozen=True)
class PriceFetcher:
    """One provider's price path: a budgeted client plus its normalizer."""

    name: str
    client: object
    namespace: str

    def fetch(self, demand: TickerDemand) -> tuple[PriceRecord, ...]:
        payload = self.client.prices_payload(  # type: ignore[attr-defined]
            demand.ticker, start=demand.fetch_start, end=demand.fetch_end
        )
        retrieved_at = datetime.now(timezone.utc)
        if self.namespace == "tiingo_prices":
            return tiingo_price_records(payload, demand.ticker, retrieved_at=retrieved_at)
        if self.namespace == "twelve_data_prices":
            return twelve_data_price_records(payload, demand.ticker, retrieved_at=retrieved_at)
        if self.namespace == "fmp_prices":
            return fmp_price_records(payload, demand.ticker, retrieved_at=retrieved_at)
        raise ValueError(f"unknown price namespace: {self.namespace}")


@dataclass
class BackfillOutcome:
    """What a prioritized prefetch pass actually accomplished."""

    prices_by_ticker: dict[str, tuple[PriceRecord, ...]] = field(default_factory=dict)
    successful_tickers: tuple[str, ...] = ()
    unsupported_tickers: tuple[str, ...] = ()
    transient_failed_tickers: tuple[str, ...] = ()
    rows_unlocked: int = 0
    symbols_attempted: int = 0
    stopped_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "successful_tickers": len(self.successful_tickers),
            "unsupported_tickers": len(self.unsupported_tickers),
            "transient_failed_tickers": len(self.transient_failed_tickers),
            "rows_unlocked": self.rows_unlocked,
            "symbols_attempted": self.symbols_attempted,
            "stopped_reason": self.stopped_reason,
            "successful_sample": list(self.successful_tickers[:25]),
            "unsupported_sample": list(self.unsupported_tickers[:25]),
            "transient_failed_sample": list(self.transient_failed_tickers[:25]),
        }


def run_price_backfill(
    plan: BackfillPlan,
    fetchers: Sequence[PriceFetcher],
    *,
    unsupported_cache=None,
    max_symbols: int | None = None,
    progress: object = None,
) -> BackfillOutcome:
    """Fetch queued symbols in priority order across the fetcher chain.

    Stops early when every fetcher has exhausted its budget or tripped a
    run-wide circuit breaker; a permanent rejection records the symbol in the
    unsupported cache so future runs skip it for free.
    """
    from explaining_markets.providers.free_historical import (
        ProviderBudgetExhausted,
        ProviderUnavailable,
    )
    from explaining_markets.providers.retry_policy import (
        TransientProviderError,
        UnsupportedSymbolError,
    )

    log = progress if callable(progress) else (lambda message: print(message, flush=True))
    outcome = BackfillOutcome()
    exhausted: set[str] = set()
    queue = plan.queue if max_symbols is None else plan.queue[: max(0, int(max_symbols))]

    for demand in queue:
        if len(exhausted) >= len(fetchers):
            outcome.stopped_reason = "all price providers exhausted or unavailable"
            break
        if unsupported_cache is not None and unsupported_cache.should_skip(demand.ticker):
            outcome.unsupported_tickers = (*outcome.unsupported_tickers, demand.ticker)
            continue

        outcome.symbols_attempted += 1
        for fetcher in fetchers:
            if fetcher.name in exhausted:
                continue
            try:
                records = fetcher.fetch(demand)
            except UnsupportedSymbolError:
                # Permanent for this symbol on this provider; try the next one.
                outcome.unsupported_tickers = (*outcome.unsupported_tickers, demand.ticker)
                continue
            except (ProviderBudgetExhausted, ProviderUnavailable) as exc:
                exhausted.add(fetcher.name)
                log(f"[V3_BACKFILL] {fetcher.name} exhausted/unavailable: {str(exc)[:200]}")
                continue
            except TransientProviderError as exc:
                outcome.transient_failed_tickers = (
                    *outcome.transient_failed_tickers,
                    demand.ticker,
                )
                log(f"[V3_BACKFILL] {fetcher.name} transient failure ticker={demand.ticker}: {str(exc)[:160]}")
                continue
            except Exception as exc:  # noqa: BLE001 - provider-specific oddity
                log(f"[V3_BACKFILL] {fetcher.name} error ticker={demand.ticker}: {type(exc).__name__}")
                continue
            if records:
                outcome.prices_by_ticker[demand.ticker] = records
                outcome.successful_tickers = (*outcome.successful_tickers, demand.ticker)
                outcome.rows_unlocked += demand.row_count
                break
    else:
        outcome.stopped_reason = outcome.stopped_reason or "queue completed"

    # De-duplicate tickers that failed on one provider but succeeded on another.
    successes = set(outcome.successful_tickers)
    outcome.unsupported_tickers = tuple(
        dict.fromkeys(t for t in outcome.unsupported_tickers if t not in successes)
    )
    outcome.transient_failed_tickers = tuple(
        dict.fromkeys(t for t in outcome.transient_failed_tickers if t not in successes)
    )
    if unsupported_cache is not None:
        unsupported_cache.save()
    return outcome


def format_backfill_stats(
    plan: BackfillPlan,
    outcome: BackfillOutcome,
    provider_stats: Sequence[object],
    *,
    coverage_after: float | None = None,
) -> str:
    """The ``=== TWELVE DATA BACKFILL ===``-style run summary."""
    lines: list[str] = []
    before = plan.total_rows_already_covered / plan.total_rows if plan.total_rows else 0.0
    for stats in provider_stats:
        data = stats.as_dict()  # type: ignore[attr-defined]
        lines.append(f"=== {str(data['vendor']).replace('_', ' ').upper()} BACKFILL ===")
        lines.append(f"symbols considered:      {plan.unique_tickers}")
        lines.append(f"already covered:         {plan.tickers_already_covered}")
        lines.append(f"unsupported skipped:     {data['symbols_skipped_unsupported']}")
        lines.append(f"symbols requested:       {data['symbols_requested']}")
        lines.append(f"network attempts:        {data['request_attempts']}")
        lines.append(f"retries performed:       {data['retries_performed']}")
        lines.append(f"successful symbols:      {data['successful_symbols']}")
        lines.append(f"transient failures:      {data['transient_failures']}")
        lines.append(f"permanent failures:      {data['permanent_failures']}")
        lines.append(f"timeout failures:        {data['timeout_failures']}")
        lines.append(f"rate-limit failures:     {data['rate_limit_failures']}")
        lines.append(f"cache hits:              {data['cache_hits']}")
        lines.append(f"API budget used:         {data['api_budget_used']}/{data['api_budget']}")
        lines.append(f"API budget remaining:    {data['api_budget_remaining']}")
        lines.append("")
    lines.append(f"training rows unlocked:  {outcome.rows_unlocked}")
    lines.append(f"cached-series row coverage before run: {before:.1%}")
    if coverage_after is not None:
        # Feature-level: fraction of rows whose eligible history reaches 5 years.
        lines.append(f"5y price feature coverage after run:  {coverage_after:.1%}")
    lines.append(f"remaining missing symbols: {max(0, plan.tickers_needing_prices - len(outcome.successful_tickers))}")
    return "\n".join(lines)


def format_plan_report(plan: BackfillPlan, *, show: int = 20) -> str:
    """Human-readable dry-run report; consumes zero API calls."""
    lines = [
        "=== V3 PRICE BACKFILL PLAN ===",
        f"total rows:                    {plan.total_rows}",
        f"unique tickers:                {plan.unique_tickers}",
        f"tickers already covered:       {plan.tickers_already_covered}",
        f"tickers needing prices:        {plan.tickers_needing_prices}",
        f"tickers skipped (unsupported): {plan.tickers_skipped_unsupported}",
        f"rows already covered:          {plan.total_rows_already_covered}"
        f" ({plan.total_rows_already_covered / plan.total_rows:.1%})" if plan.total_rows else "",
        f"rows needing prices:           {plan.total_rows_needing_prices}",
        f"rows blocked by unsupported:   {plan.rows_blocked_by_unsupported}",
        f"tickers with full 5y history:  {plan.tickers_with_5y_history}",
        f"rows with full 5y history:     {plan.rows_with_5y_history}",
        "",
        "projected rows unlocked by next N symbols:",
    ]
    for n in (10, 50, 100, 250, 500, 800):
        lines.append(f"  next {n:>4}: {plan.projected_rows_covered_by_next(n)} rows")
    lines.append("")
    lines.append(f"next {show} symbols to fetch (priority order):")
    for i, demand in enumerate(plan.next_symbols(show), start=1):
        lines.append(
            f"  {i:>3}. {demand.ticker:<8} rows={demand.row_count:<4} "
            f"span={_utc_date_key(demand.fetch_start)}..{_utc_date_key(demand.fetch_end)}"
        )
    return "\n".join(line for line in lines if line != "")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Dry-run planner CLI: ``python -m explaining_markets.backfill_planner``."""
    import argparse

    from explaining_markets.historical import load_historical_events
    from explaining_markets.historical_v3_enrichment import DEFAULT_CACHE_DIR, DiskJsonCache
    from explaining_markets.providers.unsupported_cache import (
        UnsupportedSymbolCache,
        default_unsupported_path,
    )
    from explaining_markets.v3_training_data import DEFAULT_ROWS_PATH, load_training_rows

    parser = argparse.ArgumentParser(description="Plan the V3 price backfill (no API calls).")
    parser.add_argument("--rows", default=str(DEFAULT_ROWS_PATH))
    parser.add_argument("--historical-dir", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--provider", default="twelve_data")
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--json", dest="json_out", default=None, help="write the plan as JSON")
    parser.add_argument("--retry-unsupported", action="store_true")
    args = parser.parse_args(argv)

    rows = load_training_rows(args.rows)
    events = load_historical_events(args.historical_dir)
    cache = DiskJsonCache(args.cache_dir)
    unsupported = UnsupportedSymbolCache(
        default_unsupported_path(args.cache_dir, args.provider),
        provider=args.provider,
        retry_unsupported=args.retry_unsupported,
    )
    plan = plan_price_backfill(rows, events, cache=cache, unsupported_cache=unsupported)
    print(format_plan_report(plan, show=args.show))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
        print(f"\n[V3_BACKFILL] plan written to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
