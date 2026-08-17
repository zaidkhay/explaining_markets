"""Free-provider historical V3 enrichment.

Provider preference:
* Explicit local adjusted-price CSV if supplied.
* Tiingo adjusted EOD history when cached/available.
* Twelve Data adjusted daily history for broad free US-equity coverage.
* FMP historical EOD only for symbols permitted by the account/cache.
* Finnhub free EPS-surprise + company-news endpoints -> earnings/news context.
* Alpha Vantage remains an optional cached/fallback source.
* OpenRouter is opt-in for historical reasoning; deterministic reasoning remains
  the default so a 6,299-row backfill cannot accidentally exhaust free LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from explaining_markets.backfill_planner import (
    BackfillOutcome,
    BackfillPlan,
    PriceFetcher,
    format_backfill_stats,
    plan_price_backfill,
    run_price_backfill,
)
from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.historical import HistoricalEvent, load_historical_events
from explaining_markets.historical_v3_enrichment import (
    AlphaHistoricalClient,
    ApiBudgetExhausted,
    DEFAULT_CACHE_DIR,
    DEFAULT_ENRICHED_ROWS,
    DiskJsonCache,
    LocalDailyPriceStore,
    _alpha_adjusted_prices,
    _events_by_key,
    _match_earnings,
    _normalize_broad_news,
    _news_windows_for_quarters,
    _reasoning_has_evidence,
    _timelines,
    _utcnow,
)
from explaining_markets.news_ranking import rank_news
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.providers.free_historical import (
    FinnhubHistoricalClient,
    FmpHistoricalClient,
    ProviderBudgetExhausted,
    ProviderUnavailable,
    TiingoHistoricalClient,
    TwelveDataHistoricalClient,
    finnhub_earnings_record,
    finnhub_news_records,
    fmp_price_records,
    tiingo_price_records,
    twelve_data_price_records,
)
from explaining_markets.providers.retry_policy import RetryPolicy, UnsupportedSymbolError
from explaining_markets.providers.unsupported_cache import (
    UnsupportedSymbolCache,
    default_unsupported_path,
)
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.reasoning.news_reasoner import NewsReasoner
from explaining_markets.v3_records import NewsRecord, PriceRecord, V3Context
from explaining_markets.v3_training import V3TrainingRow
from explaining_markets.v3_training_data import (
    _parse_dt,
    _prior_company_history,
    load_training_rows,
    training_data_report,
    write_training_rows,
)


@dataclass(frozen=True)
class FreeProviderEnrichmentReport:
    rows: int
    eps_matches: int
    rows_with_company_news: int
    rows_with_reasoning: int
    rows_with_prices: int
    output_path: str
    family_coverage: dict[str, float]
    alpha_api_calls: int = 0
    alpha_cache_hits: int = 0
    tiingo_api_calls: int = 0
    tiingo_cache_hits: int = 0
    twelve_data_api_calls: int = 0
    twelve_data_cache_hits: int = 0
    fmp_api_calls: int = 0
    fmp_cache_hits: int = 0
    finnhub_api_calls: int = 0
    finnhub_cache_hits: int = 0
    alpha_blocked_reason: str | None = None
    tiingo_blocked_reason: str | None = None
    twelve_data_blocked_reason: str | None = None
    fmp_blocked_reason: str | None = None
    finnhub_blocked_reason: str | None = None
    # Phase A: prioritized-backfill diagnostics and per-provider accounting.
    backfill_plan: dict | None = None
    backfill_outcome: dict | None = None
    provider_stats: dict | None = None
    backfill_summary: str | None = None

    @property
    def cache_hits(self) -> int:
        return (
            self.alpha_cache_hits
            + self.tiingo_cache_hits
            + self.twelve_data_cache_hits
            + self.fmp_cache_hits
            + self.finnhub_cache_hits
        )

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "eps_matches": self.eps_matches,
            "rows_with_company_news": self.rows_with_company_news,
            "rows_with_reasoning": self.rows_with_reasoning,
            "rows_with_prices": self.rows_with_prices,
            "output_path": self.output_path,
            "family_coverage": self.family_coverage,
            "alpha_api_calls": self.alpha_api_calls,
            "alpha_cache_hits": self.alpha_cache_hits,
            "tiingo_api_calls": self.tiingo_api_calls,
            "tiingo_cache_hits": self.tiingo_cache_hits,
            "twelve_data_api_calls": self.twelve_data_api_calls,
            "twelve_data_cache_hits": self.twelve_data_cache_hits,
            "fmp_api_calls": self.fmp_api_calls,
            "fmp_cache_hits": self.fmp_cache_hits,
            "finnhub_api_calls": self.finnhub_api_calls,
            "finnhub_cache_hits": self.finnhub_cache_hits,
            "cache_hits": self.cache_hits,
            "alpha_blocked_reason": self.alpha_blocked_reason,
            "tiingo_blocked_reason": self.tiingo_blocked_reason,
            "twelve_data_blocked_reason": self.twelve_data_blocked_reason,
            "fmp_blocked_reason": self.fmp_blocked_reason,
            "finnhub_blocked_reason": self.finnhub_blocked_reason,
            "backfill_plan": self.backfill_plan,
            "backfill_outcome": self.backfill_outcome,
            "provider_stats": self.provider_stats,
            "backfill_summary": self.backfill_summary,
        }


def _ticker_bounds(events: Iterable[HistoricalEvent]) -> dict[str, tuple[datetime, datetime]]:
    values: dict[str, list[datetime]] = {}
    for event in events:
        cutoff = _parse_dt(event.event_datetime)
        if cutoff is not None:
            values.setdefault(event.ticker.upper(), []).append(cutoff)
    return {ticker: (min(rows), max(rows)) for ticker, rows in values.items() if rows}


def _cached_tiingo_prices(cache: DiskJsonCache, ticker: str, start: datetime, end: datetime) -> tuple[PriceRecord, ...]:
    key = f"{ticker.upper()}|{start.date()}|{end.date()}"
    payload = cache.get("tiingo_prices", key)
    return tiingo_price_records(payload, ticker, retrieved_at=_utcnow()) if isinstance(payload, list) else ()


def _cached_twelve_data_prices(cache: DiskJsonCache, ticker: str, start: datetime, end: datetime) -> tuple[PriceRecord, ...]:
    key = f"{ticker.upper()}|{start.date()}|{end.date()}|adjust=all"
    payload = cache.get("twelve_data_prices", key)
    values = payload.get("values") if isinstance(payload, dict) else None
    return twelve_data_price_records(values, ticker, retrieved_at=_utcnow()) if isinstance(values, list) else ()


def _cached_fmp_prices(cache: DiskJsonCache, ticker: str, start: datetime, end: datetime) -> tuple[PriceRecord, ...]:
    key = f"{ticker.upper()}|{start.date()}|{end.date()}"
    payload = cache.get("fmp_prices", key)
    return fmp_price_records(payload, ticker, retrieved_at=_utcnow()) if isinstance(payload, list) else ()


def _cached_finnhub_earnings(cache: DiskJsonCache, ticker: str):
    payload = cache.get("finnhub_earnings", ticker.upper())
    return payload if isinstance(payload, list) else None


def _cached_finnhub_news(cache: DiskJsonCache, ticker: str, start: datetime, end: datetime):
    key = f"{ticker.upper()}|{start.date()}|{end.date()}"
    payload = cache.get("finnhub_news", key)
    return payload if isinstance(payload, list) else None


def _prices_available_by_cutoff(
    rows: tuple[PriceRecord, ...], cutoff: datetime
) -> tuple[PriceRecord, ...]:
    """Keep only price observations that were actually available by the focal cutoff."""
    return tuple(
        row
        for row in rows
        if row.value_timestamp <= cutoff and row.eligible(cutoff)
    )


def enrich_training_rows_free(
    *,
    rows_path: str | Path,
    historical_dir: str | Path,
    output_path: str | Path = DEFAULT_ENRICHED_ROWS,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    tiingo_api_key: str | None = None,
    twelve_data_api_key: str | None = None,
    fmp_api_key: str | None = None,
    finnhub_api_key: str | None = None,
    alpha_api_key: str | None = None,
    tiingo_max_api_calls: int = 40,
    twelve_data_max_api_calls: int = 0,
    fmp_max_api_calls: int = 0,
    finnhub_max_api_calls: int = 50,
    alpha_max_api_calls: int = 0,
    price_csv: str | Path | None = None,
    use_alpha_adjusted_prices: bool = False,
    include_historical_news: bool = True,
    news_chunk_days: int = 7,
    reasoning_mode: str = "deterministic",
    request_timeout: float = 10.0,
    progress_every: int = 500,
    retry_unsupported: bool = False,
    price_retry_attempts: int = 3,
    max_price_symbols: int | None = None,
    plan_only: bool = False,
) -> FreeProviderEnrichmentReport:
    seed_rows = load_training_rows(rows_path)
    events = load_historical_events(historical_dir)
    event_map = _events_by_key(events)
    timelines = _timelines(events)
    bounds = _ticker_bounds(events)
    cache = DiskJsonCache(cache_dir)
    local_prices = LocalDailyPriceStore(price_csv)

    tiingo = (
        TiingoHistoricalClient(
            tiingo_api_key,
            cache=cache,
            max_api_calls=tiingo_max_api_calls,
            timeout=request_timeout,
        )
        if tiingo_api_key
        else None
    )
    unsupported_cache = UnsupportedSymbolCache(
        default_unsupported_path(cache_dir, "twelve_data"),
        provider="twelve_data",
        retry_unsupported=retry_unsupported,
    )
    twelve_data = (
        TwelveDataHistoricalClient(
            twelve_data_api_key,
            cache=cache,
            max_api_calls=twelve_data_max_api_calls,
            timeout=request_timeout,
            retry_policy=RetryPolicy(max_attempts=max(1, int(price_retry_attempts))),
            unsupported_cache=unsupported_cache,
        )
        if twelve_data_api_key
        else None
    )
    fmp = (
        FmpHistoricalClient(
            fmp_api_key,
            cache=cache,
            max_api_calls=fmp_max_api_calls,
            timeout=request_timeout,
        )
        if fmp_api_key
        else None
    )
    finnhub = (
        FinnhubHistoricalClient(
            finnhub_api_key,
            cache=cache,
            max_api_calls=finnhub_max_api_calls,
            timeout=request_timeout,
        )
        if finnhub_api_key
        else None
    )
    alpha = (
        AlphaHistoricalClient(
            alpha_api_key,
            cache=cache,
            max_api_calls=alpha_max_api_calls,
            timeout=request_timeout,
            retries=0,
        )
        if alpha_api_key
        else None
    )

    # ---- Phase A: prioritized price backfill -----------------------------
    # Plan first (zero API calls), then fetch the highest-row-count symbols
    # that are neither already cached nor known-unsupported. The row loop below
    # performs NO network price calls, so a 6k-row pass cannot rediscover the
    # same missing ticker thousands of times.
    plan: BackfillPlan = plan_price_backfill(
        seed_rows, events, cache=cache, unsupported_cache=unsupported_cache
    )
    print(
        f"[V3_BACKFILL] rows={plan.total_rows} tickers={plan.unique_tickers} "
        f"covered={plan.tickers_already_covered} needing={plan.tickers_needing_prices} "
        f"unsupported_skipped={plan.tickers_skipped_unsupported} "
        f"row_coverage={plan.total_rows_already_covered / plan.total_rows:.1%}"
        if plan.total_rows
        else "[V3_BACKFILL] no rows",
        flush=True,
    )

    price_fetchers: list[PriceFetcher] = []
    if tiingo is not None:
        price_fetchers.append(PriceFetcher("tiingo", tiingo, "tiingo_prices"))
    if twelve_data is not None:
        price_fetchers.append(PriceFetcher("twelve_data", twelve_data, "twelve_data_prices"))
    if fmp is not None:
        price_fetchers.append(PriceFetcher("fmp", fmp, "fmp_prices"))

    outcome = BackfillOutcome()
    if price_fetchers and not plan_only:
        outcome = run_price_backfill(
            plan,
            price_fetchers,
            unsupported_cache=unsupported_cache,
            max_symbols=max_price_symbols,
        )
        print(
            f"[V3_BACKFILL] fetched={len(outcome.successful_tickers)} "
            f"unsupported={len(outcome.unsupported_tickers)} "
            f"transient_failed={len(outcome.transient_failed_tickers)} "
            f"rows_unlocked={outcome.rows_unlocked} stopped={outcome.stopped_reason}",
            flush=True,
        )
    prefetched_prices: dict[str, tuple[PriceRecord, ...]] = dict(outcome.prices_by_ticker)

    remote_reasoning = reasoning_mode == "openrouter"
    news_reasoner = NewsReasoner(use_openrouter=remote_reasoning)
    event_reasoner = EventReasoner(use_openrouter=remote_reasoning)

    broad_alpha_news: list[NewsRecord] = []
    if include_historical_news and finnhub is None and alpha is not None and alpha_max_api_calls > 0:
        for start, end in _news_windows_for_quarters((row.quarter for row in seed_rows), news_chunk_days):
            try:
                broad_alpha_news.extend(_normalize_broad_news(alpha.broad_news_payload(start, end), cutoff=end))
            except ApiBudgetExhausted:
                break
            except Exception as exc:
                print(f"[V3_ENRICH] Alpha news fallback skipped {start.date()}..{end.date()}: {type(exc).__name__}", flush=True)

    # Per-ticker memo of normalized cached prices, so a ticker's cache files are
    # parsed once per run instead of once per row.
    resolved_price_cache: dict[str, tuple[PriceRecord, ...]] = {}
    finnhub_earnings_by_ticker: dict[str, list[dict] | None] = {}
    finnhub_news_by_ticker: dict[str, tuple[NewsRecord, ...]] = {}
    finnhub_blocked = False
    alpha_blocked = False

    eps_matches = rows_with_news = rows_with_reasoning = rows_with_prices = 0
    enriched: list[V3TrainingRow] = []
    try:
        for index, seed in enumerate(seed_rows, start=1):
            if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(seed_rows)):
                print(
                    f"[V3_ENRICH] rows={index}/{len(seed_rows)} eps={eps_matches} "
                    f"news={rows_with_news} prices={rows_with_prices}",
                    flush=True,
                )
            event = event_map.get((seed.event_id, seed.ticker))
            if event is None:
                enriched.append(seed)
                continue
            cutoff = _parse_dt(event.event_datetime)
            if cutoff is None:
                enriched.append(seed)
                continue
            ticker = seed.ticker.upper()

            earnings = None
            if finnhub is not None:
                if ticker not in finnhub_earnings_by_ticker:
                    payload = None
                    if not finnhub_blocked:
                        try:
                            payload = finnhub.earnings_payload(ticker)
                        except (ProviderBudgetExhausted, ProviderUnavailable) as exc:
                            finnhub_blocked = True
                            print(f"[V3_ENRICH] Finnhub earnings switched to cache-only: {str(exc)[:300]}", flush=True)
                        except Exception as exc:
                            print(f"[V3_ENRICH] Finnhub earnings unavailable {ticker}: {type(exc).__name__}: {str(exc)[:220]}", flush=True)
                    if payload is None:
                        payload = _cached_finnhub_earnings(cache, ticker)
                    finnhub_earnings_by_ticker[ticker] = payload
                payload = finnhub_earnings_by_ticker.get(ticker)
                if payload:
                    earnings = finnhub_earnings_record(payload, event, cutoff)

            if earnings is None and alpha is not None:
                try:
                    if alpha_blocked:
                        payload = cache.get("earnings", ticker)
                    else:
                        payload = alpha.earnings_payload(ticker)
                    if isinstance(payload, dict):
                        earnings = _match_earnings(payload, event, cutoff)
                except ApiBudgetExhausted as exc:
                    alpha_blocked = True
                    payload = cache.get("earnings", ticker)
                    if isinstance(payload, dict):
                        earnings = _match_earnings(payload, event, cutoff)
                    if alpha_max_api_calls > 0:
                        print(f"[V3_ENRICH] Alpha earnings fallback switched to cache-only: {str(exc)[:300]}", flush=True)
                except Exception:
                    pass
            if earnings is not None:
                eps_matches += 1

            # Prices: local CSV -> prioritized prefetch -> normalized cache.
            # No network call happens here; the backfill pass above owns all
            # price requests, so this loop is pure CPU over cached data.
            prices = local_prices.prices(ticker)
            first, last = bounds.get(ticker, (cutoff, cutoff))
            start = first - timedelta(days=5 * 366 + 45)
            end = last

            if not prices:
                prices = prefetched_prices.get(ticker, ())

            if not prices and ticker not in resolved_price_cache:
                for reader in (
                    _cached_tiingo_prices,
                    _cached_twelve_data_prices,
                    _cached_fmp_prices,
                ):
                    cached_rows = reader(cache, ticker, start, end)
                    if cached_rows:
                        resolved_price_cache[ticker] = cached_rows
                        break
                else:
                    resolved_price_cache[ticker] = ()
            if not prices:
                prices = resolved_price_cache.get(ticker, ())

            if not prices and use_alpha_adjusted_prices and alpha is not None and not alpha_blocked:
                try:
                    prices = _alpha_adjusted_prices(alpha.adjusted_daily_payload(ticker), ticker, _utcnow())
                except ApiBudgetExhausted:
                    alpha_blocked = True
                    prices = ()
                except Exception:
                    prices = ()

            prices = _prices_available_by_cutoff(tuple(prices), cutoff)
            if prices:
                rows_with_prices += 1

            company_news: tuple[NewsRecord, ...] = ()
            if include_historical_news and finnhub is not None:
                if ticker not in finnhub_news_by_ticker:
                    first, last = bounds.get(ticker, (cutoff, cutoff))
                    news_start = first - timedelta(days=7)
                    news_end = last
                    raw_news = None
                    if not finnhub_blocked:
                        try:
                            raw_news = finnhub.company_news_payload(ticker, start=news_start, end=news_end)
                        except (ProviderBudgetExhausted, ProviderUnavailable) as exc:
                            finnhub_blocked = True
                            print(f"[V3_ENRICH] Finnhub news switched to cache-only: {str(exc)[:300]}", flush=True)
                        except Exception as exc:
                            print(f"[V3_ENRICH] Finnhub news unavailable {ticker}: {type(exc).__name__}: {str(exc)[:220]}", flush=True)
                    if raw_news is None:
                        raw_news = _cached_finnhub_news(cache, ticker, news_start, news_end)
                    finnhub_news_by_ticker[ticker] = (
                        finnhub_news_records(raw_news, ticker, retrieved_at=_utcnow()) if raw_news else ()
                    )
                company_news = tuple(
                    row for row in finnhub_news_by_ticker.get(ticker, ())
                    if cutoff - timedelta(days=7) <= row.published_at <= cutoff
                )
            elif broad_alpha_news:
                company_news = tuple(
                    row for row in broad_alpha_news
                    if ticker in {entity.upper() for entity in row.entities}
                    and cutoff - timedelta(days=7) <= row.published_at <= cutoff
                )

            ranked = rank_news(
                company_news,
                cutoff,
                targets={ticker},
                days=7,
                top_n=10,
                require_target=True,
            )
            reasoned = news_reasoner.reason_many(ranked, relation="company") if ranked else ()
            if ranked:
                rows_with_news += 1

            history = _prior_company_history(event, timelines.get(seed.ticker, []), retrieved_at=_utcnow())
            price_provider = None
            if prices:
                if prices[0].source.startswith("tiingo"):
                    price_provider = "tiingo"
                elif prices[0].source.startswith("twelve_data"):
                    price_provider = "twelve_data"
                elif prices[0].source.startswith("fmp"):
                    price_provider = "fmp"
                else:
                    price_provider = prices[0].source
            base_context = V3Context(
                ticker=seed.ticker,
                cutoff=cutoff,
                earnings=earnings,
                company_history=history,
                stock_prices=prices,
                company_news=tuple(item.record for item in ranked),
                reasoned_company_news=reasoned,
                extras={
                    "training_source": "historical_v3_enrichment_free",
                    "price_provider": price_provider,
                    "earnings_provider": earnings.source if earnings is not None else None,
                },
            )
            preliminary = build_feature_vector_v3(disclosure=list(event.disclosure), context=base_context)
            if _reasoning_has_evidence(preliminary.values):
                reasoning = event_reasoner.reason(
                    values=preliminary.values,
                    cutoff=cutoff,
                    company_news=reasoned,
                )
                final_context = replace(base_context, event_reasoning=reasoning)
            else:
                final_context = base_context

            audit = audit_context(final_context)
            vector = build_feature_vector_v3(disclosure=list(event.disclosure), context=final_context)
            if vector.values.get("has_reasoning", 0.0):
                rows_with_reasoning += 1
            enriched.append(
                V3TrainingRow(
                    event_id=seed.event_id,
                    ticker=seed.ticker,
                    quarter=seed.quarter,
                    target_percentile=seed.target_percentile,
                    values={name: float(vector.values[name]) for name in MODEL_FEATURE_NAMES_V3},
                    surprise_percentile=seed.surprise_percentile,
                    leakage_violations=int(audit.violations),
                )
            )
    finally:
        unsupported_cache.save()
        for client in (tiingo, twelve_data, fmp, finnhub, alpha):
            if client is not None:
                client.close()

    output = write_training_rows(enriched, output_path)
    coverage = training_data_report(enriched, archive_seed_only=False).family_coverage

    price_clients = [c for c in (tiingo, twelve_data, fmp) if c is not None]
    for client in price_clients:
        client.stats.rows_unlocked = outcome.rows_unlocked
        client.stats.symbols_considered = plan.unique_tickers
        client.stats.symbols_already_covered = plan.tickers_already_covered
        client.stats.check_invariants()
    coverage_after = coverage.get("price_5y")
    backfill_summary = format_backfill_stats(
        plan, outcome, [c.stats for c in price_clients], coverage_after=coverage_after
    )
    print("\n" + backfill_summary, flush=True)

    return FreeProviderEnrichmentReport(
        rows=len(enriched),
        eps_matches=eps_matches,
        rows_with_company_news=rows_with_news,
        rows_with_reasoning=rows_with_reasoning,
        rows_with_prices=rows_with_prices,
        output_path=str(output),
        family_coverage=coverage,
        alpha_api_calls=alpha.api_calls if alpha else 0,
        alpha_cache_hits=alpha.cache_hits if alpha else 0,
        tiingo_api_calls=tiingo.api_calls if tiingo else 0,
        tiingo_cache_hits=tiingo.cache_hits if tiingo else 0,
        twelve_data_api_calls=twelve_data.api_calls if twelve_data else 0,
        twelve_data_cache_hits=twelve_data.cache_hits if twelve_data else 0,
        fmp_api_calls=fmp.api_calls if fmp else 0,
        fmp_cache_hits=fmp.cache_hits if fmp else 0,
        finnhub_api_calls=finnhub.api_calls if finnhub else 0,
        finnhub_cache_hits=finnhub.cache_hits if finnhub else 0,
        alpha_blocked_reason=alpha.blocked_reason if alpha else None,
        tiingo_blocked_reason=tiingo.unavailable_reason if tiingo else None,
        twelve_data_blocked_reason=twelve_data.unavailable_reason if twelve_data else None,
        fmp_blocked_reason=fmp.unavailable_reason if fmp else None,
        finnhub_blocked_reason=finnhub.unavailable_reason if finnhub else None,
        backfill_plan=plan.as_dict(),
        backfill_outcome=outcome.as_dict(),
        provider_stats={c.vendor: c.stats.as_dict() for c in price_clients},
        backfill_summary=backfill_summary,
    )
