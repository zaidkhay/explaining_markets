"""Free-provider historical V3 enrichment.

Provider preference:
* Tiingo EOD adjusted history -> stock-price context.
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
    ProviderBudgetExhausted,
    ProviderUnavailable,
    TiingoHistoricalClient,
    finnhub_earnings_record,
    finnhub_news_records,
    tiingo_price_records,
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
    finnhub_api_calls: int = 0
    finnhub_cache_hits: int = 0
    alpha_blocked_reason: str | None = None
    tiingo_blocked_reason: str | None = None
    finnhub_blocked_reason: str | None = None

    @property
    def cache_hits(self) -> int:
        return self.alpha_cache_hits + self.tiingo_cache_hits + self.finnhub_cache_hits

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
            "finnhub_api_calls": self.finnhub_api_calls,
            "finnhub_cache_hits": self.finnhub_cache_hits,
            "cache_hits": self.cache_hits,
            "alpha_blocked_reason": self.alpha_blocked_reason,
            "tiingo_blocked_reason": self.tiingo_blocked_reason,
            "finnhub_blocked_reason": self.finnhub_blocked_reason,
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
    """Keep only price observations that were actually available by the focal cutoff.

    Tiingo is intentionally fetched once across a ticker's full archive span so
    the cache can be reused across many historical events.  The resulting bulk
    series can therefore contain observations from *after* an early event.  We
    must trim that shared series for each event before it enters V3Context; the
    point-in-time audit then verifies the already-trimmed context rather than
    seeing harmless-but-future cache rows.
    """
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
    finnhub_api_key: str | None = None,
    alpha_api_key: str | None = None,
    tiingo_max_api_calls: int = 40,
    finnhub_max_api_calls: int = 50,
    alpha_max_api_calls: int = 0,
    price_csv: str | Path | None = None,
    use_alpha_adjusted_prices: bool = False,
    include_historical_news: bool = True,
    news_chunk_days: int = 7,
    reasoning_mode: str = "deterministic",
    request_timeout: float = 10.0,
    progress_every: int = 500,
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

    remote_reasoning = reasoning_mode == "openrouter"
    news_reasoner = NewsReasoner(use_openrouter=remote_reasoning)
    event_reasoner = EventReasoner(use_openrouter=remote_reasoning)

    # Alpha broad-news is now fallback-only. Finnhub company news is preferred
    # because one cached request covers a ticker's entire current training span.
    broad_alpha_news: list[NewsRecord] = []
    if include_historical_news and finnhub is None and alpha is not None and alpha_max_api_calls > 0:
        for start, end in _news_windows_for_quarters((row.quarter for row in seed_rows), news_chunk_days):
            try:
                broad_alpha_news.extend(_normalize_broad_news(alpha.broad_news_payload(start, end), cutoff=end))
            except ApiBudgetExhausted:
                break
            except Exception as exc:
                print(f"[V3_ENRICH] Alpha news fallback skipped {start.date()}..{end.date()}: {type(exc).__name__}", flush=True)

    tiingo_prices_by_ticker: dict[str, tuple[PriceRecord, ...]] = {}
    finnhub_earnings_by_ticker: dict[str, list[dict] | None] = {}
    finnhub_news_by_ticker: dict[str, tuple[NewsRecord, ...]] = {}
    tiingo_blocked = False
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

            # --- Earnings: Finnhub free last-4-quarter surprise first; Alpha cache/fallback second.
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

            # --- Prices: explicit local CSV, then Tiingo adjusted EOD, then optional Alpha premium.
            prices = local_prices.prices(ticker)
            if not prices and tiingo is not None:
                if ticker not in tiingo_prices_by_ticker:
                    first, last = bounds.get(ticker, (cutoff, cutoff))
                    start = first - timedelta(days=5 * 366 + 45)
                    end = last
                    fetched: tuple[PriceRecord, ...] = ()
                    if not tiingo_blocked:
                        try:
                            fetched = tiingo_price_records(
                                tiingo.prices_payload(ticker, start=start, end=end),
                                ticker,
                                retrieved_at=_utcnow(),
                            )
                        except (ProviderBudgetExhausted, ProviderUnavailable) as exc:
                            tiingo_blocked = True
                            print(f"[V3_ENRICH] Tiingo switched to cache-only: {str(exc)[:300]}", flush=True)
                        except Exception as exc:
                            print(f"[V3_ENRICH] Tiingo prices unavailable {ticker}: {type(exc).__name__}: {str(exc)[:220]}", flush=True)
                    if not fetched:
                        fetched = _cached_tiingo_prices(cache, ticker, start, end)
                    tiingo_prices_by_ticker[ticker] = fetched
                prices = tiingo_prices_by_ticker.get(ticker, ())

            if not prices and use_alpha_adjusted_prices and alpha is not None and not alpha_blocked:
                try:
                    prices = _alpha_adjusted_prices(alpha.adjusted_daily_payload(ticker), ticker, _utcnow())
                except ApiBudgetExhausted:
                    alpha_blocked = True
                    prices = ()
                except Exception:
                    prices = ()

            # A ticker-level cache intentionally spans all archive events.  Trim
            # it to the focal event before building/auditing the historical context.
            prices = _prices_available_by_cutoff(tuple(prices), cutoff)
            if prices:
                rows_with_prices += 1

            # --- Company news: one Finnhub request per ticker across the archive span.
            company_news: tuple[NewsRecord, ...] = ()
            if include_historical_news and finnhub is not None:
                if ticker not in finnhub_news_by_ticker:
                    first, last = bounds.get(ticker, (cutoff, cutoff))
                    start = first - timedelta(days=7)
                    end = last
                    raw_news = None
                    if not finnhub_blocked:
                        try:
                            raw_news = finnhub.company_news_payload(ticker, start=start, end=end)
                        except (ProviderBudgetExhausted, ProviderUnavailable) as exc:
                            finnhub_blocked = True
                            print(f"[V3_ENRICH] Finnhub news switched to cache-only: {str(exc)[:300]}", flush=True)
                        except Exception as exc:
                            print(f"[V3_ENRICH] Finnhub news unavailable {ticker}: {type(exc).__name__}: {str(exc)[:220]}", flush=True)
                    if raw_news is None:
                        raw_news = _cached_finnhub_news(cache, ticker, start, end)
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
                    "price_provider": "tiingo" if prices and prices[0].source.startswith("tiingo") else None,
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
        for client in (tiingo, finnhub, alpha):
            if client is not None:
                client.close()

    output = write_training_rows(enriched, output_path)
    coverage = training_data_report(enriched, archive_seed_only=False).family_coverage
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
        finnhub_api_calls=finnhub.api_calls if finnhub else 0,
        finnhub_cache_hits=finnhub.cache_hits if finnhub else 0,
        alpha_blocked_reason=alpha.blocked_reason if alpha else None,
        tiingo_blocked_reason=tiingo.unavailable_reason if tiingo else None,
        finnhub_blocked_reason=finnhub.unavailable_reason if finnhub else None,
    )
