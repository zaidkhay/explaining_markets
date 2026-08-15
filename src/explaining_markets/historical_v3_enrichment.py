"""Historical V3 enrichment with resumable point-in-time vendor caches.

This module enriches archive seed rows without ever reading focal-event CAR1
as a feature. Alpha Vantage historical earnings/news are cached on disk and
can be resumed across API-budget-limited runs. Daily price history is sourced
only from either adjusted Alpha Vantage daily data (when the key is entitled)
or an explicit local bulk CSV; unsupported/partial price data stays missing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import httpx

from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.historical import HistoricalEvent, load_historical_events
from explaining_markets.news_ranking import rank_news
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.providers.news_provider import AlphaVantageNewsProvider
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.reasoning.news_reasoner import NewsReasoner
from explaining_markets.v3_records import EarningsRecord, NewsRecord, PriceRecord, V3Context
from explaining_markets.v3_training import V3TrainingRow
from explaining_markets.v3_training_data import _parse_dt, _prior_company_history, load_training_rows, training_data_report, write_training_rows

_API_URL = "https://www.alphavantage.co/query"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "enrichment" / "v3"
DEFAULT_ENRICHED_ROWS = Path(__file__).resolve().parents[2] / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"


class ApiBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class EnrichmentReport:
    rows: int
    eps_matches: int
    rows_with_company_news: int
    rows_with_reasoning: int
    rows_with_prices: int
    alpha_api_calls: int
    cache_hits: int
    output_path: str
    family_coverage: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "eps_matches": self.eps_matches,
            "rows_with_company_news": self.rows_with_company_news,
            "rows_with_reasoning": self.rows_with_reasoning,
            "rows_with_prices": self.rows_with_prices,
            "alpha_api_calls": self.alpha_api_calls,
            "cache_hits": self.cache_hits,
            "output_path": self.output_path,
            "family_coverage": self.family_coverage,
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _quarter_bounds(quarter: str) -> tuple[datetime, datetime]:
    year = int(quarter[:4])
    q = int(quarter[-1])
    start_month = 1 + 3 * (q - 1)
    start = datetime(year, start_month, 1, tzinfo=timezone.utc)
    if q == 4:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, start_month + 3, 1, tzinfo=timezone.utc)
    return start, end


def _serialize_news(row: NewsRecord) -> dict:
    return {
        "value_timestamp": row.value_timestamp.isoformat(),
        "available_at": row.available_at.isoformat(),
        "retrieved_at": row.retrieved_at.isoformat(),
        "source": row.source,
        "headline": row.headline,
        "published_at": row.published_at.isoformat(),
        "entities": list(row.entities),
        "url": row.url,
        "source_id": row.source_id,
        "sentiment": row.sentiment,
        "material": row.material,
        "topic": row.topic,
        "summary": row.summary,
        "excerpt": row.excerpt,
        "vendor_relevance": row.vendor_relevance,
    }


def _deserialize_news(raw: dict) -> NewsRecord:
    return NewsRecord(
        value_timestamp=datetime.fromisoformat(raw["value_timestamp"]),
        available_at=datetime.fromisoformat(raw["available_at"]),
        retrieved_at=datetime.fromisoformat(raw["retrieved_at"]),
        source=str(raw["source"]),
        headline=str(raw["headline"]),
        published_at=datetime.fromisoformat(raw["published_at"]),
        entities=tuple(raw.get("entities") or ()),
        url=raw.get("url"),
        source_id=raw.get("source_id"),
        sentiment=raw.get("sentiment"),
        material=bool(raw.get("material", False)),
        topic=raw.get("topic"),
        summary=raw.get("summary"),
        excerpt=raw.get("excerpt"),
        vendor_relevance=raw.get("vendor_relevance"),
    )


class DiskJsonCache:
    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        directory = self.root / namespace
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}.json"

    def get(self, namespace: str, key: str):
        path = self.path(namespace, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, payload) -> None:
        path = self.path(namespace, key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)


class AlphaHistoricalClient:
    """Resumable Alpha Vantage historical client with a hard per-run call budget."""

    def __init__(self, api_key: str, *, cache: DiskJsonCache, max_api_calls: int = 25, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required")
        self.api_key = api_key
        self.cache = cache
        self.max_api_calls = max(0, int(max_api_calls))
        self.timeout = float(timeout)
        self.api_calls = 0
        self.cache_hits = 0
        self.client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=self.timeout))

    def close(self) -> None:
        self.client.close()

    def _cached_request(self, namespace: str, cache_key: str, params: dict[str, str]) -> dict:
        cached = self.cache.get(namespace, cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.api_calls >= self.max_api_calls:
            raise ApiBudgetExhausted(f"Alpha Vantage per-run API budget exhausted ({self.max_api_calls})")
        response = self.client.get(_API_URL, params={**params, "apikey": self.api_key})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Alpha Vantage returned a non-object response")
        note = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        if note and not any(key in payload for key in ("quarterlyEarnings", "feed", "Time Series (Daily)")):
            raise RuntimeError(str(note))
        self.api_calls += 1
        self.cache.put(namespace, cache_key, payload)
        return payload

    def earnings_payload(self, ticker: str) -> dict:
        ticker = ticker.upper()
        return self._cached_request(
            "earnings",
            ticker,
            {"function": "EARNINGS", "symbol": ticker},
        )

    def adjusted_daily_payload(self, ticker: str) -> dict:
        ticker = ticker.upper()
        return self._cached_request(
            "adjusted_daily",
            ticker,
            {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker, "outputsize": "full"},
        )

    def broad_news_payload(self, start: datetime, end: datetime) -> dict:
        key = f"{start.isoformat()}|{end.isoformat()}|earnings"
        return self._cached_request(
            "news_windows",
            key,
            {
                "function": "NEWS_SENTIMENT",
                "topics": "earnings",
                "time_from": start.strftime("%Y%m%dT%H%M"),
                "time_to": end.strftime("%Y%m%dT%H%M"),
                "sort": "LATEST",
                "limit": "1000",
            },
        )


class LocalDailyPriceStore:
    """Bulk daily-price CSV reader.

    Required columns: ticker,date,close. Optional: volume,available_at,source.
    Dates are interpreted at 23:59 UTC unless an explicit available_at is supplied.
    The caller is responsible for supplying split/dividend-adjusted close values.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.by_ticker: dict[str, tuple[PriceRecord, ...]] = {}
        if path is None:
            return
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        rows: dict[str, list[PriceRecord]] = {}
        opener = source.open
        with opener("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"ticker", "date", "close"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("historical price CSV requires ticker,date,close columns")
            for raw in reader:
                ticker = str(raw["ticker"]).strip().upper()
                if not ticker:
                    continue
                date = datetime.fromisoformat(str(raw["date"]).strip()).replace(tzinfo=timezone.utc)
                value_ts = date.replace(hour=21, minute=0, second=0)
                available = raw.get("available_at")
                available_at = datetime.fromisoformat(available).astimezone(timezone.utc) if available else value_ts
                rows.setdefault(ticker, []).append(
                    PriceRecord(
                        value_timestamp=value_ts,
                        available_at=available_at,
                        retrieved_at=_utcnow(),
                        source=raw.get("source") or "local_adjusted_daily_csv",
                        ticker=ticker,
                        close=float(raw["close"]),
                        volume=float(raw["volume"]) if raw.get("volume") else None,
                    )
                )
        self.by_ticker = {ticker: tuple(sorted(values, key=lambda r: r.value_timestamp)) for ticker, values in rows.items()}

    def prices(self, ticker: str) -> tuple[PriceRecord, ...]:
        return self.by_ticker.get(ticker.upper(), ())


def _alpha_adjusted_prices(payload: dict, ticker: str, retrieved_at: datetime) -> tuple[PriceRecord, ...]:
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict):
        return ()
    out: list[PriceRecord] = []
    for day, raw in series.items():
        if not isinstance(raw, dict):
            continue
        try:
            value_ts = datetime.fromisoformat(day).replace(tzinfo=timezone.utc, hour=21)
            close = float(raw.get("5. adjusted close"))
        except (TypeError, ValueError):
            continue
        volume = raw.get("6. volume") or raw.get("5. volume")
        out.append(
            PriceRecord(
                value_timestamp=value_ts,
                available_at=value_ts,
                retrieved_at=retrieved_at,
                source="alpha_vantage_daily_adjusted",
                ticker=ticker.upper(),
                close=close,
                volume=float(volume) if volume not in (None, "") else None,
            )
        )
    return tuple(sorted(out, key=lambda row: row.value_timestamp))


def _match_earnings(payload: dict, event: HistoricalEvent, cutoff: datetime) -> EarningsRecord | None:
    rows = payload.get("quarterlyEarnings")
    if not isinstance(rows, list):
        return None
    candidates: list[tuple[int, dict]] = []
    for raw in rows:
        if not isinstance(raw, dict) or not raw.get("reportedDate"):
            continue
        try:
            date = datetime.fromisoformat(str(raw["reportedDate"])).date()
        except ValueError:
            continue
        distance = abs((date - cutoff.date()).days)
        if distance <= 7:
            candidates.append((distance, raw))
    if not candidates:
        return None
    _, raw = min(candidates, key=lambda pair: pair[0])

    def number(name: str) -> float | None:
        value = raw.get(name)
        if value in (None, "", "None"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    reported = number("reportedEPS")
    estimated = number("estimatedEPS")
    if reported is None or estimated is None:
        return None
    return EarningsRecord(
        value_timestamp=cutoff,
        available_at=cutoff,
        retrieved_at=_utcnow(),
        source="alpha_vantage_historical_earnings",
        ticker=event.ticker,
        reported_eps=reported,
        consensus_eps=estimated,
        event_id=event.event_id,
    )


def _normalize_broad_news(payload: dict, *, cutoff: datetime) -> tuple[NewsRecord, ...]:
    feed = payload.get("feed")
    if not isinstance(feed, list):
        return ()
    # Reuse the live provider's normalization logic without another network call.
    provider = AlphaVantageNewsProvider("offline-normalization-key")
    retrieved_at = _utcnow()
    rows = []
    for raw in feed:
        if not isinstance(raw, dict):
            continue
        row = provider._normalize(raw, cutoff=cutoff, retrieved_at=retrieved_at)
        if row is not None:
            rows.append(row)
    return tuple(rows)


def _events_by_key(events: Iterable[HistoricalEvent]) -> dict[tuple[str, str], HistoricalEvent]:
    return {(event.event_id, event.ticker): event for event in events}


def _timelines(events: Iterable[HistoricalEvent]) -> dict[str, list[HistoricalEvent]]:
    out: dict[str, list[HistoricalEvent]] = {}
    for event in events:
        out.setdefault(event.ticker, []).append(event)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    for values in out.values():
        values.sort(key=lambda e: _parse_dt(e.event_datetime) or epoch)
    return out


def _news_windows_for_quarters(quarters: Iterable[str], chunk_days: int) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    for quarter in sorted(set(quarters)):
        start, end = _quarter_bounds(quarter)
        cursor = start
        while cursor < end:
            nxt = min(end, cursor + timedelta(days=max(1, chunk_days)))
            windows.append((cursor, nxt - timedelta(minutes=1)))
            cursor = nxt
    return windows


def enrich_training_rows(
    *,
    rows_path: str | Path,
    historical_dir: str | Path,
    output_path: str | Path = DEFAULT_ENRICHED_ROWS,
    alpha_api_key: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_api_calls: int = 25,
    price_csv: str | Path | None = None,
    use_alpha_adjusted_prices: bool = False,
    include_historical_news: bool = True,
    news_chunk_days: int = 7,
    reasoning_mode: str = "deterministic",
) -> EnrichmentReport:
    seed_rows = load_training_rows(rows_path)
    events = load_historical_events(historical_dir)
    event_map = _events_by_key(events)
    timelines = _timelines(events)
    cache = DiskJsonCache(cache_dir)
    alpha = AlphaHistoricalClient(alpha_api_key, cache=cache, max_api_calls=max_api_calls) if alpha_api_key else None
    local_prices = LocalDailyPriceStore(price_csv)
    news_reasoner = NewsReasoner(use_openai=reasoning_mode == "openai")
    event_reasoner = EventReasoner(use_openai=reasoning_mode == "openai")

    broad_news: list[NewsRecord] = []
    if include_historical_news and alpha is not None:
        for start, end in _news_windows_for_quarters((row.quarter for row in seed_rows), news_chunk_days):
            try:
                payload = alpha.broad_news_payload(start, end)
            except ApiBudgetExhausted:
                break
            except Exception:
                continue
            broad_news.extend(_normalize_broad_news(payload, cutoff=end))

    eps_matches = rows_with_news = rows_with_reasoning = rows_with_prices = 0
    enriched: list[V3TrainingRow] = []
    try:
        for seed in seed_rows:
            event = event_map.get((seed.event_id, seed.ticker))
            if event is None:
                enriched.append(seed)
                continue
            cutoff = _parse_dt(event.event_datetime)
            if cutoff is None:
                enriched.append(seed)
                continue

            earnings = None
            if alpha is not None:
                try:
                    earnings = _match_earnings(alpha.earnings_payload(seed.ticker), event, cutoff)
                except (ApiBudgetExhausted, Exception):
                    earnings = None
            if earnings is not None:
                eps_matches += 1

            prices = local_prices.prices(seed.ticker)
            if not prices and use_alpha_adjusted_prices and alpha is not None:
                try:
                    prices = _alpha_adjusted_prices(alpha.adjusted_daily_payload(seed.ticker), seed.ticker, _utcnow())
                except (ApiBudgetExhausted, Exception):
                    prices = ()
            if prices:
                rows_with_prices += 1

            company_news = tuple(
                row for row in broad_news
                if seed.ticker.upper() in {entity.upper() for entity in row.entities}
                and row.published_at <= cutoff
                and row.published_at >= cutoff - timedelta(days=7)
            )
            ranked = rank_news(company_news, cutoff, targets={seed.ticker}, days=7, top_n=10, require_target=True)
            reasoned = news_reasoner.reason_many(ranked, relation="company") if ranked else ()
            if ranked:
                rows_with_news += 1

            history = _prior_company_history(
                event,
                timelines.get(seed.ticker, []),
                retrieved_at=_utcnow(),
            )
            base_context = V3Context(
                ticker=seed.ticker,
                cutoff=cutoff,
                earnings=earnings,
                company_history=history,
                stock_prices=prices,
                company_news=tuple(item.record for item in ranked),
                reasoned_company_news=reasoned,
                extras={"training_source": "historical_v3_enrichment"},
            )
            preliminary = build_feature_vector_v3(disclosure=list(event.disclosure), context=base_context)
            reasoning = event_reasoner.reason(
                values=preliminary.values,
                cutoff=cutoff,
                company_news=reasoned,
            )
            final_context = replace(base_context, event_reasoning=reasoning)
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
        if alpha is not None:
            alpha.close()

    output = write_training_rows(enriched, output_path)
    coverage = training_data_report(enriched, archive_seed_only=False).family_coverage
    return EnrichmentReport(
        rows=len(enriched),
        eps_matches=eps_matches,
        rows_with_company_news=rows_with_news,
        rows_with_reasoning=rows_with_reasoning,
        rows_with_prices=rows_with_prices,
        alpha_api_calls=alpha.api_calls if alpha is not None else 0,
        cache_hits=alpha.cache_hits if alpha is not None else 0,
        output_path=str(output),
        family_coverage=coverage,
    )
