"""Free-tier historical providers used by V3 enrichment.

Tiingo and FMP supply historical EOD prices. Finnhub supplies the last four
quarterly EPS surprises and one year of company news. Successful responses are
cached by the caller's cache object; failures never become fake zero-valued
records.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx

from explaining_markets.historical import HistoricalEvent
from explaining_markets.v3_records import EarningsRecord, NewsRecord, PriceRecord

_TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"
_FINNHUB_BASE = "https://finnhub.io/api/v1"
_FMP_BASE = "https://financialmodelingprep.com/stable"


class ProviderBudgetExhausted(RuntimeError):
    pass


class ProviderUnavailable(RuntimeError):
    pass


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class _CachedJsonClient:
    def __init__(
        self,
        *,
        cache,
        max_api_calls: int,
        timeout: float,
        min_request_interval: float,
        client: httpx.Client | None,
        progress: Callable[[str], None] | None,
        vendor: str,
    ) -> None:
        self.cache = cache
        self.max_api_calls = max(0, int(max_api_calls))
        self.timeout = max(0.5, float(timeout))
        self.min_request_interval = max(0.0, float(min_request_interval))
        self.vendor = vendor
        self.api_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.unavailable_reason: str | None = None
        self._last_request_started: float | None = None
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=self.timeout)
        self.progress = progress or (lambda message: print(message, flush=True))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _pace(self) -> None:
        if self._last_request_started is None or self.min_request_interval <= 0:
            return
        remaining = self.min_request_interval - (time.monotonic() - self._last_request_started)
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, *, namespace: str, cache_key: str, url: str, params: dict[str, Any], headers: dict[str, str]) -> Any:
        cached = self.cache.get(namespace, cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.unavailable_reason:
            raise ProviderUnavailable(self.unavailable_reason)
        if self.api_calls >= self.max_api_calls:
            raise ProviderBudgetExhausted(f"{self.vendor} per-run API budget exhausted ({self.max_api_calls})")

        self._pace()
        self.api_calls += 1
        self._last_request_started = time.monotonic()
        self.progress(
            f"[V3_ENRICH] {self.vendor} {namespace}:{cache_key} "
            f"calls={self.api_calls}/{self.max_api_calls}"
        )
        try:
            response = self.client.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code == 429:
                self.unavailable_reason = f"{self.vendor} rate limit reached (HTTP 429)"
                raise ProviderUnavailable(self.unavailable_reason)
            response.raise_for_status()
            payload = response.json()
        except ProviderUnavailable:
            self.failures += 1
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
            self.failures += 1
            raise RuntimeError(f"{self.vendor} request failed: {type(exc).__name__}") from exc

        if isinstance(payload, dict):
            error = payload.get("error") or payload.get("message") or payload.get("Error Message")
            if error and len(payload) <= 4:
                self.unavailable_reason = f"{self.vendor}: {error}"
                self.failures += 1
                raise ProviderUnavailable(self.unavailable_reason)
        self.cache.put(namespace, cache_key, payload)
        return payload


class TiingoHistoricalClient(_CachedJsonClient):
    def __init__(
        self,
        api_key: str,
        *,
        cache,
        max_api_calls: int = 40,
        timeout: float = 12.0,
        client: httpx.Client | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("TINGO_API/TIINGO_API_KEY is required")
        super().__init__(
            cache=cache,
            max_api_calls=max_api_calls,
            timeout=timeout,
            min_request_interval=0.05,
            client=client,
            progress=progress,
            vendor="tiingo",
        )
        self.api_key = api_key

    def prices_payload(self, ticker: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        symbol = ticker.upper()
        key = f"{symbol}|{start.date()}|{end.date()}"
        payload = self._request(
            namespace="tiingo_prices",
            cache_key=key,
            url=f"{_TIINGO_BASE}/{symbol}/prices",
            params={"startDate": start.date().isoformat(), "endDate": end.date().isoformat()},
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
        )
        if not isinstance(payload, list):
            raise RuntimeError("Tiingo price response was not a list")
        return payload


class FmpHistoricalClient(_CachedJsonClient):
    """Financial Modeling Prep EOD fallback for symbols not covered by Tiingo cache."""

    def __init__(
        self,
        api_key: str,
        *,
        cache,
        max_api_calls: int = 200,
        timeout: float = 12.0,
        client: httpx.Client | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("FMP_API_KEY is required")
        super().__init__(
            cache=cache,
            max_api_calls=max_api_calls,
            timeout=timeout,
            min_request_interval=0.25,
            client=client,
            progress=progress,
            vendor="fmp",
        )
        self.api_key = api_key

    def prices_payload(self, ticker: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        symbol = ticker.upper()
        key = f"{symbol}|{start.date()}|{end.date()}"
        payload = self._request(
            namespace="fmp_prices",
            cache_key=key,
            url=f"{_FMP_BASE}/historical-price-eod/full",
            params={
                "symbol": symbol,
                "from": start.date().isoformat(),
                "to": end.date().isoformat(),
                "apikey": self.api_key,
            },
            headers={"Accept": "application/json"},
        )
        if not isinstance(payload, list):
            raise RuntimeError("FMP historical-price response was not a list")
        return payload


class FinnhubHistoricalClient(_CachedJsonClient):
    def __init__(
        self,
        api_key: str,
        *,
        cache,
        max_api_calls: int = 50,
        timeout: float = 12.0,
        client: httpx.Client | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is required")
        super().__init__(
            cache=cache,
            max_api_calls=max_api_calls,
            timeout=timeout,
            min_request_interval=1.05,
            client=client,
            progress=progress,
            vendor="finnhub",
        )
        self.api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Finnhub-Token": self.api_key}

    def earnings_payload(self, ticker: str) -> list[dict[str, Any]]:
        symbol = ticker.upper()
        payload = self._request(
            namespace="finnhub_earnings",
            cache_key=symbol,
            url=f"{_FINNHUB_BASE}/stock/earnings",
            params={"symbol": symbol, "limit": 4},
            headers=self._headers,
        )
        if not isinstance(payload, list):
            raise RuntimeError("Finnhub earnings response was not a list")
        return payload

    def company_news_payload(self, ticker: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        symbol = ticker.upper()
        key = f"{symbol}|{start.date()}|{end.date()}"
        payload = self._request(
            namespace="finnhub_news",
            cache_key=key,
            url=f"{_FINNHUB_BASE}/company-news",
            params={"symbol": symbol, "from": start.date().isoformat(), "to": end.date().isoformat()},
            headers=self._headers,
        )
        if not isinstance(payload, list):
            raise RuntimeError("Finnhub company-news response was not a list")
        return payload


def tiingo_price_records(payload: list[dict[str, Any]], ticker: str, *, retrieved_at: datetime) -> tuple[PriceRecord, ...]:
    out: list[PriceRecord] = []
    for raw in payload:
        try:
            date = _parse_iso_datetime(str(raw["date"]))
            close = float(raw.get("adjClose") if raw.get("adjClose") is not None else raw["close"])
        except (KeyError, TypeError, ValueError):
            continue
        volume_raw = raw.get("adjVolume") if raw.get("adjVolume") is not None else raw.get("volume")
        value_ts = date.replace(hour=21, minute=0, second=0, microsecond=0)
        available_at = (date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        out.append(
            PriceRecord(
                value_timestamp=value_ts,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source="tiingo_eod_adjusted",
                ticker=ticker.upper(),
                close=close,
                volume=float(volume_raw) if volume_raw not in (None, "") else None,
            )
        )
    return tuple(sorted(out, key=lambda row: row.value_timestamp))


def fmp_price_records(payload: list[dict[str, Any]], ticker: str, *, retrieved_at: datetime) -> tuple[PriceRecord, ...]:
    """Normalize FMP EOD rows conservatively for pre-event use.

    FMP's stable full EOD endpoint is split-adjusted according to the provider's
    chart family. We still make each observation available on the following UTC
    day so same-day events cannot consume a completed EOD bar prematurely.
    """
    out: list[PriceRecord] = []
    for raw in payload:
        date_raw = raw.get("date")
        close_raw = raw.get("close")
        if date_raw is None or close_raw is None:
            continue
        try:
            date = _parse_iso_datetime(str(date_raw))
            close = float(close_raw)
        except (TypeError, ValueError):
            continue
        if close <= 0:
            continue
        volume_raw = raw.get("volume")
        value_ts = date.replace(hour=21, minute=0, second=0, microsecond=0)
        available_at = (date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        out.append(
            PriceRecord(
                value_timestamp=value_ts,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source="fmp_eod",
                ticker=ticker.upper(),
                close=close,
                volume=float(volume_raw) if volume_raw not in (None, "") else None,
            )
        )
    return tuple(sorted(out, key=lambda row: row.value_timestamp))


def finnhub_earnings_record(
    payload: list[dict[str, Any]],
    event: HistoricalEvent,
    cutoff: datetime,
) -> EarningsRecord | None:
    """Match a free Finnhub EPS-surprise row to an earnings event conservatively.

    Finnhub's free surprise endpoint exposes fiscal ``period`` rather than the
    release timestamp. We therefore require the fiscal period to precede the
    event and to fall within 7..150 days of the event; otherwise we fail closed.
    """
    candidates: list[tuple[int, dict[str, Any]]] = []
    for raw in payload:
        period_raw = raw.get("period")
        if not period_raw:
            continue
        try:
            period = datetime.fromisoformat(str(period_raw)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        lag = (cutoff.date() - period.date()).days
        if 7 <= lag <= 150 and raw.get("actual") is not None and raw.get("estimate") is not None:
            candidates.append((lag, raw))
    if not candidates:
        return None
    _, raw = min(candidates, key=lambda item: item[0])
    try:
        actual = float(raw["actual"])
        estimate = float(raw["estimate"])
    except (TypeError, ValueError):
        return None
    return EarningsRecord(
        value_timestamp=cutoff,
        available_at=cutoff,
        retrieved_at=datetime.now(timezone.utc),
        source="finnhub_historical_eps_surprise",
        ticker=event.ticker,
        reported_eps=actual,
        consensus_eps=estimate,
        event_id=event.event_id,
    )


def finnhub_news_records(
    payload: list[dict[str, Any]],
    ticker: str,
    *,
    retrieved_at: datetime,
) -> tuple[NewsRecord, ...]:
    out: list[NewsRecord] = []
    for raw in payload:
        try:
            published = datetime.fromtimestamp(int(raw["datetime"]), tz=timezone.utc)
            headline = str(raw["headline"]).strip()
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if not headline:
            continue
        related = str(raw.get("related") or "")
        entities = [ticker.upper()]
        entities.extend(part.strip().upper() for part in related.split(",") if part.strip())
        source_id = str(raw.get("id")) if raw.get("id") is not None else None
        out.append(
            NewsRecord(
                value_timestamp=published,
                available_at=published,
                retrieved_at=retrieved_at,
                source=str(raw.get("source") or "finnhub"),
                headline=headline,
                published_at=published,
                entities=tuple(dict.fromkeys(entities)),
                url=raw.get("url"),
                source_id=source_id,
                summary=raw.get("summary"),
                excerpt=raw.get("summary"),
                vendor_relevance=1.0,
            )
        )
    return tuple(sorted(out, key=lambda row: row.published_at))
