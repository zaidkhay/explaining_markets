"""Free-tier historical providers used by V3 enrichment.

Tiingo, Twelve Data, and FMP supply historical EOD prices. Finnhub supplies the
last four quarterly EPS surprises and one year of company news. Successful
responses are cached by the caller's cache object; failures never become fake
zero-valued records.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx

from explaining_markets.historical import HistoricalEvent
from explaining_markets.providers.retry_policy import (
    RetryPolicy,
    TransientProviderError,
    UnsupportedSymbolError,
    classify_provider_message,
    classify_status,
    is_timeout_exception,
    is_transient_exception,
    parse_retry_after,
)
from explaining_markets.providers.unsupported_cache import UnsupportedSymbolCache
from explaining_markets.v3_records import EarningsRecord, NewsRecord, PriceRecord

_TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"
_FINNHUB_BASE = "https://finnhub.io/api/v1"
_FMP_BASE = "https://financialmodelingprep.com/stable"
_TWELVE_DATA_BASE = "https://api.twelvedata.com"


class ProviderBudgetExhausted(RuntimeError):
    pass


class ProviderUnavailable(RuntimeError):
    pass


@dataclass
class ProviderStats:
    """Per-run provider accounting for the backfill statistics report.

    ``symbols_requested`` counts distinct symbol fetch operations that reached
    the network path, while ``request_attempts`` counts individual HTTP
    attempts (so a retried fetch contributes one request and several attempts).
    Every attempt consumes the API budget, because the provider's own quota
    counts attempts, not successes.
    """

    vendor: str
    symbols_considered: int = 0
    symbols_already_covered: int = 0
    symbols_skipped_unsupported: int = 0
    symbols_requested: int = 0
    request_attempts: int = 0
    successful_symbols: int = 0
    transient_failures: int = 0
    permanent_failures: int = 0
    timeout_failures: int = 0
    rate_limit_failures: int = 0
    retries_performed: int = 0
    cache_hits: int = 0
    budget_exhausted_events: int = 0
    rows_unlocked: int = 0
    max_api_calls: int = 0
    unsupported_recorded: tuple[str, ...] = ()

    @property
    def budget_used(self) -> int:
        return self.request_attempts

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_api_calls - self.request_attempts)

    def as_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "symbols_considered": self.symbols_considered,
            "symbols_already_covered": self.symbols_already_covered,
            "symbols_skipped_unsupported": self.symbols_skipped_unsupported,
            "symbols_requested": self.symbols_requested,
            "request_attempts": self.request_attempts,
            "successful_symbols": self.successful_symbols,
            "transient_failures": self.transient_failures,
            "permanent_failures": self.permanent_failures,
            "timeout_failures": self.timeout_failures,
            "rate_limit_failures": self.rate_limit_failures,
            "retries_performed": self.retries_performed,
            "cache_hits": self.cache_hits,
            "budget_exhausted_events": self.budget_exhausted_events,
            "rows_unlocked": self.rows_unlocked,
            "api_budget": self.max_api_calls,
            "api_budget_used": self.budget_used,
            "api_budget_remaining": self.budget_remaining,
            "unsupported_recorded": list(self.unsupported_recorded),
        }

    def check_invariants(self) -> None:
        """Assert the accounting identities the backfill report depends on."""
        if self.symbols_requested != (
            self.successful_symbols + self.transient_failures + self.permanent_failures
        ):
            raise AssertionError(
                f"{self.vendor} accounting mismatch: requested={self.symbols_requested} "
                f"success={self.successful_symbols} transient={self.transient_failures} "
                f"permanent={self.permanent_failures}"
            )
        if self.request_attempts < self.symbols_requested:
            raise AssertionError(
                f"{self.vendor} attempts ({self.request_attempts}) < requested "
                f"({self.symbols_requested})"
            )
        if self.request_attempts > self.max_api_calls:
            raise AssertionError(
                f"{self.vendor} exceeded API budget: {self.request_attempts} > {self.max_api_calls}"
            )


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_response_text(response: httpx.Response, *, limit: int = 300) -> str:
    """Best-effort provider error text from a failed response.

    Prefers a JSON ``message``/``error`` field, falls back to a bounded slice
    of the body. Never raises, so error handling cannot itself fail.
    """
    try:
        payload = response.json()
        if isinstance(payload, dict):
            for key in ("message", "error", "Error Message", "detail"):
                value = payload.get(key)
                if value:
                    return str(value)[:limit]
        return str(payload)[:limit]
    except Exception:  # noqa: BLE001 - error path must never raise
        try:
            return response.text[:limit]
        except Exception:  # noqa: BLE001
            return f"HTTP {response.status_code}"


def _is_run_wide_provider_error(message: str) -> bool:
    text = message.lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "too many requests",
            "api credits",
            "credits left",
            "daily limit",
            "daily request",
            "quota",
            "limit reached",
        )
    )


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
        retry_policy: RetryPolicy | None = None,
        unsupported_cache: UnsupportedSymbolCache | None = None,
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
        self.retry_policy = retry_policy or RetryPolicy()
        self.unsupported_cache = unsupported_cache
        self.stats = ProviderStats(vendor=vendor, max_api_calls=self.max_api_calls)
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

    def _request(
        self,
        *,
        namespace: str,
        cache_key: str,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
        ticker: str | None = None,
    ) -> Any:
        """Fetch with bounded retries for transient faults only.

        Retry semantics:
        * network timeouts / transport errors / HTTP 408,429,5xx -> retried
        * HTTP 429 honors ``Retry-After`` when numeric, else exponential backoff
        * provider "symbol not found"/plan messages -> ``UnsupportedSymbolError``
          (recorded in the unsupported cache, never retried)
        * run-wide quota messages -> circuit breaker via ``unavailable_reason``

        Every HTTP attempt consumes the per-run budget, so retries can never
        exceed ``max_api_calls``.
        """
        cached = self.cache.get(namespace, cache_key)
        if cached is not None:
            self.cache_hits += 1
            self.stats.cache_hits += 1
            return cached
        if self.unavailable_reason:
            raise ProviderUnavailable(self.unavailable_reason)

        symbol = (ticker or cache_key.split("|", 1)[0]).upper()
        if self.unsupported_cache is not None and self.unsupported_cache.should_skip(symbol):
            entry = self.unsupported_cache.entry(symbol)
            self.stats.symbols_skipped_unsupported += 1
            self.progress(f"[V3_ENRICH] {self.vendor} skip unsupported ticker={symbol}")
            raise UnsupportedSymbolError(
                f"{self.vendor}: {symbol} is a known unsupported symbol",
                ticker=symbol,
                vendor=self.vendor,
                reason=entry.reason if entry else "unsupported_symbol",
                status_code=entry.status_code if entry else None,
                provider_message=entry.provider_message if entry else None,
            )

        self.stats.symbols_requested += 1
        attempt = 0
        while True:
            attempt += 1
            if self.api_calls >= self.max_api_calls:
                self.stats.budget_exhausted_events += 1
                if attempt > 1:
                    # Budget ran out mid-retry: the fetch failed transiently.
                    self.stats.transient_failures += 1
                    self.failures += 1
                    raise TransientProviderError(
                        f"{self.vendor} budget exhausted while retrying {symbol}"
                    )
                self.stats.symbols_requested -= 1
                raise ProviderBudgetExhausted(
                    f"{self.vendor} per-run API budget exhausted ({self.max_api_calls})"
                )

            self._pace()
            self.api_calls += 1
            self.stats.request_attempts += 1
            self._last_request_started = time.monotonic()
            self.progress(
                f"[V3_ENRICH] {self.vendor} {namespace}:{cache_key} "
                f"calls={self.api_calls}/{self.max_api_calls}"
            )

            retry_after: float | None = None
            failure_reason: str | None = None
            try:
                response = self.client.get(url, params=params, headers=headers, timeout=self.timeout)
                status_class = classify_status(response.status_code)
                if response.status_code >= 400:
                    if status_class == "rate_limit":
                        self.stats.rate_limit_failures += 1
                        retry_after = parse_retry_after(response.headers.get("Retry-After"))
                        failure_reason = "HTTP 429"
                    elif status_class == "transient":
                        failure_reason = f"HTTP {response.status_code}"
                    else:
                        # Permanent HTTP status: classify the body for symbol vs plan.
                        self._fail_permanent(
                            symbol,
                            status_code=response.status_code,
                            message=_safe_response_text(response),
                        )
                else:
                    payload = response.json()
                    self._check_payload_error(payload, symbol=symbol, status_code=response.status_code)
                    self.cache.put(namespace, cache_key, payload)
                    self.stats.successful_symbols += 1
                    return payload
            except (UnsupportedSymbolError, ProviderUnavailable):
                raise
            except ValueError as exc:
                # Malformed JSON: treat as transient (proxies truncate bodies).
                failure_reason = type(exc).__name__
            except Exception as exc:  # noqa: BLE001 - classified immediately below
                if not is_transient_exception(exc):
                    self.stats.permanent_failures += 1
                    self.failures += 1
                    raise RuntimeError(f"{self.vendor} request failed: {type(exc).__name__}") from exc
                if is_timeout_exception(exc):
                    self.stats.timeout_failures += 1
                failure_reason = type(exc).__name__

            # ---- transient path: retry if attempts remain -----------------
            if not self.retry_policy.should_retry(attempt):
                self.stats.transient_failures += 1
                self.failures += 1
                if failure_reason == "HTTP 429":
                    # Repeated 429s are a run-wide signal, not a symbol problem.
                    self.unavailable_reason = f"{self.vendor} rate limit reached (HTTP 429)"
                    raise ProviderUnavailable(self.unavailable_reason)
                raise TransientProviderError(
                    f"{self.vendor} request failed after {attempt} attempt(s) "
                    f"for {symbol}: {failure_reason}"
                )
            self.stats.retries_performed += 1
            self.progress(
                f"[V3_ENRICH] {self.vendor} retry ticker={symbol} "
                f"attempt={attempt + 1}/{self.retry_policy.max_attempts} reason={failure_reason}"
            )
            self.retry_policy.wait(attempt, retry_after=retry_after)

    def _fail_permanent(self, symbol: str, *, status_code: int | None, message: str | None) -> None:
        """Raise a permanent failure; cache it only when confidently classified.

        An ambiguous provider error (no recognizable "symbol not found" or plan
        wording) raises a plain ``RuntimeError`` and is NOT written to the
        unsupported cache — guessing would permanently blacklist a good symbol.
        """
        self.stats.permanent_failures += 1
        self.failures += 1
        reason = classify_provider_message(message or "")
        if reason is None and status_code in {400, 401, 403, 404}:
            # Symbol-scoped 4xx with an unhelpful body is still permanent for
            # this symbol; label it by status rather than inventing wording.
            reason = "entitlement" if status_code in {401, 403} else "unsupported_symbol"
        if reason is None:
            raise RuntimeError(f"{self.vendor}: {message or f'HTTP {status_code}'}")
        if self.unsupported_cache is not None:
            self.unsupported_cache.record(
                symbol,
                reason=reason,
                status_code=status_code,
                provider_message=message,
            )
            self.stats.unsupported_recorded = (*self.stats.unsupported_recorded, symbol)
        raise UnsupportedSymbolError(
            f"{self.vendor}: {message or f'HTTP {status_code}'}",
            ticker=symbol,
            vendor=self.vendor,
            reason=reason,
            status_code=status_code,
            provider_message=message,
        )

    def _check_payload_error(self, payload: Any, *, symbol: str, status_code: int | None) -> None:
        """Raise for provider-level errors embedded in a HTTP 200 body."""
        if not isinstance(payload, dict):
            return
        error = payload.get("error") or payload.get("message") or payload.get("Error Message")
        status = str(payload.get("status") or "").lower()
        if not error or not (len(payload) <= 4 or status == "error"):
            return
        text = str(error)
        if _is_run_wide_provider_error(text):
            message = f"{self.vendor}: {error}"
            self.stats.rate_limit_failures += 1
            self.failures += 1
            self.unavailable_reason = message
            raise ProviderUnavailable(message)
        code = payload.get("code")
        self._fail_permanent(
            symbol,
            status_code=int(code) if isinstance(code, (int, float)) else status_code,
            message=text,
        )


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


class TwelveDataHistoricalClient(_CachedJsonClient):
    """Twelve Data daily adjusted time-series provider.

    The Basic plan currently exposes 8 API credits/minute and 800/day. One
    ``/time_series`` symbol costs one credit, so requests are paced just below
    the documented minute limit. Responses are cached per ticker/date span.
    """

    def __init__(
        self,
        api_key: str,
        *,
        cache,
        max_api_calls: int = 0,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
        progress: Callable[[str], None] | None = None,
        retry_policy: RetryPolicy | None = None,
        unsupported_cache: UnsupportedSymbolCache | None = None,
        min_request_interval: float = 7.6,
    ) -> None:
        if not api_key:
            raise ValueError("TWELVE_DATA_API_KEY is required")
        super().__init__(
            cache=cache,
            max_api_calls=max_api_calls,
            timeout=timeout,
            min_request_interval=min_request_interval,
            client=client,
            progress=progress,
            vendor="twelve_data",
            retry_policy=retry_policy,
            unsupported_cache=unsupported_cache,
        )
        self.api_key = api_key

    def prices_payload(self, ticker: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        symbol = ticker.upper()
        key = f"{symbol}|{start.date()}|{end.date()}|adjust=all"
        payload = self._request(
            namespace="twelve_data_prices",
            cache_key=key,
            ticker=symbol,
            url=f"{_TWELVE_DATA_BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "adjust": "all",
                "order": "ASC",
                "apikey": self.api_key,
            },
            headers={"Accept": "application/json"},
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Twelve Data time-series response was not an object")
        values = payload.get("values")
        if not isinstance(values, list):
            raise RuntimeError("Twelve Data time-series response had no values list")
        return values


class FmpHistoricalClient(_CachedJsonClient):
    """Financial Modeling Prep EOD fallback for symbols permitted by the plan."""

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
        # Free plan is 60 requests/minute; 1.05 seconds keeps us conservatively below it.
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


def twelve_data_price_records(payload: list[dict[str, Any]], ticker: str, *, retrieved_at: datetime) -> tuple[PriceRecord, ...]:
    """Normalize Twelve Data ``adjust=all`` daily bars for point-in-time use."""
    out: list[PriceRecord] = []
    for raw in payload:
        date_raw = raw.get("datetime")
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
                source="twelve_data_eod_adjust_all",
                ticker=ticker.upper(),
                close=close,
                volume=float(volume_raw) if volume_raw not in (None, "") else None,
            )
        )
    return tuple(sorted(out, key=lambda row: row.value_timestamp))


def fmp_price_records(payload: list[dict[str, Any]], ticker: str, *, retrieved_at: datetime) -> tuple[PriceRecord, ...]:
    """Normalize FMP EOD rows conservatively for pre-event use."""
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
