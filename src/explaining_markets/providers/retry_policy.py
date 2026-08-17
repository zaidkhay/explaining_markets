"""Transient-vs-permanent provider error classification and bounded retries.

Batch historical enrichment fails for two very different reasons:

* TRANSIENT — the network or the provider hiccuped (connect/read timeout,
  reset connection, HTTP 5xx, HTTP 429). The symbol itself is fine and the
  request should be retried a bounded number of times.
* PERMANENT — the provider will never serve this symbol on this plan (symbol
  not found, unsupported security, no historical series, subscription
  restriction). Retrying wastes the run's API budget on a deterministic
  failure, so the symbol is recorded as unsupported and skipped.

Conflating the two is what previously caused a single ``ConnectTimeout`` to
mark a perfectly good ticker permanently unavailable.

Determinism
-----------
``RetryPolicy`` takes injectable ``sleep`` and ``jitter`` callables so tests
can assert exact backoff sequences without real delays and without random
flakiness. Production defaults use ``time.sleep`` and ``random.random``.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import httpx

# Network-level failures that say nothing about whether the symbol exists.
TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)

# Timeout subset, tracked separately for the backfill statistics report.
TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
RATE_LIMIT_STATUS_CODES = frozenset({429})

# Provider messages that mean "this symbol will never work here".
_UNSUPPORTED_MESSAGE_TOKENS = (
    "not found",
    "not available",
    "no data",
    "invalid symbol",
    "unsupported",
    "does not exist",
    "may not be supported",
    "not supported",
    "no historical data",
    "symbol is required",
    "delisted",
    "unavailable for this symbol",
)

# Plan/entitlement rejections: permanent for the symbol under this credential,
# but deliberately distinguished from "symbol does not exist" for reporting.
_ENTITLEMENT_MESSAGE_TOKENS = (
    "not authorized",
    "unauthorized",
    "upgrade your plan",
    "subscription",
    "premium",
    "your plan",
    "grow plan",
    "pro plan",
    "access denied",
    "forbidden",
)


class TransientProviderError(RuntimeError):
    """A retryable failure that remained transient until attempts ran out."""


class UnsupportedSymbolError(RuntimeError):
    """The provider will never serve this symbol; record it and skip it."""

    def __init__(
        self,
        message: str,
        *,
        ticker: str,
        vendor: str,
        reason: str,
        status_code: int | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.vendor = vendor
        self.reason = reason
        self.status_code = status_code
        self.provider_message = provider_message


def is_transient_exception(exc: BaseException) -> bool:
    """True for network faults that justify a retry."""
    return isinstance(exc, TRANSIENT_EXCEPTIONS)


def is_timeout_exception(exc: BaseException) -> bool:
    return isinstance(exc, TIMEOUT_EXCEPTIONS)


def classify_status(status_code: int) -> str:
    """Classify an HTTP status as ``rate_limit``, ``transient`` or ``permanent``."""
    if status_code in RATE_LIMIT_STATUS_CODES:
        return "rate_limit"
    if status_code in TRANSIENT_STATUS_CODES:
        return "transient"
    return "permanent"


def looks_unsupported_symbol(message: str) -> bool:
    """True when a provider message indicates a permanently unusable symbol."""
    text = (message or "").lower()
    return any(token in text for token in _UNSUPPORTED_MESSAGE_TOKENS)


def looks_entitlement_block(message: str) -> bool:
    """True when the message indicates a plan/subscription restriction."""
    text = (message or "").lower()
    return any(token in text for token in _ENTITLEMENT_MESSAGE_TOKENS)


def classify_provider_message(message: str) -> str | None:
    """Return ``unsupported_symbol``/``entitlement`` when permanent, else None.

    Entitlement is checked first: "not authorized for this symbol" contains
    both an entitlement token and (via "not a...") no unsupported token, but an
    explicit plan message should be reported as an entitlement block rather
    than a nonexistent symbol.
    """
    if looks_entitlement_block(message):
        return "entitlement"
    if looks_unsupported_symbol(message):
        return "unsupported_symbol"
    return None


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header expressed in seconds.

    Only the numeric form is honored; an HTTP-date form returns None so the
    caller falls back to bounded exponential backoff rather than trusting a
    possibly-skewed clock difference.
    """
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return None
    return seconds


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with modest jitter.

    ``max_attempts`` counts total attempts, so ``max_attempts=3`` means one
    initial request plus two retries. Defaults are tuned for batch enrichment:
    long enough to ride out a blip, short enough not to stall a 6k-row run.
    """

    max_attempts: int = 3
    base_delay: float = 1.5
    max_delay: float = 20.0
    jitter_ratio: float = 0.25
    max_retry_after: float = 65.0
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("backoff delays must be non-negative")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be within [0, 1]")

    def should_retry(self, attempt: int) -> bool:
        """True when ``attempt`` (1-based) may be followed by another try."""
        return attempt < self.max_attempts

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Backoff before the attempt following ``attempt`` (1-based).

        A provider-supplied ``Retry-After`` wins over exponential backoff but
        is capped by ``max_retry_after`` so a hostile/misconfigured header
        cannot stall the run.
        """
        if retry_after is not None:
            return float(min(retry_after, self.max_retry_after))
        exponential = self.base_delay * (2 ** max(0, attempt - 1))
        capped = min(exponential, self.max_delay)
        # Jitter spreads retries out; it never shortens below the base delay's
        # (1 - jitter_ratio) floor, so backoff stays monotonically meaningful.
        return float(capped * (1.0 - self.jitter_ratio + self.jitter_ratio * self.jitter()))

    def wait(self, attempt: int, *, retry_after: float | None = None) -> float:
        delay = self.delay_for(attempt, retry_after=retry_after)
        if delay > 0:
            self.sleep(delay)
        return delay


NO_RETRY = RetryPolicy(max_attempts=1)
