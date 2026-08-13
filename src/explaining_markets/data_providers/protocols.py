"""Provider Protocols — the only surface the feature layer depends on.

Implementations must:

* treat ``cutoff`` as mandatory and return only records whose
  ``available_at`` is strictly before it (point-in-time discipline lives in
  the provider, not just the caller);
* never perform bulk downloads inside a live prediction path — live callers
  read the prebuilt cache (:class:`~explaining_markets.data_providers.cache.CompanyHistoryCache`),
  while vendor implementations are for OFFLINE cache building only;
* populate provenance fields honestly, and leave a field ``None`` rather than
  guessing when the vendor cannot supply it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from explaining_markets.data_providers.records import EarningsRecord, PriceBar


@runtime_checkable
class MarketDataProvider(Protocol):
    """Daily adjusted price history for one ticker, strictly before a cutoff."""

    def daily_prices_before(
        self,
        ticker: str,
        cutoff: datetime,
        *,
        max_days: int = 5 * 365,
    ) -> list[PriceBar]:
        """Return bars with ``available_at < cutoff``, ascending by session date."""
        ...


@runtime_checkable
class EarningsDataProvider(Protocol):
    """Historical earnings events for one ticker, strictly before a cutoff."""

    def earnings_before(
        self,
        ticker: str,
        cutoff: datetime,
        *,
        max_events: int = 40,
    ) -> list[EarningsRecord]:
        """Return records with ``available_at < cutoff``, ascending by event time.

        Reaction fields may still be individually unusable at the cutoff
        (``reaction_usable_at``); callers must check per-field availability.
        """
        ...
