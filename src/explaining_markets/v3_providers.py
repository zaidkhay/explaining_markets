"""Provider interfaces for point-in-time V3 data sources.

Implementations may use local caches, databases, or external APIs. Missing
credentials must produce empty results rather than fabricated observations.
"""
from __future__ import annotations

from typing import Protocol

from explaining_markets.v3_records import (
    CompanyMetadataRecord,
    EarningsRecord,
    GuidanceRecord,
    NewsRecord,
    PeerRecord,
    PriceRecord,
)


class EarningsProvider(Protocol):
    def current(self, ticker, cutoff) -> EarningsRecord | None: ...
    def history(self, ticker, cutoff, years: int = 5) -> tuple[EarningsRecord, ...]: ...


class GuidanceProvider(Protocol):
    def current(self, ticker, cutoff) -> GuidanceRecord | None: ...


class PriceProvider(Protocol):
    def history(self, ticker, cutoff, years: int = 5) -> tuple[PriceRecord, ...]: ...


class MetadataProvider(Protocol):
    def metadata(self, ticker, cutoff) -> CompanyMetadataRecord | None: ...


class PeerProvider(Protocol):
    def peers(self, ticker, cutoff, limit: int = 10) -> tuple[PeerRecord, ...]: ...


class NewsProvider(Protocol):
    def company_news(self, ticker, cutoff, days: int = 7) -> tuple[NewsRecord, ...]: ...
    def peer_news(self, tickers, cutoff, days: int = 7) -> tuple[NewsRecord, ...]: ...
    def sector_news(self, sector, cutoff, days: int = 7) -> tuple[NewsRecord, ...]: ...


class NullV3Providers:
    """Credential-free provider bundle used to express missingness safely."""

    def current(self, ticker, cutoff):
        return None

    def history(self, ticker, cutoff, years: int = 5):
        return ()

    def metadata(self, ticker, cutoff):
        return None

    def peers(self, ticker, cutoff, limit: int = 10):
        return ()

    def company_news(self, ticker, cutoff, days: int = 7):
        return ()

    def peer_news(self, tickers, cutoff, days: int = 7):
        return ()

    def sector_news(self, sector, cutoff, days: int = 7):
        return ()
