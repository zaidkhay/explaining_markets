"""In-memory provider for tests — never fabricates data outside test scope.

Satisfies both Protocols with hand-constructed fixture records, so the
feature layer's point-in-time logic can be exercised deterministically
without any vendor credentials or files.
"""

from __future__ import annotations

from datetime import datetime

from explaining_markets.data_providers.records import EarningsRecord, PriceBar


class InMemoryProvider:
    """Test double implementing MarketDataProvider + EarningsDataProvider."""

    def __init__(
        self,
        prices: list[PriceBar] | None = None,
        earnings: list[EarningsRecord] | None = None,
    ) -> None:
        self._prices = sorted(prices or [], key=lambda b: b.value_timestamp)
        self._earnings = sorted(earnings or [], key=lambda r: r.event_timestamp)

    def daily_prices_before(
        self, ticker: str, cutoff: datetime, *, max_days: int = 5 * 365
    ) -> list[PriceBar]:
        bars = [
            b for b in self._prices if b.ticker == ticker and b.usable_at(cutoff)
        ]
        return bars[-max_days:]

    def earnings_before(
        self, ticker: str, cutoff: datetime, *, max_events: int = 40
    ) -> list[EarningsRecord]:
        records = [
            r for r in self._earnings if r.ticker == ticker and r.figures_usable_at(cutoff)
        ]
        return records[-max_events:]
