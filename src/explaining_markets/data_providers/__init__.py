"""Vendor-agnostic market/earnings data provider interfaces.

Every record type carries full point-in-time provenance:

* ``value_timestamp``     — when the value refers to (e.g. the session close date)
* ``available_at``        — when the value first became publicly knowable
* ``retrieved_at``        — when we actually fetched it from the source
* ``source``              — which vendor/feed produced it

A record is usable for an event with knowledge cutoff ``T`` only when
``available_at <= T``. If ``available_at`` is unknown, callers must apply a
conservative rule (see ``company_history.py``) or exclude the record — the
system fails closed rather than leaking.

No vendor implementation ships here yet: no market-data credentials are
configured in this repository (``.env`` holds only competition/OpenAI keys).
The interfaces, SQLite cache schema, and point-in-time logic are complete and
fixture-tested so a vendor (Alpha Vantage, Polygon, FMP, Tiingo, ...) can be
added by implementing the two Protocols without touching the model layer.
"""

from explaining_markets.data_providers.records import (
    EarningsRecord,
    PriceBar,
)
from explaining_markets.data_providers.protocols import (
    EarningsDataProvider,
    MarketDataProvider,
)
from explaining_markets.data_providers.cache import CompanyHistoryCache
from explaining_markets.data_providers.fixtures import InMemoryProvider

__all__ = [
    "PriceBar",
    "EarningsRecord",
    "MarketDataProvider",
    "EarningsDataProvider",
    "CompanyHistoryCache",
    "InMemoryProvider",
]
