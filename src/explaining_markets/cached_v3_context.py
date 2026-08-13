"""Adapter from the existing offline SQLite history cache into V3 records."""
from __future__ import annotations

from explaining_markets.data_providers.cache import CompanyHistoryCache, DEFAULT_CACHE_PATH
from explaining_markets.v3_records import EarningsRecord, PriceRecord, V3Context


def _price(bar):
    return PriceRecord(
        value_timestamp=bar.value_timestamp,
        available_at=bar.available_at,
        retrieved_at=bar.retrieved_at,
        source=bar.source,
        ticker=bar.ticker,
        close=float(bar.adjusted_close),
    )


def _earnings(row, cutoff):
    reaction_available = row.reaction_available_at if row.reaction_usable_at(cutoff) else None
    available_at = row.available_at
    abnormal_return = None
    if reaction_available is not None:
        available_at = max(available_at, reaction_available)
        abnormal_return = row.abnormal_return
    return EarningsRecord(
        value_timestamp=row.event_timestamp,
        available_at=available_at,
        retrieved_at=row.retrieved_at,
        source=row.source,
        ticker=row.ticker,
        reported_eps=row.eps_actual,
        consensus_eps=row.eps_estimate,
        reported_revenue=row.revenue_actual,
        consensus_revenue=row.revenue_estimate,
        abnormal_return=abnormal_return,
    )


def context_from_existing_cache(ticker, cutoff, *, market_ticker="SPY") -> V3Context:
    """Use only already-built cache data; never create/rebuild history live."""
    if not DEFAULT_CACHE_PATH.exists():
        return V3Context(ticker=ticker, cutoff=cutoff)
    cache = CompanyHistoryCache(DEFAULT_CACHE_PATH)
    try:
        stock = tuple(_price(x) for x in cache.daily_prices_before(ticker, cutoff, max_days=5 * 366))
        market = tuple(_price(x) for x in cache.daily_prices_before(market_ticker, cutoff, max_days=5 * 366))
        old_history = cache.earnings_before(ticker, cutoff, max_events=40)
        history = tuple(_earnings(x, cutoff) for x in old_history)
        history = tuple(x for x in history if x.eligible(cutoff))
        return V3Context(
            ticker=ticker,
            cutoff=cutoff,
            company_history=history,
            stock_prices=stock,
            market_prices=market,
            extras={"cache_source": str(DEFAULT_CACHE_PATH)},
        )
    finally:
        cache.close()
