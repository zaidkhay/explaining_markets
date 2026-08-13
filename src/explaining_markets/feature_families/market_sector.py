"""Market and sector context derived only from pre-cutoff observations."""
from __future__ import annotations

from explaining_markets.feature_families.price_context import eligible_prices, trailing_return
from explaining_markets.v3_records import PriceRecord

MARKET_SECTOR_FEATURE_NAMES = (
    "market_return_1d", "market_return_5d", "sector_return_1d", "sector_return_5d",
    "sector_return_20d", "sector_volatility_20d", "stock_minus_market_5d",
    "stock_minus_market_20d", "stock_minus_market_60d", "stock_minus_sector_20d",
    "stock_minus_sector_60d",
)


def _vol20(rows):
    if len(rows) < 3:
        return 0.0
    rets = [rows[i].close / rows[i - 1].close - 1.0 for i in range(1, len(rows[-21:]))]
    if len(rets) < 2:
        return 0.0
    mu = sum(rets) / len(rets)
    variance = sum((x - mu) ** 2 for x in rets) / len(rets)
    return (variance ** 0.5) * (252.0 ** 0.5)


def market_sector_features(
    stock: tuple[PriceRecord, ...], market: tuple[PriceRecord, ...], sector: tuple[PriceRecord, ...], cutoff
) -> dict[str, float]:
    s = eligible_prices(stock, cutoff)
    m = eligible_prices(market, cutoff)
    sec = eligible_prices(sector, cutoff)
    sr = {w: trailing_return(s, w) for w in (5, 20, 60)}
    mr = {w: trailing_return(m, w) for w in (1, 5, 20, 60)}
    cr = {w: trailing_return(sec, w) for w in (1, 5, 20, 60)}
    return {
        "market_return_1d": mr[1],
        "market_return_5d": mr[5],
        "sector_return_1d": cr[1],
        "sector_return_5d": cr[5],
        "sector_return_20d": cr[20],
        "sector_volatility_20d": _vol20(sec),
        "stock_minus_market_5d": sr[5] - mr[5],
        "stock_minus_market_20d": sr[20] - mr[20],
        "stock_minus_market_60d": sr[60] - mr[60],
        "stock_minus_sector_20d": sr[20] - cr[20],
        "stock_minus_sector_60d": sr[60] - cr[60],
    }
