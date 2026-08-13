from datetime import datetime, timedelta, timezone

from explaining_markets.feature_families.market_sector import market_sector_features
from explaining_markets.feature_families.price_context import price_context_features
from explaining_markets.v3_records import PriceRecord

T = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)


def series(ticker, n=300, base=100.0):
    rows = []
    for i in range(n):
        ts = T - timedelta(days=n-i)
        rows.append(PriceRecord(value_timestamp=ts, available_at=ts, retrieved_at=T, source="fixture", ticker=ticker, close=base+i*0.1))
    return tuple(rows)


def test_price_windows_and_relative_returns():
    stock = series("XYZ", base=100)
    market = series("SPY", base=200)
    sector = series("SEC", base=150)
    features = price_context_features(stock, T)
    assert features["return_20d"] > 0
    assert features["has_5y_price_history"] == 0.0
    relative = market_sector_features(stock, market, sector, T)
    assert "stock_minus_market_20d" in relative
    assert relative["sector_volatility_20d"] >= 0
