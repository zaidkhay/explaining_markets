"""Pre-event price context. All inputs are filtered by available_at <= cutoff."""
from __future__ import annotations

import math
from statistics import pstdev

from explaining_markets.v3_records import PriceRecord

PRICE_CONTEXT_FEATURE_NAMES = (
    "return_1d_pre_event", "return_5d", "return_20d", "return_60d", "return_120d",
    "return_252d", "return_3y", "return_5y", "realized_volatility_20d",
    "realized_volatility_60d", "realized_volatility_252d", "max_drawdown_60d",
    "max_drawdown_252d", "distance_from_20d_high", "distance_from_52w_high",
    "distance_from_20d_low", "distance_from_52w_low", "volume_zscore", "has_5y_price_history",
)


def eligible_prices(rows: tuple[PriceRecord, ...], cutoff) -> list[PriceRecord]:
    return sorted((r for r in rows if r.eligible(cutoff) and r.close > 0), key=lambda r: r.value_timestamp)


def trailing_return(rows: list[PriceRecord], sessions: int) -> float:
    if len(rows) <= sessions:
        return 0.0
    return rows[-1].close / rows[-1 - sessions].close - 1.0


def _daily_returns(rows: list[PriceRecord]) -> list[float]:
    return [rows[i].close / rows[i - 1].close - 1.0 for i in range(1, len(rows)) if rows[i - 1].close > 0]


def _vol(rows: list[PriceRecord], sessions: int) -> float:
    if len(rows) < 3:
        return 0.0
    rets = _daily_returns(rows[-(sessions + 1):])
    return pstdev(rets) * math.sqrt(252.0) if len(rets) >= 2 else 0.0


def _drawdown(rows: list[PriceRecord], sessions: int) -> float:
    values = [r.close for r in rows[-sessions:]]
    if len(values) < 2:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _distance(rows: list[PriceRecord], sessions: int, high: bool) -> float:
    values = [r.close for r in rows[-sessions:]]
    if not values:
        return 0.0
    anchor = max(values) if high else min(values)
    return rows[-1].close / anchor - 1.0 if anchor else 0.0


def _volume_zscore(rows: list[PriceRecord]) -> float:
    vols = [float(r.volume) for r in rows[-60:] if r.volume is not None]
    if len(vols) < 10:
        return 0.0
    mu = sum(vols) / len(vols)
    sd = pstdev(vols)
    return (vols[-1] - mu) / sd if sd > 1e-12 else 0.0


def price_context_features(records: tuple[PriceRecord, ...], cutoff) -> dict[str, float]:
    rows = eligible_prices(records, cutoff)
    out = {name: 0.0 for name in PRICE_CONTEXT_FEATURE_NAMES}
    if not rows:
        return out
    windows = {1: "return_1d_pre_event", 5: "return_5d", 20: "return_20d", 60: "return_60d", 120: "return_120d", 252: "return_252d", 756: "return_3y", 1260: "return_5y"}
    for sessions, name in windows.items():
        out[name] = trailing_return(rows, sessions)
    out.update({
        "realized_volatility_20d": _vol(rows, 20),
        "realized_volatility_60d": _vol(rows, 60),
        "realized_volatility_252d": _vol(rows, 252),
        "max_drawdown_60d": _drawdown(rows, 60),
        "max_drawdown_252d": _drawdown(rows, 252),
        "distance_from_20d_high": _distance(rows, 20, True),
        "distance_from_52w_high": _distance(rows, 252, True),
        "distance_from_20d_low": _distance(rows, 20, False),
        "distance_from_52w_low": _distance(rows, 252, False),
        "volume_zscore": _volume_zscore(rows),
        "has_5y_price_history": float(len(rows) >= 1261),
    })
    return out
