"""Deterministic peer/sympathy aggregation using only eligible peer records."""
from __future__ import annotations

from statistics import mean, pstdev

from explaining_markets.feature_families.price_context import eligible_prices, trailing_return
from explaining_markets.v3_records import EarningsRecord, PeerRecord, PriceRecord

PEER_FEATURE_NAMES = (
    "peer_mean_return_1d", "peer_mean_return_5d", "peer_mean_return_20d",
    "peer_abnormal_return_1d", "peer_dispersion_1d", "peer_dispersion_5d",
    "recent_peer_earnings_count", "recent_peer_eps_surprise_mean",
    "recent_peer_revenue_surprise_mean", "recent_peer_guidance_signal_mean",
    "recent_peer_post_earnings_abnormal_return_mean", "has_peer_data",
)


def select_peers(records: tuple[PeerRecord, ...], cutoff, limit: int = 10) -> tuple[PeerRecord, ...]:
    eligible = [r for r in records if r.eligible(cutoff)]
    eligible.sort(key=lambda r: (-r.score, r.peer_ticker))
    return tuple(eligible[:limit])


def _mean(values):
    return mean(values) if values else 0.0


def peer_sympathy_features(
    peers: tuple[PeerRecord, ...],
    peer_prices: dict[str, tuple[PriceRecord, ...]],
    peer_earnings: tuple[EarningsRecord, ...],
    market_return_1d: float,
    cutoff,
) -> dict[str, float]:
    selected = select_peers(peers, cutoff)
    returns = {1: [], 5: [], 20: []}
    names = {p.peer_ticker for p in selected}
    for ticker in names:
        rows = eligible_prices(peer_prices.get(ticker, ()), cutoff)
        for window in returns:
            returns[window].append(trailing_return(rows, window))
    recent = [r for r in peer_earnings if r.ticker in names and r.eligible(cutoff)]
    eps = []
    rev = []
    reactions = []
    for row in recent:
        if row.reported_eps is not None and row.consensus_eps is not None:
            eps.append((row.reported_eps - row.consensus_eps) / max(abs(row.consensus_eps), 0.05))
        if row.reported_revenue is not None and row.consensus_revenue is not None:
            rev.append((row.reported_revenue - row.consensus_revenue) / max(abs(row.consensus_revenue), 1.0))
        if row.abnormal_return is not None:
            reactions.append(float(row.abnormal_return))
    return {
        "peer_mean_return_1d": _mean(returns[1]),
        "peer_mean_return_5d": _mean(returns[5]),
        "peer_mean_return_20d": _mean(returns[20]),
        "peer_abnormal_return_1d": _mean(returns[1]) - market_return_1d,
        "peer_dispersion_1d": pstdev(returns[1]) if len(returns[1]) >= 2 else 0.0,
        "peer_dispersion_5d": pstdev(returns[5]) if len(returns[5]) >= 2 else 0.0,
        "recent_peer_earnings_count": float(len(recent)),
        "recent_peer_eps_surprise_mean": _mean(eps),
        "recent_peer_revenue_surprise_mean": _mean(rev),
        "recent_peer_guidance_signal_mean": 0.0,
        "recent_peer_post_earnings_abnormal_return_mean": _mean(reactions),
        "has_peer_data": float(bool(selected)),
    }
