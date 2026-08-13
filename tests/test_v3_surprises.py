from datetime import datetime, timedelta, timezone

from explaining_markets.feature_families.earnings_surprise import earnings_surprise_features
from explaining_markets.feature_families.revenue_results import revenue_surprise_features
from explaining_markets.v3_records import EarningsRecord

T = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)


def row(**values):
    base = dict(value_timestamp=T-timedelta(minutes=5), available_at=T-timedelta(minutes=4), retrieved_at=T, source="fixture", ticker="XYZ")
    base.update(values)
    return EarningsRecord(**base)


def test_eps_negative_consensus_beat():
    features = earnings_surprise_features(row(reported_eps=-0.10, consensus_eps=-0.20), (), T)
    assert features["is_eps_beat"] == 1.0
    assert features["eps_surprise_percent"] > 0


def test_eps_near_zero_consensus_is_bounded_by_floor():
    features = earnings_surprise_features(row(reported_eps=0.02, consensus_eps=0.0), (), T)
    assert features["has_eps_surprise"] == 1.0
    assert abs(features["eps_surprise_percent"]) < 1.0


def test_revenue_double_beat_and_double_miss():
    beat = row(reported_eps=1.2, consensus_eps=1.0, reported_revenue=110, consensus_revenue=100)
    miss = row(reported_eps=0.8, consensus_eps=1.0, reported_revenue=90, consensus_revenue=100)
    assert revenue_surprise_features(beat, (), T)["eps_beat_and_revenue_beat"] == 1.0
    assert revenue_surprise_features(miss, (), T)["eps_miss_and_revenue_miss"] == 1.0
