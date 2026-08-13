from datetime import datetime, timezone

from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.v3_records import V3Context


def test_v3_feature_order_and_missingness():
    cutoff = datetime(2026, 8, 1, tzinfo=timezone.utc)
    vector = build_feature_vector_v3(disclosure=[], context=V3Context(ticker="XYZ", cutoff=cutoff))
    assert tuple(vector.values) == MODEL_FEATURE_NAMES_V3
    assert vector.values["has_eps_surprise"] == 0.0
    assert vector.values["has_revenue_surprise"] == 0.0
    assert vector.values["has_company_earnings_history"] == 0.0
    assert vector.values["has_peer_data"] == 0.0
    assert vector.values["has_company_news"] == 0.0
