from datetime import datetime, timezone

import pytest

from explaining_markets.disclosure_results_v3 import parse_disclosure_records
from explaining_markets.feature_families.earnings_surprise import earnings_surprise_features
from explaining_markets.feature_families.guidance_expectations import guidance_expectation_features
from explaining_markets.feature_families.revenue_results import revenue_surprise_features
from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.v3_records import V3Context


CUTOFF = datetime(2026, 8, 18, 21, 0, tzinfo=timezone.utc)


def test_percentage_beat_miss_facts_become_structured_surprises():
    parsed = parse_disclosure_records(
        [
            "Revenue missed consensus by 8%.",
            "EPS missed consensus by 12%.",
        ],
        ticker="TEST",
        cutoff=CUTOFF,
    )
    eps = earnings_surprise_features(parsed.earnings, (), CUTOFF)
    rev = revenue_surprise_features(parsed.earnings, (), CUTOFF)
    assert eps["has_eps_surprise"] == 1.0
    assert eps["eps_surprise_percent"] == pytest.approx(-0.12)
    assert eps["is_eps_miss"] == 1.0
    assert rev["has_revenue_surprise"] == 1.0
    assert rev["revenue_surprise_percent"] == pytest.approx(-0.08)
    assert rev["is_revenue_miss"] == 1.0


def test_inline_facts_encode_zero_surprise_not_missing():
    parsed = parse_disclosure_records(
        [
            "Revenue was in line with consensus.",
            "EPS matched consensus.",
        ],
        ticker="TEST",
        cutoff=CUTOFF,
    )
    eps = earnings_surprise_features(parsed.earnings, (), CUTOFF)
    rev = revenue_surprise_features(parsed.earnings, (), CUTOFF)
    assert eps["has_eps_surprise"] == 1.0
    assert eps["eps_surprise_percent"] == pytest.approx(0.0)
    assert rev["has_revenue_surprise"] == 1.0
    assert rev["revenue_surprise_percent"] == pytest.approx(0.0)


def test_exact_actual_consensus_values_and_units_are_supported():
    parsed = parse_disclosure_records(
        [
            "Adjusted EPS was $2.20 vs. consensus $2.00.",
            "Revenue was $4.4 billion versus consensus $4.0 billion.",
        ],
        ticker="TEST",
        cutoff=CUTOFF,
    )
    eps = earnings_surprise_features(parsed.earnings, (), CUTOFF)
    rev = revenue_surprise_features(parsed.earnings, (), CUTOFF)
    assert eps["eps_surprise_percent"] == pytest.approx(0.10)
    assert rev["revenue_surprise_percent"] == pytest.approx(0.10)


def test_directional_guidance_is_parsed_point_in_time():
    parsed = parse_disclosure_records(
        ["The company raised full-year guidance after the quarter."],
        ticker="TEST",
        cutoff=CUTOFF,
    )
    guide = guidance_expectation_features(parsed.guidance, CUTOFF)
    assert parsed.guidance is not None
    assert parsed.guidance.available_at == CUTOFF
    assert guide["numeric_guidance_raised"] == 1.0


def test_v3_feature_builder_uses_disclosure_when_provider_context_is_empty():
    disclosure = [
        "Revenue beat consensus by 8%.",
        "EPS beat consensus by 12%.",
        "The company raised full-year guidance.",
    ]
    vector = build_feature_vector_v3(
        disclosure=disclosure,
        context=V3Context(ticker="TEST", cutoff=CUTOFF),
    )
    assert vector.values["has_eps_surprise"] == 1.0
    assert vector.values["eps_surprise_percent"] == pytest.approx(0.12)
    assert vector.values["has_revenue_surprise"] == 1.0
    assert vector.values["revenue_surprise_percent"] == pytest.approx(0.08)
    assert vector.values["numeric_guidance_raised"] == 1.0


def test_unrelated_growth_percentage_does_not_fake_consensus_surprise():
    parsed = parse_disclosure_records(
        ["Revenue grew 8% year over year and gross margin expanded."],
        ticker="TEST",
        cutoff=CUTOFF,
    )
    assert parsed.earnings is None
