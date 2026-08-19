from scripts.build_v3_lite_candidate import (
    CORE_RESULT_FEATURES,
    EPS_DIRECTIONAL_FEATURES,
    LIVE_CANDIDATE_FEATURE_SETS,
    REVENUE_DIRECTIONAL_FEATURES,
)


def test_availability_only_ablation_is_not_a_live_candidate():
    assert "fls_plus_availability" not in LIVE_CANDIDATE_FEATURE_SETS


def test_live_candidates_include_direct_result_direction_sets():
    assert "fls_plus_core_results" in LIVE_CANDIDATE_FEATURE_SETS
    assert "fls_plus_core_results_reasoning" in LIVE_CANDIDATE_FEATURE_SETS
    assert "fls_plus_results" in LIVE_CANDIDATE_FEATURE_SETS
    assert "fls_plus_results_reasoning" in LIVE_CANDIDATE_FEATURE_SETS
    assert "eps_surprise_percent" in LIVE_CANDIDATE_FEATURE_SETS["fls_plus_results"]
    assert "revenue_surprise_percent" in LIVE_CANDIDATE_FEATURE_SETS["fls_plus_results"]
    assert "is_eps_beat" in LIVE_CANDIDATE_FEATURE_SETS["fls_plus_results"]
    assert "is_revenue_miss" in LIVE_CANDIDATE_FEATURE_SETS["fls_plus_results"]


def test_core_result_set_is_small_scale_invariant_and_directional():
    assert CORE_RESULT_FEATURES == (
        "eps_surprise_percent",
        "has_eps_surprise",
        "revenue_surprise_percent",
        "has_revenue_surprise",
    )


def test_emergency_feature_sets_exclude_source_scale_dependent_raw_amounts():
    assert "reported_eps" not in EPS_DIRECTIONAL_FEATURES
    assert "consensus_eps" not in EPS_DIRECTIONAL_FEATURES
    assert "eps_surprise_absolute" not in EPS_DIRECTIONAL_FEATURES
    assert "reported_revenue" not in REVENUE_DIRECTIONAL_FEATURES
    assert "consensus_revenue" not in REVENUE_DIRECTIONAL_FEATURES
    assert "revenue_surprise_absolute" not in REVENUE_DIRECTIONAL_FEATURES
