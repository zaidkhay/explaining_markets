from scripts.build_v3_lite_candidate import (
    AUTO_CANDIDATE_ABLATIONS,
    _directional_disclosure_ablation,
)


def test_availability_only_ablation_is_not_directional():
    assert _directional_disclosure_ablation("fls_plus_availability") is False


def test_auto_candidates_are_live_safe_and_directional():
    assert set(AUTO_CANDIDATE_ABLATIONS) == {"fls_plus_eps", "fls_plus_reasoning"}
    assert all(_directional_disclosure_ablation(name) for name in AUTO_CANDIDATE_ABLATIONS)


def test_directional_eps_and_reasoning_ablations_are_accepted():
    assert _directional_disclosure_ablation("fls_plus_eps") is True
    assert _directional_disclosure_ablation("fls_plus_reasoning") is True
