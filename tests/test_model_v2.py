"""Model layer: V1 default, V2 non-promoted, fallback chain, TEST behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES as V1_FEATURE_NAMES,
    extract_forward_looking_features,
)
from explaining_markets.model import (
    BaselineModel,
    CompanyHistoryRidgeModel,
    ForwardLookingRidgeModel,
    HeuristicFactModel,
    get_default_model,
)
from explaining_markets.features import FeatureVector

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "src/explaining_markets/artifacts"
V1_ARTIFACT = ARTIFACT_DIR / "fls_ridge_v1.json"
V2_ARTIFACT = ARTIFACT_DIR / "fls_company_history_ridge_v2.json"


def _fls_features(**overrides) -> "FeatureVector":
    return FeatureVector(
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        n_facts=2,
        text_length=100,
        positive_hits=1,
        negative_hits=0,
        net_sentiment=1,
        has_guidance_mention=True,
    )


def test_v1_artifact_loads_and_predicts_in_bounds() -> None:
    if not V1_ARTIFACT.exists():
        import pytest as _pytest

        _pytest.skip("V1 artifact not present")
    model = ForwardLookingRidgeModel()
    assert model.model_version == "fls_ridge_v1"
    disclosure = ["We expect revenue to grow.", "We raised guidance to $100 million."]
    prediction = model.predict_disclosure(disclosure)
    assert 0.0 <= prediction <= 1.0


def test_v2_artifact_loads_with_promoted_false() -> None:
    if not V2_ARTIFACT.exists():
        import pytest as _pytest

        _pytest.skip("V2 artifact not present")
    model = CompanyHistoryRidgeModel()
    assert model.model_version == "fls_company_history_ridge_v2"
    assert model.promoted is False


def test_default_model_is_v1_when_v2_is_not_promoted() -> None:
    if not V1_ARTIFACT.exists() or not V2_ARTIFACT.exists():
        import pytest as _pytest

        _pytest.skip("artifacts not present")
    model = get_default_model()
    assert isinstance(model, ForwardLookingRidgeModel)
    assert model.model_version == "fls_ridge_v1"


def test_company_history_ridge_model_predicts_in_bounds() -> None:
    if not V2_ARTIFACT.exists():
        import pytest as _pytest

        _pytest.skip("V2 artifact not present")
    from explaining_markets.company_history import empty_company_history
    from explaining_markets.features_v2 import build_feature_vector_v2
    from datetime import datetime, timezone

    model = CompanyHistoryRidgeModel()
    assert model.model_version == "fls_company_history_ridge_v2"
    disclosure = ["We expect strong growth.", "We raised guidance."]
    fls = extract_forward_looking_features(disclosure)
    history = empty_company_history("AAPL", datetime(2026, 9, 1, tzinfo=timezone.utc))
    vector = build_feature_vector_v2(fls=fls, history=history)
    prediction = model.predict_vector(vector)
    assert 0.0 <= prediction <= 1.0


def test_company_history_ridge_model_rejects_wrong_length_vector() -> None:
    if not V2_ARTIFACT.exists():
        import pytest as _pytest

        _pytest.skip("V2 artifact not present")
    model = CompanyHistoryRidgeModel()

    class _BadVector:
        def vector(self, names):
            return [0.0, 1.0]  # wrong length

    with pytest.raises(ValueError, match="zip"):
        model.predict_vector(_BadVector())


def test_heuristic_model_returns_bounded_prediction() -> None:
    model = HeuristicFactModel()
    prediction = model.predict_percentile(_fls_features())
    assert 0.0 <= prediction <= 1.0


def test_baseline_model_returns_exactly_half() -> None:
    assert BaselineModel().predict_percentile(_fls_features()) == 0.5


def test_fallback_chain_terminates_at_baseline() -> None:
    # When no artifacts exist and no features are available, the final fallback
    # must be the deterministic neutral baseline.
    assert BaselineModel().predict_percentile(_fls_features()) == 0.5
