from __future__ import annotations

import json

import pytest

from explaining_markets.forward_looking_features import extract_forward_looking_features
from explaining_markets.features_v3 import FeatureVectorV3, MODEL_FEATURE_NAMES_V3
from explaining_markets.model import ForwardLookingRidgeModel, get_default_model
from explaining_markets.model_v3_lite import V3LiteCandidateModel


def _artifact(*, promoted: bool = False, operator_override: bool = True) -> dict:
    return {
        "model_version": "v3_lite_operator_test",
        "feature_spec_version": "v3_lite_v1",
        "ablation": "fls_plus_reasoning",
        "feature_names": ["fls_count", "has_reasoning"],
        "means": [0.0, 0.0],
        "standard_deviations": [1.0, 1.0],
        "coefficients": [0.05, 0.10],
        "intercept": 0.45,
        "clip_bounds": [0.05, 0.95],
        "calibration": {
            "method": "empirical_oos_midrank_cdf",
            "version": "calibration_v1",
            "source": "2026Q1 validation predictions from elastic_net fitted on 2025Q4 only (ablation=fls_plus_reasoning)",
            "n_fitted": 5,
            "n_knots": 5,
            "bounds": [0.01, 0.99],
            "knots": [0.30, 0.40, 0.50, 0.60, 0.70],
        },
        "promoted": promoted,
        "operator_override": operator_override,
        "production_candidate": True,
        "structured_only": False,
        "training_metadata": {"normal_promotion_gate_passed": False},
    }


def _write(tmp_path, payload: dict):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _vector(*, fls_count: float, has_reasoning: float) -> FeatureVectorV3:
    values = {name: 0.0 for name in MODEL_FEATURE_NAMES_V3}
    values["fls_count"] = fls_count
    values["has_reasoning"] = has_reasoning
    return FeatureVectorV3(values=values, fls=extract_forward_looking_features([]))


def test_candidate_keeps_normal_promotion_false_and_uses_operator_metadata(tmp_path):
    model = V3LiteCandidateModel(_write(tmp_path, _artifact()))
    assert model.promoted is False
    assert model.operator_override is True
    assert model.production_candidate is True


def test_candidate_rejects_fake_promoted_artifact(tmp_path):
    with pytest.raises(ValueError, match="must not claim promoted"):
        V3LiteCandidateModel(_write(tmp_path, _artifact(promoted=True)))


def test_candidate_rejects_missing_operator_override(tmp_path):
    with pytest.raises(ValueError, match="operator-override"):
        V3LiteCandidateModel(_write(tmp_path, _artifact(operator_override=False)))


def test_candidate_reasoning_changes_live_score(tmp_path):
    model = V3LiteCandidateModel(_write(tmp_path, _artifact()))
    neutral = model.predict_vector(_vector(fls_count=0.0, has_reasoning=0.0))
    reasoned = model.predict_vector(_vector(fls_count=0.0, has_reasoning=1.0))
    assert reasoned > neutral
    assert 0.0 <= neutral <= 1.0
    assert 0.0 <= reasoned <= 1.0


def test_default_model_prefers_candidate_when_artifact_exists(tmp_path, monkeypatch):
    import explaining_markets.model_v3_lite as runtime

    path = _write(tmp_path, _artifact())
    monkeypatch.setattr(runtime, "DEFAULT_V3_LITE_CANDIDATE_PATH", path)
    monkeypatch.setenv("PRODUCTION_MODEL", "v3_lite_candidate")
    model = get_default_model()
    assert isinstance(model, V3LiteCandidateModel)


def test_v1_environment_switch_is_immediate_rollback(tmp_path, monkeypatch):
    import explaining_markets.model_v3_lite as runtime

    path = _write(tmp_path, _artifact())
    monkeypatch.setattr(runtime, "DEFAULT_V3_LITE_CANDIDATE_PATH", path)
    monkeypatch.setenv("PRODUCTION_MODEL", "v1")
    model = get_default_model()
    assert isinstance(model, ForwardLookingRidgeModel)
