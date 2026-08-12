from __future__ import annotations

import json
import math

import explaining_markets.model as model_module
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES, extract_forward_looking_features
from explaining_markets.model import ForwardLookingRidgeModel, HeuristicFactModel, get_default_model


def _artifact(path, coefficients=None) -> None:
    n = len(MODEL_FEATURE_NAMES)
    coef = [0.0] * n if coefficients is None else list(coefficients)
    payload = {
        "model_version": "fls_ridge_v1",
        "feature_names": list(MODEL_FEATURE_NAMES),
        "means": [0.0] * n,
        "standard_deviations": [1.0] * n,
        "coefficients": coef,
        "intercept": 0.5,
        "selected_alpha": 1.0,
        "clip_bounds": [0.05, 0.95],
        "training_metadata": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_artifact_feature_order_exactly_matches_extractor(tmp_path) -> None:
    path = tmp_path / "model.json"
    _artifact(path)
    model = ForwardLookingRidgeModel(path)
    assert model.feature_names == MODEL_FEATURE_NAMES
    f = extract_forward_looking_features(["We expect EPS of $2.50."])
    assert len(f.vector(model.feature_names)) == len(MODEL_FEATURE_NAMES)


def test_standardization_is_deterministic(tmp_path) -> None:
    path = tmp_path / "model.json"
    _artifact(path)
    model = ForwardLookingRidgeModel(path)
    f = extract_forward_looking_features(["We expect revenue growth of 12%."])
    assert model.standardize(f) == model.standardize(f)


def test_model_returns_finite_bounded_prediction(tmp_path) -> None:
    path = tmp_path / "model.json"
    coef = [1000.0] * len(MODEL_FEATURE_NAMES)
    _artifact(path, coef)
    model = ForwardLookingRidgeModel(path)
    p = model.predict_disclosure(["We expect EPS of $4.00 and stronger growth of 20%."])
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
    assert 0.05 <= p <= 0.95


def test_positive_and_negative_disclosures_are_differentiated(tmp_path) -> None:
    path = tmp_path / "model.json"
    coef = [0.0] * len(MODEL_FEATURE_NAMES)
    coef[MODEL_FEATURE_NAMES.index("signed_forward_tone")] = 0.2
    coef[MODEL_FEATURE_NAMES.index("guidance_direction")] = 0.1
    _artifact(path, coef)
    model = ForwardLookingRidgeModel(path)
    positive = model.predict_disclosure(["We raised guidance and expect stronger profit growth."])
    negative = model.predict_disclosure(["We lowered guidance and expect weaker profit and demand."])
    assert positive > negative


def test_invalid_artifact_falls_back_to_heuristic(monkeypatch, tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(model_module, "DEFAULT_ARTIFACT_PATH", broken)
    assert isinstance(get_default_model(), HeuristicFactModel)


def test_mismatched_feature_order_is_rejected(tmp_path) -> None:
    path = tmp_path / "model.json"
    _artifact(path)
    payload = json.loads(path.read_text())
    payload["feature_names"] = list(reversed(payload["feature_names"]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        ForwardLookingRidgeModel(path)
    except ValueError as exc:
        assert "feature order" in str(exc)
    else:
        raise AssertionError("mismatched artifact feature order was accepted")
