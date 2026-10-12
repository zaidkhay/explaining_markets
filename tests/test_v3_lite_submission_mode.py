"""Artifact `submission` mode: calibrated (default) vs raw_clipped."""

from __future__ import annotations

import json

import pytest

from explaining_markets.model_v3_lite import DEFAULT_V3_LITE_CANDIDATE_PATH, V3LiteCandidateModel
from explaining_markets.v3_lite_live_gate import evaluate_v3_lite_live_gate


def _model(tmp_path, **overrides) -> V3LiteCandidateModel:
    payload = json.loads(DEFAULT_V3_LITE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return V3LiteCandidateModel(path)


def test_default_is_calibrated_when_field_absent(tmp_path) -> None:
    payload = json.loads(DEFAULT_V3_LITE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    payload.pop("submission", None)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    model = V3LiteCandidateModel(path)
    assert model.submission == "calibrated"
    assert model.submission_value(0.5) == model.calibrator.calibrate(0.5)


def test_raw_clipped_submits_the_raw_score(tmp_path) -> None:
    model = _model(tmp_path, submission="raw_clipped")
    for raw in (0.05, 0.31, 0.5, 0.95):
        assert model.submission_value(raw) == raw


def test_unknown_mode_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="submission mode"):
        _model(tmp_path, submission="shrunk")


def test_gate_is_equally_strict_in_both_modes(tmp_path) -> None:
    """Separation is judged in rank space, so switching what is submitted must
    not loosen (or tighten) the gate."""
    calibrated = evaluate_v3_lite_live_gate(_model(tmp_path, submission="calibrated"))
    raw = evaluate_v3_lite_live_gate(_model(tmp_path, submission="raw_clipped"))
    assert calibrated.passed and raw.passed
    assert raw.negative_neutral_gap == calibrated.negative_neutral_gap
    assert raw.neutral_positive_gap == calibrated.neutral_positive_gap
    assert raw.submitted_spread == calibrated.submitted_spread
    for scenario in raw.scenarios:
        assert scenario.submitted == scenario.raw
    for scenario in calibrated.scenarios:
        assert scenario.submitted == scenario.rank


def test_gate_still_fails_a_constant_model(tmp_path, monkeypatch) -> None:
    model = _model(tmp_path, submission="raw_clipped")
    monkeypatch.setattr(model, "predict_raw_vector", lambda vector: 0.4946)
    assert not evaluate_v3_lite_live_gate(model).passed
