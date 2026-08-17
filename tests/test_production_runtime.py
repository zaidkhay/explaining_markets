"""Production V1 calibration, explanation, and live wiring tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import predict as predict_module
from explaining_markets.model import ForwardLookingRidgeModel
from explaining_markets.production_runtime import (
    build_v1_explanation,
    load_production_calibrator,
    persist_production_explanation,
    production_scenario_report,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_default_production_calibrator_is_enabled_and_monotonic(monkeypatch) -> None:
    monkeypatch.delenv("PRODUCTION_CALIBRATION_ENABLED", raising=False)
    cal = load_production_calibrator()
    assert cal is not None
    assert cal.model_version == "fls_ridge_v1"
    assert cal.method == "affine_oos_scale_v1"
    assert cal.scale > 1.0
    assert cal.calibrate(0.49) < cal.calibrate(0.50) < cal.calibrate(0.51)
    assert cal.calibrate(0.50) == pytest.approx(0.50)
    assert abs(cal.calibrate(0.51) - 0.50) > abs(0.51 - 0.50)


def test_production_calibration_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("PRODUCTION_CALIBRATION_ENABLED", "0")
    assert load_production_calibrator() is None


def test_production_scenarios_are_ordered_and_differentiated(monkeypatch) -> None:
    monkeypatch.delenv("PRODUCTION_CALIBRATION_ENABLED", raising=False)
    report = production_scenario_report()
    assert report["model_version"] == "fls_ridge_v1"
    assert report["calibration_loaded"] is True
    assert report["ordered"] is True
    assert report["meaningfully_differentiated"] is True
    assert report["spread"] >= 0.10
    negative = report["scenarios"]["negative"]["final"]
    neutral = report["scenarios"]["neutral"]["final"]
    positive = report["scenarios"]["positive"]["final"]
    assert negative < neutral < positive


def test_explanation_packet_persists_write_once(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PRODUCTION_CALIBRATION_ENABLED", raising=False)
    model = ForwardLookingRidgeModel()
    raw, features = model.predict_with_features(
        ["We raised full-year guidance and expect earnings growth of 20%."]
    )
    cal = load_production_calibrator(expected_model_version=model.model_version)
    packet = build_v1_explanation(
        model=model,
        features=features,
        ticker="AAPL",
        raw_prediction=raw,
        calibrator=cal,
        disclosure_available=True,
    )
    cutoff = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)
    path = persist_production_explanation(
        packet=packet,
        event_id="evt-1",
        cutoff=cutoff,
        information_url_fetch_success=True,
        directory=tmp_path,
    )
    first = path.read_text(encoding="utf-8")
    persist_production_explanation(
        packet=packet,
        event_id="evt-1",
        cutoff=cutoff,
        information_url_fetch_success=False,
        directory=tmp_path,
    )
    assert path.read_text(encoding="utf-8") == first
    payload = json.loads(first)
    assert payload["submitted_percentile"] == pytest.approx(packet.calibrated_percentile)
    assert payload["explanation"]["raw_prediction"] == pytest.approx(raw)
    assert payload["explanation"]["key_drivers"]


def test_live_predict_submits_calibrated_v1_score(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PRODUCTION_CALIBRATION_ENABLED", raising=False)
    monkeypatch.setenv("V3_EVIDENCE_DIR", str(tmp_path))
    predict_module._production_calibrator_cache.clear()
    disclosure = "We raised full-year guidance and expect earnings growth of 20%."
    monkeypatch.setattr(
        predict_module.httpx,
        "get",
        lambda *a, **k: _FakeResponse({"summary": disclosure}),
    )
    event = {
        "id": "evt-prod",
        "event_id": "evt-prod",
        "event_type": "EARNINGS_RELEASE",
        "event_datetime": "2026-08-16T20:00:00Z",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "AAPL"}],
        "information_url": "https://example.test/disclosure",
    }
    raw = ForwardLookingRidgeModel().predict_disclosure([disclosure])
    cal = load_production_calibrator()
    assert cal is not None
    expected = cal.calibrate(raw)
    result = predict_module.predict(event)
    assert result[0]["predicted_percentile"] == pytest.approx(expected)
    assert result[0]["predicted_percentile"] != pytest.approx(raw)
