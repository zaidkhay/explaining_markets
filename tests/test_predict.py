"""predict() shape and fallback behavior for the MVP prediction pipeline.

`predict()` fetches the event's disclosure, extracts features, and asks the
default model (`explaining_markets.model.get_default_model()`) for a
percentile per focal asset. Every test here is fully offline (the disclosure
fetch is stubbed) and exercises the "must never fail" contract: a neutral
disclosure, a broken fetch, or a broken model must all still return a
well-formed prediction rather than raising.
"""

from __future__ import annotations

import httpx
import pytest

import predict as predict_module


SAMPLE_EVENT = {
    "id": "evt_test_1",
    "event_id": "11111111-1111-1111-1111-111111111111",
    "event_type": "EARNINGS_RELEASE",
    "timing_category": "SCHEDULED",
    "event_datetime": "2026-01-15T21:00:00Z",
    "focal_assets": [
        {"identifier_type": "TICKER", "identifier_value": "AAPL"},
        {"identifier_type": "TICKER", "identifier_value": "MSFT"},
    ],
    "information_url": "https://example.test/disclosure",
    "prediction_deadline": "2026-01-15T21:05:00Z",
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - stub
        return None

    def json(self) -> dict:
        return self._payload


def _assert_well_formed(preds, expected_tickers) -> None:
    assert isinstance(preds, list)
    assert len(preds) == len(expected_tickers)
    returned = {p["identifier_value"] for p in preds}
    assert returned == set(expected_tickers)
    for p in preds:
        assert set(p) == {"identifier_value", "predicted_percentile"}
        assert isinstance(p["predicted_percentile"], float)
        assert 0.0 <= p["predicted_percentile"] <= 1.0


def test_predict_neutral_disclosure_yields_baseline_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        predict_module.httpx,
        "get",
        lambda *a, **k: _FakeResponse({"summary": "Quarterly results in line with expectations."}),
    )

    preds = predict_module.predict(SAMPLE_EVENT)

    _assert_well_formed(preds, ["AAPL", "MSFT"])
    for p in preds:
        assert p["predicted_percentile"] == 0.5  # neutral text -> no sentiment signal


def test_predict_positive_disclosure_raises_percentile_above_half(monkeypatch) -> None:
    monkeypatch.setattr(
        predict_module.httpx,
        "get",
        lambda *a, **k: _FakeResponse(
            {"summary": "Revenue beat expectations. Guidance raised for the full year."}
        ),
    )

    preds = predict_module.predict(SAMPLE_EVENT)

    _assert_well_formed(preds, ["AAPL", "MSFT"])
    for p in preds:
        assert p["predicted_percentile"] > 0.5


def test_predict_never_fails_when_information_url_fetch_raises(monkeypatch) -> None:
    def _raise(*a, **k):
        raise httpx.ConnectError("network is unreachable")

    monkeypatch.setattr(predict_module.httpx, "get", _raise)

    preds = predict_module.predict(SAMPLE_EVENT)

    _assert_well_formed(preds, ["AAPL", "MSFT"])
    for p in preds:
        assert p["predicted_percentile"] == 0.5  # deterministic baseline, not a crash


def test_predict_never_fails_when_information_url_is_missing(monkeypatch) -> None:
    event = {**SAMPLE_EVENT, "information_url": None}
    calls = []
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: calls.append(1))

    preds = predict_module.predict(event)

    _assert_well_formed(preds, ["AAPL", "MSFT"])
    assert calls == []  # no network call was even attempted


def test_predict_never_fails_when_model_raises(monkeypatch) -> None:
    class _BrokenModel:
        def predict_percentile(self, features):  # noqa: ANN001
            raise RuntimeError("boom")

    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: _FakeResponse({"summary": "x"}))
    monkeypatch.setattr(predict_module, "get_default_model", lambda: _BrokenModel())

    preds = predict_module.predict(SAMPLE_EVENT)

    _assert_well_formed(preds, ["AAPL", "MSFT"])
    for p in preds:
        assert p["predicted_percentile"] == 0.5


def test_predict_handles_no_focal_assets() -> None:
    event = {**SAMPLE_EVENT, "focal_assets": []}
    assert predict_module.predict(event) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"disclosure": {"items": [{"kind": "facts", "content": ["Revenue beat."]}]}},
        {"facts": ["Revenue beat."]},
        {"summary": "Revenue beat."},
    ],
)
def test_predict_accepts_every_documented_information_url_shape(monkeypatch, payload) -> None:
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: _FakeResponse(payload))
    preds = predict_module.predict(SAMPLE_EVENT)
    _assert_well_formed(preds, ["AAPL", "MSFT"])
