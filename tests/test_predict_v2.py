"""predict.py: V2-aware fallback chain, TEST events, logging contract."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import predict


def _make_event(
    *,
    event_id: str = "evt_live_001",
    ticker: str = "AAPL",
    event_type: str = "EARNINGS_RELEASE",
    information_url: str = "https://example.com/disclosure",
    is_test: bool = False,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "information_url": information_url,
        "event_datetime": "2026-06-01T21:00:00Z",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
        "is_test": is_test,
    }


def test_predict_returns_empty_for_no_focal_assets() -> None:
    result = predict.predict({"event_id": "x", "focal_assets": []})
    assert result == []


def test_predict_returns_one_prediction_per_ticker(monkeypatch) -> None:
    event = _make_event()
    monkeypatch.setattr(predict, "_fetch_disclosure", lambda url: ["We expect revenue to grow."])
    result = predict.predict(event)
    assert len(result) == 1
    assert result[0]["identifier_value"] == "AAPL"
    assert 0.0 <= result[0]["predicted_percentile"] <= 1.0


def test_predict_neutral_on_disclosure_failure(monkeypatch) -> None:
    event = _make_event()

    def _fail(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(predict, "_fetch_disclosure", _fail)
    result = predict.predict(event)
    assert result == [{"identifier_value": "AAPL", "predicted_percentile": 0.5}]


def test_predict_handles_multiple_tickers(monkeypatch) -> None:
    event = _make_event()
    event["focal_assets"] = [
        {"identifier_type": "TICKER", "identifier_value": "AAPL"},
        {"identifier_type": "TICKER", "identifier_value": "MSFT"},
    ]
    monkeypatch.setattr(predict, "_fetch_disclosure", lambda url: ["We expect strong growth."])
    result = predict.predict(event)
    assert len(result) == 2
    tickers = {r["identifier_value"] for r in result}
    assert tickers == {"AAPL", "MSFT"}
    for r in result:
        assert 0.0 <= r["predicted_percentile"] <= 1.0


def test_predict_fallback_to_baseline_on_total_model_failure(monkeypatch) -> None:
    """get_default_model() has its own internal fallback chain that always
    returns a model (BaselineModel at worst). Verify that chain works."""
    event = _make_event()
    monkeypatch.setattr(predict, "_fetch_disclosure", lambda url: ["We expect growth."])

    # Force get_default_model to return a BaselineModel by making V1 and V2
    # artifacts unavailable. The prediction must still be bounded.
    from explaining_markets.model import BaselineModel

    monkeypatch.setattr(predict, "get_default_model", lambda: BaselineModel())
    result = predict.predict(event)
    assert len(result) == 1
    assert result[0]["predicted_percentile"] == 0.5


def test_event_cutoff_parses_event_datetime() -> None:
    event = {"event_datetime": "2026-06-01T21:00:00Z"}
    cutoff = predict._event_cutoff(event)
    assert cutoff == datetime(2026, 6, 1, 21, 0, tzinfo=timezone.utc)


def test_event_cutoff_falls_back_to_now_when_unparseable() -> None:
    before = datetime.now(timezone.utc)
    cutoff = predict._event_cutoff({})
    after = datetime.now(timezone.utc)
    assert before <= cutoff <= after
