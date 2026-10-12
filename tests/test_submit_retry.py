"""Submission retries: transient failures are retried, permanent ones are not."""

from __future__ import annotations

import httpx
import pytest

import explaining_markets.client as client_mod
from explaining_markets.client import (
    PredictionSubmissionError,
    is_transient_submit_error,
    submit_predictions_with_retry,
)


def _run(monkeypatch, outcomes, attempts=3):
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_submit(*, event_id, predictions, config):
        calls.append(event_id)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(client_mod, "submit_predictions", fake_submit)
    result = None
    error = None
    try:
        result = submit_predictions_with_retry(
            event_id="ev", predictions=[], config=None, attempts=attempts, sleep=sleeps.append
        )
    except Exception as exc:  # noqa: BLE001
        error = exc
    return result, error, calls, sleeps


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
        PredictionSubmissionError("busy", status_code=503),
        PredictionSubmissionError("slow down", status_code=429),
    ],
)
def test_transient_failure_then_success_is_retried(monkeypatch, exc) -> None:
    result, error, calls, sleeps = _run(monkeypatch, [exc, {"ok": True}])
    assert error is None and result == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_rejection_is_not_retried(monkeypatch, status) -> None:
    exc = PredictionSubmissionError("nope", status_code=status)
    _, error, calls, sleeps = _run(monkeypatch, [exc, {"ok": True}])
    assert error is exc
    assert len(calls) == 1 and sleeps == []


def test_gives_up_after_the_last_attempt(monkeypatch) -> None:
    exc = httpx.ReadTimeout("slow")
    _, error, calls, sleeps = _run(monkeypatch, [exc, exc, exc])
    assert error is exc
    assert len(calls) == 3
    assert sleeps == [1.0, 3.0]


def test_unexpected_errors_are_not_transient() -> None:
    assert not is_transient_submit_error(RuntimeError("bug"))
    assert not is_transient_submit_error(PredictionSubmissionError("x"))
