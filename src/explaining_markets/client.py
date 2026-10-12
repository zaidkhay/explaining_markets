"""Tiny client for submitting predictions to the competition API.

One call: ``POST {EM_API_BASE_URL}/predictions`` authenticated with the
``X-API-Key`` header. The API always accepts well-formed predictions with 201;
late, duplicate, and pre-broadcast submissions are tagged for scoring rather than
rejected. Only your first submission per event is scored — a re-POST is accepted
but does not overwrite it.
"""

from __future__ import annotations

import time

import httpx

from explaining_markets.config import Config


class PredictionSubmissionError(Exception):
    """Raised when the API rejects a prediction submission."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# 429 and 5xx are the API or its edge hiccuping; any other 4xx is a request the
# API will keep rejecting, so retrying it only burns the prediction window.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def submit_predictions(
    *,
    event_id: str,
    predictions: list[dict],
    config: Config | None = None,
    timeout: float = 15.0,
) -> dict:
    """POST predictions for one event. Returns the parsed response body.

    ``predictions`` is a list of
    ``{"identifier_value": str, "predicted_percentile": float}``.
    """
    cfg = config or Config.from_env()
    url = f"{cfg.api_base_url}/predictions"
    body = {"event_id": event_id, "predictions": predictions}

    resp = httpx.post(
        url,
        json=body,
        headers={"X-API-Key": cfg.api_key, "Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code >= 300:
        raise PredictionSubmissionError(
            f"prediction submission failed: {resp.status_code} {resp.text}",
            status_code=resp.status_code,
        )
    return resp.json()


def is_transient_submit_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):  # timeouts, connect/read errors
        return True
    return isinstance(exc, PredictionSubmissionError) and exc.status_code in _TRANSIENT_STATUS


def submit_predictions_with_retry(
    *,
    event_id: str,
    predictions: list[dict],
    config: Config | None = None,
    attempts: int = 3,
    backoff_seconds: tuple[float, ...] = (1.0, 3.0),
    sleep=None,
) -> dict:
    """``submit_predictions`` retried on transient failures only.

    A re-POST is safe: only the first submission per event is scored, so a
    retry after a lost response cannot overwrite or double-count anything.
    Worst case is ``attempts`` x 15s timeout plus backoff -- well inside the
    5-minute prediction window.
    """
    for attempt in range(1, attempts + 1):
        try:
            return submit_predictions(event_id=event_id, predictions=predictions, config=config)
        except Exception as exc:
            if attempt == attempts or not is_transient_submit_error(exc):
                raise
            delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
            print(
                f"[SUBMIT_RETRY] event={event_id} attempt={attempt}/{attempts} "
                f"error={type(exc).__name__} retry_in={delay:.0f}s"
            )
            (sleep or time.sleep)(delay)
    raise AssertionError("unreachable")
