"""★ THIS IS THE ONLY FILE YOU NEED TO EDIT. ★

`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

The current strategy is the MVP prediction pipeline in
`src/explaining_markets/{features,model,historical,backtest}.py`:

    fetch disclosure -> extract_features() -> model.predict_percentile()

`features.py` builds a small, transparent feature set from the event's own
disclosure text (never from a realized outcome — see that module's docstring),
and `model.py` maps those features to a percentile via a simple, fully
auditable rule (`HeuristicFactModel`), swappable for a trained model later
without touching this file. See `backtest.py` for how to evaluate a model
offline against `data/historical/` before changing `model.get_default_model`.

Nothing here can fail the whole prediction: a missing/broken disclosure fetch,
or any model error, degrades to a deterministic 0.5 baseline rather than
raising, so the deploy -> receive -> submit round trip always completes.
"""

from __future__ import annotations

import httpx

from explaining_markets.features import extract_features
from explaining_markets.model import PercentileModel, get_default_model

# Sized against the 5-minute prediction window that opens when your handler
# ACKs the webhook. This pipeline is a single, fast local computation (no LLM
# call), so the only network I/O is this one disclosure fetch.
SUMMARY_TIMEOUT_SECONDS = 15.0

_NEUTRAL_BASELINE = 0.5


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL with the event summary JSON
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — where you predict the asset's
    next-day abnormal (market-adjusted) return will rank across all of the
    quarter's event outcomes: 0 = the quarter's most negative reaction,
    0.50 = median, 1 = its most positive. It's a cross-sectional rank across the
    quarter's events, not a percentile within the asset's own history.
    """
    model = get_default_model()
    disclosure = _fetch_disclosure(event.get("information_url"))
    event_type = event.get("event_type", "UNKNOWN")

    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _predict_one(
                model=model,
                ticker=asset["identifier_value"],
                event_type=event_type,
                disclosure=disclosure,
            ),
        }
        for asset in event.get("focal_assets", [])
    ]


def _fetch_disclosure(information_url: str | None) -> list[str]:
    """Best-effort fetch of the event's disclosure/summary facts.

    Never raises: any missing URL, network error, or unexpected payload shape
    returns `[]` — a neutral, empty disclosure — rather than failing `predict()`.
    The model layer treats an empty disclosure as "no signal" (see `model.py`),
    which is exactly the deterministic-baseline behavior required when
    upstream data is unavailable.
    """
    if not information_url:
        return []
    try:
        resp = httpx.get(information_url, timeout=SUMMARY_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"[WARN] failed to fetch information_url ({exc}); using neutral baseline")
        return []
    return _facts_from_payload(payload)


def _facts_from_payload(payload: object) -> list[str]:
    """Extract fact/summary sentences from an `information_url` response body.

    Handles the two shapes documented across the starter and the disclosure
    schema: a `{"disclosure": {"items": [...]}}` bundle (kind="facts"), a bare
    `{"facts": [...]}` list, or a `{"summary": "..."}` string. Anything else
    yields `[]`.
    """
    if not isinstance(payload, dict):
        return []
    items = (payload.get("disclosure") or {}).get("items") or []
    for item in items:
        if isinstance(item, dict) and item.get("kind") == "facts":
            return [str(f) for f in (item.get("content") or [])]
    facts = payload.get("facts")
    if isinstance(facts, list):
        return [str(f) for f in facts]
    summary = payload.get("summary")
    if isinstance(summary, str) and summary:
        return [summary]
    return []


def _predict_one(
    *, model: PercentileModel, ticker: str, event_type: str, disclosure: list[str]
) -> float:
    """Run the model for one asset; never raises.

    Any feature-extraction or model failure falls back to the deterministic
    0.5 baseline, so a bug in a future (e.g. trained) model can never take
    down the whole prediction — the competition scores nothing worse than a
    neutral, unscored-looking guess for that asset.
    """
    try:
        features = extract_features(ticker=ticker, event_type=event_type, disclosure=disclosure)
        percentile = float(model.predict_percentile(features))
    except Exception as exc:  # noqa: BLE001 - model must never crash the pipeline
        print(f"[WARN] model prediction failed for {ticker} ({exc}); using 0.5 baseline")
        return _NEUTRAL_BASELINE
    return max(0.0, min(1.0, percentile))
