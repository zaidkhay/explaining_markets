"""Live prediction entry point.

Real events fetch the competition disclosure facts, run the paper-informed
ForwardLookingRidgeModel, and return one CAR1 percentile per focal ticker.
The Modal worker keeps synthetic TEST events on its existing neutral 0.50 path.
"""

from __future__ import annotations

from typing import Any

import httpx

from explaining_markets.features import extract_features
from explaining_markets.model import (
    BaselineModel,
    ForwardLookingRidgeModel,
    HeuristicFactModel,
    get_default_model,
)

_FETCH_TIMEOUT_SECONDS = 20.0


def predict(event: dict) -> list[dict]:
    tickers = [
        str(asset.get("identifier_value"))
        for asset in (event.get("focal_assets") or [])
        if asset.get("identifier_value")
    ]
    if not tickers:
        return []

    try:
        disclosure = _fetch_disclosure(event.get("information_url"))
    except Exception as exc:
        print(f"[PREDICT] disclosure fetch failed: {type(exc).__name__}; using neutral baseline")
        return _neutral(tickers)

    model = get_default_model()
    out: list[dict] = []
    for ticker in tickers:
        prediction = _predict_one(
            model=model,
            ticker=ticker,
            event_type=str(event.get("event_type") or "UNKNOWN"),
            disclosure=disclosure,
        )
        out.append({"identifier_value": ticker, "predicted_percentile": float(prediction)})
    return out


def _predict_one(*, model, ticker: str, event_type: str, disclosure: list[str]) -> float:
    # Healthy production path: paper-informed extraction + serialized Ridge.
    if isinstance(model, ForwardLookingRidgeModel):
        try:
            prediction, fls = model.predict_with_features(disclosure)
            values = fls.values
            print(
                "[PREDICT] "
                f"ticker={ticker} model={model.model_version} "
                f"fls_ratio={values['fls_ratio']:.3f} "
                f"quant_earnings_ratio={values['quant_earnings_fls_ratio']:.3f} "
                f"other_fls_ratio={values['other_fls_ratio']:.3f} "
                f"tone={values['signed_forward_tone']:.3f} "
                f"guidance={values['guidance_direction']:.0f} "
                f"prediction={prediction:.4f}"
            )
            return _bounded(prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} fls_ridge failed: {type(exc).__name__}; using heuristic")
            model = HeuristicFactModel()

    # First fallback: the pre-existing disclosure heuristic.
    if isinstance(model, HeuristicFactModel):
        try:
            features = extract_features(ticker=ticker, event_type=event_type, disclosure=disclosure)
            prediction = model.predict_percentile(features)
            print(
                f"[PREDICT] ticker={ticker} model=heuristic_fact "
                f"net_sentiment={features.net_sentiment} prediction={prediction:.4f}"
            )
            return _bounded(prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} heuristic failed: {type(exc).__name__}; using baseline")

    # Final fallback: deterministic 0.50. This path is only for unexpected
    # artifact/feature/model failures, not the healthy live default.
    baseline = BaselineModel()
    features = extract_features(ticker=ticker, event_type=event_type, disclosure=[])
    return baseline.predict_percentile(features)


def _fetch_disclosure(information_url: str | None) -> list[str]:
    if not information_url:
        return []
    response = httpx.get(information_url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return [str(payload)]

    disclosure = payload.get("disclosure") or {}
    if isinstance(disclosure, dict):
        for item in disclosure.get("items") or []:
            if isinstance(item, dict) and item.get("kind") == "facts":
                content = item.get("content") or []
                return [str(x) for x in content]

    facts = payload.get("facts")
    if isinstance(facts, list):
        return [str(x) for x in facts]
    if isinstance(facts, str):
        return [facts]

    summary = payload.get("summary")
    if isinstance(summary, str):
        return [summary]
    if isinstance(summary, list):
        return [str(x) for x in summary]

    # Last-resort textual fields only; never inspect realized outcome keys.
    forbidden = {"car1", "earnings_surprise", "event_returns", "baseline_predictions"}
    return [str(v) for k, v in payload.items() if k not in forbidden and isinstance(v, str)]


def _bounded(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _neutral(tickers: list[str]) -> list[dict]:
    return [{"identifier_value": ticker, "predicted_percentile": 0.5} for ticker in tickers]
