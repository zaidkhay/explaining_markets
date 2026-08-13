"""Live prediction entry point.

Real events fetch the competition disclosure facts and run the model chain
(promoted V2 with company history -> fls_ridge_v1 -> heuristic -> 0.50),
returning one CAR1 percentile per focal ticker. Every prediction logs which
model actually produced it — fallbacks are never silent. The Modal worker
keeps synthetic TEST events on its existing neutral 0.50 path.

Company history for V2 comes from the packaged archive snapshot
(millisecond load, no network); no bulk historical download ever happens
inside this call.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from explaining_markets.features import extract_features
from explaining_markets.model import (
    BaselineModel,
    CompanyHistoryRidgeModel,
    ForwardLookingRidgeModel,
    HeuristicFactModel,
    get_default_model,
)

_FETCH_TIMEOUT_SECONDS = 20.0

_history_provider_cache: list = []  # lazy singleton; [provider-or-None] once loaded


def _history_provider():
    """Load the packaged snapshot provider once; None when unavailable."""
    if not _history_provider_cache:
        try:
            from explaining_markets.competition_history import SnapshotCompanyHistoryProvider

            _history_provider_cache.append(SnapshotCompanyHistoryProvider())
        except Exception as exc:
            print(f"[PREDICT] history snapshot unavailable: {type(exc).__name__}")
            _history_provider_cache.append(None)
    return _history_provider_cache[0]


def _event_cutoff(event: dict) -> datetime:
    """The prediction knowledge cutoff for history eligibility.

    Uses the event's own timestamp when parseable (never later information),
    falling back to "now" — for a live event, now is always at/after the
    event, and the history snapshot only contains sealed pre-2026Q3 outcomes,
    so both choices are point-in-time safe.
    """
    raw = event.get("event_datetime")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)


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
    cutoff = _event_cutoff(event)
    out: list[dict] = []
    for ticker in tickers:
        prediction = _predict_one(
            model=model,
            ticker=ticker,
            event_type=str(event.get("event_type") or "UNKNOWN"),
            disclosure=disclosure,
            cutoff=cutoff,
        )
        out.append({"identifier_value": ticker, "predicted_percentile": float(prediction)})
    return out


def _predict_one(*, model, ticker: str, event_type: str, disclosure: list[str], cutoff=None) -> float:
    # Healthiest path: promoted V2 (FLS + company history from the packaged
    # snapshot). Falls back explicitly — never silently — to V1 on any error.
    if isinstance(model, CompanyHistoryRidgeModel):
        try:
            provider = _history_provider()
            if provider is not None and cutoff is not None:
                history = provider.history_before(ticker, cutoff)
                history_source = "cache"
            else:
                from explaining_markets.company_history import empty_company_history

                history = empty_company_history(ticker, cutoff or datetime.now(timezone.utc))
                history_source = "missing"
            prediction, vector = model.predict(disclosure=disclosure, history=history)
            values = vector.values
            print(
                "[PREDICT] "
                f"ticker={ticker} model={model.model_version} "
                f"history_source={history_source} "
                f"prior_earnings_count={values['prior_earnings_count']:.0f} "
                f"mean_prior_earnings_abnormal_return={values['mean_prior_earnings_abnormal_return']:.4f} "
                f"last_prior_competition_car1={values['last_prior_competition_car1']:.4f} "
                f"has_competition_history={values['has_competition_history']:.0f} "
                f"fls_ratio={values['fls_ratio']:.3f} "
                f"signed_forward_tone={values['signed_forward_tone']:.3f} "
                f"guidance_direction={values['guidance_direction']:.0f} "
                f"prediction={prediction:.4f}"
            )
            return _bounded(prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} v2 failed: {type(exc).__name__}; falling back to fls_ridge_v1")
            try:
                model = ForwardLookingRidgeModel()
            except Exception as v1_exc:
                print(f"[PREDICT] ticker={ticker} fls_ridge_v1 load failed: {type(v1_exc).__name__}")
                model = HeuristicFactModel()

    # Production path today: paper-informed extraction + serialized V1 Ridge.
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
