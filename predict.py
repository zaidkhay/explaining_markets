"""Live prediction entry point.

Production selection is promotion-gated V3 -> fls_ridge_v1 -> heuristic ->
0.50. Every fallback is logged. Synthetic TEST events remain neutral in the
Modal worker before this function is called.
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
_history_provider_cache: list = []


def _history_provider():
    if not _history_provider_cache:
        try:
            from explaining_markets.competition_history import SnapshotCompanyHistoryProvider

            _history_provider_cache.append(SnapshotCompanyHistoryProvider())
        except Exception as exc:
            print(f"[PREDICT] history snapshot unavailable: {type(exc).__name__}")
            _history_provider_cache.append(None)
    return _history_provider_cache[0]


def _event_cutoff(event: dict) -> datetime:
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

    fetch_success = True
    try:
        disclosure = _fetch_disclosure(event.get("information_url"))
    except Exception as exc:
        fetch_success = False
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
            information_url_fetch_success=fetch_success,
        )
        out.append({"identifier_value": ticker, "predicted_percentile": float(prediction)})
    return out


def _predict_one(
    *, model, ticker: str, event_type: str, disclosure: list[str], cutoff=None,
    information_url_fetch_success: bool = True,
) -> float:
    try:
        from explaining_markets.model_v3 import MultiSignalV3Model
    except Exception:
        MultiSignalV3Model = ()  # type: ignore[assignment]

    if MultiSignalV3Model and isinstance(model, MultiSignalV3Model):
        try:
            from explaining_markets.cached_v3_context import context_from_existing_cache
            from explaining_markets.features_v3 import build_feature_vector_v3, family_availability
            from explaining_markets.point_in_time_audit_v3 import audit_context

            actual_cutoff = cutoff or datetime.now(timezone.utc)
            context = context_from_existing_cache(ticker, actual_cutoff)
            audit_context(context)
            vector = build_feature_vector_v3(disclosure=disclosure, context=context)
            prediction = model.predict_vector(vector)
            availability = family_availability(vector)
            unavailable = ",".join(name for name, value in availability.items() if not value) or "none"
            print(
                "[PREDICT] "
                f"ticker={ticker} model={model.model_version} "
                f"unavailable_families={unavailable} "
                f"eps_surprise_pct={vector.values['eps_surprise_percent']:.4f} "
                f"revenue_surprise_pct={vector.values['revenue_surprise_percent']:.4f} "
                f"guidance_vs_consensus={vector.values['guidance_surprise_percent']:.4f} "
                f"prior_earnings_count={vector.values['prior_earnings_count']:.0f} "
                f"return_20d={vector.values['return_20d']:.4f} "
                f"prediction={prediction:.4f}"
            )
            return _bounded(prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} V3 failed: {type(exc).__name__}; falling back to fls_ridge_v1")
            try:
                model = ForwardLookingRidgeModel()
            except Exception as v1_exc:
                print(f"[PREDICT] ticker={ticker} fls_ridge_v1 load failed: {type(v1_exc).__name__}")
                model = HeuristicFactModel()

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
                f"ticker={ticker} model={model.model_version} history_source={history_source} "
                f"prior_earnings_count={values['prior_earnings_count']:.0f} "
                f"fls_ratio={values['fls_ratio']:.3f} prediction={prediction:.4f}"
            )
            return _bounded(prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} v2 failed: {type(exc).__name__}; falling back to fls_ridge_v1")
            try:
                model = ForwardLookingRidgeModel()
            except Exception as v1_exc:
                print(f"[PREDICT] ticker={ticker} fls_ridge_v1 load failed: {type(v1_exc).__name__}")
                model = HeuristicFactModel()

    if isinstance(model, ForwardLookingRidgeModel):
        try:
            prediction, fls = model.predict_with_features(disclosure)
            values = fls.values
            vector = [float(values[name]) for name in model.feature_names]
            nonzero = sum(abs(x) > 1e-12 for x in vector)
            norm = sum(x * x for x in vector) ** 0.5
            print(
                "[PREDICT] "
                f"ticker={ticker} model={model.model_version} "
                f"disclosure_fact_count={len(disclosure)} "
                f"non_zero_fls_feature_count={nonzero} fls_vector_norm={norm:.4f} "
                f"information_url_fetch_success={int(information_url_fetch_success)} "
                f"empty_disclosure_flag={int(not disclosure)} "
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

    forbidden = {"car1", "earnings_surprise", "event_returns", "baseline_predictions"}
    return [str(v) for k, v in payload.items() if k not in forbidden and isinstance(v, str)]


def _bounded(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _neutral(tickers: list[str]) -> list[dict]:
    return [{"identifier_value": ticker, "predicted_percentile": 0.5} for ticker in tickers]
