"""Live prediction entry point for the production V3-lite system."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from explaining_markets.features import extract_features
from explaining_markets.model import BaselineModel, ForwardLookingRidgeModel, HeuristicFactModel, get_default_model

_FETCH_TIMEOUT_SECONDS = 20.0
_production_calibrator_cache: list = []


def _production_calibrator(model_version: str):
    if not _production_calibrator_cache:
        try:
            from explaining_markets.production_runtime import load_production_calibrator

            _production_calibrator_cache.append(load_production_calibrator(expected_model_version=model_version))
        except Exception as exc:
            print(f"[PROD_CALIBRATION] load failed error={type(exc).__name__}; using raw V1")
            _production_calibrator_cache.append(None)
    return _production_calibrator_cache[0]


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


def _log_live_input(event: dict, tickers: list[str], disclosure: list[str], cutoff: datetime) -> None:
    payload = {
        "event_id": str(event.get("event_id") or event.get("id") or "unknown"),
        "event_type": str(event.get("event_type") or "UNKNOWN"),
        "event_datetime": event.get("event_datetime"),
        "cutoff": cutoff.isoformat(),
        "tickers": tickers,
        "information_url": event.get("information_url"),
        "disclosure_fact_count": len(disclosure),
        "disclosure": [str(item) for item in disclosure],
    }
    print("[LIVE_INPUT] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


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
    _log_live_input(event, tickers, disclosure, cutoff)
    out = []
    for ticker in tickers:
        prediction = _predict_one(
            model=model,
            ticker=ticker,
            event_type=str(event.get("event_type") or "UNKNOWN"),
            disclosure=disclosure,
            cutoff=cutoff,
            event=event,
        )
        out.append({"identifier_value": ticker, "predicted_percentile": float(prediction)})
    return out


def _predict_one(*, model, ticker: str, event_type: str, disclosure: list[str], cutoff=None, event=None) -> float:
    try:
        from explaining_markets.model_v3_lite import V3LiteCandidateModel
    except Exception:
        V3LiteCandidateModel = ()  # type: ignore[assignment]

    if V3LiteCandidateModel and isinstance(model, V3LiteCandidateModel):
        try:
            from explaining_markets.evidence_bundle import persist_evidence_bundle
            from explaining_markets.features_v3 import build_feature_vector_v3, family_availability
            from explaining_markets.live_v3_context import build_live_v3_context, feed_diagnostics
            from explaining_markets.point_in_time_audit_v3 import audit_context
            from explaining_markets.prediction_diagnostics import build_prediction_diagnostics
            from explaining_markets.providers.live_context import default_provider_bundle_from_env
            from explaining_markets.v3_providers import V3ProviderBundle

            actual_cutoff = cutoff or datetime.now(timezone.utc)
            live_event = dict(event or {})
            live_event["disclosure"] = list(disclosure)
            providers = default_provider_bundle_from_env() if live_event.get("information_url") else V3ProviderBundle.null()
            context = build_live_v3_context(ticker=ticker, event=live_event, cutoff=actual_cutoff, providers=providers)
            audit = audit_context(context)
            vector = build_feature_vector_v3(disclosure=disclosure, context=context)
            prediction = model.predict_vector(vector)
            feed = feed_diagnostics(context)
            unavailable = ",".join(name for name, value in family_availability(vector).items() if not value) or "none"
            print(
                "[V3_FEED] "
                f"ticker={ticker} cutoff={actual_cutoff.isoformat()} "
                f"earnings_received={feed['earnings_received']} revenue_received={feed['revenue_received']} "
                f"guidance_received={feed['guidance_received']} price_rows={feed['price_rows']} "
                f"peer_count={feed['peer_count']} company_news_count={feed['company_news_count']} "
                f"peer_news_count={feed['peer_news_count']} sector_news_count={feed['sector_news_count']} "
                f"reasoned_news_count={feed['reasoned_news_count']} cutoff_audit=PASS records_checked={audit.records_checked}"
            )
            reasoning = context.event_reasoning
            if reasoning is not None:
                print(
                    "[V3_REASONING] "
                    f"ticker={ticker} earnings_quality={reasoning.earnings_quality:.3f} "
                    f"revenue_quality={reasoning.revenue_quality:.3f} guidance_quality={reasoning.guidance_quality:.3f} "
                    f"overall_event_signal={reasoning.overall_event_signal:.3f} "
                    f"materiality={reasoning.materiality:.3f} confidence={reasoning.confidence:.3f}"
                )
            print(
                f"[V3_PREDICT] ticker={ticker} model={model.model_version} "
                f"prediction={prediction:.4f} fallback=none unavailable_families={unavailable}"
            )
            try:
                diagnostics = build_prediction_diagnostics(
                    model=model,
                    vector=vector,
                    disclosure=disclosure,
                    context=context,
                )
                persist_evidence_bundle(
                    context=context,
                    vector=vector,
                    event_id=str(live_event.get("event_id") or live_event.get("id") or "unknown"),
                    model_version=model.model_version,
                    prediction=prediction,
                    raw_prediction=model.last_raw_prediction,
                    disclosure=disclosure,
                    prediction_diagnostics=diagnostics,
                )
            except Exception as evidence_exc:
                print(f"[V3_EVIDENCE] ticker={ticker} status=error error={type(evidence_exc).__name__}")
            return _bounded(prediction)
        except Exception as exc:
            print(f"[V3_PREDICT] ticker={ticker} fallback=fls_ridge_v1 error={type(exc).__name__}")
            try:
                model = ForwardLookingRidgeModel()
            except Exception as v1_exc:
                print(f"[PREDICT] ticker={ticker} emergency V1 load failed: {type(v1_exc).__name__}")
                model = HeuristicFactModel()

    if isinstance(model, ForwardLookingRidgeModel):
        try:
            raw_prediction, fls = model.predict_with_features(disclosure)
            final_prediction = float(raw_prediction)
            calibration_status = "raw"
            calibrator = _production_calibrator(model.model_version)
            if calibrator is not None:
                final_prediction = calibrator.calibrate(raw_prediction)
                calibration_status = calibrator.version
            print(
                "[V1_ROLLBACK] "
                f"ticker={ticker} raw={raw_prediction:.4f} submitted={final_prediction:.4f} "
                f"fls_ratio={fls.values['fls_ratio']:.3f} calibration={calibration_status}"
            )
            return _bounded(final_prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} emergency V1 failed: {type(exc).__name__}; using heuristic")
            model = HeuristicFactModel()

    if isinstance(model, HeuristicFactModel):
        try:
            features = extract_features(ticker=ticker, event_type=event_type, disclosure=disclosure)
            prediction = model.predict_percentile(features)
            print(f"[PREDICT] ticker={ticker} model=heuristic_fact prediction={prediction:.4f}")
            return _bounded(prediction)
        except Exception as exc:
            print(f"[PREDICT] ticker={ticker} heuristic failed: {type(exc).__name__}; using baseline")

    return BaselineModel().predict_percentile(
        extract_features(ticker=ticker, event_type=event_type, disclosure=[])
    )


def _fetch_disclosure(information_url: str | None) -> list[str]:
    if not information_url:
        return []
    response = httpx.get(information_url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return [str(payload)]

    def facts_from_items(items) -> list[str]:
        if not isinstance(items, list):
            return []
        for item in items:
            if not isinstance(item, dict) or item.get("kind") != "facts":
                continue
            content = item.get("content")
            if isinstance(content, list):
                return [str(x) for x in content if str(x).strip()]
            if isinstance(content, str) and content.strip():
                return [content]
        return []

    facts = facts_from_items(payload.get("items"))
    if facts:
        return facts
    disclosure = payload.get("disclosure") or {}
    if isinstance(disclosure, dict):
        facts = facts_from_items(disclosure.get("items"))
        if facts:
            return facts
    facts = payload.get("facts")
    if isinstance(facts, list):
        return [str(x) for x in facts if str(x).strip()]
    if isinstance(facts, str) and facts.strip():
        return [facts]
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return [summary]
    if isinstance(summary, list):
        return [str(x) for x in summary if str(x).strip()]
    return []


def _bounded(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _neutral(tickers: list[str]) -> list[dict]:
    return [{"identifier_value": ticker, "predicted_percentile": 0.5} for ticker in tickers]
