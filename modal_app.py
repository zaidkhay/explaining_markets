"""Modal deployment for the Explaining Markets starter.

This is orchestration only: verify/ACK/dedupe/spawn, prediction, submission,
and non-public feed/production diagnostics. Business logic stays under
``src/explaining_markets``.

Deploy:      uv run modal deploy modal_app.py
Dev/local:   uv run modal serve modal_app.py
Feed check:  uv run modal run modal_app.py::check_v3_feed --ticker AAPL
Prod check:  uv run modal run modal_app.py::check_production
"""

import modal

app = modal.App("explaining-markets-starter")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "httpx", "openai", "pydantic")
    .add_local_python_source("explaining_markets", "predict", ignore=[])
)

seen_webhooks = modal.Dict.from_name("em-webhook-dedupe", create_if_missing=True)
v3_data = modal.Volume.from_name("em-v3-data", create_if_missing=True)
secrets = [modal.Secret.from_dotenv(".env")]


def _claim(webhook_id):
    if not webhook_id:
        return True
    return seen_webhooks.put(webhook_id, "in_flight", skip_if_exists=True)


async def _claim_aio(webhook_id):
    if not webhook_id:
        return True
    return await seen_webhooks.put.aio(webhook_id, "in_flight", skip_if_exists=True)


def _release(webhook_id, submitted):
    if not webhook_id:
        return
    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


@app.function(image=image, secrets=secrets, volumes={"/v3-data": v3_data}, timeout=600, retries=0)
def predict_and_submit(event: dict, webhook_id: str | None = None):
    import os

    os.environ.setdefault("V3_HISTORY_CACHE_PATH", "/v3-data/company_history.sqlite")
    os.environ.setdefault("V3_EVIDENCE_DIR", "/v3-data/evidence")

    from explaining_markets.client import submit_predictions
    from explaining_markets.config import Config
    from explaining_markets.event_utils import is_test, neutral_predictions
    from predict import predict

    submitted = False
    try:
        predictions = neutral_predictions(event) if is_test(event) else predict(event)
        submit_predictions(
            event_id=event["event_id"],
            predictions=predictions,
            config=Config.from_env(),
        )
        submitted = True
    except Exception as exc:
        print(f"[ERROR] prediction failed for event {event.get('event_id')}: {exc}")
    finally:
        _release(webhook_id, submitted)


@app.function(image=image, secrets=secrets, volumes={"/v3-data": v3_data}, timeout=120)
def check_v3_feed(ticker: str):
    """Non-public Modal diagnostic. Never prints credential values.

    Live providers are production-bounded. OpenRouter is disabled by default
    in the provider bundle unless V3_LIVE_USE_OPENROUTER=1 is explicitly set.
    """
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    os.environ.setdefault("V3_HISTORY_CACHE_PATH", "/v3-data/company_history.sqlite")
    os.environ.setdefault("V3_EVIDENCE_DIR", "/v3-data/evidence")

    from explaining_markets.live_v3_context import build_live_v3_context, feed_diagnostics
    from explaining_markets.point_in_time_audit_v3 import audit_context
    from explaining_markets.providers.live_context import default_provider_bundle_from_env

    ticker = ticker.upper()
    cutoff = datetime.now(timezone.utc)
    providers = default_provider_bundle_from_env(production_safe=True)
    event = {"event_id": f"modal-diagnostic-{ticker}", "disclosure": []}
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit = audit_context(context)
    diag = feed_diagnostics(context)
    result = {
        "ticker": ticker,
        "cutoff": cutoff.isoformat(),
        "alpha_vantage_configured": bool(os.getenv("ALPHAVANTAGE_API_KEY")),
        "finnhub_configured": bool(os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUBB_API")),
        "twelve_data_configured": bool(os.getenv("TWELVE_DATA_API_KEY")),
        "tiingo_configured": bool(os.getenv("TINGO_API") or os.getenv("TIINGO_API_KEY")),
        "openrouter_configured": bool(os.getenv("OPEN_ROUTER_API_KEY")),
        "openrouter_live_enabled": os.getenv("V3_LIVE_USE_OPENROUTER", "0"),
        "historical_cache_mounted": Path(os.environ["V3_HISTORY_CACHE_PATH"]).exists(),
        "company_news_count": diag["company_news_count"],
        "peer_news_count": diag["peer_news_count"],
        "sector_news_count": diag["sector_news_count"],
        "peer_count": diag["peer_count"],
        "reasoned_news_count": diag["reasoned_news_count"],
        "structured_reasoning_returned": bool(context.event_reasoning),
        "cutoff_audit": "PASS",
        "records_checked": audit.records_checked,
    }
    print("[V3_MODAL_DIAGNOSTIC] " + " ".join(f"{key}={value}" for key, value in result.items()))
    return result


@app.function(image=image, secrets=secrets, volumes={"/v3-data": v3_data}, timeout=120)
def check_production():
    """Non-submitting deployed V3-lite production diagnostic.

    This intentionally FAILS when the candidate artifact is missing or Modal
    selects V1, preventing a silent rollback from being mistaken for a V3
    deployment.
    """
    import os

    os.environ.setdefault("V3_HISTORY_CACHE_PATH", "/v3-data/company_history.sqlite")
    os.environ.setdefault("V3_EVIDENCE_DIR", "/v3-data/evidence")

    from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
    from explaining_markets.features_v3 import FeatureVectorV3, MODEL_FEATURE_NAMES_V3
    from explaining_markets.forward_looking_features import extract_forward_looking_features
    from explaining_markets.model import get_default_model
    from explaining_markets.model_v3_lite import V3LiteCandidateModel

    model = get_default_model()
    if not isinstance(model, V3LiteCandidateModel):
        result = {
            "status": "FAIL",
            "model": getattr(model, "model_version", type(model).__name__),
            "expected": "v3_lite_operator_2026_08_18",
            "detail": "V3-lite operator candidate was not selected",
        }
        print("[PROD_MODAL_DIAGNOSTIC] " + " ".join(f"{k}={v}" for k, v in result.items()))
        return result

    def vector(direction: int):
        values = {name: 0.0 for name in MODEL_FEATURE_NAMES_V3}
        for name, mean, sd, coef in zip(
            model.feature_names,
            model.means,
            model.standard_deviations,
            model.coefficients,
            strict=True,
        ):
            if name not in REASONING_FEATURE_NAMES:
                continue
            if abs(coef) <= 1e-12:
                values[name] = mean
                continue
            sign = 1.0 if coef > 0 else -1.0
            values[name] = mean + direction * sign * sd
        return FeatureVectorV3(values=values, fls=extract_forward_looking_features([]))

    negative_raw = model.predict_raw_vector(vector(-1))
    neutral_raw = model.predict_raw_vector(vector(0))
    positive_raw = model.predict_raw_vector(vector(1))
    negative = model.calibrator.calibrate(negative_raw)
    neutral = model.calibrator.calibrate(neutral_raw)
    positive = model.calibrator.calibrate(positive_raw)
    reasoning_nonzero = sum(
        1
        for name, coef in zip(model.feature_names, model.coefficients, strict=True)
        if name in REASONING_FEATURE_NAMES and abs(coef) > 1e-12
    )
    ordered = negative_raw < neutral_raw < positive_raw and negative <= neutral <= positive
    spread = positive - negative
    status = "PASS" if reasoning_nonzero > 0 and ordered and spread > 0.05 else "FAIL"
    result = {
        "status": status,
        "model": model.model_version,
        "ablation": model.ablation,
        "operator_override": model.operator_override,
        "promoted": model.promoted,
        "calibration": model.calibrator.version,
        "reasoning_nonzero": reasoning_nonzero,
        "negative": negative,
        "neutral": neutral,
        "positive": positive,
        "spread": spread,
        "ordered": ordered,
        "openrouter_live_enabled": os.getenv("V3_LIVE_USE_OPENROUTER", "0"),
        "em_api_configured": bool(os.getenv("EM_API_KEY")),
        "webhook_secret_configured": bool(os.getenv("EM_WEBHOOK_SECRET")),
    }
    print("[PROD_MODAL_DIAGNOSTIC] " + " ".join(f"{key}={value}" for key, value in result.items()))
    return result


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label="explaining-markets")
def web():
    from fastapi import FastAPI, Request, Response

    from explaining_markets import WebhookVerificationError, verify_webhook
    from explaining_markets.config import Config
    from explaining_markets.event_utils import log_deadline

    api = FastAPI(title="Explaining Markets starter")

    @api.get("/")
    def health() -> dict:
        return {"ok": True, "service": "explaining-markets-starter"}

    @api.post("/")
    @api.post("/competition/webhook")
    async def competition_webhook(request: Request) -> Response:
        config = Config.from_env()
        raw_body = await request.body()
        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            return Response(content=str(exc), status_code=401)

        webhook_id = event.get("id")
        if not await _claim_aio(webhook_id):
            return Response(status_code=200)

        log_deadline(event)
        await predict_and_submit.spawn.aio(event, webhook_id)
        return Response(status_code=200)

    return api
