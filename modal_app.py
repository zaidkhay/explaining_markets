"""Modal deployment for the Explaining Markets starter.

This is orchestration only: verify/ACK/dedupe/spawn, prediction, submission,
and a non-public V3 feed diagnostic. Business logic stays under
``src/explaining_markets``.

Deploy:    uv run modal deploy modal_app.py
Dev/local: uv run modal serve modal_app.py
Diagnostic: uv run modal run modal_app.py::check_v3_feed --ticker AAPL
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
    """Non-public Modal diagnostic. Never prints credential values."""
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
    providers = default_provider_bundle_from_env()
    event = {"event_id": f"modal-diagnostic-{ticker}", "disclosure": []}
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit = audit_context(context)
    diag = feed_diagnostics(context)
    result = {
        "ticker": ticker,
        "cutoff": cutoff.isoformat(),
        "news_secret_configured": bool(os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("NEWS_API_KEY")),
        "openai_secret_configured": bool(os.getenv("OPENAI_API_KEY")),
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
