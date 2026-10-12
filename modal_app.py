"""Modal deployment for the Explaining Markets V3-lite production service.

Deploy:      uv run modal deploy modal_app.py
Feed check:  uv run modal run modal_app.py::check_v3_feed --ticker AAPL
Prod check:  uv run modal run modal_app.py::check_production

Routes: the original submission is served at POST / (and /competition/webhook)
with EM_API_KEY / EM_WEBHOOK_SECRET. Up to four more submissions are served at
POST /s/<n> (n = 2..5) with EM_API_KEY_S<n> / EM_WEBHOOK_SECRET_S<n>. Every
route runs the same model; only the credentials differ.
"""

import modal

app = modal.App("explaining-markets-starter")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "httpx", "pydantic")
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


def _dedupe_key(webhook_id, slot):
    """Namespace claims per slot so two submissions never share a claim.
    The original submission keeps its bare key, unchanged from before slots."""
    if not webhook_id or slot is None:
        return webhook_id
    return f"s{slot}:{webhook_id}"


def _release(webhook_id, submitted):
    if not webhook_id:
        return
    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


@app.function(image=image, secrets=secrets, volumes={"/v3-data": v3_data}, timeout=600, retries=0)
def predict_and_submit(event: dict, webhook_id: str | None = None, slot: int | None = None):
    import os

    os.environ.setdefault("V3_HISTORY_CACHE_PATH", "/v3-data/company_history.sqlite")
    os.environ.setdefault("V3_EVIDENCE_DIR", "/v3-data/evidence")

    from explaining_markets.client import submit_predictions_with_retry
    from explaining_markets.config import Config
    from explaining_markets.event_utils import is_test, neutral_predictions
    from predict import predict

    # Always submit something (CLAUDE.md §8.3): a model failure degrades to a
    # neutral 0.5 per focal asset and logs loudly; it never skips submission.
    submitted = False
    source = "test_neutral" if is_test(event) else "model"
    try:
        if is_test(event):
            predictions = neutral_predictions(event)
        else:
            try:
                predictions = predict(event)
            except Exception as exc:
                print(
                    f"[FALLBACK] slot={slot or 1} event={event.get('event_id')} "
                    f"model failed ({type(exc).__name__}: {exc}); submitting neutral 0.5"
                )
                predictions = neutral_predictions(event)
                source = "fallback_neutral"
        submit_predictions_with_retry(
            event_id=event["event_id"],
            predictions=predictions,
            config=Config.from_env(slot),
        )
        submitted = True
        # Positive confirmation: without it a successful submit and a worker
        # that never ran look identical in the logs (CLAUDE.md §9).
        values = ",".join(f"{p['identifier_value']}={p['predicted_percentile']:.4f}" for p in predictions)
        print(f"[SUBMIT] ok slot={slot or 1} event={event.get('event_id')} source={source} predictions={values or 'none'}")
    except Exception as exc:
        print(f"[ERROR] slot={slot or 1} submission failed for event {event.get('event_id')}: {exc}")
    finally:
        _release(webhook_id, submitted)


@app.function(image=image, secrets=secrets, volumes={"/v3-data": v3_data}, timeout=120)
def check_v3_feed(ticker: str):
    """Non-public bounded provider/cutoff diagnostic."""
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
    """Non-submitting deployed V3-lite model/live-gate diagnostic."""
    import os

    os.environ.setdefault("V3_HISTORY_CACHE_PATH", "/v3-data/company_history.sqlite")
    os.environ.setdefault("V3_EVIDENCE_DIR", "/v3-data/evidence")

    from explaining_markets.config import configured_slots
    from explaining_markets.model import get_default_model
    from explaining_markets.model_v3_lite import V3LiteCandidateModel
    from explaining_markets.v3_lite_live_gate import evaluate_v3_lite_live_gate

    expected_model = os.getenv("EXPECTED_V3_LITE_MODEL", "v3_lite_operator_2026_08_19")
    model = get_default_model()
    if not isinstance(model, V3LiteCandidateModel):
        result = {
            "status": "FAIL",
            "model": getattr(model, "model_version", type(model).__name__),
            "expected": expected_model,
            "detail": "V3-lite operator candidate was not selected",
        }
        print("[PROD_MODAL_DIAGNOSTIC] " + " ".join(f"{k}={v}" for k, v in result.items()))
        return result

    gate = evaluate_v3_lite_live_gate(model, min_submitted_spread=0.05, min_adjacent_rank_steps=5)
    scenario_by_label = {scenario.label: scenario for scenario in gate.scenarios}
    model_matches = model.model_version == expected_model
    metadata_ok = model.operator_override and model.production_candidate and not model.promoted
    status = "PASS" if model_matches and metadata_ok and gate.passed else "FAIL"
    result = {
        "status": status,
        "model": model.model_version,
        "expected": expected_model,
        "ablation": model.ablation,
        "operator_override": model.operator_override,
        "production_candidate": model.production_candidate,
        "promoted": model.promoted,
        "submission": model.submission,
        "calibration": model.calibrator.version,
        "calibration_n_fitted": model.calibrator.n_fitted,
        "parser": model.disclosure_parser_version,
        "negative": scenario_by_label["negative"].submitted,
        "neutral": scenario_by_label["neutral"].submitted,
        "positive": scenario_by_label["positive"].submitted,
        "negative_neutral_gap": gate.negative_neutral_gap,
        "neutral_positive_gap": gate.neutral_positive_gap,
        "required_adjacent_gap": gate.minimum_adjacent_gap_required,
        "required_adjacent_ranks": gate.adjacent_rank_steps_required,
        "spread": gate.submitted_spread,
        "ordered": gate.ordered,
        "parsed_ok": gate.parsed_ok,
        "zero_fls": gate.zero_fls,
        "live_gate_passed": gate.passed,
        "openrouter_live_enabled": os.getenv("V3_LIVE_USE_OPENROUTER", "0"),
        "em_api_configured": bool(os.getenv("EM_API_KEY")),
        "webhook_secret_configured": bool(os.getenv("EM_WEBHOOK_SECRET")),
        "slots_configured": ",".join(str(slot) for slot in configured_slots()) or "none",
    }
    print("[PROD_MODAL_DIAGNOSTIC] " + " ".join(f"{key}={value}" for key, value in result.items()))
    return result


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label="explaining-markets")
def web():
    from fastapi import FastAPI, Request, Response

    from explaining_markets import WebhookVerificationError, verify_webhook
    from explaining_markets.config import EXTRA_SLOTS, Config, configured_slots
    from explaining_markets.event_utils import log_deadline

    api = FastAPI(title="Explaining Markets")

    @api.get("/")
    def health() -> dict:
        return {"ok": True, "service": "explaining-markets-starter"}

    @api.post("/")
    @api.post("/competition/webhook")
    async def competition_webhook(request: Request) -> Response:
        return await _handle(request, slot=None)

    @api.post("/s/{slot}")
    async def slot_webhook(slot: int, request: Request) -> Response:
        if slot not in EXTRA_SLOTS or slot not in configured_slots():
            return Response(content="submission slot not configured", status_code=404)
        return await _handle(request, slot=slot)

    async def _handle(request: Request, slot: int | None) -> Response:
        config = Config.from_env(slot)
        raw_body = await request.body()
        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            return Response(content=str(exc), status_code=401)

        webhook_id = _dedupe_key(event.get("id"), slot)
        if not await _claim_aio(webhook_id):
            return Response(status_code=200)

        log_deadline(event)
        await predict_and_submit.spawn.aio(event, webhook_id, slot)
        return Response(status_code=200)

    return api
