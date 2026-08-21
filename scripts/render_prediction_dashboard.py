"""Generate a standalone HTML + JSON dashboard for one live-style V3-lite event.

Example:
    uv run python scripts/render_prediction_dashboard.py \
      --ticker WMT \
      --cutoff 2026-08-20T11:00:00Z \
      --event-id ea_WMT_Q2_2027 \
      --url 'https://fresh-signed-information-url...'

Use --no-external to inspect only the supplied disclosure and local cache.
Optional realized outcomes can be attached later with --realized-car1 and
--realized-percentile.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.live_v3_context import build_live_v3_context
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.prediction_dashboard import render_prediction_dashboard, write_report_json
from explaining_markets.prediction_diagnostics import build_prediction_diagnostics
from explaining_markets.providers.live_context import default_provider_bundle_from_env
from explaining_markets.v3_providers import V3ProviderBundle
from predict import _fetch_disclosure


def _cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cutoff must include timezone, for example 2026-08-20T13:00:00Z")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--url", required=True, help="Fresh competition information_url")
    parser.add_argument("--event-id", default="manual_inspection")
    parser.add_argument("--event-type", default="EARNINGS_RELEASE")
    parser.add_argument("--output-dir", default="data/diagnostics")
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--realized-car1", type=float)
    parser.add_argument("--realized-percentile", type=float)
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    cutoff = _cutoff(args.cutoff)
    disclosure = _fetch_disclosure(args.url)
    event = {
        "event_id": args.event_id,
        "event_type": args.event_type,
        "event_datetime": cutoff.isoformat(),
        "information_url": args.url,
        "disclosure": disclosure,
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
    }
    providers = V3ProviderBundle.null() if args.no_external else default_provider_bundle_from_env()
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit_context(context)
    vector = build_feature_vector_v3(disclosure=disclosure, context=context)
    model = V3LiteCandidateModel()
    report = build_prediction_diagnostics(model=model, vector=vector, disclosure=disclosure, context=context)
    report["event"] = {
        "event_id": args.event_id,
        "event_type": args.event_type,
        "ticker": ticker,
        "cutoff": cutoff.isoformat(),
        "disclosure_fact_count": len(disclosure),
    }
    if args.realized_car1 is not None or args.realized_percentile is not None:
        report["realized"] = {
            "car1": args.realized_car1,
            "realized_percentile": args.realized_percentile,
        }

    root = Path(args.output_dir)
    safe_event = "".join(c for c in str(args.event_id) if c.isalnum() or c in "-_") or "manual"
    stem = f"{safe_event}__{ticker}"
    json_path = write_report_json(report, root / f"{stem}.json")
    html_path = render_prediction_dashboard(report, root / f"{stem}.html")

    score = report["score"]
    print("=== V3-LITE PREDICTION DASHBOARD ===")
    print(f"ticker: {ticker}")
    print(f"claims: {len(disclosure)}")
    print(f"raw_score: {score['raw_score']:.6f}")
    print(f"submitted_percentile: {score['submitted_percentile']:.6f}")
    print(f"json: {json_path}")
    print(f"html: {html_path}")
    print("Open the HTML file in your browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
