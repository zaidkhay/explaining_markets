"""Inspect exactly how the deployed V3-lite model scores one event.

Example:
    uv run python scripts/inspect_v3_prediction.py \
      --ticker WMT \
      --cutoff 2026-08-20T11:00:00Z \
      --url 'https://...'

The report shows:
- the disclosure facts fetched from the competition information URL
- parser matches and live data-family availability
- every feature used by the deployed artifact
- standardization, coefficient, and additive raw-score contribution
- the raw model score and empirical-CDF calibrated submitted percentile
"""
from __future__ import annotations

import argparse
from datetime import datetime

from explaining_markets.disclosure_results_v3 import parse_disclosure_records
from explaining_markets.features_v3 import build_feature_vector_v3, family_availability
from explaining_markets.live_v3_context import build_live_v3_context, feed_diagnostics
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.providers.live_context import default_provider_bundle_from_env


def _parse_cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cutoff must include a timezone, e.g. 2026-08-20T13:00:00Z")
    return parsed


def _fetch_disclosure(url: str) -> list[str]:
    # Reuse the exact production parser for the competition information URL.
    from predict import _fetch_disclosure as production_fetch

    return production_fetch(url)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--cutoff", required=True, help="ISO-8601 event cutoff, e.g. 2026-08-20T13:00:00Z")
    parser.add_argument("--url", required=True, help="Fresh competition information_url")
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()
    cutoff = _parse_cutoff(args.cutoff)
    disclosure = _fetch_disclosure(args.url)

    event = {
        "event_id": f"inspect_{ticker}",
        "event_type": "EARNINGS_RELEASE",
        "event_datetime": cutoff.isoformat(),
        "information_url": args.url,
        "disclosure": disclosure,
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
    }

    providers = default_provider_bundle_from_env()
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit = audit_context(context)
    vector = build_feature_vector_v3(disclosure=disclosure, context=context)
    parsed = parse_disclosure_records(disclosure, ticker=ticker, cutoff=cutoff)
    model = V3LiteCandidateModel()

    rows = []
    for name, mean, sd, coefficient in zip(
        model.feature_names,
        model.means,
        model.standard_deviations,
        model.coefficients,
        strict=True,
    ):
        value = float(vector.values[name])
        z = (value - mean) / sd
        contribution = coefficient * z
        rows.append((name, value, mean, sd, z, coefficient, contribution))

    raw_unclipped = model.intercept + sum(row[-1] for row in rows)
    raw = model.predict_raw_vector(vector)
    submitted = model.calibrator.calibrate(raw)

    print("=== EVENT ===")
    print(f"ticker: {ticker}")
    print(f"cutoff: {cutoff.isoformat()}")
    print(f"model: {model.model_version}")
    print(f"ablation: {model.ablation}")
    print(f"features_used: {len(model.feature_names)}")
    print(f"cutoff_audit: PASS ({audit.records_checked} records checked)")

    print("\n=== DISCLOSURE FACTS THE MODEL RECEIVES ===")
    if not disclosure:
        print("(none)")
    for i, fact in enumerate(disclosure, 1):
        print(f"{i:02d}. {fact}")

    print("\n=== DETERMINISTIC DISCLOSURE PARSER ===")
    print("matched_fields:", list(parsed.matched_fields))
    if parsed.earnings is None:
        print("earnings_record: none")
    else:
        print(
            "earnings_record:",
            {
                "reported_eps": parsed.earnings.reported_eps,
                "consensus_eps": parsed.earnings.consensus_eps,
                "reported_revenue": parsed.earnings.reported_revenue,
                "consensus_revenue": parsed.earnings.consensus_revenue,
            },
        )
    print("guidance_direction:", None if parsed.guidance is None else parsed.guidance.direction)

    print("\n=== LIVE CONTEXT AVAILABILITY ===")
    feed = feed_diagnostics(context)
    for key, value in feed.items():
        print(f"{key}: {value}")
    print("feature_family_availability:")
    for key, value in family_availability(vector).items():
        print(f"  {key}: {int(bool(value))}")

    print("\n=== IMPORTANT LIMITATION ===")
    print("The deployed artifact is fls_plus_revenue.")
    print("It uses 30 forward-looking-language features + 9 revenue/result-direction features.")
    print("EPS-only, price, peer, news, sector, company-history, and reasoning features may be built in V3 context")
    print("but DO NOT enter this model unless their feature name appears below.")

    print("\n=== MODEL INPUTS AND RAW-SCORE CONTRIBUTIONS ===")
    print("feature                                    value        z       coef     contribution")
    print("-" * 92)
    for name, value, _mean, _sd, z, coefficient, contribution in sorted(
        rows, key=lambda row: abs(row[-1]), reverse=True
    ):
        print(f"{name:<42} {value:>10.4f} {z:>8.3f} {coefficient:>10.6f} {contribution:>13.6f}")

    print("\n=== SCORE CONSTRUCTION ===")
    print(f"intercept: {model.intercept:.6f}")
    print(f"sum_feature_contributions: {sum(row[-1] for row in rows):+.6f}")
    print(f"raw_unclipped: {raw_unclipped:.6f}")
    print(f"raw_after_[{model.clip_lower:.2f},{model.clip_upper:.2f}]_clip: {raw:.6f}")
    print(
        "calibration: empirical OOS mid-rank CDF against "
        f"{model.calibrator.n_fitted} validation predictions"
    )
    print(f"submitted_percentile: {submitted:.6f}")

    print("\n=== TOP UP / DOWN DRIVERS ===")
    positive = sorted((r for r in rows if r[-1] > 0), key=lambda r: r[-1], reverse=True)[:5]
    negative = sorted((r for r in rows if r[-1] < 0), key=lambda r: r[-1])[:5]
    print("UP:")
    for row in positive:
        print(f"  {row[0]}: {row[-1]:+.6f}")
    print("DOWN:")
    for row in negative:
        print(f"  {row[0]}: {row[-1]:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
