#!/usr/bin/env python3
"""Score live tickers with the fitted, unpromoted V3 research artifact.

This command never submits a competition prediction and never changes the
production selector. It is for checking whether the *trained* V3 artifact
produces useful dispersion on current pre-cutoff inputs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from explaining_markets.features_v3 import build_feature_vector_v3, family_availability
from explaining_markets.live_v3_context import build_live_v3_context, feed_diagnostics
from explaining_markets.model_v3 import MultiSignalV3Model
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.providers.live_context import default_provider_bundle_from_env
from explaining_markets.v3_research_training import DEFAULT_RESEARCH_ARTIFACT


def score_one(
    ticker: str,
    *,
    artifact: Path,
    sector: str | None,
    sector_ticker: str | None,
    peers: tuple[str, ...],
) -> dict:
    cutoff = datetime.now(timezone.utc)
    event = {
        "event_id": f"v3-shadow-{ticker}-{cutoff:%Y%m%dT%H%M%S}",
        "sector": sector,
        "sector_ticker": sector_ticker,
        "peer_tickers": list(peers),
        "disclosure": [],
    }
    providers = default_provider_bundle_from_env()
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit = audit_context(context)
    vector = build_feature_vector_v3(disclosure=[], context=context)
    model = MultiSignalV3Model(artifact)
    if model.promoted:
        raise RuntimeError("research shadow scorer refuses a promoted artifact")
    diagnostics = feed_diagnostics(context)
    reasoning = context.event_reasoning
    return {
        "ticker": ticker,
        "cutoff": cutoff.isoformat(),
        "research_model_version": model.model_version,
        "research_score_not_submitted": model.predict_vector(vector),
        "records_checked": audit.records_checked,
        "family_availability": family_availability(vector),
        "company_news_count": diagnostics["company_news_count"],
        "peer_news_count": diagnostics["peer_news_count"],
        "sector_news_count": diagnostics["sector_news_count"],
        "reasoned_news_count": diagnostics["reasoned_news_count"],
        "overall_event_signal": None if reasoning is None else reasoning.overall_event_signal,
        "reasoning_confidence": None if reasoning is None else reasoning.confidence,
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Shadow-score live tickers with trained V3 research artifact")
    parser.add_argument("tickers", nargs="+", help="one or more tickers")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_RESEARCH_ARTIFACT)
    parser.add_argument("--sector")
    parser.add_argument("--sector-ticker")
    parser.add_argument("--peers", default="")
    args = parser.parse_args()

    if not args.artifact.exists():
        print(f"Research artifact not found: {args.artifact}")
        print("Run: uv run python scripts/train_v3_model.py --archive-seed --run-tests")
        return 2

    peers = tuple(value.strip().upper() for value in args.peers.split(",") if value.strip())
    results = []
    for ticker in args.tickers:
        result = score_one(
            ticker.upper(),
            artifact=args.artifact,
            sector=args.sector,
            sector_ticker=args.sector_ticker,
            peers=peers,
        )
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))

    scores = [float(result["research_score_not_submitted"]) for result in results]
    if len(scores) >= 2:
        print()
        print("=== TRAINED V3 SHADOW DISPERSION ===")
        print(f"min: {min(scores):.4f}")
        print(f"max: {max(scores):.4f}")
        print(f"spread: {max(scores) - min(scores):.4f}")
    print()
    print("NOT SUBMITTED: production model selection is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
