"""Runnable V3 verification suite.

Examples:
    uv run python scripts/verify_v3_end_to_end.py --synthetic
    uv run python scripts/verify_v3_end_to_end.py --openai
    uv run python scripts/verify_v3_end_to_end.py --live AAPL --sector Technology --sector-ticker XLK --peers MSFT,NVDA,GOOGL,META
    uv run python scripts/verify_v3_end_to_end.py --all AAPL --sector Technology --sector-ticker XLK --peers MSFT,NVDA,GOOGL,META

Diagnostic V3 scores are never submitted to the competition. They use a
short-lived synthetic artifact solely to prove that the V3 feature/inference
machinery does not mechanically collapse to ~0.49 when inputs differ.
"""
from __future__ import annotations

import argparse
import json
import sys

from explaining_markets.v3_verification import (
    run_synthetic_suite,
    summarize_scores,
    verify_live_ticker,
    verify_openai_structured_output,
)


def _print_synthetic() -> None:
    rows = run_synthetic_suite()
    print("\n=== V3 SYNTHETIC OUTCOME MATRIX ===")
    print("NOTE: diagnostic scores below are NOT production predictions and are NOT submitted.\n")
    header = f"{'scenario':34} {'score':>7} {'signal':>7} {'eps%':>8} {'rev%':>8} {'guide%':>8} {'priced':>7} {'news':>7} {'contr':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.name:34} {row.score:7.3f} {row.overall_event_signal:7.3f} "
            f"{row.eps_surprise_percent:8.3f} {row.revenue_surprise_percent:8.3f} "
            f"{row.guidance_surprise_percent:8.3f} {row.priced_in_score:7.3f} "
            f"{row.company_news_signal:7.3f} {row.contradiction_score:7.3f}"
        )
    summary = summarize_scores(rows)
    print("\nSynthetic score distribution:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    print("\nSYNTHETIC V3 VERIFICATION: PASS")
    print("  - strong positive > neutral > strong negative")
    print("  - positive news > negative news")
    print("  - beat+cut contradiction detected")
    print("  - priced-in context changes scores")
    print("  - post-cutoff news is rejected")
    print("  - score distribution is not collapsed around 0.49")


def _print_openai() -> bool:
    print("\n=== OPENAI STRUCTURED-OUTPUT SMOKE TEST ===")
    result = verify_openai_structured_output()
    print(json.dumps(result, indent=2, default=str))
    if result.get("ok"):
        print("OPENAI STRUCTURED REASONING: PASS")
        return True
    print("OPENAI STRUCTURED REASONING: FAIL")
    return False


def _print_live(args) -> dict:
    peers = tuple(x.strip().upper() for x in (args.peers or "").split(",") if x.strip())
    print("\n=== LIVE V3 FEED / REASONING CHECK ===")
    result = verify_live_ticker(
        args.live_ticker,
        sector=args.sector,
        sector_ticker=args.sector_ticker,
        peers=peers,
    )
    print(json.dumps(result, indent=2, default=str))
    print("\nLive interpretation:")
    print(f"  company news: {result['company_news_count']}")
    print(f"  peer news: {result['peer_news_count']}")
    print(f"  sector news: {result['sector_news_count']}")
    print(f"  reasoned articles: {result['reasoned_news_count']}")
    print(f"  price rows: {result['price_rows']}")
    print(f"  history rows: {result['history_rows']}")
    print(f"  event reasoning: {result['reasoning_available']}")
    print(f"  overall event signal: {result['overall_event_signal']}")
    print(f"  DIAGNOSTIC V3 score (NOT SENT): {result['diagnostic_v3_score_not_submitted']:.4f}")
    print(f"  production model currently selected: {result['production_model']}")
    print(f"  promoted V3 selected: {result['production_v3_selected']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Exhaustive V3 verification suite")
    parser.add_argument("--synthetic", action="store_true", help="run deterministic V3 scenario matrix")
    parser.add_argument("--openai", action="store_true", help="make one real OpenAI structured-output smoke-test call")
    parser.add_argument("--live", dest="live_ticker", metavar="TICKER", help="run real live-feed V3 verification for one ticker")
    parser.add_argument("--all", dest="all_ticker", metavar="TICKER", help="run synthetic + OpenAI + live checks")
    parser.add_argument("--sector", default=None)
    parser.add_argument("--sector-ticker", default=None)
    parser.add_argument("--peers", default="", help="comma-separated peer tickers")
    parser.add_argument("--require-production-v3", action="store_true", help="fail unless a promoted V3 artifact is the current production model")
    args = parser.parse_args()

    if args.all_ticker:
        args.synthetic = True
        args.openai = True
        args.live_ticker = args.all_ticker
    if not (args.synthetic or args.openai or args.live_ticker):
        args.synthetic = True

    ok = True
    if args.synthetic:
        try:
            _print_synthetic()
        except Exception as exc:
            ok = False
            print(f"\nSYNTHETIC V3 VERIFICATION: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.openai:
        ok = _print_openai() and ok

    live = None
    if args.live_ticker:
        try:
            live = _print_live(args)
            if live["company_news_count"] <= 0:
                print("LIVE WARNING: company news is empty")
            if not live["reasoning_available"]:
                ok = False
                print("LIVE FAIL: event reasoning is unavailable", file=sys.stderr)
        except Exception as exc:
            ok = False
            print(f"\nLIVE V3 VERIFICATION: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.require_production_v3:
        if not live:
            print("--require-production-v3 requires --live or --all", file=sys.stderr)
            ok = False
        elif not live["production_v3_selected"]:
            print("PRODUCTION V3 CHECK: FAIL — production is not using a promoted V3 artifact.", file=sys.stderr)
            ok = False
        else:
            print("PRODUCTION V3 CHECK: PASS")

    print("\n=== FINAL RESULT ===")
    print("PASS" if ok else "FAIL")
    if live and not live["production_v3_selected"]:
        print("Important: V3 pipeline verification can pass while production submissions remain V1 (~0.49).")
        print("To prove non-0.49 scores are actually being SUBMITTED, a trained/promoted V3 artifact must first pass the promotion gate and be deployed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
