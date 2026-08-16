"""Runnable V3 verification suite.

Examples:
    uv run python scripts/verify_v3_end_to_end.py --synthetic
    uv run python scripts/verify_v3_end_to_end.py --openrouter
    uv run python scripts/verify_v3_end_to_end.py --live AAPL --sector Technology --sector-ticker XLK --peers MSFT,NVDA,GOOGL,META
    uv run python scripts/verify_v3_end_to_end.py --all AAPL --sector Technology --sector-ticker XLK --peers MSFT,NVDA,GOOGL,META

Diagnostic V3 scores are never submitted to the competition.
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from explaining_markets.reasoning.openrouter_client import openrouter_api_key, openrouter_model, structured_json
from explaining_markets.v3_verification import run_synthetic_suite, summarize_scores, verify_live_ticker


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
    print("  - score distribution is not collapsed around 0.49")
    print("  - post-cutoff news is rejected")


def _print_openrouter() -> bool:
    load_dotenv()
    print("\n=== OPENROUTER STRUCTURED-OUTPUT SMOKE TEST ===")
    if not openrouter_api_key():
        print(json.dumps({"configured": False, "ok": False, "detail": "OPEN_ROUTER_API_KEY missing"}, indent=2))
        print("OPENROUTER STRUCTURED REASONING: FAIL")
        return False
    try:
        result = structured_json(
            schema_name="v3_reasoning_smoke",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "direction": {"type": "number", "minimum": -1, "maximum": 1},
                    "materiality": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "maxLength": 180},
                },
                "required": ["direction", "materiality", "confidence", "rationale"],
            },
            system_prompt="Use only the supplied synthetic headline. Return bounded structured market-event features.",
            user_payload={"headline": "Synthetic company beats EPS expectations and raises guidance."},
        )
        print(json.dumps({"configured": True, "ok": True, "model": openrouter_model(), "result": result}, indent=2))
        print("OPENROUTER STRUCTURED REASONING: PASS")
        return True
    except Exception as exc:
        print(json.dumps({"configured": True, "ok": False, "model": openrouter_model(), "detail": f"{type(exc).__name__}: {exc}"}, indent=2))
        print("OPENROUTER STRUCTURED REASONING: FAIL")
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
    parser.add_argument("--openrouter", action="store_true", help="make one real OpenRouter structured-output smoke-test call")
    parser.add_argument("--openai", action="store_true", help=argparse.SUPPRESS)  # backwards-compatible alias
    parser.add_argument("--live", dest="live_ticker", metavar="TICKER", help="run real live-feed V3 verification for one ticker")
    parser.add_argument("--all", dest="all_ticker", metavar="TICKER", help="run synthetic + OpenRouter + live checks")
    parser.add_argument("--sector", default=None)
    parser.add_argument("--sector-ticker", default=None)
    parser.add_argument("--peers", default="", help="comma-separated peer tickers")
    parser.add_argument("--require-production-v3", action="store_true", help="fail unless a promoted V3 artifact is the current production model")
    args = parser.parse_args()

    if args.openai:
        args.openrouter = True
    if args.all_ticker:
        args.synthetic = True
        args.openrouter = True
        args.live_ticker = args.all_ticker
    if not (args.synthetic or args.openrouter or args.live_ticker):
        args.synthetic = True

    ok = True
    if args.synthetic:
        try:
            _print_synthetic()
        except Exception as exc:
            ok = False
            print(f"\nSYNTHETIC V3 VERIFICATION: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.openrouter:
        ok = _print_openrouter() and ok

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
