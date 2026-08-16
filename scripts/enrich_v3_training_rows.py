#!/usr/bin/env python3
"""Enrich V3 archive rows using the free-provider stack."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from explaining_markets.historical_v3_enrichment import DEFAULT_CACHE_DIR, DEFAULT_ENRICHED_ROWS
from explaining_markets.historical_v3_enrichment_free import enrich_training_rows_free
from explaining_markets.v3_training_data import DEFAULT_ROWS_PATH

DEFAULT_HISTORICAL_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"


def _coverage_gate(
    coverage: dict[str, float],
    *,
    min_eps: float,
    min_news: float,
    min_reasoning: float,
    min_price: float,
) -> tuple[bool, list[str]]:
    failures = []
    checks = {
        "eps": (coverage.get("eps", 0.0), min_eps),
        "company_news": (coverage.get("company_news", 0.0), min_news),
        "reasoning": (coverage.get("reasoning", 0.0), min_reasoning),
        "price_5y": (coverage.get("price_5y", 0.0), min_price),
    }
    for name, (actual, required) in checks.items():
        if actual < required:
            failures.append(f"{name}={actual:.3f} < required {required:.3f}")
    return not failures, failures


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build point-in-time-enriched V3 historical training rows")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--historical-dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENRICHED_ROWS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    # Free-provider budgets are deliberately conservative. Tiingo Starter is
    # 50 requests/hour; Finnhub Free is 60 requests/minute.
    parser.add_argument("--tiingo-api-calls", type=int, default=40)
    parser.add_argument("--finnhub-api-calls", type=int, default=50)

    # Backwards-compatible Alpha flags. Alpha is fallback-only now and defaults
    # to zero new historical calls so its 25/day free allowance is preserved.
    parser.add_argument("--earnings-api-calls", type=int, default=0, help="Alpha fallback EARNINGS calls")
    parser.add_argument("--news-api-calls", type=int, default=0, help="Alpha fallback NEWS calls")

    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--price-csv", type=Path, default=None, help="optional adjusted daily CSV fallback")
    parser.add_argument("--alpha-adjusted-prices", action="store_true", help="try premium Alpha adjusted prices as a final fallback")
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--news-chunk-days", type=int, default=7, help="Alpha fallback broad-news window size")
    parser.add_argument("--reasoning-mode", choices=("deterministic", "openrouter"), default="deterministic")
    parser.add_argument("--openrouter-max-calls", type=int, default=25, help="process-wide LLM call cap when --reasoning-mode=openrouter")
    parser.add_argument("--retrain", action="store_true", help="run research V3 training only after coverage gates pass")

    parser.add_argument("--min-eps-coverage", type=float, default=0.30)
    parser.add_argument("--min-news-coverage", type=float, default=0.20)
    parser.add_argument("--min-reasoning-coverage", type=float, default=0.20)
    parser.add_argument("--min-price-coverage", type=float, default=0.50)
    args = parser.parse_args()

    for name in ("tiingo_api_calls", "finnhub_api_calls", "earnings_api_calls", "news_api_calls", "openrouter_max_calls"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    if args.reasoning_mode == "openrouter":
        os.environ["OPEN_ROUTER_MAX_CALLS"] = str(args.openrouter_max_calls)

    tiingo_key = _env("TINGO_API", "TIINGO_API_KEY", "TIINGO_API")
    finnhub_key = _env("FINNHUB_API_KEY", "FINNHUBB_API")
    openrouter_key = _env("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY")
    alpha_key = _env("ALPHAVANTAGE_API_KEY", "NEWS_API_KEY")

    print("=== V3 PROVIDER CONFIG ===")
    print(f"tiingo_configured: {bool(tiingo_key)}")
    print(f"finnhub_configured: {bool(finnhub_key)}")
    print(f"openrouter_configured: {bool(openrouter_key)}")
    print(f"alpha_fallback_configured: {bool(alpha_key)}")
    print(f"reasoning_mode: {args.reasoning_mode}")

    report = enrich_training_rows_free(
        rows_path=args.rows,
        historical_dir=args.historical_dir,
        output_path=args.output,
        cache_dir=args.cache_dir,
        tiingo_api_key=tiingo_key,
        finnhub_api_key=finnhub_key,
        alpha_api_key=alpha_key,
        tiingo_max_api_calls=args.tiingo_api_calls,
        finnhub_max_api_calls=args.finnhub_api_calls,
        alpha_max_api_calls=args.earnings_api_calls + args.news_api_calls,
        price_csv=args.price_csv,
        use_alpha_adjusted_prices=args.alpha_adjusted_prices,
        include_historical_news=not args.no_news,
        news_chunk_days=args.news_chunk_days,
        reasoning_mode=args.reasoning_mode,
        request_timeout=args.request_timeout,
        progress_every=args.progress_every,
    )

    print("=== V3 HISTORICAL ENRICHMENT ===")
    print(f"rows: {report.rows}")
    print(f"eps_matches: {report.eps_matches}")
    print(f"rows_with_company_news: {report.rows_with_company_news}")
    print(f"rows_with_reasoning: {report.rows_with_reasoning}")
    print(f"rows_with_prices: {report.rows_with_prices}")
    print(f"tiingo_api_calls_this_run: {report.tiingo_api_calls}")
    print(f"tiingo_cache_hits: {report.tiingo_cache_hits}")
    print(f"finnhub_api_calls_this_run: {report.finnhub_api_calls}")
    print(f"finnhub_cache_hits: {report.finnhub_cache_hits}")
    print(f"alpha_api_calls_this_run: {report.alpha_api_calls}")
    print(f"alpha_cache_hits: {report.alpha_cache_hits}")
    print(f"output: {report.output_path}")
    print("family_coverage:")
    for name, value in sorted(report.family_coverage.items()):
        print(f"  {name}: {value:.3f}")

    passed, failures = _coverage_gate(
        report.family_coverage,
        min_eps=args.min_eps_coverage,
        min_news=args.min_news_coverage,
        min_reasoning=args.min_reasoning_coverage,
        min_price=args.min_price_coverage,
    )
    print(f"coverage_gate_passed: {passed}")
    for failure in failures:
        print(f"  FAIL: {failure}")

    metadata_path = args.output.with_suffix(args.output.suffix + ".report.json")
    metadata_path.write_text(
        json.dumps(
            {
                **report.as_dict(),
                "coverage_gate_passed": passed,
                "coverage_gate_failures": failures,
                "reasoning_mode": args.reasoning_mode,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"coverage_report: {metadata_path}")

    if args.retrain:
        if not passed:
            print("RETRAINING REFUSED: historical enrichment coverage gate did not pass.")
            return 3
        print("Coverage gate passed; running research-only V3 retraining...")
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_v3_model.py")),
            "--rows", str(args.output),
            "--run-tests",
        ]
        return subprocess.run(command, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
