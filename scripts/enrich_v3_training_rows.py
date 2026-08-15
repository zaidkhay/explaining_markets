#!/usr/bin/env python3
"""Enrich V3 archive seed rows with historical EPS, prices, news and reasoning."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from explaining_markets.historical_v3_enrichment import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ENRICHED_ROWS,
    enrich_training_rows,
)
from explaining_markets.v3_training_data import DEFAULT_ROWS_PATH

DEFAULT_HISTORICAL_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"


def _coverage_gate(coverage: dict[str, float], *, min_eps: float, min_news: float, min_reasoning: float) -> tuple[bool, list[str]]:
    failures = []
    checks = {
        "eps": (coverage.get("eps", 0.0), min_eps),
        "company_news": (coverage.get("company_news", 0.0), min_news),
        "reasoning": (coverage.get("reasoning", 0.0), min_reasoning),
    }
    for name, (actual, required) in checks.items():
        if actual < required:
            failures.append(f"{name}={actual:.3f} < required {required:.3f}")
    return not failures, failures


def _run_enrichment(args, *, max_api_calls: int, include_news: bool):
    return enrich_training_rows(
        rows_path=args.rows,
        historical_dir=args.historical_dir,
        output_path=args.output,
        alpha_api_key=os.environ.get("ALPHAVANTAGE_API_KEY"),
        cache_dir=args.cache_dir,
        max_api_calls=max_api_calls,
        price_csv=args.price_csv,
        use_alpha_adjusted_prices=args.alpha_adjusted_prices,
        include_historical_news=include_news,
        news_chunk_days=args.news_chunk_days,
        reasoning_mode=args.reasoning_mode,
        request_timeout=args.request_timeout,
        request_retries=args.request_retries,
        progress_every=args.progress_every,
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build point-in-time-enriched V3 historical training rows")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--historical-dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENRICHED_ROWS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--earnings-api-calls", type=int, default=19, help="Alpha calls reserved for historical EARNINGS/cache progress")
    parser.add_argument("--news-api-calls", type=int, default=6, help="Alpha calls reserved for historical NEWS_SENTIMENT/cache progress")
    parser.add_argument("--request-timeout", type=float, default=8.0, help="per-request Alpha Vantage timeout in seconds")
    parser.add_argument("--request-retries", type=int, default=0, help="retries per uncached Alpha request; each attempt consumes budget")
    parser.add_argument("--progress-every", type=int, default=500, help="print row progress every N rows; 0 disables row progress")
    parser.add_argument("--price-csv", type=Path, default=None, help="bulk adjusted daily CSV with ticker,date,close[,volume,available_at,source]")
    parser.add_argument("--alpha-adjusted-prices", action="store_true", help="try premium TIME_SERIES_DAILY_ADJUSTED when entitled")
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--news-chunk-days", type=int, default=7, help="broad earnings-news cache window size")
    parser.add_argument("--reasoning-mode", choices=("deterministic", "openai"), default="deterministic")
    parser.add_argument("--retrain", action="store_true", help="run research V3 training only after the coverage gate passes")
    parser.add_argument("--min-eps-coverage", type=float, default=0.30)
    parser.add_argument("--min-news-coverage", type=float, default=0.20)
    parser.add_argument("--min-reasoning-coverage", type=float, default=0.20)
    args = parser.parse_args()

    if args.earnings_api_calls < 0 or args.news_api_calls < 0:
        parser.error("API call budgets must be non-negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.request_retries < 0:
        parser.error("--request-retries must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    # Phase 1 prioritizes EARNINGS. Successful responses are cached by ticker,
    # so repeated runs naturally advance to the next uncached companies.
    earnings_report = _run_enrichment(
        args,
        max_api_calls=args.earnings_api_calls,
        include_news=False,
    )

    # Phase 2 spends only the separate news budget. It rebuilds the rows using
    # all cached earnings plus newly cached broad historical news windows.
    if not args.no_news and args.news_api_calls > 0:
        report = _run_enrichment(
            args,
            max_api_calls=args.news_api_calls,
            include_news=True,
        )
        total_calls = earnings_report.alpha_api_calls + report.alpha_api_calls
        total_hits = earnings_report.cache_hits + report.cache_hits
    else:
        report = earnings_report
        total_calls = earnings_report.alpha_api_calls
        total_hits = earnings_report.cache_hits

    print("=== V3 HISTORICAL ENRICHMENT ===")
    print(f"rows: {report.rows}")
    print(f"eps_matches: {report.eps_matches}")
    print(f"rows_with_company_news: {report.rows_with_company_news}")
    print(f"rows_with_reasoning: {report.rows_with_reasoning}")
    print(f"rows_with_prices: {report.rows_with_prices}")
    print(f"alpha_api_calls_this_run: {total_calls}")
    print(f"alpha_cache_hits_across_phases: {total_hits}")
    print(f"output: {report.output_path}")
    print("family_coverage:")
    for name, value in sorted(report.family_coverage.items()):
        print(f"  {name}: {value:.3f}")

    passed, failures = _coverage_gate(
        report.family_coverage,
        min_eps=args.min_eps_coverage,
        min_news=args.min_news_coverage,
        min_reasoning=args.min_reasoning_coverage,
    )
    print(f"coverage_gate_passed: {passed}")
    for failure in failures:
        print(f"  FAIL: {failure}")

    metadata_path = args.output.with_suffix(args.output.suffix + ".report.json")
    metadata_path.write_text(
        json.dumps(
            {
                **report.as_dict(),
                "alpha_api_calls_this_run_total": total_calls,
                "coverage_gate_passed": passed,
                "coverage_gate_failures": failures,
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
        result = subprocess.run(command, check=False)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
