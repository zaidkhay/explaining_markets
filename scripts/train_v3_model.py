#!/usr/bin/env python3
"""Train/evaluate V3 from prebuilt point-in-time training rows.

By default this writes only an unpromoted research artifact under
``data/processed``. Production serialization is available only with
``--promote`` and remains subject to every gate in ``v3_training.evaluate``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from explaining_markets.v3_research_training import (
    DEFAULT_RESEARCH_ARTIFACT,
    serialize_research_linear_artifact,
)
from explaining_markets.v3_training import DEFAULT_ARTIFACT, evaluate, serialize_linear_artifact
from explaining_markets.v3_training_data import (
    DEFAULT_ROWS_PATH,
    load_training_rows,
    training_data_report,
)

DEFAULT_EVALUATION_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "v3_evaluation.json"


def _run_tests() -> bool:
    print("Running pytest before model evaluation...")
    result = subprocess.run([sys.executable, "-m", "pytest"], check=False)
    return result.returncode == 0


def _metric_line(label: str, payload: dict | None) -> None:
    if not payload:
        print(f"{label}: unavailable")
        return
    metrics = payload.get("metrics", payload)
    print(
        f"{label}: pearson={metrics.get('pearson')} "
        f"spearman={metrics.get('spearman')} "
        f"mae={metrics.get('mae')} "
        f"std={metrics.get('prediction_std')} "
        f"near_0.5={metrics.get('fraction_between_048_052')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the multi-signal V3 model")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--research-artifact", type=Path, default=DEFAULT_RESEARCH_ARTIFACT)
    parser.add_argument("--production-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--archive-seed", action="store_true", help="label coverage report as archive-seed-only")
    parser.add_argument("--skip-research-artifact", action="store_true")
    parser.add_argument("--promote", action="store_true", help="serialize production artifact only if every promotion gate passes")
    parser.add_argument("--local-feed-verified", action="store_true")
    parser.add_argument("--modal-feed-verified", action="store_true")
    parser.add_argument("--news-coverage-nonzero", action="store_true")
    parser.add_argument("--reasoning-valid", action="store_true")
    parser.add_argument("--latency-ok", action="store_true")
    args = parser.parse_args()

    rows = load_training_rows(args.rows)
    report = training_data_report(rows, archive_seed_only=args.archive_seed)
    tests_passing = _run_tests() if args.run_tests else False

    evaluation = evaluate(
        rows,
        tests_passing=tests_passing,
        local_feed_verified=args.local_feed_verified,
        modal_feed_verified=args.modal_feed_verified,
        news_coverage_nonzero=args.news_coverage_nonzero,
        reasoning_valid=args.reasoning_valid,
        latency_ok=args.latency_ok,
    )
    evaluation["training_data_report"] = report.as_dict()
    evaluation["tests_invoked_by_training_cli"] = bool(args.run_tests)

    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== V3 TRAINING / EVALUATION ===")
    print(f"rows: {report.rows}")
    print(f"quarter_counts: {report.quarter_counts}")
    print(f"family_coverage: {report.family_coverage}")
    print(f"active_non_fls_features: {report.active_non_fls_features}")
    print(f"evaluation_output: {args.evaluation_output}")
    print()

    baseline = evaluation["ablations"]["v1_fls_only"]["selected"]
    full = evaluation["ablations"]["full_v3"]["selected"]
    _metric_line("validation V1 FLS", baseline)
    _metric_line("validation full V3", full)
    print(f"validation_gain_over_v1: {evaluation.get('validation_gain_over_v1')}")
    _metric_line("legacy holdout 2026Q2", evaluation.get("legacy_holdout"))
    _metric_line("honest holdout 2026Q3", evaluation.get("honest_holdout"))
    print(f"selected full-V3 model: {full.get('kind')} {full.get('params')}")
    print(f"promotion_gate_passed: {evaluation.get('promoted', False)}")

    if not args.skip_research_artifact:
        research = serialize_research_linear_artifact(rows, evaluation, args.research_artifact)
        print()
        print("RESEARCH ARTIFACT WRITTEN (NOT PROMOTED / NOT USED BY PRODUCTION)")
        print(f"path: {args.research_artifact}")
        print(f"model: {research['selected_linear_candidate']['kind']} {research['selected_linear_candidate']['params']}")
        print(f"training_quarters: {research['training_quarters']}")
        print(f"training_rows: {research['training_rows']}")

    if args.promote:
        if not evaluation.get("promoted"):
            print()
            print("PRODUCTION PROMOTION REFUSED: one or more predeclared gates failed.")
            return 3
        serialize_linear_artifact(rows, evaluation, args.production_artifact)
        print()
        print(f"PRODUCTION V3 ARTIFACT WRITTEN: {args.production_artifact}")

    if report.archive_seed_only:
        print()
        print("WARNING: archive-seed-only rows intentionally leave many V3 families missing.")
        print("Use this artifact for research/shadow inference only; do not promote it as full V3.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
