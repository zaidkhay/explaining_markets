#!/usr/bin/env python3
"""Build the explicitly operator-selected V3-lite candidate for live use.

This never changes the normal promotion gate.  It requires refreshed historical
rows with meaningful realized-disclosure coverage and records the operator
override in the artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from explaining_markets.calibration import PercentileCalibrator
from explaining_markets.v3_lite_operator import DEFAULT_OPERATOR_ARTIFACT, serialize_operator_candidate
from explaining_markets.v3_lite_training import (
    ABLATIONS,
    _active_features,
    evaluate_v3_lite,
    fit_predict,
)
from explaining_markets.v3_training import TRAIN_QUARTER, VALIDATION_QUARTER
from explaining_markets.v3_training_data import load_training_rows, training_data_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"
DEFAULT_BASE_ROWS = ROOT / "data" / "processed" / "v3_training_rows.jsonl.gz"


def _default_rows() -> Path:
    return DEFAULT_ENRICHED_ROWS if DEFAULT_ENRICHED_ROWS.exists() else DEFAULT_BASE_ROWS


def _score(payload: dict) -> tuple[float, float]:
    metrics = payload["selected"]["metrics"]
    spear = metrics.get("spearman")
    return (-1e9 if spear is None else float(spear), -float(metrics["mae"]))


def _disclosure_capable_ablation(name: str) -> bool:
    names = ABLATIONS[name]
    return (
        "has_eps_surprise" in names
        or "has_revenue_surprise" in names
        or "reasoning_earnings_quality" in names
        or "reasoning_revenue_quality" in names
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build operator-selected V3-lite live candidate")
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OPERATOR_ARTIFACT)
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), default=None)
    parser.add_argument("--min-disclosure-result-coverage", type=float, default=0.10)
    args = parser.parse_args()

    rows_path = args.rows or _default_rows()
    if not rows_path.exists():
        raise SystemExit(f"training rows not found: {rows_path}")

    rows = load_training_rows(rows_path)
    report = training_data_report(rows, archive_seed_only=False)
    eps_cov = float(report.family_coverage.get("eps", 0.0))
    rev_cov = float(report.family_coverage.get("revenue", 0.0))
    if max(eps_cov, rev_cov) < args.min_disclosure_result_coverage:
        raise SystemExit(
            "refusing to build V3-lite operator artifact: historical rows do not appear "
            "to have been refreshed with the disclosure-results parser. "
            f"eps_coverage={eps_cov:.3f} revenue_coverage={rev_cov:.3f}; "
            "rerun scripts/enrich_v3_training_rows.py cache-only first"
        )

    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    if args.ablation:
        ablation = args.ablation
    else:
        eligible = [
            name for name, payload in results["ablations"].items()
            if "selected" in payload and _disclosure_capable_ablation(name)
        ]
        if not eligible:
            raise SystemExit("no disclosure-capable V3-lite ablation is available")
        ablation = max(eligible, key=lambda name: _score(results["ablations"][name]))

    selected = results["ablations"][ablation]["selected"]
    v1 = results["ablations"]["v1_fls_only"]["selected"]["metrics"]
    selected_spear = selected["metrics"].get("spearman")
    v1_spear = v1.get("spearman")
    if selected_spear is None or v1_spear is None or float(selected_spear) <= float(v1_spear):
        raise SystemExit(
            "refusing operator V3-lite artifact: selected disclosure-capable model does not "
            f"beat V1 validation Spearman (selected={selected_spear}, v1={v1_spear})"
        )

    train = [row for row in rows if row.quarter == TRAIN_QUARTER]
    validation = [row for row in rows if row.quarter == VALIDATION_QUARTER]
    if not train or not validation:
        raise SystemExit("V3-lite candidate requires 2025Q4 train and 2026Q1 validation rows")

    active = _active_features(train + validation, ABLATIONS[ablation])
    fit = fit_predict(
        train,
        validation,
        active,
        str(selected["kind"]),
        dict(selected["params"]),
    )
    calibrator = PercentileCalibrator.fit(
        fit.predictions.tolist(),
        source=(
            f"{VALIDATION_QUARTER} validation predictions from {selected['kind']} "
            f"fitted on {TRAIN_QUARTER} only (ablation={ablation})"
        ),
    )

    path = serialize_operator_candidate(
        rows,
        feature_names=active,
        kind=str(selected["kind"]),
        params=dict(selected["params"]),
        ablation=ablation,
        calibrator=calibrator,
        validation_metrics=dict(selected["metrics"]),
        legacy_metrics=(results.get("legacy_evaluation") or {}).get("selected_raw"),
        artifact_path=args.output,
        operator_reason=(
            "User-authorized 2026-08-18 production switch after live V1 received non-empty "
            "disclosures but produced all-zero FLS vectors and identical 0.4946 raw scores. "
            "Candidate was rebuilt after realized disclosure facts were mapped into the same "
            "point-in-time V3 feature families used for historical enrichment."
        ),
    )

    print("=== V3-LITE OPERATOR CANDIDATE ===")
    print(f"rows: {rows_path}")
    print(f"eps_coverage: {eps_cov:.3f}")
    print(f"revenue_coverage: {rev_cov:.3f}")
    print(f"ablation: {ablation}")
    print(f"model: {selected['kind']} {selected['params']}")
    print(f"features: {len(active)}")
    print(f"v1_validation_spearman: {v1_spear}")
    print(f"validation_spearman: {selected_spear}")
    print(f"validation_pearson: {selected['metrics'].get('pearson')}")
    print(f"calibration: {calibrator.version}")
    print("normal_promotion_gate: NOT PASSED (untouched holdout unavailable)")
    print("operator_override: ENABLED AND RECORDED")
    print(f"artifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
