#!/usr/bin/env python3
"""Build the explicitly operator-selected V3-lite candidate for live use.

This does NOT alter or bypass the normal promotion serializer.  The generated
artifact remains ``promoted=false`` and records the operator override and the
missing untouched holdout.  It is intended for the user-authorized emergency
switch away from a live V1 model whose real-event FLS vectors are all zero.
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
from explaining_markets.v3_training_data import load_training_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"
DEFAULT_BASE_ROWS = ROOT / "data" / "processed" / "v3_training_rows.jsonl.gz"
ABLATION = "fls_plus_reasoning"


def _default_rows() -> Path:
    return DEFAULT_ENRICHED_ROWS if DEFAULT_ENRICHED_ROWS.exists() else DEFAULT_BASE_ROWS


def main() -> int:
    parser = argparse.ArgumentParser(description="Build operator-selected V3-lite live candidate")
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OPERATOR_ARTIFACT)
    args = parser.parse_args()

    rows_path = args.rows or _default_rows()
    if not rows_path.exists():
        raise SystemExit(f"training rows not found: {rows_path}")

    rows = load_training_rows(rows_path)
    results, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    selected = results["ablations"][ABLATION]["selected"]

    train = [row for row in rows if row.quarter == TRAIN_QUARTER]
    validation = [row for row in rows if row.quarter == VALIDATION_QUARTER]
    if not train or not validation:
        raise SystemExit("V3-lite candidate requires 2025Q4 train and 2026Q1 validation rows")

    active = _active_features(train + validation, ABLATIONS[ABLATION])
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
            f"fitted on {TRAIN_QUARTER} only (ablation={ABLATION})"
        ),
    )

    path = serialize_operator_candidate(
        rows,
        feature_names=active,
        kind=str(selected["kind"]),
        params=dict(selected["params"]),
        calibrator=calibrator,
        validation_metrics=dict(selected["metrics"]),
        legacy_metrics=(results.get("legacy_evaluation") or {}).get("selected_raw"),
        artifact_path=args.output,
        operator_reason=(
            "User-authorized 2026-08-18 production switch because live V1 received "
            "non-empty disclosures but produced all-zero FLS vectors and identical "
            "0.4946 raw predictions across real events."
        ),
    )

    print("=== V3-LITE OPERATOR CANDIDATE ===")
    print(f"rows: {rows_path}")
    print(f"ablation: {ABLATION}")
    print(f"model: {selected['kind']} {selected['params']}")
    print(f"features: {len(active)}")
    print(f"validation_spearman: {selected['metrics'].get('spearman')}")
    print(f"validation_pearson: {selected['metrics'].get('pearson')}")
    print(f"calibration: {calibrator.version}")
    print("normal_promotion_gate: NOT PASSED (untouched holdout unavailable)")
    print("operator_override: ENABLED AND RECORDED")
    print(f"artifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
