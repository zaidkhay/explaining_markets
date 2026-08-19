#!/usr/bin/env python3
"""Audit whether the current V3-lite artifact still collapses many events to one score.

This is intentionally descriptive rather than a promotion gate. It evaluates the
serialized production candidate against the point-in-time historical matrix and
reports how often the model sees no active signal, plus concentration of raw and
calibrated predictions at the same display precision used in live diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from explaining_markets.feature_families.revenue_results import REVENUE_SURPRISE_FEATURE_NAMES
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.v3_training_data import load_training_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"


def _raw(model: V3LiteCandidateModel, values: dict[str, float]) -> float:
    prediction = model.intercept + sum(
        coef * (float(values[name]) - mean) / sd
        for name, coef, mean, sd in zip(
            model.feature_names,
            model.coefficients,
            model.means,
            model.standard_deviations,
            strict=True,
        )
    )
    return float(max(model.clip_lower, min(model.clip_upper, prediction)))


def _largest_bucket(values: list[float], digits: int) -> tuple[float, int, float]:
    if not values:
        return 0.0, 0, 0.0
    counts = Counter(round(value, digits) for value in values)
    score, count = counts.most_common(1)[0]
    return float(score), int(count), float(count / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V3-lite repeated-score / zero-signal risk")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    args = parser.parse_args()

    rows = load_training_rows(args.rows)
    model = V3LiteCandidateModel()

    fls_names = tuple(name for name in MODEL_FEATURE_NAMES if name in model.feature_names)
    revenue_names = tuple(name for name in REVENUE_SURPRISE_FEATURE_NAMES if name in model.feature_names)

    raw_scores: list[float] = []
    submitted_scores: list[float] = []
    zero_model_rows = 0
    zero_fls_rows = 0
    zero_revenue_rows = 0
    zero_fls_and_revenue_rows = 0

    by_quarter: dict[str, dict[str, object]] = defaultdict(
        lambda: {"rows": 0, "zero_model": 0, "raw": [], "submitted": []}
    )

    for row in rows:
        values = row.values
        raw = _raw(model, values)
        submitted = model.calibrator.calibrate(raw)
        raw_scores.append(raw)
        submitted_scores.append(submitted)

        model_zero = all(abs(float(values[name])) <= 1e-12 for name in model.feature_names)
        fls_zero = all(abs(float(values[name])) <= 1e-12 for name in fls_names)
        revenue_zero = all(abs(float(values[name])) <= 1e-12 for name in revenue_names)

        zero_model_rows += int(model_zero)
        zero_fls_rows += int(fls_zero)
        zero_revenue_rows += int(revenue_zero)
        zero_fls_and_revenue_rows += int(fls_zero and revenue_zero)

        quarter = by_quarter[row.quarter]
        quarter["rows"] = int(quarter["rows"]) + 1
        quarter["zero_model"] = int(quarter["zero_model"]) + int(model_zero)
        quarter["raw"].append(raw)
        quarter["submitted"].append(submitted)

    n = len(rows)
    print("=== V3-LITE COLLAPSE-RISK AUDIT ===")
    print(f"model: {model.model_version}")
    print(f"ablation: {model.ablation}")
    print(f"rows: {n}")
    print(f"features: {len(model.feature_names)}")
    print(f"fls_features_in_model: {len(fls_names)}")
    print(f"revenue_features_in_model: {len(revenue_names)}")
    print()
    print("SIGNAL AVAILABILITY")
    print(f"all_model_features_zero: {zero_model_rows}/{n} ({zero_model_rows / n:.3%})")
    print(f"all_fls_features_zero: {zero_fls_rows}/{n} ({zero_fls_rows / n:.3%})")
    print(f"all_revenue_features_zero: {zero_revenue_rows}/{n} ({zero_revenue_rows / n:.3%})")
    print(
        "fls_and_revenue_both_zero: "
        f"{zero_fls_and_revenue_rows}/{n} ({zero_fls_and_revenue_rows / n:.3%})"
    )
    print()

    print("SCORE CONCENTRATION")
    print(f"unique_raw_exact: {len(set(raw_scores))}")
    print(f"unique_submitted_exact: {len(set(submitted_scores))}")
    for digits in (4, 3, 2):
        raw_score, raw_count, raw_frac = _largest_bucket(raw_scores, digits)
        sub_score, sub_count, sub_frac = _largest_bucket(submitted_scores, digits)
        print(
            f"raw_largest_bucket_{digits}dp: score={raw_score:.{digits}f} "
            f"count={raw_count} fraction={raw_frac:.3%}"
        )
        print(
            f"submitted_largest_bucket_{digits}dp: score={sub_score:.{digits}f} "
            f"count={sub_count} fraction={sub_frac:.3%}"
        )
    near_half = sum(abs(value - 0.5) <= 0.02 for value in submitted_scores)
    print(f"submitted_between_0.48_0.52: {near_half}/{n} ({near_half / n:.3%})")
    print()

    print("TOP SUBMITTED BUCKETS (2dp)")
    for score, count in Counter(round(value, 2) for value in submitted_scores).most_common(10):
        print(f"  {score:.2f}: {count} ({count / n:.3%})")
    print()

    print("BY QUARTER")
    for quarter_name in sorted(by_quarter):
        quarter = by_quarter[quarter_name]
        qn = int(quarter["rows"])
        qzero = int(quarter["zero_model"])
        score, count, frac = _largest_bucket(list(quarter["submitted"]), 2)
        print(
            f"  {quarter_name}: rows={qn} all_features_zero={qzero / qn:.3%} "
            f"largest_submitted_2dp={score:.2f} ({count}/{qn}, {frac:.3%})"
        )

    print()
    print("=== END AUDIT ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
