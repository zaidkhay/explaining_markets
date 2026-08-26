"""Compare deployed V3-lite against V1 using the official competition metric.

Reports the frozen Explaining Markets Delta-R^2 score on:
- 2026Q1 chronological validation
- 2026Q2 legacy evaluation (diagnostic only; not pristine)

For V3-lite it evaluates both the raw linear score and the actually submitted
empirical-CDF calibrated score. Because the competition metric is invariant to
uniform affine transforms but not arbitrary nonlinear transforms, this directly
measures whether the CDF calibration helps or hurts the leaderboard objective.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from explaining_markets.competition_scoring import score_complete_predictions
from explaining_markets.model import ForwardLookingRidgeModel
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.v3_training import LEGACY_HOLDOUT_QUARTER, VALIDATION_QUARTER
from explaining_markets.v3_training_data import load_training_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"
DEFAULT_BASE_ROWS = ROOT / "data" / "processed" / "v3_training_rows.jsonl.gz"


def _default_rows() -> Path:
    return DEFAULT_ENRICHED_ROWS if DEFAULT_ENRICHED_ROWS.exists() else DEFAULT_BASE_ROWS


def _linear_predict(row, *, names, means, stds, coefficients, intercept, low, high) -> float:
    raw = float(intercept) + sum(
        float(coef) * (float(row.values[name]) - float(mean)) / float(sd)
        for name, mean, sd, coef in zip(names, means, stds, coefficients, strict=True)
    )
    return float(max(low, min(high, raw)))


def _score(rows, predictions) -> dict:
    pairs = [
        (row, float(pred))
        for row, pred in zip(rows, predictions, strict=True)
        if row.surprise_percentile is not None
    ]
    if len(pairs) < 3:
        return {
            "n": len(pairs),
            "r_squared_surprise": None,
            "r_squared": None,
            "delta_r_squared": None,
            "beta": None,
            "beta_surprise": None,
            "alpha": None,
            "mse": None,
        }
    return score_complete_predictions(
        [pred for _row, pred in pairs],
        [float(row.target_percentile) for row, _pred in pairs],
        [float(row.surprise_percentile) for row, _pred in pairs],
    )


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"


def _report_period(label: str, rows, v3: V3LiteCandidateModel, v1: ForwardLookingRidgeModel) -> None:
    v3_raw = [
        _linear_predict(
            row,
            names=v3.feature_names,
            means=v3.means,
            stds=v3.standard_deviations,
            coefficients=v3.coefficients,
            intercept=v3.intercept,
            low=v3.clip_lower,
            high=v3.clip_upper,
        )
        for row in rows
    ]
    v3_cal = [v3.calibrator.calibrate(x) for x in v3_raw]
    v1_raw = [
        _linear_predict(
            row,
            names=v1.feature_names,
            means=v1.means,
            stds=v1.standard_deviations,
            coefficients=v1.coefficients,
            intercept=v1.intercept,
            low=v1.clip_lower,
            high=v1.clip_upper,
        )
        for row in rows
    ]

    blocks = {
        "v1_raw": _score(rows, v1_raw),
        "v3_raw": _score(rows, v3_raw),
        "v3_submitted_cdf": _score(rows, v3_cal),
    }
    print(f"\n=== {label} ===")
    print(f"rows total: {len(rows)}")
    print(f"rows with surprise benchmark: {blocks['v3_raw']['n']}")
    print(f"{'model':<20}{'Delta R2':>14}{'full R2':>14}{'surprise R2':>14}{'beta pred':>14}")
    for name, block in blocks.items():
        print(
            f"{name:<20}{_fmt(block['delta_r_squared']):>14}"
            f"{_fmt(block['r_squared']):>14}{_fmt(block['r_squared_surprise']):>14}"
            f"{_fmt(block['beta']):>14}"
        )

    raw_delta = blocks["v3_raw"]["delta_r_squared"]
    cal_delta = blocks["v3_submitted_cdf"]["delta_r_squared"]
    v1_delta = blocks["v1_raw"]["delta_r_squared"]
    if raw_delta is not None and cal_delta is not None:
        print(f"CDF effect on Delta R2: {float(cal_delta) - float(raw_delta):+.6f}")
    if cal_delta is not None and v1_delta is not None:
        print(f"submitted V3 gain vs V1: {float(cal_delta) - float(v1_delta):+.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=None)
    args = parser.parse_args()
    path = args.rows or _default_rows()
    rows = load_training_rows(path)
    v3 = V3LiteCandidateModel()
    v1 = ForwardLookingRidgeModel()

    print("=== OFFICIAL-METRIC MODEL COMPARISON ===")
    print(f"rows: {path}")
    print(f"V3 model: {v3.model_version} ({v3.ablation})")
    print(f"V3 calibration: {v3.calibrator.version} n={v3.calibrator.n_fitted}")
    print("metric: Delta R-squared above earnings-surprise percentile benchmark")

    validation = [row for row in rows if row.quarter == VALIDATION_QUARTER]
    legacy = [row for row in rows if row.quarter == LEGACY_HOLDOUT_QUARTER]
    _report_period(f"{VALIDATION_QUARTER} VALIDATION", validation, v3, v1)
    if legacy:
        _report_period(f"{LEGACY_HOLDOUT_QUARTER} LEGACY (NOT PRISTINE)", legacy, v3, v1)

    print("\nInterpretation:")
    print("- Higher Delta R2 is better; this is the competition objective.")
    print("- A positive prediction beta means the numeric percentile direction matches realized percentile direction.")
    print("- Uniform affine rescaling cannot improve Delta R2; only changing cross-event information/ranking structure can.")
    print("- 2026Q2 is diagnostic only because it was already used during research.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
