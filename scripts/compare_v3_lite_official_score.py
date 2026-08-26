"""Chronologically compare the production V3-lite *spec* with a V1 baseline.

Important: the deployed operator artifact is refit on 2025Q4+2026Q1+2026Q2.
Scoring those fitted coefficients back on Q1/Q2 would be in-sample and invalid.
This script therefore reads only the production artifact's selected
ablation/model/hyperparameters, then reconstructs them chronologically:

  2025Q4 -> fit -> 2026Q1 validation
  2025Q4+2026Q1 -> fit -> 2026Q2 legacy evaluation

The empirical-CDF calibrator is always fit on Q1 predictions produced by the
2025Q4-only model, so Q2 never enters calibration.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from explaining_markets.calibration import PercentileCalibrator
from explaining_markets.competition_scoring import score_complete_predictions
from explaining_markets.constrained_linear import fit_sign_constrained_ridge
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.v3_lite_training import RIDGE_ALPHAS, _active_features, candidate_specs, fit_predict
from explaining_markets.v3_training import LEGACY_HOLDOUT_QUARTER, TRAIN_QUARTER, VALIDATION_QUARTER
from explaining_markets.v3_training_data import load_training_rows

# Keep this in sync with the read-only official sweep. Importing from scripts is
# intentional here: both are operator/research entry points, not production code.
from sweep_v3_lite_official_candidates import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"
DEFAULT_BASE_ROWS = ROOT / "data" / "processed" / "v3_training_rows.jsonl.gz"


def _default_rows() -> Path:
    return DEFAULT_ENRICHED_ROWS if DEFAULT_ENRICHED_ROWS.exists() else DEFAULT_BASE_ROWS


def _official(predictions, rows) -> dict:
    selected = [
        (float(pred), row)
        for pred, row in zip(predictions, rows, strict=True)
        if row.surprise_percentile is not None
    ]
    if len(selected) < 3:
        return {"n": len(selected), "delta_r_squared": None, "r_squared": None, "r_squared_surprise": None, "beta": None}
    return score_complete_predictions(
        [p for p, _r in selected],
        [float(r.target_percentile) for _p, r in selected],
        [float(r.surprise_percentile) for _p, r in selected],
    )


def _fit(train, evaluate, names, kind: str, params: dict):
    active = _active_features(list(train) + list(evaluate), names)
    if kind == "constrained_ridge":
        return fit_sign_constrained_ridge(
            train, evaluate, active, alpha=float(params["alpha"])
        )
    return fit_predict(train, evaluate, active, kind, params)


def _v1_validation_candidates(train, validation):
    active = _active_features(list(train) + list(validation), MODEL_FEATURE_NAMES)
    fits = [fit_predict(train, validation, active, kind, params) for kind, params in candidate_specs(include_nonlinear=False)]
    # Constrained ridge is equivalent to ordinary constrained fitting here
    # because the FLS-only feature names have no semantic sign constraints, but
    # include it for symmetry with the V3 sweep.
    fits.extend(
        fit_sign_constrained_ridge(train, validation, active, alpha=float(alpha))
        for alpha in RIDGE_ALPHAS
    )
    out = []
    for fit in fits:
        raw = [float(x) for x in fit.predictions]
        calibrator = PercentileCalibrator.fit(
            raw,
            source=f"{VALIDATION_QUARTER} OOS V1 predictions fitted on {TRAIN_QUARTER}",
        )
        submitted = calibrator.calibrate_many(raw)
        out.append((fit, calibrator, _official(raw, validation), _official(submitted, validation)))
    return out


def _score_key(item) -> tuple[float, float]:
    fit, _cal, raw, submitted = item
    delta = submitted.get("delta_r_squared")
    raw_delta = raw.get("delta_r_squared")
    return (
        -math.inf if delta is None else float(delta),
        -math.inf if raw_delta is None else float(raw_delta),
    )


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"


def _print_row(name: str, raw: dict, submitted: dict) -> None:
    print(
        f"{name:<28}{_fmt(submitted.get('delta_r_squared')):>14}"
        f"{_fmt(raw.get('delta_r_squared')):>14}"
        f"{_fmt(submitted.get('r_squared')):>14}"
        f"{_fmt(submitted.get('r_squared_surprise')):>14}"
        f"{_fmt(submitted.get('beta')):>14}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=None)
    args = parser.parse_args()
    path = args.rows or _default_rows()
    rows = load_training_rows(path)
    train = [r for r in rows if r.quarter == TRAIN_QUARTER]
    validation = [r for r in rows if r.quarter == VALIDATION_QUARTER]
    legacy = [r for r in rows if r.quarter == LEGACY_HOLDOUT_QUARTER]
    if not train or not validation:
        raise SystemExit("requires 2025Q4 train and 2026Q1 validation")

    production = V3LiteCandidateModel()
    ablation = production.ablation
    if ablation not in FEATURE_SETS:
        raise SystemExit(f"production ablation is not in research feature sets: {ablation}")
    kind = str(production.training_metadata.get("selected_kind") or "")
    params = dict(production.training_metadata.get("selected_params") or {})
    if not kind:
        raise SystemExit("production artifact has no selected_kind metadata")

    # Production V3 spec: fit on Q4 only, predict Q1.
    v3_val_fit = _fit(train, validation, FEATURE_SETS[ablation], kind, params)
    v3_val_raw = [float(x) for x in v3_val_fit.predictions]
    v3_cal = PercentileCalibrator.fit(
        v3_val_raw,
        source=f"{VALIDATION_QUARTER} OOS production-spec predictions fitted on {TRAIN_QUARTER}",
    )
    v3_val_sub = v3_cal.calibrate_many(v3_val_raw)
    v3_val_raw_score = _official(v3_val_raw, validation)
    v3_val_sub_score = _official(v3_val_sub, validation)

    # Select the V1 reference by the same official submitted validation metric.
    v1_candidates = _v1_validation_candidates(train, validation)
    v1_fit, v1_cal, v1_val_raw_score, v1_val_sub_score = max(v1_candidates, key=_score_key)

    print("=== CHRONOLOGICAL OFFICIAL-METRIC MODEL COMPARISON ===")
    print(f"rows: {path}")
    print(f"train: {TRAIN_QUARTER} n={len(train)}")
    print(f"validation: {VALIDATION_QUARTER} n={len(validation)}")
    print(f"production spec: {ablation} {kind} {params}")
    print(f"V1 reference: {v1_fit.kind} {v1_fit.params}")
    print("metric: Delta R-squared above earnings-surprise percentile benchmark")
    print("NOTE: coefficients below are reconstructed chronologically; deployed refit coefficients are NOT scored on their own training quarters.")

    print(f"\n=== {VALIDATION_QUARTER} OUT-OF-SAMPLE VALIDATION ===")
    print(f"{'model':<28}{'DeltaR2 sub':>14}{'DeltaR2 raw':>14}{'full R2 sub':>14}{'surprise R2':>14}{'beta pred':>14}")
    _print_row("v1_best_official", v1_val_raw_score, v1_val_sub_score)
    _print_row("production_v3_spec", v3_val_raw_score, v3_val_sub_score)
    if v3_val_sub_score.get("delta_r_squared") is not None and v1_val_sub_score.get("delta_r_squared") is not None:
        print(
            "submitted V3 gain vs V1: "
            f"{float(v3_val_sub_score['delta_r_squared']) - float(v1_val_sub_score['delta_r_squared']):+.6f}"
        )
    if v3_val_raw_score.get("delta_r_squared") is not None and v3_val_sub_score.get("delta_r_squared") is not None:
        print(
            "V3 CDF effect: "
            f"{float(v3_val_sub_score['delta_r_squared']) - float(v3_val_raw_score['delta_r_squared']):+.6f}"
        )

    if legacy:
        development = train + validation
        v3_legacy_fit = _fit(development, legacy, FEATURE_SETS[ablation], kind, params)
        v3_legacy_raw = [float(x) for x in v3_legacy_fit.predictions]
        v3_legacy_sub = v3_cal.calibrate_many(v3_legacy_raw)
        v3_legacy_raw_score = _official(v3_legacy_raw, legacy)
        v3_legacy_sub_score = _official(v3_legacy_sub, legacy)

        v1_legacy_fit = _fit(development, legacy, MODEL_FEATURE_NAMES, v1_fit.kind, v1_fit.params)
        v1_legacy_raw = [float(x) for x in v1_legacy_fit.predictions]
        v1_legacy_sub = v1_cal.calibrate_many(v1_legacy_raw)
        v1_legacy_raw_score = _official(v1_legacy_raw, legacy)
        v1_legacy_sub_score = _official(v1_legacy_sub, legacy)

        print(f"\n=== {LEGACY_HOLDOUT_QUARTER} CHRONOLOGICAL LEGACY READ (NOT PRISTINE) ===")
        print(f"{'model':<28}{'DeltaR2 sub':>14}{'DeltaR2 raw':>14}{'full R2 sub':>14}{'surprise R2':>14}{'beta pred':>14}")
        _print_row("v1_best_official", v1_legacy_raw_score, v1_legacy_sub_score)
        _print_row("production_v3_spec", v3_legacy_raw_score, v3_legacy_sub_score)

    print("\nInterpretation:")
    print("- Q1 is the valid chronological model-selection comparison.")
    print("- Q2 is chronological but not pristine because prior research already inspected it.")
    print("- CDF calibration is nonlinear, so it can change Delta R2; raw vs submitted should both be tracked.")
    print("- Do not evaluate the deployed refit artifact on Q1/Q2: those quarters are in its fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
