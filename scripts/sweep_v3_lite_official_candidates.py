"""Read-only V3-lite candidate sweep ranked by the official Delta-R^2 metric.

This script NEVER writes the production artifact. It is a research comparison
that answers: among live-safe feature families and model forms, which candidates
add the most validation explanatory power beyond the earnings-surprise benchmark?
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from explaining_markets.calibration import PercentileCalibrator
from explaining_markets.competition_scoring import score_complete_predictions
from explaining_markets.constrained_linear import fit_sign_constrained_ridge
from explaining_markets.feature_families.earnings_surprise import EARNINGS_SURPRISE_FEATURE_NAMES
from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.feature_families.revenue_results import REVENUE_SURPRISE_FEATURE_NAMES
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.v3_lite_training import RIDGE_ALPHAS, _active_features, candidate_specs, fit_predict, metric_block
from explaining_markets.v3_training import TRAIN_QUARTER, VALIDATION_QUARTER
from explaining_markets.v3_training_data import load_training_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"
DEFAULT_BASE_ROWS = ROOT / "data" / "processed" / "v3_training_rows.jsonl.gz"

EPS_DIRECTIONAL_FEATURES = tuple(
    name for name in EARNINGS_SURPRISE_FEATURE_NAMES
    if name not in {"reported_eps", "consensus_eps", "eps_surprise_absolute"}
)
REVENUE_DIRECTIONAL_FEATURES = tuple(
    name for name in REVENUE_SURPRISE_FEATURE_NAMES
    if name not in {"reported_revenue", "consensus_revenue", "revenue_surprise_absolute"}
)
CORE_RESULT_FEATURES = (
    "eps_surprise_percent", "has_eps_surprise",
    "revenue_surprise_percent", "has_revenue_surprise",
)

FEATURE_SETS = {
    "v1_fls_only": MODEL_FEATURE_NAMES,
    "fls_plus_core_results": (*MODEL_FEATURE_NAMES, *CORE_RESULT_FEATURES),
    "fls_plus_core_results_reasoning": (*MODEL_FEATURE_NAMES, *CORE_RESULT_FEATURES, *REASONING_FEATURE_NAMES),
    "fls_plus_eps": (*MODEL_FEATURE_NAMES, *EPS_DIRECTIONAL_FEATURES),
    "fls_plus_revenue": (*MODEL_FEATURE_NAMES, *REVENUE_DIRECTIONAL_FEATURES),
    "fls_plus_results": (*MODEL_FEATURE_NAMES, *EPS_DIRECTIONAL_FEATURES, *REVENUE_DIRECTIONAL_FEATURES),
    "fls_plus_reasoning": (*MODEL_FEATURE_NAMES, *REASONING_FEATURE_NAMES),
    "fls_plus_results_reasoning": (*MODEL_FEATURE_NAMES, *EPS_DIRECTIONAL_FEATURES, *REVENUE_DIRECTIONAL_FEATURES, *REASONING_FEATURE_NAMES),
}


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
        [pred for pred, _row in selected],
        [float(row.target_percentile) for _pred, row in selected],
        [float(row.surprise_percentile) for _pred, row in selected],
    )


def _rank_key(row: dict) -> tuple[float, float, float]:
    delta = row["submitted_official"].get("delta_r_squared")
    spear = row["ordinary_metrics"].get("spearman")
    mae = row["ordinary_metrics"].get("mae")
    return (
        -math.inf if delta is None else float(delta),
        -math.inf if spear is None else float(spear),
        -math.inf if mae is None else -float(mae),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    path = args.rows or _default_rows()
    rows = load_training_rows(path)
    train = [row for row in rows if row.quarter == TRAIN_QUARTER]
    validation = [row for row in rows if row.quarter == VALIDATION_QUARTER]
    if not train or not validation:
        raise SystemExit("requires 2025Q4 train and 2026Q1 validation")

    candidates: list[dict] = []
    for ablation, requested in FEATURE_SETS.items():
        active = _active_features(train + validation, requested)
        if not active:
            continue
        fits = []
        for kind, params in candidate_specs(include_nonlinear=False):
            fits.append(fit_predict(train, validation, active, kind, params))
        for alpha in RIDGE_ALPHAS:
            fits.append(fit_sign_constrained_ridge(train, validation, active, alpha=float(alpha)))

        for fit in fits:
            raw = [float(x) for x in fit.predictions]
            calibrator = PercentileCalibrator.fit(
                raw,
                source=f"{VALIDATION_QUARTER} validation predictions from {fit.kind} fitted on {TRAIN_QUARTER} only (ablation={ablation})",
            )
            submitted = calibrator.calibrate_many(raw)
            candidates.append({
                "ablation": ablation,
                "kind": fit.kind,
                "params": dict(fit.params),
                "features": len(active),
                "ordinary_metrics": metric_block(raw, validation),
                "raw_official": _official(raw, validation),
                "submitted_official": _official(submitted, validation),
            })

    candidates.sort(key=_rank_key, reverse=True)
    print("=== V3-LITE OFFICIAL-METRIC CANDIDATE SWEEP ===")
    print(f"rows: {path}")
    print(f"train={len(train)} validation={len(validation)} candidates={len(candidates)}")
    print("ranking: submitted validation Delta-R2, then Spearman, then MAE")
    print()
    print(f"{'rank':>4} {'ablation':<34} {'model':<24} {'DeltaR2 sub':>12} {'DeltaR2 raw':>12} {'beta':>9} {'spear':>9} {'mae':>8}")
    for index, row in enumerate(candidates[: max(1, args.top)], 1):
        params = ",".join(f"{k}={v}" for k, v in row["params"].items()) or "-"
        model = f"{row['kind']} {params}"
        sub = row["submitted_official"].get("delta_r_squared")
        raw = row["raw_official"].get("delta_r_squared")
        beta = row["submitted_official"].get("beta")
        spear = row["ordinary_metrics"].get("spearman")
        mae = row["ordinary_metrics"].get("mae")
        def f(value): return "n/a" if value is None else f"{float(value):.6f}"
        print(
            f"{index:>4} {row['ablation']:<34} {model:<24} {f(sub):>12} {f(raw):>12} "
            f"{f(beta):>9} {f(spear):>9} {f(mae):>8}"
        )

    if candidates:
        winner = candidates[0]
        print("\nBEST BY OFFICIAL VALIDATION OBJECTIVE")
        print(f"ablation: {winner['ablation']}")
        print(f"model: {winner['kind']} {winner['params']}")
        print(f"features: {winner['features']}")
        print(f"submitted Delta R2: {winner['submitted_official'].get('delta_r_squared')}")
        print(f"raw Delta R2: {winner['raw_official'].get('delta_r_squared')}")
        print(f"prediction beta: {winner['submitted_official'].get('beta')}")
        print("\nThis script is read-only. Do not promote the winner until it also passes the realized-disclosure live gate and the existing point-in-time/tests/deployment checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
