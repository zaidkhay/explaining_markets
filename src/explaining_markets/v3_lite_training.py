"""V3-lite: find the strongest honestly-validated subset of AVAILABLE families.

Motivation
----------
Full V3 assumes EPS/revenue/guidance/peer/news coverage that historical
enrichment has not yet reached (see the family-coverage report). Waiting for
100% coverage before modelling would waste the sealed quarters we already
have, so this module evaluates ablations over the families that are actually
populated, and represents availability explicitly rather than dropping rows.

Chronology and holdout honesty
------------------------------
    2025Q4 -> TRAIN
    2026Q1 -> VALIDATION (model + hyperparameter selection)
    2026Q2 -> EVALUATION, explicitly NOT pristine

2026Q2 has already guided earlier research decisions in this repository (V2 and
V3 work both read it), so it is reported as a *legacy* evaluation quarter and
can never satisfy the promotion gate on its own. ``HONEST_HOLDOUT_QUARTER``
(2026Q3) is the first untouched split; until those outcomes exist, promotion of
a new production ranking model is deliberately impossible.

Calibration is fitted only on VALIDATION predictions produced by a model fitted
on TRAIN alone, so no row contributes to both the model and its calibration.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

from explaining_markets.calibration import PercentileCalibrator, spearman
from explaining_markets.dispersion_diagnostics import dispersion_report
from explaining_markets.feature_families.company_history_v3 import COMPANY_HISTORY_V3_FEATURE_NAMES
from explaining_markets.feature_families.earnings_surprise import EARNINGS_SURPRISE_FEATURE_NAMES
from explaining_markets.feature_families.news import NEWS_FEATURE_NAMES
from explaining_markets.feature_families.price_context import PRICE_CONTEXT_FEATURE_NAMES
from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.features_v3 import MODEL_FEATURE_NAMES_V3
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.point_in_time_audit_v3 import audit_feature_names
from explaining_markets.v3_training import (
    HONEST_HOLDOUT_QUARTER,
    LEGACY_HOLDOUT_QUARTER,
    TRAIN_QUARTER,
    VALIDATION_QUARTER,
    V3TrainingRow,
    surprise_incremental_r2,
)

CLIP_BOUNDS = (0.05, 0.95)
RIDGE_ALPHAS = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
ELASTIC_SPECS = tuple(
    {"alpha": a, "l1_ratio": r} for a in (0.0005, 0.001, 0.005) for r in (0.1, 0.5)
)
MODEL_VERSION = "v3_lite"
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "data" / "processed" / "v3_lite_evaluation.json"

# Availability indicators: always retained so "missing" stays explicit.
AVAILABILITY_NAMES = tuple(
    name for name in MODEL_FEATURE_NAMES_V3 if name.startswith("has_")
)

ABLATIONS: dict[str, tuple[str, ...]] = {
    "v1_fls_only": MODEL_FEATURE_NAMES,
    "fls_plus_company_history": (*MODEL_FEATURE_NAMES, *COMPANY_HISTORY_V3_FEATURE_NAMES),
    "fls_plus_price": (*MODEL_FEATURE_NAMES, *PRICE_CONTEXT_FEATURE_NAMES),
    "fls_plus_history_price": (
        *MODEL_FEATURE_NAMES, *COMPANY_HISTORY_V3_FEATURE_NAMES, *PRICE_CONTEXT_FEATURE_NAMES,
    ),
    "fls_plus_reasoning": (*MODEL_FEATURE_NAMES, *REASONING_FEATURE_NAMES),
    "fls_plus_history_reasoning": (
        *MODEL_FEATURE_NAMES, *COMPANY_HISTORY_V3_FEATURE_NAMES, *REASONING_FEATURE_NAMES,
    ),
    "fls_history_price_reasoning": (
        *MODEL_FEATURE_NAMES, *COMPANY_HISTORY_V3_FEATURE_NAMES,
        *PRICE_CONTEXT_FEATURE_NAMES, *REASONING_FEATURE_NAMES,
    ),
    "fls_plus_eps": (*MODEL_FEATURE_NAMES, *EARNINGS_SURPRISE_FEATURE_NAMES),
    "fls_plus_news": (*MODEL_FEATURE_NAMES, *NEWS_FEATURE_NAMES),
    "fls_plus_availability": (*MODEL_FEATURE_NAMES, *AVAILABILITY_NAMES),
    "full_v3_available": MODEL_FEATURE_NAMES_V3,
}


def _active_features(rows: Sequence[V3TrainingRow], names: Sequence[str]) -> tuple[str, ...]:
    """Drop features that are constant across ALL rows.

    A family with zero coverage contributes a column of identical values; it
    carries no information, would be standardized by a zero std, and only adds
    regularisation noise. Dropping it is not the same as dropping rows.
    """
    if len(rows) < 2:
        return tuple(names)
    keep: list[str] = []
    for name in names:
        column = [float(r.values[name]) for r in rows]
        if max(column) - min(column) > 1e-12:
            keep.append(name)
    return tuple(keep)


@dataclass
class FitResult:
    kind: str
    params: dict
    predictions: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    model: object
    feature_names: tuple[str, ...]


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds <= 1e-12] = 1.0
    return means, stds


def fit_predict(
    train: Sequence[V3TrainingRow],
    evaluate: Sequence[V3TrainingRow],
    names: Sequence[str],
    kind: str,
    params: dict,
) -> FitResult:
    """Fit on ``train`` only and predict ``evaluate``. Never sees eval targets."""
    feature_names = tuple(names)
    X_train = np.asarray([r.x(feature_names) for r in train], dtype=float)
    y_train = np.asarray([r.target_percentile for r in train], dtype=float)
    X_eval = np.asarray([r.x(feature_names) for r in evaluate], dtype=float)
    means, stds = _standardize(X_train)
    Z_train, Z_eval = (X_train - means) / stds, (X_eval - means) / stds
    if kind == "ridge":
        model = Ridge(alpha=params["alpha"])
    elif kind == "elastic_net":
        model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=20000)
    elif kind == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, learning_rate=0.05, random_state=7
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.fit(Z_train, y_train)
    predictions = np.clip(model.predict(Z_eval), *CLIP_BOUNDS)
    return FitResult(
        kind=kind, params=dict(params), predictions=predictions,
        means=means, stds=stds, model=model, feature_names=feature_names,
    )


def candidate_specs(include_nonlinear: bool = True):
    for alpha in RIDGE_ALPHAS:
        yield "ridge", {"alpha": alpha}
    for spec in ELASTIC_SPECS:
        yield "elastic_net", dict(spec)
    if include_nonlinear:
        yield "hist_gradient_boosting", {}


def metric_block(
    predictions: Sequence[float],
    rows: Sequence[V3TrainingRow],
) -> dict:
    """Full metric set: correlation, error, dispersion and surprise increment."""
    p = np.asarray([float(x) for x in predictions], dtype=float)
    y = np.asarray([r.target_percentile for r in rows], dtype=float)
    pearson = None
    if len(p) >= 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
        pearson = float(np.corrcoef(p, y)[0, 1])
    dispersion = dispersion_report(p.tolist())
    return {
        "n": int(len(p)),
        "pearson": pearson,
        "spearman": spearman(p.tolist(), y.tolist()),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "dispersion": dispersion.as_dict(),
        "surprise_r2": surprise_incremental_r2(list(rows), p),
    }


def _score_key(block: dict) -> tuple[float, float]:
    """Selection score: Spearman first (ranking power), then negative MAE.

    Spearman is used because the competition target is a rank/percentile and
    Spearman is invariant to the calibration transform applied later.
    """
    spear = block.get("spearman")
    return (
        -math.inf if spear is None else float(spear),
        -float(block["mae"]),
    )


def coverage_buckets(rows: Sequence[V3TrainingRow]) -> dict[str, list[int]]:
    """Index rows by which data families are available (explicit missingness)."""
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        history = float(row.values.get("has_company_earnings_history", 0.0)) > 0
        price = float(row.values.get("has_5y_price_history", 0.0)) > 0
        reasoning = float(row.values.get("has_reasoning", 0.0)) > 0
        parts = ["fls"]
        if history:
            parts.append("history")
        if price:
            parts.append("price")
        if reasoning:
            parts.append("reasoning")
        buckets.setdefault("+".join(parts), []).append(index)
    return dict(sorted(buckets.items()))


def evaluate_v3_lite(
    rows: Sequence[V3TrainingRow],
    *,
    include_nonlinear: bool = True,
) -> dict:
    """Run the full chronological ablation + calibration study."""
    audit_feature_names()
    if any(r.leakage_violations for r in rows):
        raise ValueError("V3-lite refuses to train on rows with audit violations")

    by_quarter = {
        q: [r for r in rows if r.quarter == q]
        for q in (TRAIN_QUARTER, VALIDATION_QUARTER, LEGACY_HOLDOUT_QUARTER, HONEST_HOLDOUT_QUARTER)
    }
    train, validation = by_quarter[TRAIN_QUARTER], by_quarter[VALIDATION_QUARTER]
    legacy = by_quarter[LEGACY_HOLDOUT_QUARTER]
    honest = by_quarter[HONEST_HOLDOUT_QUARTER]
    if not train or not validation:
        raise RuntimeError("V3-lite requires 2025Q4 train and 2026Q1 validation rows")

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_legacy_evaluation": len(legacy),
        "n_honest_holdout": len(honest),
        "legacy_holdout_is_pristine": False,
        "legacy_holdout_note": (
            "2026Q2 already informed earlier V2/V3 research decisions in this "
            "repository; it is reported for continuity but cannot satisfy the "
            "promotion gate."
        ),
        "honest_holdout_quarter": HONEST_HOLDOUT_QUARTER,
        "honest_holdout_available": bool(honest),
        "ablations": {},
    }

    # ---- baselines ---------------------------------------------------
    constant = np.full(len(validation), 0.5)
    results["baselines"] = {
        "constant_0.5_validation": metric_block(constant.tolist(), validation),
    }
    surprise_rows = [r for r in validation if r.surprise_percentile is not None]
    if len(surprise_rows) >= 3:
        results["baselines"]["surprise_benchmark_validation"] = metric_block(
            [float(r.surprise_percentile) for r in surprise_rows], surprise_rows
        )

    # ---- ablation sweep on VALIDATION only ---------------------------
    best_name, best_block, best_fit = None, None, None
    for name, names in ABLATIONS.items():
        active = _active_features(train + validation, names)
        if not active:
            results["ablations"][name] = {"skipped": "no varying features available"}
            continue
        candidates = []
        for kind, params in candidate_specs(include_nonlinear):
            fit = fit_predict(train, validation, active, kind, params)
            block = metric_block(fit.predictions.tolist(), validation)
            candidates.append({"kind": kind, "params": params, "metrics": block, "_fit": fit})
        chosen = max(candidates, key=lambda c: _score_key(c["metrics"]))
        results["ablations"][name] = {
            "n_features_requested": len(names),
            "n_features_active": len(active),
            "dropped_constant_features": len(names) - len(active),
            "selected": {
                "kind": chosen["kind"],
                "params": chosen["params"],
                "metrics": chosen["metrics"],
            },
            "candidates": [
                {"kind": c["kind"], "params": c["params"], "metrics": c["metrics"]}
                for c in candidates
            ],
        }
        if best_block is None or _score_key(chosen["metrics"]) > _score_key(best_block):
            best_name, best_block, best_fit = name, chosen["metrics"], chosen["_fit"]

    if best_fit is None:
        raise RuntimeError("no V3-lite ablation produced a usable candidate")
    results["selected_ablation"] = best_name
    results["selected_model"] = {
        "kind": best_fit.kind,
        "params": best_fit.params,
        "feature_names": list(best_fit.feature_names),
        "validation_metrics": best_block,
    }

    # ---- calibration fitted on VALIDATION (out-of-sample) ------------
    calibrator = PercentileCalibrator.fit(
        best_fit.predictions.tolist(),
        source=(
            f"{VALIDATION_QUARTER} validation predictions from {best_fit.kind} "
            f"fitted on {TRAIN_QUARTER} only (ablation={best_name})"
        ),
    )
    calibrated_validation = calibrator.calibrate_many(best_fit.predictions.tolist())
    results["calibration"] = {
        "artifact": {k: v for k, v in calibrator.as_dict().items() if k != "knots"},
        "validation_raw": best_block,
        "validation_calibrated": metric_block(calibrated_validation, validation),
    }
    # The mid-rank CDF itself is exactly rank-preserving, but the final clamp
    # to ``bounds`` can create ties at the extremes, causing a tiny Spearman
    # drop. We tolerate up to 1e-4 degradation from clamping; anything larger
    # would indicate a non-monotonic transform or a fitting bug.
    raw_spear = results["calibration"]["validation_raw"]["spearman"]
    cal_spear = results["calibration"]["validation_calibrated"]["spearman"]
    results["calibration"]["preserves_ranking"] = (
        raw_spear is None
        or abs((cal_spear or 0.0) - (raw_spear or 0.0)) < 1e-4
    )

    # ---- legacy 2026Q2 evaluation (NOT pristine) ---------------------
    v1_names = _active_features(train + validation, MODEL_FEATURE_NAMES)
    v1_choice = results["ablations"]["v1_fls_only"]["selected"]
    if legacy:
        development = train + validation
        selected_legacy = fit_predict(
            development, legacy, best_fit.feature_names, best_fit.kind, best_fit.params
        )
        v1_legacy = fit_predict(
            development, legacy, v1_names, v1_choice["kind"], v1_choice["params"]
        )
        # Calibration for the legacy read is re-fitted out-of-sample on
        # validation predictions from a TRAIN-only model (already above).
        results["legacy_evaluation"] = {
            "selected_raw": metric_block(selected_legacy.predictions.tolist(), legacy),
            "selected_calibrated": metric_block(
                calibrator.calibrate_many(selected_legacy.predictions.tolist()), legacy
            ),
            "v1_raw": metric_block(v1_legacy.predictions.tolist(), legacy),
            "constant_0.5": metric_block([0.5] * len(legacy), legacy),
        }
        legacy_surprise = [r for r in legacy if r.surprise_percentile is not None]
        if len(legacy_surprise) >= 3:
            results["legacy_evaluation"]["surprise_benchmark"] = metric_block(
                [float(r.surprise_percentile) for r in legacy_surprise], legacy_surprise
            )

    # ---- coverage-bucket performance ---------------------------------
    buckets = coverage_buckets(validation)
    bucket_report = {}
    for label, indices in buckets.items():
        if len(indices) < 30:
            bucket_report[label] = {"n": len(indices), "skipped": "fewer than 30 rows"}
            continue
        subset = [validation[i] for i in indices]
        bucket_report[label] = {
            "n": len(indices),
            "selected_raw": metric_block(
                [float(best_fit.predictions[i]) for i in indices], subset
            ),
            "selected_calibrated": metric_block(
                [calibrated_validation[i] for i in indices], subset
            ),
        }
    results["coverage_buckets"] = bucket_report
    results["coverage_bucket_sizes"] = {k: len(v) for k, v in buckets.items()}
    return results, calibrator, best_fit


def write_report(results: dict, path: str | Path = DEFAULT_REPORT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return output


def format_summary(results: dict) -> str:
    lines = [
        "=== V3-LITE CHRONOLOGICAL EVALUATION ===",
        f"train={results['n_train']} validation={results['n_validation']} "
        f"legacy_2026Q2={results['n_legacy_evaluation']} honest_holdout={results['n_honest_holdout']}",
        f"honest holdout ({results['honest_holdout_quarter']}) available: "
        f"{results['honest_holdout_available']}",
        "",
        f"{'ablation':<32}{'model':<24}{'spearman':>10}{'pearson':>10}{'mae':>8}{'std':>8}{'near.5':>8}",
    ]
    for name, payload in results["ablations"].items():
        if "selected" not in payload:
            lines.append(f"{name:<32}{'skipped':<24}")
            continue
        sel = payload["selected"]
        m = sel["metrics"]
        d = m["dispersion"]
        spear = "n/a" if m["spearman"] is None else f"{m['spearman']:.4f}"
        pear = "n/a" if m["pearson"] is None else f"{m['pearson']:.4f}"
        params = ",".join(f"{k}={v}" for k, v in sel["params"].items()) or "-"
        lines.append(
            f"{name:<32}{sel['kind'] + ' ' + params:<24}{spear:>10}{pear:>10}"
            f"{m['mae']:>8.4f}{d['prediction_std']:>8.4f}{d['fraction_between_048_052']:>8.3f}"
        )
    lines.append("")
    lines.append(f"selected ablation: {results['selected_ablation']}")
    cal = results["calibration"]
    for label in ("validation_raw", "validation_calibrated"):
        m = cal[label]
        d = m["dispersion"]
        lines.append(
            f"  {label:<24} spearman={m['spearman']:.4f} pearson={m['pearson']:.4f} "
            f"mae={m['mae']:.4f} std={d['prediction_std']:.4f} "
            f"near0.5={d['fraction_between_048_052']:.3f} unique={d['unique_predictions_4dp']}"
        )
    lines.append(f"  calibration preserves ranking: {cal['preserves_ranking']}")
    if "legacy_evaluation" in results:
        lines.append("")
        lines.append("legacy 2026Q2 (NOT pristine):")
        for label, m in results["legacy_evaluation"].items():
            d = m["dispersion"]
            spear = "n/a" if m["spearman"] is None else f"{m['spearman']:.4f}"
            pear = "n/a" if m["pearson"] is None else f"{m['pearson']:.4f}"
            lines.append(
                f"  {label:<22} spearman={spear} pearson={pear} mae={m['mae']:.4f} "
                f"std={d['prediction_std']:.4f} near0.5={d['fraction_between_048_052']:.3f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Promotion gate and artifact serialization
# ---------------------------------------------------------------------------

V3_LITE_PROMOTION_GATE = {
    "min_validation_pearson_gain_over_v1": 0.01,
    "max_honest_holdout_pearson_regression": 0.01,
    "min_prediction_std": 0.01,
    "max_fraction_near_0_5": 0.90,
    "require_honest_holdout_available": True,
    "require_zero_leakage_violations": True,
    "require_calibration_preserves_ranking": True,
    "require_all_tests_passing": True,
    "require_local_feed_verified": True,
    "require_modal_feed_verified": True,
    "require_nonzero_news_coverage": True,
    "require_reasoning_valid": True,
    "require_latency_ok": True,
}

DEFAULT_V3_LITE_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "src" / "explaining_markets" / "artifacts" / "v3_lite.json"
)


def evaluate_promotion_gate(
    results: dict,
    *,
    tests_passing: bool = False,
    local_feed_verified: bool = False,
    modal_feed_verified: bool = False,
    news_coverage_nonzero: bool = False,
    reasoning_valid: bool = False,
    latency_ok: bool = False,
) -> dict:
    """Evaluate the predeclared promotion gate strictly.

    The gate CANNOT pass without an honest holdout (2026Q3). This is by design:
    2026Q2 is explicitly not pristine and cannot satisfy the gate alone.

    No gate threshold is loosened after observing results. If the candidate
    fails, ``promoted`` is ``False`` and the evidence is recorded honestly.
    """
    selected_metrics = results["selected_model"]["validation_metrics"]
    v1_metrics = results["ablations"]["v1_fls_only"]["selected"]["metrics"]
    validation_gain = None
    if selected_metrics["pearson"] is not None and v1_metrics["pearson"] is not None:
        validation_gain = selected_metrics["pearson"] - v1_metrics["pearson"]

    honest_available = results.get("honest_holdout_available", False)
    honest_regression = None
    if honest_available and "honest_holdout_evaluation" in results:
        honest = results["honest_holdout_evaluation"]
        v3_p = honest["selected_raw"]["pearson"]
        v1_p = honest["v1_raw"]["pearson"]
        if v3_p is not None and v1_p is not None:
            honest_regression = v1_p - v3_p

    cal_ok = results.get("calibration", {}).get("preserves_ranking", False)
    pred_std = selected_metrics["dispersion"]["prediction_std"]
    near_05 = selected_metrics["dispersion"]["fraction_between_048_052"]

    evidence = {
        "validation_pearson_gain_over_v1": validation_gain,
        "honest_holdout_available": honest_available,
        "honest_holdout_pearson_regression": honest_regression,
        "prediction_std": pred_std,
        "fraction_near_0_5": near_05,
        "calibration_preserves_ranking": cal_ok,
        "zero_leakage_violations": 0,
        "tests_passing": bool(tests_passing),
        "local_feed_verified": bool(local_feed_verified),
        "modal_feed_verified": bool(modal_feed_verified),
        "news_coverage_nonzero": bool(news_coverage_nonzero),
        "reasoning_valid": bool(reasoning_valid),
        "latency_ok": bool(latency_ok),
    }

    gate = V3_LITE_PROMOTION_GATE
    checks = {
        "validation_gain_ok": (
            validation_gain is not None
            and validation_gain >= gate["min_validation_pearson_gain_over_v1"]
        ),
        "honest_holdout_available": honest_available,
        "honest_holdout_regression_ok": (
            honest_regression is not None
            and honest_regression <= gate["max_honest_holdout_pearson_regression"]
        ),
        "prediction_std_ok": pred_std >= gate["min_prediction_std"],
        "near_05_ok": near_05 <= gate["max_fraction_near_0_5"],
        "calibration_ok": cal_ok,
        "tests_passing": bool(tests_passing),
        "local_feed_verified": bool(local_feed_verified),
        "modal_feed_verified": bool(modal_feed_verified),
        "news_coverage_nonzero": bool(news_coverage_nonzero),
        "reasoning_valid": bool(reasoning_valid),
        "latency_ok": bool(latency_ok),
    }
    promoted = all(checks.values())
    return {
        "gate": gate,
        "evidence": evidence,
        "checks": checks,
        "promoted": promoted,
    }


def serialize_v3_lite_artifact(
    rows: Sequence[V3TrainingRow],
    results: dict,
    calibrator: PercentileCalibrator,
    fit: FitResult,
    promotion: dict,
    artifact_path: str | Path = DEFAULT_V3_LITE_ARTIFACT,
) -> Path:
    """Serialize a V3-lite artifact ONLY if the promotion gate passed.

    Refuses to write if ``promotion['promoted']`` is ``False``. This prevents
    any manual override of the gate. The artifact includes calibration data
    so production inference can apply the same monotonic transform.
    """
    if not promotion.get("promoted"):
        raise RuntimeError(
            "refusing to serialize V3-lite artifact: promotion gate did not pass. "
            f"Failed checks: {[k for k, v in promotion['checks'].items() if not v]}"
        )
    if fit.kind not in {"ridge", "elastic_net"}:
        raise RuntimeError(
            "pure-Python production artifact currently supports linear V3-lite selections only"
        )

    # Refit on all development data (train + validation + legacy) for the
    # final production model. Honest holdout rows are NEVER included.
    development_quarters = {TRAIN_QUARTER, VALIDATION_QUARTER, LEGACY_HOLDOUT_QUARTER}
    dev_rows = [r for r in rows if r.quarter in development_quarters]
    X = np.asarray([r.x(fit.feature_names) for r in dev_rows], dtype=float)
    y = np.asarray([r.target_percentile for r in dev_rows], dtype=float)
    means, stds = _standardize(X)
    Z = (X - means) / stds
    if fit.kind == "ridge":
        model = Ridge(alpha=fit.params["alpha"]).fit(Z, y)
    else:
        model = ElasticNet(
            alpha=fit.params["alpha"], l1_ratio=fit.params["l1_ratio"], max_iter=20000
        ).fit(Z, y)

    artifact = {
        "model_version": MODEL_VERSION,
        "feature_spec_version": "v3_lite_v1",
        "ablation": results["selected_ablation"],
        "feature_names": list(fit.feature_names),
        "means": [float(x) for x in means],
        "standard_deviations": [float(x) for x in stds],
        "coefficients": [float(x) for x in model.coef_],
        "intercept": float(model.intercept_),
        "clip_bounds": list(CLIP_BOUNDS),
        "calibration": calibrator.as_dict(),
        "promoted": True,
        "structured_only": False,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_quarters": sorted(development_quarters),
        "training_metadata": {
            "n_development": len(dev_rows),
            "evaluation": results,
            "promotion": promotion,
        },
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path
