#!/usr/bin/env python3
"""Build an explicitly operator-selected, live-safe V3-lite candidate.

The normal V3 promotion gate remains untouched.  This emergency builder uses
chronological validation for model selection and a separate production-realism
constraint derived from the live failure we observed: realized negative,
neutral and positive disclosure facts must remain distinguishable when the
legacy FLS block is zero.

The live-realism scenarios are a veto only.  They are not used as training
labels or as an optimization objective; among candidates that satisfy the
invariant, chronological validation Spearman remains the selection criterion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from explaining_markets.calibration import PercentileCalibrator
from explaining_markets.feature_families.earnings_surprise import EARNINGS_SURPRISE_FEATURE_NAMES
from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.feature_families.revenue_results import REVENUE_SURPRISE_FEATURE_NAMES
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.v3_lite_live_gate import evaluate_v3_lite_live_gate
from explaining_markets.v3_lite_operator import DEFAULT_OPERATOR_ARTIFACT, serialize_operator_candidate
from explaining_markets.v3_lite_training import (
    _active_features,
    candidate_specs,
    evaluate_v3_lite,
    fit_predict,
    metric_block,
)
from explaining_markets.v3_training import TRAIN_QUARTER, VALIDATION_QUARTER
from explaining_markets.v3_training_data import load_training_rows, training_data_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_ROWS = ROOT / "data" / "processed" / "v3_training_rows_enriched.jsonl.gz"
DEFAULT_BASE_ROWS = ROOT / "data" / "processed" / "v3_training_rows.jsonl.gz"

# Raw reported/consensus amounts are intentionally excluded from the emergency
# live sets.  A disclosure that says "beat consensus by 12%" is represented by
# a normalized 1.12/1.00 pair, whereas vendor records may contain dollar values.
# The scale-invariant surprise/direction fields below have identical semantics
# across both sources.
EPS_DIRECTIONAL_FEATURES = tuple(
    name
    for name in EARNINGS_SURPRISE_FEATURE_NAMES
    if name not in {"reported_eps", "consensus_eps", "eps_surprise_absolute"}
)
REVENUE_DIRECTIONAL_FEATURES = tuple(
    name
    for name in REVENUE_SURPRISE_FEATURE_NAMES
    if name not in {"reported_revenue", "consensus_revenue", "revenue_surprise_absolute"}
)

LIVE_CANDIDATE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "fls_plus_eps": (*MODEL_FEATURE_NAMES, *EPS_DIRECTIONAL_FEATURES),
    "fls_plus_revenue": (*MODEL_FEATURE_NAMES, *REVENUE_DIRECTIONAL_FEATURES),
    "fls_plus_results": (
        *MODEL_FEATURE_NAMES,
        *EPS_DIRECTIONAL_FEATURES,
        *REVENUE_DIRECTIONAL_FEATURES,
    ),
    "fls_plus_reasoning": (*MODEL_FEATURE_NAMES, *REASONING_FEATURE_NAMES),
    "fls_plus_results_reasoning": (
        *MODEL_FEATURE_NAMES,
        *EPS_DIRECTIONAL_FEATURES,
        *REVENUE_DIRECTIONAL_FEATURES,
        *REASONING_FEATURE_NAMES,
    ),
}


def _default_rows() -> Path:
    return DEFAULT_ENRICHED_ROWS if DEFAULT_ENRICHED_ROWS.exists() else DEFAULT_BASE_ROWS


def _score(metrics: dict) -> tuple[float, float]:
    spear = metrics.get("spearman")
    return (-1e9 if spear is None else float(spear), -float(metrics["mae"]))


def _calibrator(fit, *, ablation: str) -> PercentileCalibrator:
    return PercentileCalibrator.fit(
        fit.predictions.tolist(),
        source=(
            f"{VALIDATION_QUARTER} validation predictions from {fit.kind} "
            f"fitted on {TRAIN_QUARTER} only (ablation={ablation})"
        ),
    )


def _temporary_runtime(
    rows,
    *,
    ablation: str,
    active: tuple[str, ...],
    kind: str,
    params: dict,
    calibrator: PercentileCalibrator,
    validation_metrics: dict,
    path: Path,
) -> V3LiteCandidateModel:
    serialize_operator_candidate(
        rows,
        feature_names=active,
        kind=kind,
        params=params,
        ablation=ablation,
        calibrator=calibrator,
        validation_metrics=validation_metrics,
        legacy_metrics=None,
        artifact_path=path,
        operator_reason="temporary pre-serialization live-realism gate",
    )
    return V3LiteCandidateModel(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build operator-selected V3-lite live candidate")
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OPERATOR_ARTIFACT)
    parser.add_argument("--ablation", choices=tuple(LIVE_CANDIDATE_FEATURE_SETS), default=None)
    parser.add_argument("--min-disclosure-result-coverage", type=float, default=0.10)
    parser.add_argument("--min-live-spread", type=float, default=0.05)
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

    train = [row for row in rows if row.quarter == TRAIN_QUARTER]
    validation = [row for row in rows if row.quarter == VALIDATION_QUARTER]
    if not train or not validation:
        raise SystemExit("V3-lite candidate requires 2025Q4 train and 2026Q1 validation rows")

    # Use the established chronological study for an apples-to-apples V1
    # benchmark.  Custom live-safe feature sets below are evaluated on the same
    # train/validation split and with the same linear hyperparameter grid.
    research, _, _ = evaluate_v3_lite(rows, include_nonlinear=False)
    v1_metrics = research["ablations"]["v1_fls_only"]["selected"]["metrics"]
    v1_spear = v1_metrics.get("spearman")
    if v1_spear is None:
        raise SystemExit("V1 validation Spearman is unavailable")

    feature_sets = LIVE_CANDIDATE_FEATURE_SETS
    if args.ablation:
        feature_sets = {args.ablation: LIVE_CANDIDATE_FEATURE_SETS[args.ablation]}

    candidates: list[dict] = []
    for ablation, requested_names in feature_sets.items():
        active = _active_features(train + validation, requested_names)
        if not active:
            continue
        for kind, params in candidate_specs(include_nonlinear=False):
            fit = fit_predict(train, validation, active, kind, params)
            metrics = metric_block(fit.predictions.tolist(), validation)
            spear = metrics.get("spearman")
            if spear is None or float(spear) <= float(v1_spear):
                continue
            candidates.append(
                {
                    "ablation": ablation,
                    "active": active,
                    "kind": kind,
                    "params": dict(params),
                    "fit": fit,
                    "metrics": metrics,
                    "calibrator": _calibrator(fit, ablation=ablation),
                }
            )

    if not candidates:
        raise SystemExit(
            "refusing operator V3-lite artifact: no live-safe directional candidate "
            f"beats V1 validation Spearman ({v1_spear})"
        )

    candidates.sort(key=lambda item: _score(item["metrics"]), reverse=True)
    chosen = None
    rejected: list[dict] = []
    with TemporaryDirectory(prefix="v3_lite_live_gate_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, candidate in enumerate(candidates):
            temp_path = temp_root / f"candidate_{index}.json"
            runtime = _temporary_runtime(
                rows,
                ablation=candidate["ablation"],
                active=candidate["active"],
                kind=candidate["kind"],
                params=candidate["params"],
                calibrator=candidate["calibrator"],
                validation_metrics=candidate["metrics"],
                path=temp_path,
            )
            gate = evaluate_v3_lite_live_gate(
                runtime, min_submitted_spread=args.min_live_spread
            )
            candidate["live_gate"] = gate
            if gate.passed:
                chosen = candidate
                break
            rejected.append(candidate)

    if chosen is None:
        print("=== V3-LITE LIVE-GATE REJECTIONS ===")
        for candidate in rejected[:12]:
            gate = candidate["live_gate"]
            print(
                f"{candidate['ablation']} {candidate['kind']} {candidate['params']} "
                f"validation_spearman={candidate['metrics'].get('spearman')} "
                f"ordered={gate.ordered} spread={gate.submitted_spread:.4f}"
            )
        raise SystemExit(
            "refusing operator V3-lite artifact: every validation-improving directional "
            "candidate failed the realized-disclosure live gate"
        )

    gate = chosen["live_gate"]
    path = serialize_operator_candidate(
        rows,
        feature_names=chosen["active"],
        kind=chosen["kind"],
        params=chosen["params"],
        ablation=chosen["ablation"],
        calibrator=chosen["calibrator"],
        validation_metrics=chosen["metrics"],
        legacy_metrics=None,
        artifact_path=args.output,
        operator_reason=(
            "User-authorized 2026-08-19 production switch after live V1 received non-empty "
            "disclosures but produced all-zero FLS vectors and identical 0.4946 raw scores. "
            "Historical rows were refreshed with the point-in-time realized-disclosure parser. "
            "Candidate selection required both improved chronological validation Spearman over "
            "V1 and a separate negative<neutral<positive live-realism gate with FLS forced to zero."
        ),
    )

    print("=== V3-LITE OPERATOR CANDIDATE ===")
    print(f"rows: {rows_path}")
    print(f"eps_coverage: {eps_cov:.3f}")
    print(f"revenue_coverage: {rev_cov:.3f}")
    print(f"ablation: {chosen['ablation']}")
    print(f"model: {chosen['kind']} {chosen['params']}")
    print(f"features: {len(chosen['active'])}")
    print(f"v1_validation_spearman: {v1_spear}")
    print(f"validation_spearman: {chosen['metrics'].get('spearman')}")
    print(f"validation_pearson: {chosen['metrics'].get('pearson')}")
    print(f"calibration: {chosen['calibrator'].version}")
    print("selection_policy: validation improvement + realized-disclosure live gate")
    print("live_gate:")
    for scenario in gate.scenarios:
        print(
            f"  {scenario.label:<8} eps={scenario.eps_surprise:+.3f} "
            f"revenue={scenario.revenue_surprise:+.3f} "
            f"raw={scenario.raw:.4f} submitted={scenario.submitted:.4f}"
        )
    print(f"  parsed_ok: {gate.parsed_ok}")
    print(f"  zero_fls: {gate.zero_fls}")
    print(f"  ordered: {gate.ordered}")
    print(f"  submitted_spread: {gate.submitted_spread:.4f}")
    print("normal_promotion_gate: NOT PASSED (untouched holdout unavailable)")
    print("operator_override: ENABLED AND RECORDED")
    print(f"artifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
