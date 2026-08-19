#!/usr/bin/env python3
"""Verify the disclosure-aware operator V3-lite artifact before deployment."""
from __future__ import annotations

from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.v3_lite_live_gate import evaluate_v3_lite_live_gate


def main() -> int:
    model = V3LiteCandidateModel()
    gate = evaluate_v3_lite_live_gate(
        model,
        min_submitted_spread=0.05,
        min_adjacent_submitted_gap=0.02,
    )

    reasoning_features = [
        (name, coef)
        for name, coef in zip(model.feature_names, model.coefficients, strict=True)
        if name in REASONING_FEATURE_NAMES and abs(coef) > 1e-12
    ]

    print("=== V3-LITE PRODUCTION CANDIDATE VERIFICATION ===")
    print(f"model: {model.model_version}")
    print(f"ablation: {model.ablation}")
    print(f"features: {len(model.feature_names)}")
    print(f"reasoning_features_with_nonzero_weight: {len(reasoning_features)}")
    print(f"operator_override: {model.operator_override}")
    print(f"promoted: {model.promoted}")
    print(f"calibration: {model.calibrator.version}")
    print(f"parser: {model.disclosure_parser_version}")
    print()
    print("REALIZED DISCLOSURE TEST (FLS intentionally zero)")
    for scenario in gate.scenarios:
        print(
            f"{scenario.label:<8} eps={scenario.eps_surprise:+.3f} "
            f"revenue={scenario.revenue_surprise:+.3f} "
            f"earnings_quality={scenario.earnings_quality:+.3f} "
            f"revenue_quality={scenario.revenue_quality:+.3f} "
            f"raw={scenario.raw:.4f} submitted={scenario.submitted:.4f}"
        )
    print(f"parsed_ok: {gate.parsed_ok}")
    print(f"zero_fls: {gate.zero_fls}")
    print(f"realized_ordered: {gate.ordered}")
    print(f"negative_to_neutral_gap: {gate.negative_neutral_gap:.4f}")
    print(f"neutral_to_positive_gap: {gate.neutral_positive_gap:.4f}")
    print(f"realized_submitted_spread: {gate.submitted_spread:.4f}")
    print()
    if reasoning_features:
        print("top reasoning weights:")
        for name, coef in sorted(
            reasoning_features, key=lambda item: abs(item[1]), reverse=True
        )[:8]:
            print(f"  {name}: {coef:+.6f}")
        print()
    else:
        print("reasoning path: not required by selected ablation; direct result features drive ranking")
        print()
    print("=== FINAL ===")
    print("PASS" if gate.passed else "FAIL")
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
