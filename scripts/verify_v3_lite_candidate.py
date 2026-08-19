#!/usr/bin/env python3
"""Verify the operator V3-lite artifact before Modal deployment."""
from __future__ import annotations

from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.features_v3 import FeatureVectorV3, MODEL_FEATURE_NAMES_V3
from explaining_markets.forward_looking_features import extract_forward_looking_features
from explaining_markets.model_v3_lite import V3LiteCandidateModel


def _vector(model: V3LiteCandidateModel, direction: int) -> FeatureVectorV3:
    values = {name: 0.0 for name in MODEL_FEATURE_NAMES_V3}
    # Keep FLS at zero to reproduce the exact live failure mode observed on
    # 2026-08-18.  Move only reasoning features by one training standard
    # deviation in the coefficient-consistent direction.
    for name, mean, sd, coef in zip(
        model.feature_names,
        model.means,
        model.standard_deviations,
        model.coefficients,
        strict=True,
    ):
        if name not in REASONING_FEATURE_NAMES:
            continue
        if abs(coef) <= 1e-12:
            values[name] = mean
            continue
        sign = 1.0 if coef > 0 else -1.0
        values[name] = mean + direction * sign * sd
    return FeatureVectorV3(values=values, fls=extract_forward_looking_features([]))


def main() -> int:
    model = V3LiteCandidateModel()
    negative_vector = _vector(model, -1)
    neutral_vector = _vector(model, 0)
    positive_vector = _vector(model, 1)

    negative_raw = model.predict_raw_vector(negative_vector)
    neutral_raw = model.predict_raw_vector(neutral_vector)
    positive_raw = model.predict_raw_vector(positive_vector)
    negative = model.calibrator.calibrate(negative_raw)
    neutral = model.calibrator.calibrate(neutral_raw)
    positive = model.calibrator.calibrate(positive_raw)

    reasoning_features = [
        (name, coef)
        for name, coef in zip(model.feature_names, model.coefficients, strict=True)
        if name in REASONING_FEATURE_NAMES and abs(coef) > 1e-12
    ]
    ordered = negative_raw < neutral_raw < positive_raw and negative <= neutral <= positive
    raw_spread = positive_raw - negative_raw
    final_spread = positive - negative
    pass_status = bool(reasoning_features) and ordered and raw_spread > 0.01 and final_spread > 0.05

    print("=== V3-LITE PRODUCTION CANDIDATE VERIFICATION ===")
    print(f"model: {model.model_version}")
    print(f"ablation: {model.ablation}")
    print(f"features: {len(model.feature_names)}")
    print(f"reasoning_features_with_nonzero_weight: {len(reasoning_features)}")
    print(f"operator_override: {model.operator_override}")
    print(f"promoted: {model.promoted}")
    print(f"calibration: {model.calibrator.version}")
    print()
    print(f"reasoning-negative raw={negative_raw:.4f} submitted={negative:.4f}")
    print(f"reasoning-neutral  raw={neutral_raw:.4f} submitted={neutral:.4f}")
    print(f"reasoning-positive raw={positive_raw:.4f} submitted={positive:.4f}")
    print(f"ordered: {ordered}")
    print(f"raw_spread: {raw_spread:.4f}")
    print(f"submitted_spread: {final_spread:.4f}")
    print()
    print("top reasoning weights:")
    for name, coef in sorted(reasoning_features, key=lambda item: abs(item[1]), reverse=True)[:8]:
        print(f"  {name}: {coef:+.6f}")
    print()
    print("=== FINAL ===")
    print("PASS" if pass_status else "FAIL")
    return 0 if pass_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
