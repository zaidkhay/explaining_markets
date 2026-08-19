#!/usr/bin/env python3
"""Verify the disclosure-aware operator V3-lite artifact before deployment."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.features_v3 import FeatureVectorV3, MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.forward_looking_features import extract_forward_looking_features
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.v3_records import V3Context


def _reasoning_vector(model: V3LiteCandidateModel, direction: int) -> FeatureVectorV3:
    values = {name: 0.0 for name in MODEL_FEATURE_NAMES_V3}
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


def _realized_vector(model: V3LiteCandidateModel, disclosure: list[str], cutoff):
    base = V3Context(ticker="TEST", cutoff=cutoff)
    before = build_feature_vector_v3(disclosure=disclosure, context=base)
    reasoner = EventReasoner(use_openrouter=False)
    reasoning = reasoner.reason(values=before.values, cutoff=cutoff)
    final = replace(base, event_reasoning=reasoning)
    vector = build_feature_vector_v3(disclosure=disclosure, context=final)
    raw = model.predict_raw_vector(vector)
    submitted = model.calibrator.calibrate(raw)
    return vector, reasoning, raw, submitted


def main() -> int:
    model = V3LiteCandidateModel()

    # Coefficient-level sensitivity: confirms the fitted artifact has a usable
    # reasoning path even when the legacy FLS block is all zero.
    neg_vec = _reasoning_vector(model, -1)
    neu_vec = _reasoning_vector(model, 0)
    pos_vec = _reasoning_vector(model, 1)
    sensitivity_raw = [
        model.predict_raw_vector(neg_vec),
        model.predict_raw_vector(neu_vec),
        model.predict_raw_vector(pos_vec),
    ]
    sensitivity_submitted = [model.calibrator.calibrate(x) for x in sensitivity_raw]
    reasoning_features = [
        (name, coef)
        for name, coef in zip(model.feature_names, model.coefficients, strict=True)
        if name in REASONING_FEATURE_NAMES and abs(coef) > 1e-12
    ]

    # Production-realism test: exactly the type of realized facts that caused
    # V1 to emit an all-zero FLS vector.  No external APIs are involved.
    cutoff = datetime.now(timezone.utc)
    scenarios = {
        "negative": [
            "Revenue missed consensus by 8%.",
            "EPS missed consensus by 12%.",
        ],
        "neutral": [
            "Revenue was in line with consensus.",
            "EPS matched consensus.",
        ],
        "positive": [
            "Revenue beat consensus by 8%.",
            "EPS beat consensus by 12%.",
        ],
    }
    realized = {}
    for label, disclosure in scenarios.items():
        vector, reasoning, raw, submitted = _realized_vector(model, disclosure, cutoff)
        realized[label] = {
            "vector": vector,
            "reasoning": reasoning,
            "raw": raw,
            "submitted": submitted,
        }

    real_raw = [realized[name]["raw"] for name in ("negative", "neutral", "positive")]
    real_submitted = [realized[name]["submitted"] for name in ("negative", "neutral", "positive")]
    parsed_ok = all(
        item["vector"].values["has_eps_surprise"] == 1.0
        and item["vector"].values["has_revenue_surprise"] == 1.0
        for item in realized.values()
    )
    zero_fls = all(
        sum(abs(x) > 1e-12 for x in item["vector"].fls.values.values()) == 0
        for item in realized.values()
    )
    real_ordered = real_raw[0] < real_raw[1] < real_raw[2] and real_submitted[0] <= real_submitted[1] <= real_submitted[2]
    real_spread = real_submitted[2] - real_submitted[0]

    sensitivity_ordered = sensitivity_raw[0] < sensitivity_raw[1] < sensitivity_raw[2]
    sensitivity_spread = sensitivity_submitted[2] - sensitivity_submitted[0]
    pass_status = (
        parsed_ok
        and zero_fls
        and real_ordered
        and real_spread > 0.05
        and sensitivity_ordered
        and sensitivity_spread > 0.05
    )

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
    for label in ("negative", "neutral", "positive"):
        item = realized[label]
        v = item["vector"].values
        r = item["reasoning"]
        print(
            f"{label:<8} eps={v['eps_surprise_percent']:+.3f} "
            f"revenue={v['revenue_surprise_percent']:+.3f} "
            f"earnings_quality={r.earnings_quality:+.3f} "
            f"revenue_quality={r.revenue_quality:+.3f} "
            f"raw={item['raw']:.4f} submitted={item['submitted']:.4f}"
        )
    print(f"parsed_ok: {parsed_ok}")
    print(f"zero_fls: {zero_fls}")
    print(f"realized_ordered: {real_ordered}")
    print(f"realized_submitted_spread: {real_spread:.4f}")
    print()
    print("REASONING SENSITIVITY TEST")
    print(f"negative raw={sensitivity_raw[0]:.4f} submitted={sensitivity_submitted[0]:.4f}")
    print(f"neutral  raw={sensitivity_raw[1]:.4f} submitted={sensitivity_submitted[1]:.4f}")
    print(f"positive raw={sensitivity_raw[2]:.4f} submitted={sensitivity_submitted[2]:.4f}")
    print(f"ordered: {sensitivity_ordered}")
    print(f"submitted_spread: {sensitivity_spread:.4f}")
    print()
    print("=== FINAL ===")
    print("PASS" if pass_status else "FAIL")
    return 0 if pass_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
