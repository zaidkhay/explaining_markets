"""Auditable diagnostics for one V3-lite prediction.

These diagnostics intentionally distinguish three different concepts:

* interpretation_confidence: how explicitly the deterministic parser can map a
  supplied disclosure claim into the live feature schema;
* model_relevance: whether the currently deployed artifact actually consumes
  the signal family affected by that claim;
* prediction strength: how far the raw score sits from the historical OOS raw
  prediction distribution used for validation/calibration.

None of these is a probability that a disclosure claim is true.  The
competition disclosure is treated as supplied evidence, not independently
fact-checked by this model.
"""
from __future__ import annotations

import math
import re
from typing import Any

from explaining_markets.disclosure_results_v3 import parse_disclosure_records
from explaining_markets.features_v3 import FeatureVectorV3, family_availability
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES, classify_statement
from explaining_markets.model_v3_lite import V3LiteCandidateModel

_EXPLICIT_EXPECTATION = re.compile(
    r"\b(?:consensus|estimate|estimates|expected|expectations|guidance|outlook|forecast|plan)\b",
    re.I,
)
_EXPLICIT_COMPARISON = re.compile(
    r"\b(?:beat|beats|beating|miss|missed|misses|above|below|exceeded|fell short|in[- ]?line|matched|versus|vs\.?)\b",
    re.I,
)
_EXPLICIT_NUMBER = re.compile(r"(?:[$€£]\s*)?-?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|bps|basis points|[KMBT]|million|billion|trillion)?", re.I)


def _feature_family(name: str) -> str:
    if name in MODEL_FEATURE_NAMES:
        return "forward_looking"
    if name.startswith("revenue_") or name in {
        "is_revenue_beat", "is_revenue_miss", "has_revenue_surprise",
        "eps_beat_and_revenue_beat", "eps_beat_revenue_miss",
        "eps_miss_revenue_beat", "eps_miss_and_revenue_miss",
    }:
        return "revenue_results"
    return "other"


def feature_contributions(model: V3LiteCandidateModel, vector: FeatureVectorV3) -> dict[str, Any]:
    """Return the exact standardized linear contribution of every used feature."""
    rows: list[dict[str, Any]] = []
    total = 0.0
    for name, mean, sd, coefficient in zip(
        model.feature_names,
        model.means,
        model.standard_deviations,
        model.coefficients,
        strict=True,
    ):
        value = float(vector.values[name])
        standardized = (value - mean) / sd
        contribution = coefficient * standardized
        total += contribution
        rows.append({
            "feature": name,
            "family": _feature_family(name),
            "value": value,
            "training_mean": float(mean),
            "training_std": float(sd),
            "standardized_value": float(standardized),
            "coefficient": float(coefficient),
            "raw_score_contribution": float(contribution),
        })

    raw_unclipped = float(model.intercept + total)
    raw = float(max(model.clip_lower, min(model.clip_upper, raw_unclipped)))
    submitted = float(model.calibrator.calibrate(raw))
    rows.sort(key=lambda row: abs(float(row["raw_score_contribution"])), reverse=True)
    return {
        "intercept": float(model.intercept),
        "sum_feature_contributions": float(total),
        "raw_unclipped": raw_unclipped,
        "raw_score": raw,
        "submitted_percentile": submitted,
        "features": rows,
    }


def _claim_direction(text: str, parsed, statement) -> tuple[str, int]:
    if parsed.guidance is not None:
        direction = (parsed.guidance.direction or "").lower()
        if direction == "raised":
            return "positive", 1
        if direction == "lowered":
            return "negative", -1
        if direction in {"reaffirmed", "maintained"}:
            return "neutral", 0
    if parsed.earnings is not None:
        revenue_delta = None
        if parsed.earnings.reported_revenue is not None and parsed.earnings.consensus_revenue is not None:
            revenue_delta = parsed.earnings.reported_revenue - parsed.earnings.consensus_revenue
        eps_delta = None
        if parsed.earnings.reported_eps is not None and parsed.earnings.consensus_eps is not None:
            eps_delta = parsed.earnings.reported_eps - parsed.earnings.consensus_eps
        deltas = [x for x in (revenue_delta, eps_delta) if x is not None]
        if deltas and all(x > 0 for x in deltas):
            return "positive", 1
        if deltas and all(x < 0 for x in deltas):
            return "negative", -1
    if statement.directional_score > 0:
        return "positive", 1
    if statement.directional_score < 0:
        return "negative", -1
    return "neutral", 0


def _interpretation_confidence(text: str, parsed, statement) -> tuple[float, str]:
    """Rule-strength score, not a probability that the claim is true."""
    has_expectation = bool(_EXPLICIT_EXPECTATION.search(text))
    has_comparison = bool(_EXPLICIT_COMPARISON.search(text))
    numeric = len(_EXPLICIT_NUMBER.findall(text))
    matched = set(parsed.matched_fields)

    if ("eps" in matched or "revenue" in matched) and has_expectation and has_comparison and numeric >= 1:
        return 0.98, "explicit realized-vs-expectation result"
    if ("eps" in matched or "revenue" in matched) and has_expectation and numeric >= 2:
        return 0.95, "explicit actual/expectation numeric pair"
    if any(name.startswith("guidance_") for name in matched):
        return 0.92, "explicit guidance direction"
    if statement.forward_looking and statement.directional_score != 0 and statement.quantitative:
        return 0.86, "quantitative forward-looking directional statement"
    if statement.forward_looking and statement.directional_score != 0:
        return 0.78, "forward-looking directional statement"
    if statement.forward_looking:
        return 0.68, "forward-looking statement without clear direction"
    if has_comparison and numeric:
        return 0.55, "numeric comparison not mapped into a deployed structured feature"
    return 0.25, "claim is largely outside the deployed parser/model schema"


def _claim_model_relevance(model: V3LiteCandidateModel, parsed, statement) -> tuple[float, list[str]]:
    used = set(model.feature_names)
    affected: list[str] = []

    if statement.forward_looking:
        affected.extend(name for name in MODEL_FEATURE_NAMES if name in used)

    matched = set(parsed.matched_fields)
    if "revenue" in matched:
        revenue_names = {
            "revenue_surprise_percent", "revenue_surprise_percentile_company",
            "is_revenue_beat", "is_revenue_miss", "has_revenue_surprise",
            "eps_beat_and_revenue_beat", "eps_beat_revenue_miss",
            "eps_miss_revenue_beat", "eps_miss_and_revenue_miss",
        }
        affected.extend(sorted(revenue_names.intersection(used)))

    # The current artifact has no standalone EPS family. EPS only matters when
    # it participates in one of the revenue/EPS interaction flags.
    if "eps" in matched:
        eps_interactions = {
            "eps_beat_and_revenue_beat", "eps_beat_revenue_miss",
            "eps_miss_revenue_beat", "eps_miss_and_revenue_miss",
        }
        affected.extend(sorted(eps_interactions.intersection(used)))

    affected = list(dict.fromkeys(affected))
    if not affected:
        return 0.05, []

    coefficient_by_name = dict(zip(model.feature_names, model.coefficients, strict=True))
    total_weight = sum(abs(x) for x in model.coefficients) or 1.0
    relevant_weight = sum(abs(coefficient_by_name[name]) for name in affected)
    # Weight share alone understates broad FLS claims because they can alter
    # multiple correlated count/ratio/interaction features.  Blend it with
    # affected-feature coverage to create a bounded descriptive relevance score.
    coverage = len(affected) / len(model.feature_names)
    score = min(1.0, 0.65 * min(1.0, relevant_weight / (0.25 * total_weight)) + 0.35 * min(1.0, coverage / 0.5))
    return float(score), affected


def claim_diagnostics(disclosure: list[str], *, ticker: str, cutoff, model: V3LiteCandidateModel) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for index, raw in enumerate(disclosure, 1):
        text = " ".join(str(raw).split())
        statement = classify_statement(text)
        parsed = parse_disclosure_records([text], ticker=ticker, cutoff=cutoff)
        confidence, confidence_reason = _interpretation_confidence(text, parsed, statement)
        relevance, affected = _claim_model_relevance(model, parsed, statement)
        direction, direction_score = _claim_direction(text, parsed, statement)
        claims.append({
            "claim_index": index,
            "text": text,
            "direction": direction,
            "direction_score": direction_score,
            "forward_looking": bool(statement.forward_looking),
            "earnings_related": bool(statement.earnings_related),
            "quantitative": bool(statement.quantitative),
            "fls_category": statement.category,
            "parser_matched_fields": list(parsed.matched_fields),
            "interpretation_confidence": float(confidence),
            "interpretation_confidence_reason": confidence_reason,
            "model_relevance": float(relevance),
            "affected_deployed_features": affected,
            "truth_confidence": None,
            "truth_confidence_note": "not assessed; supplied competition disclosure is treated as evidence",
        })
    return claims


def build_prediction_diagnostics(
    *,
    model: V3LiteCandidateModel,
    vector: FeatureVectorV3,
    disclosure: list[str],
    context,
) -> dict[str, Any]:
    score = feature_contributions(model, vector)
    claims = claim_diagnostics(disclosure, ticker=context.ticker, cutoff=context.cutoff, model=model)
    validation = dict((model.training_metadata.get("validation_metrics") or {}))
    dispersion = dict(validation.get("dispersion") or {})
    median = float(dispersion.get("median", 0.5))
    pred_std = float(dispersion.get("prediction_std", 0.0))
    raw_signal_z = 0.0 if pred_std <= 1e-12 else abs(float(score["raw_score"]) - median) / pred_std

    used_families = sorted({_feature_family(name) for name in model.feature_names})
    nonzero_used = sum(abs(float(vector.values[name])) > 1e-12 for name in model.feature_names)
    availability = family_availability(vector)
    provider_receipts = list(context.extras.get("provider_receipts", ()) or ())
    provider_successes = sum(1 for row in provider_receipts if row.get("status") == "ok")
    provider_errors = sum(1 for row in provider_receipts if row.get("status") == "error")

    return {
        "model": {
            "version": model.model_version,
            "ablation": model.ablation,
            "feature_count": len(model.feature_names),
            "used_families": used_families,
            "calibration_method": model.calibrator.method,
            "calibration_n_fitted": model.calibrator.n_fitted,
        },
        "score": score,
        "claims": claims,
        "external_context": {
            "family_availability": availability,
            "provider_successes": provider_successes,
            "provider_errors": provider_errors,
            "provider_receipts": provider_receipts,
            "nonzero_deployed_features": nonzero_used,
            "deployed_feature_count": len(model.feature_names),
        },
        "historical_validation": {
            "n": validation.get("n"),
            "spearman": validation.get("spearman"),
            "pearson": validation.get("pearson"),
            "mae": validation.get("mae"),
            "rmse": validation.get("rmse"),
            "prediction_std": dispersion.get("prediction_std"),
            "raw_prediction_median": dispersion.get("median"),
        },
        "diagnostic_signal": {
            "raw_signal_z_vs_validation": float(raw_signal_z),
            "submitted_extremeness": float(min(1.0, 2.0 * abs(float(score["submitted_percentile"]) - 0.5))),
            "mean_claim_interpretation_confidence": float(
                sum(float(row["interpretation_confidence"]) for row in claims) / max(len(claims), 1)
            ),
            "mean_claim_model_relevance": float(
                sum(float(row["model_relevance"]) for row in claims) / max(len(claims), 1)
            ),
            "note": "diagnostic strength/support metrics; not calibrated probability of correctness",
        },
    }
