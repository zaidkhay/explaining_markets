"""Quantify recent live V3-lite results and turn errors into testable improvements.

This module is deliberately diagnostic, not an automatic production tuner.
Recent live events are too small and non-stationary a sample to fit directly.
Instead we:

1. score current predictions against realized CAR1 percentiles;
2. compare against neutral and fixed shrinkage counterfactuals;
3. join per-event prediction diagnostics when available;
4. measure failure modes by signal availability / model relevance;
5. generate explicit hypotheses that must still pass chronological historical
   validation before any production model change is promoted.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from explaining_markets.calibration import spearman


SHRINK_FACTORS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    if xx <= 1e-12 or yy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(xx * yy)


def metric_block(predicted: Iterable[float], realized: Iterable[float]) -> dict[str, float | int | None]:
    p = [float(x) for x in predicted]
    y = [float(x) for x in realized]
    if len(p) != len(y):
        raise ValueError("predicted and realized lengths differ")
    if not p:
        return {
            "n": 0,
            "spearman": None,
            "pearson": None,
            "mae": None,
            "rmse": None,
            "mean_signed_error": None,
            "direction_accuracy": None,
            "mean_predicted_extremeness": None,
            "mean_realized_extremeness": None,
        }
    errors = [a - b for a, b in zip(p, y, strict=True)]
    direction = []
    for a, b in zip(p, y, strict=True):
        if abs(a - 0.5) <= 1e-12 or abs(b - 0.5) <= 1e-12:
            direction.append(0.5)
        else:
            direction.append(float((a - 0.5) * (b - 0.5) > 0))
    return {
        "n": len(p),
        "spearman": spearman(p, y),
        "pearson": _pearson(p, y),
        "mae": mean(abs(e) for e in errors),
        "rmse": math.sqrt(mean(e * e for e in errors)),
        "mean_signed_error": mean(errors),
        "direction_accuracy": mean(direction),
        "mean_predicted_extremeness": mean(abs(x - 0.5) for x in p),
        "mean_realized_extremeness": mean(abs(x - 0.5) for x in y),
    }


def shrink_prediction(value: float, factor: float) -> float:
    return float(0.5 + float(factor) * (float(value) - 0.5))


def _read_diagnostic(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _diagnostic_path(row: dict[str, Any], diagnostics_dir: Path | None) -> Path | None:
    explicit = row.get("diagnostic_json")
    if explicit:
        return Path(str(explicit))
    if diagnostics_dir is None:
        return None
    event_id = str(row.get("event_id") or "").strip()
    ticker = str(row.get("ticker") or row.get("identifier_value") or "").upper().strip()
    if not event_id or not ticker:
        return None
    safe_event = "".join(c for c in event_id if c.isalnum() or c in "-_") or "manual"
    return diagnostics_dir / f"{safe_event}__{ticker}.json"


def _family_contributions(diag: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not diag:
        return out
    features = ((diag.get("score") or {}).get("features") or [])
    for feature in features:
        if not isinstance(feature, dict):
            continue
        family = str(feature.get("family") or "unknown")
        try:
            value = float(feature.get("raw_score_contribution") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        out[family] = out.get(family, 0.0) + value
    return out


def _normalize_row(raw: dict[str, Any], diagnostics_dir: Path | None) -> dict[str, Any] | None:
    row = dict(raw)
    path = _diagnostic_path(row, diagnostics_dir)
    diag = _read_diagnostic(path)

    predicted = row.get("predicted_percentile")
    realized = row.get("realized_percentile")
    if predicted in (None, "") and diag:
        predicted = (diag.get("score") or {}).get("submitted_percentile")
    if realized in (None, "") and diag:
        realized = (diag.get("realized") or {}).get("realized_percentile")
    if predicted in (None, "") or realized in (None, ""):
        return None

    try:
        predicted_f = float(predicted)
        realized_f = float(realized)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= predicted_f <= 1.0 and 0.0 <= realized_f <= 1.0):
        return None

    ticker = str(row.get("ticker") or row.get("identifier_value") or "?").upper()
    event_id = str(row.get("event_id") or ((diag or {}).get("event") or {}).get("event_id") or "")
    diagnostic_signal = (diag or {}).get("diagnostic_signal") or {}
    external = (diag or {}).get("external_context") or {}
    availability = external.get("family_availability") or {}
    claims = (diag or {}).get("claims") or []

    interpretation = diagnostic_signal.get("mean_claim_interpretation_confidence")
    relevance = diagnostic_signal.get("mean_claim_model_relevance")
    if interpretation is None and claims:
        interpretation = mean(float(x.get("interpretation_confidence") or 0.0) for x in claims if isinstance(x, dict))
    if relevance is None and claims:
        relevance = mean(float(x.get("model_relevance") or 0.0) for x in claims if isinstance(x, dict))

    families = _family_contributions(diag)
    top_features = []
    if diag:
        features = ((diag.get("score") or {}).get("features") or [])[:5]
        for feature in features:
            if isinstance(feature, dict):
                top_features.append({
                    "feature": feature.get("feature"),
                    "family": feature.get("family"),
                    "contribution": feature.get("raw_score_contribution"),
                })

    return {
        **row,
        "ticker": ticker,
        "event_id": event_id,
        "predicted_percentile": predicted_f,
        "realized_percentile": realized_f,
        "absolute_error": abs(predicted_f - realized_f),
        "signed_error": predicted_f - realized_f,
        "diagnostic_json": str(path) if path is not None and path.exists() else None,
        "interpretation_confidence": None if interpretation is None else float(interpretation),
        "model_relevance": None if relevance is None else float(relevance),
        "provider_errors": int(external.get("provider_errors") or 0),
        "nonzero_deployed_features": int(external.get("nonzero_deployed_features") or 0),
        "revenue_available": bool(float(availability.get("revenue") or 0.0)),
        "eps_available": bool(float(availability.get("eps") or 0.0)),
        "guidance_available": bool(float(availability.get("guidance") or 0.0)),
        "family_contributions": families,
        "top_features": top_features,
        "raw_signal_z": None if diagnostic_signal.get("raw_signal_z_vs_validation") is None else float(diagnostic_signal["raw_signal_z_vs_validation"]),
    }


def _group_metrics(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    subset = [row for row in rows if predicate(row)]
    return metric_block(
        [row["predicted_percentile"] for row in subset],
        [row["realized_percentile"] for row in subset],
    )


def _corr(values: list[float], realized: list[float]) -> dict[str, float | None]:
    if len(values) < 3 or max(values, default=0.0) - min(values, default=0.0) <= 1e-12:
        return {"spearman": None, "pearson": None}
    return {"spearman": spearman(values, realized), "pearson": _pearson(values, realized)}


def _recommendations(
    rows: list[dict[str, Any]],
    current: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    family_signal: dict[str, dict[str, Any]],
    validation_metrics: dict[str, Any] | None,
) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    n = len(rows)

    if n < 20:
        recs.append({
            "priority": "guardrail",
            "action": "Do not fit production coefficients directly to this live sample.",
            "evidence": f"Only {n} realized live events are present; use them to form hypotheses, then test those hypotheses on chronological historical splits.",
        })

    neutral = comparisons.get("shrink_0.00") or {}
    current_mae = current.get("mae")
    neutral_mae = neutral.get("mae")
    if current_mae is not None and neutral_mae is not None and float(current_mae) > float(neutral_mae) + 0.02:
        recs.append({
            "priority": "high",
            "action": "Reduce calibration extremeness unless stronger signal support is present.",
            "evidence": f"Current live MAE {current_mae:.3f} is worse than neutral-0.5 MAE {neutral_mae:.3f} by {current_mae-neutral_mae:.3f}.",
        })

    shrink_candidates = [(name, block) for name, block in comparisons.items() if block.get("mae") is not None]
    if shrink_candidates and current_mae is not None:
        best_name, best = min(shrink_candidates, key=lambda item: float(item[1]["mae"]))
        if best_name != "shrink_1.00" and float(current_mae) - float(best["mae"]) >= 0.015:
            recs.append({
                "priority": "high",
                "action": "Backtest support-aware shrinkage toward 0.5.",
                "evidence": f"Retrospective {best_name} lowers recent MAE from {current_mae:.3f} to {best['mae']:.3f}. This is diagnostic only until reproduced out-of-sample.",
            })

    rev_yes = groups.get("revenue_available") or {}
    rev_no = groups.get("revenue_missing") or {}
    if rev_yes.get("n", 0) >= 3 and rev_no.get("n", 0) >= 3 and rev_yes.get("mae") is not None and rev_no.get("mae") is not None:
        gap = float(rev_no["mae"]) - float(rev_yes["mae"])
        if gap >= 0.05:
            recs.append({
                "priority": "high",
                "action": "Add a missing-revenue fallback or confidence gate before allowing extreme predictions.",
                "evidence": f"Revenue-missing MAE is {rev_no['mae']:.3f} versus {rev_yes['mae']:.3f} when revenue is available (gap {gap:.3f}).",
            })

    low_rel = groups.get("low_model_relevance") or {}
    high_rel = groups.get("high_model_relevance") or {}
    if low_rel.get("n", 0) >= 3 and high_rel.get("n", 0) >= 3 and low_rel.get("mae") is not None and high_rel.get("mae") is not None:
        gap = float(low_rel["mae"]) - float(high_rel["mae"])
        if gap >= 0.05:
            recs.append({
                "priority": "high",
                "action": "Expand the deployed feature set to signals already present in disclosure but currently ignored.",
                "evidence": f"Low-relevance events have MAE {low_rel['mae']:.3f} versus {high_rel['mae']:.3f} for high-relevance events.",
            })

    extreme = groups.get("extreme_predictions") or {}
    middle = groups.get("non_extreme_predictions") or {}
    if extreme.get("n", 0) >= 3 and middle.get("n", 0) >= 3 and extreme.get("mae") is not None and middle.get("mae") is not None:
        gap = float(extreme["mae"]) - float(middle["mae"])
        if gap >= 0.05:
            recs.append({
                "priority": "medium",
                "action": "Revisit empirical-CDF calibration in the tails.",
                "evidence": f"Predictions outside [0.20,0.80] have MAE {extreme['mae']:.3f}, {gap:.3f} worse than non-extreme predictions.",
            })

    for family, block in family_signal.items():
        corr = block.get("pearson")
        if block.get("n", 0) >= 8 and corr is not None and float(corr) < -0.10:
            recs.append({
                "priority": "medium",
                "action": f"Audit the {family} feature-family signs and interactions.",
                "evidence": f"Recent {family} aggregate raw contribution has Pearson correlation {corr:.3f} with realized centered percentile, opposite the intended direction.",
            })

    if validation_metrics and current.get("spearman") is not None and validation_metrics.get("spearman") is not None and n >= 8:
        live_s = float(current["spearman"])
        val_s = float(validation_metrics["spearman"])
        if live_s < val_s - 0.05:
            recs.append({
                "priority": "high",
                "action": "Treat the current live feature distribution as a regime/coverage shift and rerun chronological ablations.",
                "evidence": f"Recent live Spearman {live_s:.3f} trails stored validation Spearman {val_s:.3f} by {val_s-live_s:.3f}.",
            })

    if not any(r["priority"] in {"high", "medium"} for r in recs):
        recs.append({
            "priority": "monitor",
            "action": "Accumulate more realized events before changing production behavior.",
            "evidence": "No failure mode crossed the pre-defined diagnostic thresholds in the available live sample.",
        })
    return recs


def analyze_recent_results(
    raw_rows: Iterable[dict[str, Any]],
    *,
    diagnostics_dir: str | Path | None = "data/diagnostics",
    validation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(diagnostics_dir) if diagnostics_dir is not None else None
    rows = [normalized for raw in raw_rows if (normalized := _normalize_row(dict(raw), root)) is not None]
    if not rows:
        raise ValueError("no rows have both predicted_percentile and realized_percentile")

    predicted = [row["predicted_percentile"] for row in rows]
    realized = [row["realized_percentile"] for row in rows]
    current = metric_block(predicted, realized)

    comparisons: dict[str, dict[str, Any]] = {}
    for factor in SHRINK_FACTORS:
        shrunk = [shrink_prediction(p, factor) for p in predicted]
        comparisons[f"shrink_{factor:.2f}"] = metric_block(shrunk, realized)

    groups = {
        "revenue_available": _group_metrics(rows, lambda r: r["revenue_available"]),
        "revenue_missing": _group_metrics(rows, lambda r: not r["revenue_available"]),
        "eps_available": _group_metrics(rows, lambda r: r["eps_available"]),
        "eps_missing": _group_metrics(rows, lambda r: not r["eps_available"]),
        "guidance_available": _group_metrics(rows, lambda r: r["guidance_available"]),
        "guidance_missing": _group_metrics(rows, lambda r: not r["guidance_available"]),
        "provider_errors": _group_metrics(rows, lambda r: r["provider_errors"] > 0),
        "no_provider_errors": _group_metrics(rows, lambda r: r["provider_errors"] == 0),
        "high_model_relevance": _group_metrics(rows, lambda r: r["model_relevance"] is not None and r["model_relevance"] >= 0.50),
        "low_model_relevance": _group_metrics(rows, lambda r: r["model_relevance"] is not None and r["model_relevance"] < 0.50),
        "high_interpretation_confidence": _group_metrics(rows, lambda r: r["interpretation_confidence"] is not None and r["interpretation_confidence"] >= 0.70),
        "low_interpretation_confidence": _group_metrics(rows, lambda r: r["interpretation_confidence"] is not None and r["interpretation_confidence"] < 0.70),
        "extreme_predictions": _group_metrics(rows, lambda r: r["predicted_percentile"] <= 0.20 or r["predicted_percentile"] >= 0.80),
        "non_extreme_predictions": _group_metrics(rows, lambda r: 0.20 < r["predicted_percentile"] < 0.80),
    }

    families = sorted({family for row in rows for family in row["family_contributions"]})
    family_signal: dict[str, dict[str, Any]] = {}
    centered_realized = [row["realized_percentile"] - 0.5 for row in rows]
    for family in families:
        values = [float(row["family_contributions"].get(family, 0.0)) for row in rows]
        corr = _corr(values, centered_realized)
        family_signal[family] = {
            "n": len(values),
            "mean_contribution": mean(values),
            "mean_abs_contribution": mean(abs(x) for x in values),
            **corr,
        }

    largest_misses = sorted(rows, key=lambda r: r["absolute_error"], reverse=True)[:10]
    best_calls = sorted(rows, key=lambda r: r["absolute_error"])[:10]
    recommendations = _recommendations(rows, current, comparisons, groups, family_signal, validation_metrics)

    return {
        "summary": current,
        "validation_reference": dict(validation_metrics or {}),
        "counterfactual_comparisons": comparisons,
        "groups": groups,
        "feature_family_signal": family_signal,
        "largest_misses": largest_misses,
        "best_calls": best_calls,
        "recommendations": recommendations,
        "rows": rows,
        "methodology": {
            "note": "Recent live results are diagnostic only; production changes must be retested on chronological historical validation/holdout data.",
            "shrinkage": "shrink_k = 0.5 + k * (submitted_percentile - 0.5); k is evaluated retrospectively and is not auto-promoted.",
            "realized_target": "Use official within-quarter realized percentile when available. Subset ranks from a handful of recent CAR1 values are not equivalent to the competition target.",
        },
    }
