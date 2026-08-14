"""Chronological experiments and promotion gate for the multi-signal V3 model.

V3 training consumes pre-built point-in-time rows. It deliberately does not
fabricate missing vendor data from the competition archive. 2026Q2 is treated
as a legacy research holdout because prior work has already inspected it;
promotion requires a later untouched holdout (default: 2026Q3).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

from explaining_markets.feature_families.company_history_v3 import COMPANY_HISTORY_V3_FEATURE_NAMES
from explaining_markets.feature_families.earnings_surprise import EARNINGS_SURPRISE_FEATURE_NAMES
from explaining_markets.feature_families.guidance_expectations import GUIDANCE_FEATURE_NAMES
from explaining_markets.feature_families.market_sector import MARKET_SECTOR_FEATURE_NAMES
from explaining_markets.feature_families.news import NEWS_FEATURE_NAMES
from explaining_markets.feature_families.peer_sympathy import PEER_FEATURE_NAMES
from explaining_markets.feature_families.price_context import PRICE_CONTEXT_FEATURE_NAMES
from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES
from explaining_markets.feature_families.revenue_results import REVENUE_SURPRISE_FEATURE_NAMES
from explaining_markets.features_v3 import FEATURE_SPEC_VERSION_V3, MODEL_FEATURE_NAMES_V3
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES
from explaining_markets.point_in_time_audit_v3 import audit_feature_names

TRAIN_QUARTER = "2025Q4"
VALIDATION_QUARTER = "2026Q1"
LEGACY_HOLDOUT_QUARTER = "2026Q2"
HONEST_HOLDOUT_QUARTER = "2026Q3"
MODEL_VERSION = "multi_signal_v3"
DEFAULT_ARTIFACT = Path(__file__).with_name("artifacts") / "multi_signal_v3.json"
CLIP_BOUNDS = (0.05, 0.95)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0)
ELASTIC_ALPHAS = (0.0005, 0.001, 0.005, 0.01)
ELASTIC_L1_RATIOS = (0.1, 0.5, 0.9)

PROMOTION_GATE = {
    "min_validation_pearson_gain_over_v1": 0.01,
    "max_honest_holdout_pearson_regression": 0.01,
    "min_prediction_std": 0.01,
    "max_fraction_near_0_5": 0.90,
    "require_zero_leakage_violations": True,
    "require_all_tests_passing": True,
    "require_local_feed_verified": True,
    "require_modal_feed_verified": True,
    "require_nonzero_news_coverage": True,
    "require_reasoning_valid": True,
    "require_latency_ok": True,
}

COMPANY_NEWS_NAMES = tuple(n for n in NEWS_FEATURE_NAMES if n.startswith("company_") or n == "has_company_news")
PEER_NEWS_NAMES = tuple(n for n in NEWS_FEATURE_NAMES if n.startswith("peer_") or n == "has_peer_news")
SECTOR_NEWS_NAMES = tuple(n for n in NEWS_FEATURE_NAMES if n.startswith("sector_") or n == "has_sector_news")
BASE_STRUCTURED_NAMES = tuple(n for n in MODEL_FEATURE_NAMES_V3 if n not in NEWS_FEATURE_NAMES and n not in REASONING_FEATURE_NAMES)
ARTICLE_REASONING_NAMES = tuple(
    n for n in REASONING_FEATURE_NAMES
    if n in {
        "reasoning_company_news_signal", "reasoning_peer_signal", "reasoning_sector_signal",
        "reasoning_materiality", "reasoning_confidence", "has_reasoning",
    }
)

ABLATIONS: dict[str, tuple[str, ...]] = {
    "v1_fls_only": MODEL_FEATURE_NAMES,
    "fls_plus_eps": (*MODEL_FEATURE_NAMES, *EARNINGS_SURPRISE_FEATURE_NAMES),
    "fls_plus_eps_revenue": (*MODEL_FEATURE_NAMES, *EARNINGS_SURPRISE_FEATURE_NAMES, *REVENUE_SURPRISE_FEATURE_NAMES),
    "fls_plus_guidance": (*MODEL_FEATURE_NAMES, *GUIDANCE_FEATURE_NAMES),
    "fls_plus_company_history": (*MODEL_FEATURE_NAMES, *COMPANY_HISTORY_V3_FEATURE_NAMES),
    "fls_plus_price_context": (*MODEL_FEATURE_NAMES, *PRICE_CONTEXT_FEATURE_NAMES),
    "fls_plus_market_sector": (*MODEL_FEATURE_NAMES, *MARKET_SECTOR_FEATURE_NAMES),
    "fls_plus_peers": (*MODEL_FEATURE_NAMES, *PEER_FEATURE_NAMES),
    "v3_without_news": BASE_STRUCTURED_NAMES,
    "company_news_only": (*BASE_STRUCTURED_NAMES, *COMPANY_NEWS_NAMES),
    "company_peer_news": (*BASE_STRUCTURED_NAMES, *COMPANY_NEWS_NAMES, *PEER_NEWS_NAMES),
    "company_peer_sector_news": (*BASE_STRUCTURED_NAMES, *NEWS_FEATURE_NAMES),
    "peer_news_off": (*BASE_STRUCTURED_NAMES, *COMPANY_NEWS_NAMES, *SECTOR_NEWS_NAMES),
    "peer_news_on": (*BASE_STRUCTURED_NAMES, *NEWS_FEATURE_NAMES),
    "v3_deterministic_news": (*BASE_STRUCTURED_NAMES, *NEWS_FEATURE_NAMES),
    "v3_article_reasoning": (*BASE_STRUCTURED_NAMES, *NEWS_FEATURE_NAMES, *ARTICLE_REASONING_NAMES),
    "v3_event_reasoning": (*BASE_STRUCTURED_NAMES, *NEWS_FEATURE_NAMES, *REASONING_FEATURE_NAMES),
    "full_v3": MODEL_FEATURE_NAMES_V3,
}


@dataclass(frozen=True)
class V3TrainingRow:
    event_id: str
    ticker: str
    quarter: str
    target_percentile: float
    values: dict[str, float]
    surprise_percentile: float | None = None
    leakage_violations: int = 0

    def x(self, names: tuple[str, ...]) -> list[float]:
        return [float(self.values[name]) for name in names]


@dataclass(frozen=True)
class Metrics:
    pearson: float | None
    spearman: float | None
    mae: float
    rmse: float
    prediction_std: float
    minimum_prediction: float
    maximum_prediction: float
    prediction_p05: float
    prediction_p95: float
    fraction_between_048_052: float


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def compute_metrics(predicted, actual) -> Metrics:
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(actual, dtype=float)
    if len(p) == 0:
        raise ValueError("cannot compute V3 metrics for zero rows")
    return Metrics(
        pearson=_corr(p, y),
        spearman=_corr(_rank(p), _rank(y)),
        mae=float(np.mean(np.abs(p - y))),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        prediction_std=float(np.std(p)),
        minimum_prediction=float(np.min(p)),
        maximum_prediction=float(np.max(p)),
        prediction_p05=float(np.quantile(p, 0.05)),
        prediction_p95=float(np.quantile(p, 0.95)),
        fraction_between_048_052=float(np.mean((p >= 0.48) & (p <= 0.52))),
    )


def _r2(y: np.ndarray, predictors: np.ndarray) -> float | None:
    if len(y) < 3:
        return None
    X = np.column_stack([np.ones(len(y)), predictors])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    if ss_total <= 1e-12:
        return None
    return 1.0 - float(np.sum((y - fitted) ** 2)) / ss_total


def surprise_incremental_r2(rows: list[V3TrainingRow], predicted) -> dict[str, float | None]:
    pairs = [(r, float(p)) for r, p in zip(rows, predicted, strict=True) if r.surprise_percentile is not None]
    if len(pairs) < 3:
        return {"r2_surprise_only": None, "r2_prediction_plus_surprise": None, "delta_r2": None}
    y = np.asarray([r.target_percentile for r, _ in pairs])
    s = np.asarray([float(r.surprise_percentile) for r, _ in pairs])
    p = np.asarray([pred for _, pred in pairs])
    r2_s = _r2(y, s[:, None])
    r2_both = _r2(y, np.column_stack([p, s]))
    return {
        "r2_surprise_only": r2_s,
        "r2_prediction_plus_surprise": r2_both,
        "delta_r2": None if r2_s is None or r2_both is None else r2_both - r2_s,
    }


def _fit_predict(train, evaluate_rows, names, kind, params):
    X_train = np.asarray([r.x(names) for r in train], dtype=float)
    y_train = np.asarray([r.target_percentile for r in train], dtype=float)
    X_eval = np.asarray([r.x(names) for r in evaluate_rows], dtype=float)
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds <= 1e-12] = 1.0
    Z_train, Z_eval = (X_train - means) / stds, (X_eval - means) / stds
    if kind == "ridge":
        model = Ridge(alpha=params["alpha"])
    elif kind == "elastic_net":
        model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=20000)
    elif kind == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=0.05, random_state=7)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.fit(Z_train, y_train)
    return np.clip(model.predict(Z_eval), *CLIP_BOUNDS), model, means, stds


def _candidate_specs():
    for alpha in RIDGE_ALPHAS:
        yield "ridge", {"alpha": alpha}
    for alpha in ELASTIC_ALPHAS:
        for ratio in ELASTIC_L1_RATIOS:
            yield "elastic_net", {"alpha": alpha, "l1_ratio": ratio}
    yield "hist_gradient_boosting", {}


def evaluate(
    rows: list[V3TrainingRow],
    *,
    tests_passing: bool = False,
    local_feed_verified: bool = False,
    modal_feed_verified: bool = False,
    news_coverage_nonzero: bool = False,
    reasoning_valid: bool = False,
    latency_ok: bool = False,
) -> dict:
    audit_feature_names()
    if any(r.leakage_violations for r in rows):
        raise ValueError("V3 training rows contain point-in-time audit violations")
    for row in rows:
        missing = set(MODEL_FEATURE_NAMES_V3).difference(row.values)
        if missing:
            raise ValueError(f"row {row.event_id} missing V3 features: {sorted(missing)}")

    by_q = {q: [r for r in rows if r.quarter == q] for q in (TRAIN_QUARTER, VALIDATION_QUARTER, LEGACY_HOLDOUT_QUARTER, HONEST_HOLDOUT_QUARTER)}
    if not by_q[TRAIN_QUARTER] or not by_q[VALIDATION_QUARTER]:
        raise RuntimeError("V3 requires 2025Q4 train and 2026Q1 validation rows")

    results = {"ablations": {}, "legacy_holdout_is_pristine": False}
    selected = None
    for ablation, names in ABLATIONS.items():
        candidates = []
        for kind, params in _candidate_specs():
            pred, _, _, _ = _fit_predict(by_q[TRAIN_QUARTER], by_q[VALIDATION_QUARTER], names, kind, params)
            metrics = compute_metrics(pred, [r.target_percentile for r in by_q[VALIDATION_QUARTER]])
            candidates.append({"kind": kind, "params": params, "metrics": asdict(metrics), "surprise_r2": surprise_incremental_r2(by_q[VALIDATION_QUARTER], pred)})
        best = max(candidates, key=lambda x: (-math.inf if x["metrics"]["pearson"] is None else x["metrics"]["pearson"], -x["metrics"]["mae"]))
        results["ablations"][ablation] = {"selected": best, "candidates": candidates}
        if ablation == "full_v3":
            selected = best

    if selected is None:
        raise RuntimeError("full V3 candidate was not evaluated")
    base = results["ablations"]["v1_fls_only"]["selected"]
    results["selected_model"] = selected
    results["validation_gain_over_v1"] = None if selected["metrics"]["pearson"] is None or base["metrics"]["pearson"] is None else selected["metrics"]["pearson"] - base["metrics"]["pearson"]

    development = by_q[TRAIN_QUARTER] + by_q[VALIDATION_QUARTER]
    for label, quarter in (("legacy_holdout", LEGACY_HOLDOUT_QUARTER), ("honest_holdout", HONEST_HOLDOUT_QUARTER)):
        target = by_q[quarter]
        if not target:
            results[label] = None
            continue
        pred, _, _, _ = _fit_predict(development, target, MODEL_FEATURE_NAMES_V3, selected["kind"], selected["params"])
        results[label] = {"metrics": asdict(compute_metrics(pred, [r.target_percentile for r in target])), "surprise_r2": surprise_incremental_r2(target, pred), "n": len(target)}

    honest = results["honest_holdout"]
    validation_gain = results["validation_gain_over_v1"]
    promoted = False
    if honest is not None and validation_gain is not None:
        base_pred, _, _, _ = _fit_predict(development, by_q[HONEST_HOLDOUT_QUARTER], MODEL_FEATURE_NAMES, base["kind"], base["params"])
        base_honest = compute_metrics(base_pred, [r.target_percentile for r in by_q[HONEST_HOLDOUT_QUARTER]])
        results["honest_holdout_v1"] = asdict(base_honest)
        v3_p = honest["metrics"]["pearson"]
        v1_p = base_honest.pearson
        regression = None if v3_p is None or v1_p is None else v1_p - v3_p
        results["promotion_observed"] = {
            "validation_pearson_gain": validation_gain,
            "honest_holdout_pearson_regression": regression,
            "prediction_std": honest["metrics"]["prediction_std"],
            "fraction_near_0_5": honest["metrics"]["fraction_between_048_052"],
            "zero_leakage_violations": True,
            "tests_passing": bool(tests_passing),
            "local_feed_verified": bool(local_feed_verified),
            "modal_feed_verified": bool(modal_feed_verified),
            "news_coverage_nonzero": bool(news_coverage_nonzero),
            "reasoning_valid": bool(reasoning_valid),
            "latency_ok": bool(latency_ok),
        }
        promoted = bool(
            validation_gain >= PROMOTION_GATE["min_validation_pearson_gain_over_v1"]
            and regression is not None
            and regression <= PROMOTION_GATE["max_honest_holdout_pearson_regression"]
            and honest["metrics"]["prediction_std"] >= PROMOTION_GATE["min_prediction_std"]
            and honest["metrics"]["fraction_between_048_052"] <= PROMOTION_GATE["max_fraction_near_0_5"]
            and tests_passing
            and local_feed_verified
            and modal_feed_verified
            and news_coverage_nonzero
            and reasoning_valid
            and latency_ok
        )
    results["promotion_gate"] = PROMOTION_GATE
    results["promoted"] = promoted
    return results


def serialize_linear_artifact(rows: list[V3TrainingRow], evaluation: dict, artifact_path: str | Path = DEFAULT_ARTIFACT) -> dict:
    if not evaluation.get("promoted"):
        raise RuntimeError("refusing to serialize production V3 artifact: promotion gate did not pass")
    selected = evaluation["selected_model"]
    if selected["kind"] not in {"ridge", "elastic_net"}:
        raise RuntimeError("pure-Python production artifact currently supports linear V3 selections only")
    final_rows = [r for r in rows if r.quarter in {TRAIN_QUARTER, VALIDATION_QUARTER, LEGACY_HOLDOUT_QUARTER, HONEST_HOLDOUT_QUARTER}]
    X = np.asarray([r.x(MODEL_FEATURE_NAMES_V3) for r in final_rows], dtype=float)
    y = np.asarray([r.target_percentile for r in final_rows], dtype=float)
    means, stds = X.mean(axis=0), X.std(axis=0)
    stds[stds <= 1e-12] = 1.0
    Z = (X - means) / stds
    if selected["kind"] == "ridge":
        model = Ridge(alpha=selected["params"]["alpha"]).fit(Z, y)
    else:
        model = ElasticNet(alpha=selected["params"]["alpha"], l1_ratio=selected["params"]["l1_ratio"], max_iter=20000).fit(Z, y)
    artifact = {
        "model_version": MODEL_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION_V3,
        "feature_names": list(MODEL_FEATURE_NAMES_V3),
        "means": [float(x) for x in means],
        "standard_deviations": [float(x) for x in stds],
        "coefficients": [float(x) for x in model.coef_],
        "intercept": float(model.intercept_),
        "clip_bounds": list(CLIP_BOUNDS),
        "promoted": True,
        "structured_only": False,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_metadata": evaluation,
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
