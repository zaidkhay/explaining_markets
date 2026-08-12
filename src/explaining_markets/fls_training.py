"""Offline training/evaluation for the production paper-informed FLS Ridge.

Chronology is fixed by design:
2025Q4 TRAIN -> 2026Q1 VALIDATION/model selection -> 2026Q2 LOCKED HOLDOUT.
The holdout is touched only after alpha selection, then the unchanged
specification is retrained on all three sealed quarters for the live artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from explaining_markets.backtest import percentile_ranks
from explaining_markets.features import extract_features
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES, extract_forward_looking_features
from explaining_markets.historical import HistoricalEvent, labeled_events, load_historical_events
from explaining_markets.model import HeuristicFactModel

TRAIN_QUARTER = "2025Q4"
VALIDATION_QUARTER = "2026Q1"
HOLDOUT_QUARTER = "2026Q2"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
CLIP_BOUNDS = (0.05, 0.95)
MODEL_VERSION = "fls_ridge_v1"
DEFAULT_ARTIFACT = Path(__file__).with_name("artifacts") / "fls_ridge_v1.json"


@dataclass(frozen=True)
class Row:
    event: HistoricalEvent
    y: float
    surprise_percentile: float | None
    x: tuple[float, ...]


@dataclass(frozen=True)
class Metrics:
    n_obs: int
    pearson: float | None
    spearman: float | None
    mae: float
    rmse: float
    prediction_std: float


@dataclass(frozen=True)
class OfficialMetrics:
    n_obs: int
    r_squared_surprise: float | None
    r_squared_prediction_plus_surprise: float | None
    delta_r_squared: float | None


@dataclass(frozen=True)
class Standardizer:
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        # Constant training columns are harmless under Ridge. Keep their
        # standardized values at zero rather than dividing by zero.
        stds = np.where(stds > 1e-12, stds, 1.0)
        return cls(tuple(map(float, means)), tuple(map(float, stds)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - np.asarray(self.means)) / np.asarray(self.standard_deviations)


def build_rows(events: list[HistoricalEvent]) -> list[Row]:
    """Construct safe disclosure features and official within-quarter y labels."""
    rows: list[Row] = []
    by_quarter: dict[str, list[HistoricalEvent]] = {}
    for event in labeled_events(events):
        if event.quarter in {TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER}:
            by_quarter.setdefault(str(event.quarter), []).append(event)

    for quarter in (TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER):
        quarter_events = by_quarter.get(quarter, [])
        y = percentile_ranks([float(e.car1) for e in quarter_events if e.car1 is not None])
        surprise_idx = [i for i, e in enumerate(quarter_events) if e.earnings_surprise is not None]
        surprise_ranks = percentile_ranks([float(quarter_events[i].earnings_surprise) for i in surprise_idx])
        surprise_by_idx = dict(zip(surprise_idx, surprise_ranks, strict=True))
        for i, (event, target) in enumerate(zip(quarter_events, y, strict=True)):
            fls = extract_forward_looking_features(event.disclosure)
            x = tuple(fls.vector(MODEL_FEATURE_NAMES))
            rows.append(Row(event=event, y=float(target), surprise_percentile=surprise_by_idx.get(i), x=x))
    return rows


def train_and_serialize(
    source: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict:
    events = load_historical_events(source)
    rows = build_rows(events)
    by_q = {q: [r for r in rows if r.event.quarter == q] for q in (TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER)}
    if any(not by_q[q] for q in by_q):
        raise RuntimeError("all three sealed quarters are required to train fls_ridge_v1")

    train = by_q[TRAIN_QUARTER]
    validation = by_q[VALIDATION_QUARTER]
    holdout = by_q[HOLDOUT_QUARTER]

    X_train, y_train = _xy(train)
    X_val, y_val = _xy(validation)
    train_scaler = Standardizer.fit(X_train)
    X_train_z, X_val_z = train_scaler.transform(X_train), train_scaler.transform(X_val)

    candidates: list[dict] = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(X_train_z, y_train)
        pred = np.clip(model.predict(X_val_z), *CLIP_BOUNDS)
        ordinary = compute_metrics(pred, y_val)
        official = compute_official_metrics(validation, pred)
        candidates.append({
            "alpha": alpha,
            "metrics": asdict(ordinary),
            "official": asdict(official),
        })

    # Competition-aligned selection on VALIDATION only. If official delta R²
    # is unavailable because surprise is missing/degenerate, fall back to
    # Pearson. HOLDOUT is not referenced anywhere in this decision.
    def candidate_key(item: dict) -> tuple[float, float, float]:
        delta = item["official"]["delta_r_squared"]
        pearson = item["metrics"]["pearson"]
        return (
            -math.inf if delta is None else float(delta),
            -math.inf if pearson is None else float(pearson),
            -float(item["alpha"]),
        )

    chosen = max(candidates, key=candidate_key)
    selected_alpha = float(chosen["alpha"])

    # Lock selection, then refit on TRAIN+VALIDATION and evaluate Q2 once.
    development = train + validation
    X_dev, y_dev = _xy(development)
    dev_scaler = Standardizer.fit(X_dev)
    dev_model = Ridge(alpha=selected_alpha).fit(dev_scaler.transform(X_dev), y_dev)
    X_hold, y_hold = _xy(holdout)
    hold_pred = np.clip(dev_model.predict(dev_scaler.transform(X_hold)), *CLIP_BOUNDS)

    validation_comparison = _comparison(validation, np.asarray(chosen_predictions(
        train=train,
        evaluate=validation,
        alpha=selected_alpha,
    )))
    holdout_comparison = _comparison(holdout, hold_pred)

    # Final live artifact: exact unchanged feature specification/alpha, now fit
    # on all three sealed quarters after the single locked-holdout read.
    final_rows = train + validation + holdout
    X_final, y_final = _xy(final_rows)
    final_scaler = Standardizer.fit(X_final)
    final_model = Ridge(alpha=selected_alpha).fit(final_scaler.transform(X_final), y_final)

    artifact = {
        "model_version": MODEL_VERSION,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "means": list(final_scaler.means),
        "standard_deviations": list(final_scaler.standard_deviations),
        "coefficients": [float(x) for x in final_model.coef_],
        "intercept": float(final_model.intercept_),
        "selected_alpha": selected_alpha,
        "clip_bounds": list(CLIP_BOUNDS),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_quarters": [TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER],
        "training_metadata": {
            "n_train": len(train),
            "n_validation": len(validation),
            "n_holdout": len(holdout),
            "n_final": len(final_rows),
            "selection_rule": "max validation official delta_r_squared; Pearson fallback",
            "validation_candidates": candidates,
            "validation_comparison": validation_comparison,
            "locked_holdout_comparison": holdout_comparison,
        },
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def chosen_predictions(*, train: list[Row], evaluate: list[Row], alpha: float) -> list[float]:
    X_train, y_train = _xy(train)
    X_eval, _ = _xy(evaluate)
    scaler = Standardizer.fit(X_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)
    return list(np.clip(model.predict(scaler.transform(X_eval)), *CLIP_BOUNDS))


def _comparison(rows: list[Row], ridge_pred: np.ndarray) -> dict:
    y = [r.y for r in rows]
    constant = np.full(len(rows), 0.5)
    heuristic = np.asarray([
        HeuristicFactModel().predict_percentile(
            extract_features(ticker=r.event.ticker, event_type=r.event.event_type, disclosure=r.event.disclosure)
        )
        for r in rows
    ])
    surprise_rows = [(r, r.surprise_percentile) for r in rows if r.surprise_percentile is not None]
    surprise_y = [r.y for r, _ in surprise_rows]
    surprise_pred = np.asarray([float(s) for _, s in surprise_rows])
    return {
        "constant_0.5": asdict(compute_metrics(constant, y)),
        "heuristic_fact": asdict(compute_metrics(heuristic, y)),
        "fls_ridge": asdict(compute_metrics(ridge_pred, y)),
        "surprise_benchmark": asdict(compute_metrics(surprise_pred, surprise_y)),
        "fls_ridge_official": asdict(compute_official_metrics(rows, ridge_pred)),
    }


def compute_metrics(predicted: np.ndarray | list[float], realized: np.ndarray | list[float]) -> Metrics:
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(realized, dtype=float)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("metrics require equal non-empty arrays")
    return Metrics(
        n_obs=len(p),
        pearson=_pearson(p, y),
        spearman=_spearman(p, y),
        mae=float(np.mean(np.abs(p - y))),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        prediction_std=float(np.std(p)),
    )


def compute_official_metrics(rows: list[Row], predicted: np.ndarray | list[float]) -> OfficialMetrics:
    p = np.asarray(predicted, dtype=float)
    idx = [i for i, r in enumerate(rows) if r.surprise_percentile is not None]
    if len(idx) < 3:
        return OfficialMetrics(len(idx), None, None, None)
    y = np.asarray([rows[i].y for i in idx], dtype=float)
    s = np.asarray([float(rows[i].surprise_percentile) for i in idx], dtype=float)
    pp = p[idx]
    r2_surprise = _ols_r2(y, np.column_stack([np.ones(len(y)), s]))
    r2_full = _ols_r2(y, np.column_stack([np.ones(len(y)), pp, s]))
    return OfficialMetrics(
        n_obs=len(idx),
        r_squared_surprise=r2_surprise,
        r_squared_prediction_plus_surprise=r2_full,
        delta_r_squared=None if r2_surprise is None or r2_full is None else r2_full - r2_surprise,
    )


def _xy(rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([r.x for r in rows], dtype=float), np.asarray([r.y for r in rows], dtype=float)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ra = np.asarray(percentile_ranks(list(map(float, a))))
    rb = np.asarray(percentile_ranks(list(map(float, b))))
    return _pearson(ra, rb)


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float | None:
    if len(y) < X.shape[1] or np.var(y) <= 1e-18:
        return None
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    sse = float(residual @ residual)
    centered = y - np.mean(y)
    sst = float(centered @ centered)
    return None if sst <= 1e-18 else 1.0 - sse / sst
