"""Offline training/evaluation for ``fls_company_history_ridge_v2``.

Identical chronological discipline to ``fls_training``:

1. Fit standardization + candidate Ridges on TRAIN (2025Q4) only.
2. Select alpha on VALIDATION (2026Q1) only, by official delta R²
   (Pearson fallback), for every ablation variant.
3. Freeze the specification; refit TRAIN+VALIDATION; evaluate the LOCKED
   HOLDOUT (2026Q2) exactly once, for the full V2 and the V1-equivalent
   FLS-only benchmark.
4. Apply the predeclared promotion gate (see ``PROMOTION_GATE`` below).
5. Refit the unchanged specification on all three sealed quarters for the
   live artifact — regardless of promotion, so the artifact exists for
   explicit evaluation, but ``model.get_default_model`` only prefers it when
   the artifact records ``promoted: true``.

The leakage audit (``leakage_audit``) runs over every row BEFORE any fit and
raises on violations. Company-history features come exclusively from
``competition_history.walk_forward_history`` (sealed archive, conservative
availability lag); price/current-surprise families have no legal data source
yet and are neutral-with-indicator-0 in both training and live.
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
from explaining_markets.company_history import (
    AVAILABILITY_FEATURE_NAMES,
    CURRENT_VS_HISTORY_FEATURE_NAMES,
    EARNINGS_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    RECENCY_FEATURE_NAMES,
    SIMILAR_FEATURE_NAMES,
    SURPRISE_FEATURE_NAMES,
)
from explaining_markets.competition_history import (
    AVAILABILITY_LAG_DAYS,
    COMPETITION_FEATURE_NAMES,
    CUTOFF_GUARD_DAYS,
    parse_event_datetime,
    walk_forward_history,
)
from explaining_markets.features_v2 import (
    MODEL_FEATURE_NAMES_V2,
    build_feature_vector_v2,
)
from explaining_markets.fls_training import (
    Standardizer,
    compute_metrics,
    compute_official_metrics,
)
from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES,
    extract_forward_looking_features,
)
from explaining_markets.historical import HistoricalEvent, labeled_events, load_historical_events
from explaining_markets.leakage_audit import (
    AuditResult,
    audit_current_surprise_neutrality,
    audit_feature_names,
    audit_history_row,
)

TRAIN_QUARTER = "2025Q4"
VALIDATION_QUARTER = "2026Q1"
HOLDOUT_QUARTER = "2026Q2"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0)
CLIP_BOUNDS = (0.05, 0.95)
MODEL_VERSION = "fls_company_history_ridge_v2"
DEFAULT_ARTIFACT = Path(__file__).with_name("artifacts") / "fls_company_history_ridge_v2.json"

# Predeclared BEFORE the holdout is read (Part 14/30). V2 is promoted only if
# ALL hold:
#   validation delta_r_squared(V2) >= delta_r_squared(FLS-only) + MIN_VALIDATION_GAIN
#   holdout    delta_r_squared(V2) >= delta_r_squared(FLS-only) - MAX_HOLDOUT_REGRESSION
#   holdout    prediction_std(V2)  >= MIN_PREDICTION_STD
PROMOTION_GATE = {
    "min_validation_delta_r2_gain": 0.002,
    "max_holdout_delta_r2_regression": 0.002,
    "min_prediction_std": 0.01,
}

# Ablation variants: name -> feature subset (Part 15).
HISTORY_FAMILY = (
    *PRICE_FEATURE_NAMES, *EARNINGS_FEATURE_NAMES, *SURPRISE_FEATURE_NAMES,
    *CURRENT_VS_HISTORY_FEATURE_NAMES, *SIMILAR_FEATURE_NAMES, *RECENCY_FEATURE_NAMES,
    *AVAILABILITY_FEATURE_NAMES, *COMPETITION_FEATURE_NAMES,
)
ABLATIONS: dict[str, tuple[str, ...]] = {
    "fls_only": MODEL_FEATURE_NAMES,
    "history_only": HISTORY_FAMILY,
    "fls_plus_price": (*MODEL_FEATURE_NAMES, *PRICE_FEATURE_NAMES),
    "fls_plus_earnings_reaction": (
        *MODEL_FEATURE_NAMES, *EARNINGS_FEATURE_NAMES, *SURPRISE_FEATURE_NAMES,
        *RECENCY_FEATURE_NAMES,
    ),
    "fls_plus_current_surprise": (
        *MODEL_FEATURE_NAMES, *CURRENT_VS_HISTORY_FEATURE_NAMES, *SIMILAR_FEATURE_NAMES,
    ),
    "fls_plus_competition": (*MODEL_FEATURE_NAMES, *COMPETITION_FEATURE_NAMES,
                             *AVAILABILITY_FEATURE_NAMES),
    "all_v2": MODEL_FEATURE_NAMES_V2,
}


@dataclass(frozen=True)
class RowV2:
    event: HistoricalEvent
    y: float
    surprise_percentile: float | None
    values: dict[str, float]  # full V2 feature dict, frozen order via names

    def x(self, names: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(self.values[n] for n in names)


def build_rows(events: list[HistoricalEvent]) -> tuple[list[RowV2], AuditResult]:
    """Assemble V2 rows + run the full leakage audit. Raises LeakageError."""
    audit_feature_names(MODEL_FEATURE_NAMES_V2)
    history_by_key = walk_forward_history(events)

    by_quarter: dict[str, list[HistoricalEvent]] = {}
    for event in labeled_events(events):
        if event.quarter in {TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER}:
            by_quarter.setdefault(str(event.quarter), []).append(event)

    rows: list[RowV2] = []
    n_sources = 0
    n_with_history = 0
    for quarter in (TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER):
        quarter_events = by_quarter.get(quarter, [])
        y = percentile_ranks([float(e.car1) for e in quarter_events])
        surprise_idx = [i for i, e in enumerate(quarter_events) if e.earnings_surprise is not None]
        surprise_ranks = percentile_ranks(
            [float(quarter_events[i].earnings_surprise) for i in surprise_idx]
        )
        surprise_by_idx = dict(zip(surprise_idx, surprise_ranks, strict=True))
        for i, (event, target) in enumerate(zip(quarter_events, y, strict=True)):
            history = history_by_key.get(f"{event.event_id}:{event.ticker}")
            focal_dt = parse_event_datetime(event)
            if history is None or focal_dt is None:
                continue  # unparseable timestamp: fail closed, drop the row
            n_sources += audit_history_row(
                focal_event_id=event.event_id,
                focal_ticker=event.ticker,
                focal_event_datetime=focal_dt,
                history=history,
            )
            if history.source_events:
                n_with_history += 1
            fls = extract_forward_looking_features(event.disclosure)
            vec = build_feature_vector_v2(fls=fls, history=history)
            audit_current_surprise_neutrality(vec.values)
            rows.append(
                RowV2(
                    event=event,
                    y=float(target),
                    surprise_percentile=surprise_by_idx.get(i),
                    values=vec.values,
                )
            )
    return rows, AuditResult(
        n_rows=len(rows), n_source_records=n_sources, n_rows_with_history=n_with_history
    )


def _fit_eval(
    train: list[RowV2],
    evaluate: list[RowV2],
    names: tuple[str, ...],
    alpha: float,
) -> np.ndarray:
    X_train = np.asarray([r.x(names) for r in train], dtype=float)
    y_train = np.asarray([r.y for r in train], dtype=float)
    X_eval = np.asarray([r.x(names) for r in evaluate], dtype=float)
    scaler = Standardizer.fit(X_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)
    return np.clip(model.predict(scaler.transform(X_eval)), *CLIP_BOUNDS)


def _select_alpha(train: list[RowV2], validation: list[RowV2], names: tuple[str, ...]) -> dict:
    """Competition-aligned alpha selection on VALIDATION only (v1's rule)."""
    candidates = []
    for alpha in ALPHAS:
        pred = _fit_eval(train, validation, names, alpha)
        candidates.append(
            {
                "alpha": alpha,
                "metrics": asdict(compute_metrics(pred, [r.y for r in validation])),
                "official": asdict(_official(validation, pred)),
            }
        )

    def key(item: dict) -> tuple[float, float, float]:
        delta = item["official"]["delta_r_squared"]
        pearson = item["metrics"]["pearson"]
        return (
            -math.inf if delta is None else float(delta),
            -math.inf if pearson is None else float(pearson),
            -float(item["alpha"]),
        )

    chosen = max(candidates, key=key)
    return {"selected_alpha": float(chosen["alpha"]), "candidates": candidates,
            "validation_metrics": chosen["metrics"], "validation_official": chosen["official"]}


def _official(rows: list[RowV2], predicted: np.ndarray) -> object:
    class _Adapter:
        __slots__ = ("y", "surprise_percentile")

        def __init__(self, row: RowV2) -> None:
            self.y = row.y
            self.surprise_percentile = row.surprise_percentile

    return compute_official_metrics([_Adapter(r) for r in rows], predicted)


def train_and_serialize(
    source: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict:
    events = load_historical_events(source)
    rows, audit = build_rows(events)
    by_q = {
        q: [r for r in rows if r.event.quarter == q]
        for q in (TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER)
    }
    if any(not by_q[q] for q in by_q):
        raise RuntimeError("all three sealed quarters are required to train v2")

    train, validation, holdout = by_q[TRAIN_QUARTER], by_q[VALIDATION_QUARTER], by_q[HOLDOUT_QUARTER]

    # ---- Phase A: per-ablation alpha selection on VALIDATION only ---------
    ablation_results: dict[str, dict] = {}
    for name, feature_names in ABLATIONS.items():
        ablation_results[name] = _select_alpha(train, validation, feature_names)

    v2_alpha = ablation_results["all_v2"]["selected_alpha"]
    v1_equiv_alpha = ablation_results["fls_only"]["selected_alpha"]

    # ---- Phase B: freeze; refit TRAIN+VALIDATION; single locked-holdout read
    development = train + validation
    holdout_reads: dict[str, dict] = {}
    for name, feature_names, alpha in (
        ("all_v2", MODEL_FEATURE_NAMES_V2, v2_alpha),
        ("fls_only", MODEL_FEATURE_NAMES, v1_equiv_alpha),
    ):
        pred = _fit_eval(development, holdout, feature_names, alpha)
        holdout_reads[name] = {
            "metrics": asdict(compute_metrics(pred, [r.y for r in holdout])),
            "official": asdict(_official(holdout, pred)),
        }
    holdout_reads["constant_0.5"] = {
        "metrics": asdict(compute_metrics(np.full(len(holdout), 0.5), [r.y for r in holdout])),
    }
    surprise_rows = [r for r in holdout if r.surprise_percentile is not None]
    holdout_reads["surprise_benchmark"] = {
        "metrics": asdict(
            compute_metrics(
                np.asarray([r.surprise_percentile for r in surprise_rows]),
                [r.y for r in surprise_rows],
            )
        ),
    }

    # ---- Phase C: predeclared promotion gate --------------------------------
    val_v2 = ablation_results["all_v2"]["validation_official"]["delta_r_squared"]
    val_v1 = ablation_results["fls_only"]["validation_official"]["delta_r_squared"]
    hold_v2 = holdout_reads["all_v2"]["official"]["delta_r_squared"]
    hold_v1 = holdout_reads["fls_only"]["official"]["delta_r_squared"]
    std_v2 = holdout_reads["all_v2"]["metrics"]["prediction_std"]
    gate = {
        "validation_gain": None if val_v2 is None or val_v1 is None else val_v2 - val_v1,
        "holdout_regression": None if hold_v2 is None or hold_v1 is None else hold_v1 - hold_v2,
        "holdout_prediction_std": std_v2,
    }
    promoted = bool(
        gate["validation_gain"] is not None
        and gate["validation_gain"] >= PROMOTION_GATE["min_validation_delta_r2_gain"]
        and gate["holdout_regression"] is not None
        and gate["holdout_regression"] <= PROMOTION_GATE["max_holdout_delta_r2_regression"]
        and std_v2 >= PROMOTION_GATE["min_prediction_std"]
    )

    # ---- Phase D: final refit on all sealed data (unchanged spec) -----------
    final_rows = train + validation + holdout
    X_final = np.asarray([r.x(MODEL_FEATURE_NAMES_V2) for r in final_rows], dtype=float)
    y_final = np.asarray([r.y for r in final_rows], dtype=float)
    final_scaler = Standardizer.fit(X_final)
    final_model = Ridge(alpha=v2_alpha).fit(final_scaler.transform(X_final), y_final)

    artifact = {
        "model_version": MODEL_VERSION,
        "feature_spec_version": "v2",
        "feature_names": list(MODEL_FEATURE_NAMES_V2),
        "means": [float(x) for x in final_scaler.means],
        "standard_deviations": [float(x) for x in final_scaler.standard_deviations],
        "coefficients": [float(x) for x in final_model.coef_],
        "intercept": float(final_model.intercept_),
        "selected_alpha": v2_alpha,
        "clip_bounds": list(CLIP_BOUNDS),
        "promoted": promoted,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_quarters": [TRAIN_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER],
        "training_metadata": {
            "n_train": len(train),
            "n_validation": len(validation),
            "n_holdout": len(holdout),
            "n_final": len(final_rows),
            "selection_rule": "max validation official delta_r_squared; Pearson fallback",
            "promotion_gate": PROMOTION_GATE,
            "promotion_gate_observed": gate,
            "ablation_validation": ablation_results,
            "locked_holdout": holdout_reads,
            "leakage_audit": asdict(audit),
        },
        "data_provenance": {
            "history_source": "sealed Explaining Markets archive (data/historical/)",
            "availability_rule": (
                f"prior outcome usable only if prior_event_datetime + {AVAILABILITY_LAG_DAYS}d"
                f" < focal_event_datetime - {CUTOFF_GUARD_DAYS}d"
            ),
            "price_history_source": "none configured — features neutral with indicator 0",
            "current_surprise_source": "none legally available pre-cutoff — neutral, indicator 0",
        },
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - manual entry point
    artifact = train_and_serialize()
    md = artifact["training_metadata"]
    print(f"model_version={artifact['model_version']} alpha={artifact['selected_alpha']} "
          f"promoted={artifact['promoted']}")
    print(f"audit: {md['leakage_audit']}")
    print("validation (delta_r2 by ablation):")
    for name, res in md["ablation_validation"].items():
        official = res["validation_official"]
        print(f"  {name:28s} alpha={res['selected_alpha']:<6} "
              f"delta_r2={official['delta_r_squared']} pearson={res['validation_metrics']['pearson']}")
    print("locked holdout:")
    for name, res in md["locked_holdout"].items():
        line = f"  {name:28s} {res['metrics']}"
        if "official" in res:
            line += f" official={res['official']}"
        print(line)
    print(f"promotion gate observed: {md['promotion_gate_observed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
