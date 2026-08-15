"""Research-only V3 artifact serialization.

Production serialization in :mod:`explaining_markets.v3_training` is gated and
may only emit a promoted artifact after the honest holdout and all live-feed
checks pass. This module provides a separate, explicitly unpromoted artifact
for local/shadow inference while V3 is still being developed.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

from explaining_markets.features_v3 import FEATURE_SPEC_VERSION_V3, MODEL_FEATURE_NAMES_V3
from explaining_markets.v3_training import (
    CLIP_BOUNDS,
    LEGACY_HOLDOUT_QUARTER,
    TRAIN_QUARTER,
    VALIDATION_QUARTER,
    V3TrainingRow,
)

DEFAULT_RESEARCH_ARTIFACT = Path(__file__).resolve().parents[2] / "data" / "processed" / "multi_signal_v3_research.json"


def _candidate_score(candidate: dict) -> tuple[float, float]:
    metrics = candidate.get("metrics") or {}
    pearson = metrics.get("pearson")
    mae = metrics.get("mae")
    return (
        -math.inf if pearson is None else float(pearson),
        -math.inf if mae is None else -float(mae),
    )


def select_research_linear_candidate(evaluation: dict) -> dict:
    full = ((evaluation.get("ablations") or {}).get("full_v3") or {}).get("candidates") or []
    linear = [candidate for candidate in full if candidate.get("kind") in {"ridge", "elastic_net"}]
    if not linear:
        raise RuntimeError("evaluation contains no linear full-V3 candidate")
    return max(linear, key=_candidate_score)


def serialize_research_linear_artifact(
    rows: list[V3TrainingRow],
    evaluation: dict,
    artifact_path: str | Path = DEFAULT_RESEARCH_ARTIFACT,
) -> dict:
    """Fit a shadow V3 linear model and mark it permanently unpromoted.

    The research model is fit on 2025Q4 + 2026Q1 + the already-inspected
    2026Q2 legacy holdout when those rows exist. The honest 2026Q3 holdout is
    deliberately excluded so it remains untouched for promotion evaluation.
    """
    selected = select_research_linear_candidate(evaluation)
    development_quarters = {TRAIN_QUARTER, VALIDATION_QUARTER, LEGACY_HOLDOUT_QUARTER}
    fit_rows = [row for row in rows if row.quarter in development_quarters]
    if not fit_rows:
        raise RuntimeError("no V3 development rows are available for research artifact fitting")

    X = np.asarray([row.x(MODEL_FEATURE_NAMES_V3) for row in fit_rows], dtype=float)
    y = np.asarray([row.target_percentile for row in fit_rows], dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds <= 1e-12] = 1.0
    Z = (X - means) / stds

    kind = str(selected["kind"])
    params = dict(selected.get("params") or {})
    if kind == "ridge":
        model = Ridge(alpha=float(params["alpha"])).fit(Z, y)
    elif kind == "elastic_net":
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=20000,
        ).fit(Z, y)
    else:  # pragma: no cover - guarded by selector
        raise RuntimeError(f"unsupported research linear model: {kind}")

    artifact = {
        "model_version": "multi_signal_v3_research",
        "feature_spec_version": FEATURE_SPEC_VERSION_V3,
        "feature_names": list(MODEL_FEATURE_NAMES_V3),
        "means": [float(value) for value in means],
        "standard_deviations": [float(value) for value in stds],
        "coefficients": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "clip_bounds": list(CLIP_BOUNDS),
        "promoted": False,
        "structured_only": False,
        "research_only": True,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_quarters": sorted({row.quarter for row in fit_rows}),
        "training_rows": len(fit_rows),
        "selected_linear_candidate": selected,
        "training_metadata": {
            "research_only": True,
            "honest_holdout_excluded_from_fit": True,
            "evaluation_promoted": bool(evaluation.get("promoted", False)),
        },
    }
    output = Path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
