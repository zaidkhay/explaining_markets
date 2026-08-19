"""Offline serialization for an explicitly operator-selected V3-lite candidate.

The normal V3-lite promotion serializer remains untouched and still refuses to
write without the untouched holdout.  This module exists for an emergency
operator decision: it records the failed promotion state honestly and writes a
separate candidate artifact that production may opt into explicitly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

from explaining_markets.calibration import PercentileCalibrator
from explaining_markets.v3_lite_training import CLIP_BOUNDS, _standardize
from explaining_markets.v3_training import (
    LEGACY_HOLDOUT_QUARTER,
    TRAIN_QUARTER,
    VALIDATION_QUARTER,
    V3TrainingRow,
)

DEFAULT_OPERATOR_ARTIFACT = (
    Path(__file__).resolve().parent / "artifacts" / "v3_lite_candidate.json"
)


def serialize_operator_candidate(
    rows: Sequence[V3TrainingRow],
    *,
    feature_names: Sequence[str],
    kind: str,
    params: dict,
    calibrator: PercentileCalibrator,
    validation_metrics: dict,
    legacy_metrics: dict | None,
    artifact_path: str | Path = DEFAULT_OPERATOR_ARTIFACT,
    operator_reason: str,
) -> Path:
    """Write an unpromoted, explicitly operator-selected candidate artifact."""
    if kind not in {"ridge", "elastic_net"}:
        raise RuntimeError("operator V3-lite runtime supports linear candidates only")
    if not operator_reason.strip():
        raise ValueError("operator_reason is required")

    development_quarters = {TRAIN_QUARTER, VALIDATION_QUARTER, LEGACY_HOLDOUT_QUARTER}
    dev_rows = [row for row in rows if row.quarter in development_quarters]
    if not dev_rows:
        raise RuntimeError("no V3-lite development rows are available")

    names = tuple(str(name) for name in feature_names)
    X = np.asarray([row.x(names) for row in dev_rows], dtype=float)
    y = np.asarray([row.target_percentile for row in dev_rows], dtype=float)
    means, stds = _standardize(X)
    Z = (X - means) / stds

    if kind == "ridge":
        model = Ridge(alpha=float(params["alpha"])).fit(Z, y)
    else:
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=20000,
        ).fit(Z, y)

    artifact = {
        "model_version": "v3_lite_operator_2026_08_18",
        "feature_spec_version": "v3_lite_v1",
        "ablation": "fls_plus_reasoning",
        "feature_names": list(names),
        "means": [float(x) for x in means],
        "standard_deviations": [float(x) for x in stds],
        "coefficients": [float(x) for x in model.coef_],
        "intercept": float(model.intercept_),
        "clip_bounds": list(CLIP_BOUNDS),
        "calibration": calibrator.as_dict(),
        "promoted": False,
        "operator_override": True,
        "production_candidate": True,
        "structured_only": False,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_quarters": sorted(development_quarters),
        "training_metadata": {
            "n_development": len(dev_rows),
            "normal_promotion_gate_passed": False,
            "normal_promotion_blocker": "untouched 2026Q3 holdout unavailable",
            "operator_reason": operator_reason,
            "selected_kind": kind,
            "selected_params": dict(params),
            "validation_metrics": validation_metrics,
            "legacy_metrics": legacy_metrics,
            "calibration_source": calibrator.source,
            "honest_holdout_in_fit": False,
        },
    }

    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
