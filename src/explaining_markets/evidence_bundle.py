"""Immutable point-in-time evidence bundles for live V3 predictions."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from explaining_markets.features_v3 import FEATURE_SPEC_VERSION_V3, FeatureVectorV3, family_availability
from explaining_markets.v3_records import V3Context


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _default_dir() -> Path:
    configured = os.getenv("V3_EVIDENCE_DIR")
    if configured:
        return Path(configured)
    return Path("data") / "evidence"


def build_evidence_bundle(
    *,
    context: V3Context,
    vector: FeatureVectorV3,
    event_id: str,
    model_version: str,
    prediction: float,
    raw_prediction: float | None = None,
    disclosure: list[str] | tuple[str, ...] | None = None,
    prediction_diagnostics: dict[str, Any] | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "ticker": context.ticker,
        "cutoff": context.cutoff.isoformat(),
        "created_at": datetime.now(context.cutoff.tzinfo).isoformat(),
        "disclosure": [str(x) for x in (disclosure or ())],
        "provider_receipts": _jsonable(context.extras.get("provider_receipts", ())),
        "feature_availability": family_availability(vector),
        # Persist the exact point-in-time V3 values used to build the deployed
        # model input. This makes later realized-outcome attribution exact even
        # after signed information URLs expire.
        "feature_values": {name: float(value) for name, value in vector.values.items()},
        "earnings": _jsonable(context.earnings),
        "guidance": _jsonable(context.guidance),
        "price_context": {
            name: vector.values[name]
            for name in vector.values
            if name.startswith("return_")
            or name.startswith("realized_volatility_")
            or name.startswith("distance_from_")
        },
        "company_history_summary": {
            name: vector.values[name]
            for name in vector.values
            if name.startswith("prior_earnings")
            or name.startswith("mean_prior_")
            or name.startswith("similar_")
        },
        "peer_summary": {
            name: vector.values[name]
            for name in vector.values
            if name.startswith("peer_") or name.startswith("recent_peer_")
        },
        "selected_company_news": _jsonable(context.reasoned_company_news),
        "selected_peer_news": _jsonable(context.reasoned_peer_news),
        "selected_sector_news": _jsonable(context.reasoned_sector_news),
        "reasoning": _jsonable(context.event_reasoning),
        "model_version": model_version,
        "feature_spec_version": FEATURE_SPEC_VERSION_V3,
        "raw_prediction": None if raw_prediction is None else float(raw_prediction),
        "prediction": float(prediction),
        "prediction_diagnostics": _jsonable(prediction_diagnostics) if prediction_diagnostics is not None else None,
    }


def persist_evidence_bundle(
    *,
    context: V3Context,
    vector: FeatureVectorV3,
    event_id: str,
    model_version: str,
    prediction: float,
    raw_prediction: float | None = None,
    disclosure: list[str] | tuple[str, ...] | None = None,
    prediction_diagnostics: dict[str, Any] | None = None,
    directory: str | Path | None = None,
) -> Path:
    """Write once. A retry never mutates an already frozen evidence bundle."""
    root = Path(directory) if directory is not None else _default_dir()
    root.mkdir(parents=True, exist_ok=True)
    safe_event = "".join(c for c in str(event_id) if c.isalnum() or c in "-_") or "unknown"
    safe_ticker = "".join(c for c in context.ticker.upper() if c.isalnum() or c in ".-_")
    path = root / f"{safe_event}__{safe_ticker}.json"
    bundle = build_evidence_bundle(
        context=context,
        vector=vector,
        event_id=str(event_id),
        model_version=model_version,
        prediction=prediction,
        raw_prediction=raw_prediction,
        disclosure=disclosure,
        prediction_diagnostics=prediction_diagnostics,
    )
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        pass
    return path
