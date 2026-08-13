"""V2 feature vector assembly, ordering, imputation, and dimensionality."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from explaining_markets.company_history import (
    COMPANY_HISTORY_FEATURE_NAMES,
    empty_company_history,
)
from explaining_markets.competition_history import COMPETITION_FEATURE_NAMES, competition_feature_values
from explaining_markets.features_v2 import (
    FORBIDDEN_FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    MODEL_FEATURE_NAMES_V2,
    build_feature_vector_v2,
)
from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES as V1_FEATURE_NAMES,
    extract_forward_looking_features,
)

UTC = timezone.utc
CUTOFF = datetime(2026, 6, 1, tzinfo=UTC)


def test_v2_feature_names_preserve_v1_order_then_append_history() -> None:
    names = MODEL_FEATURE_NAMES_V2
    assert names[: len(V1_FEATURE_NAMES)] == tuple(V1_FEATURE_NAMES)
    assert names[len(V1_FEATURE_NAMES): len(V1_FEATURE_NAMES) + len(COMPANY_HISTORY_FEATURE_NAMES)] == tuple(
        COMPANY_HISTORY_FEATURE_NAMES
    )
    assert names[-len(COMPETITION_FEATURE_NAMES):] == tuple(COMPETITION_FEATURE_NAMES)
    assert len(names) == len(set(names))  # no duplicates
    assert len(names) == len(V1_FEATURE_NAMES) + len(COMPANY_HISTORY_FEATURE_NAMES) + len(COMPETITION_FEATURE_NAMES)


def test_build_feature_vector_v2_preserves_v1_block_then_history_block() -> None:
    disclosure = ["We expect revenue to grow.", "We raised guidance to $100 million."]
    fls = extract_forward_looking_features(disclosure)
    history = empty_company_history("AAPL", CUTOFF)
    vector = build_feature_vector_v2(fls=fls, history=history)
    assert len(vector.values) == len(MODEL_FEATURE_NAMES_V2)
    # V1 block is populated from FLS extraction.
    for name in V1_FEATURE_NAMES:
        assert name in vector.values
        assert math.isfinite(vector.values[name])
    # History block follows imputation policy: None -> 0.0.
    for name in COMPANY_HISTORY_FEATURE_NAMES:
        assert vector.values[name] == 0.0 or math.isfinite(vector.values[name])
    # Competition block is empty (no source events).
    assert vector.values["prior_competition_event_count"] == 0.0
    assert vector.values["has_competition_history"] == 0.0


def test_build_feature_vector_v2_vector_method_matches_names() -> None:
    disclosure = ["We expect strong growth."]
    fls = extract_forward_looking_features(disclosure)
    history = empty_company_history("AAPL", CUTOFF)
    vector = build_feature_vector_v2(fls=fls, history=history)
    raw = vector.vector(MODEL_FEATURE_NAMES_V2)
    assert len(raw) == len(MODEL_FEATURE_NAMES_V2)
    assert all(math.isfinite(v) for v in raw)


def test_forbidden_feature_names_are_not_in_v2_spec() -> None:
    assert FORBIDDEN_FEATURE_NAMES.isdisjoint(MODEL_FEATURE_NAMES_V2)


def test_feature_spec_version_is_v2() -> None:
    assert FEATURE_SPEC_VERSION == "v2"


def test_competition_features_are_a_subset_of_v2_names() -> None:
    comp = competition_feature_values([])
    for name in comp:
        assert name in MODEL_FEATURE_NAMES_V2


def test_v2_artifact_feature_order_matches_contract() -> None:
    """The trained artifact must use the same feature order as live assembly."""
    from pathlib import Path

    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "src/explaining_markets/artifacts/fls_company_history_ridge_v2.json"
    )
    if not artifact_path.exists():
        import pytest as _pytest

        _pytest.skip("V2 artifact not present in this environment")
    artifact = json.loads(artifact_path.read_text())
    artifact_names = tuple(artifact["feature_names"])
    assert artifact_names == MODEL_FEATURE_NAMES_V2
    assert len(artifact["coefficients"]) == len(MODEL_FEATURE_NAMES_V2)
    assert len(artifact["means"]) == len(MODEL_FEATURE_NAMES_V2)
    assert len(artifact["standard_deviations"]) == len(MODEL_FEATURE_NAMES_V2)
