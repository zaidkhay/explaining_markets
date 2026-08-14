import json
from datetime import datetime, timezone

import pytest

from explaining_markets.evidence_bundle import persist_evidence_bundle
from explaining_markets.features_v3 import FEATURE_SPEC_VERSION_V3, MODEL_FEATURE_NAMES_V3, build_feature_vector_v3
from explaining_markets.model_v3 import MultiSignalV3Model
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.v3_records import V3Context


def test_promoted_artifact_requires_live_gate_evidence(tmp_path):
    n = len(MODEL_FEATURE_NAMES_V3)
    artifact = {
        "model_version": "multi_signal_v3",
        "feature_spec_version": FEATURE_SPEC_VERSION_V3,
        "feature_names": list(MODEL_FEATURE_NAMES_V3),
        "means": [0.0] * n,
        "standard_deviations": [1.0] * n,
        "coefficients": [0.0] * n,
        "intercept": 0.5,
        "clip_bounds": [0.05, 0.95],
        "promoted": True,
        "training_metadata": {"promotion_observed": {"tests_passing": True}},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="promotion evidence"):
        MultiSignalV3Model(path)


def test_evidence_bundle_is_write_once(tmp_path):
    cutoff = datetime(2026, 8, 13, 20, tzinfo=timezone.utc)
    reasoning = EventReasoner(use_openai=False).reason(values={}, cutoff=cutoff)
    context = V3Context(ticker="AAPL", cutoff=cutoff, event_reasoning=reasoning)
    vector = build_feature_vector_v3(disclosure=[], context=context)
    path = persist_evidence_bundle(
        context=context,
        vector=vector,
        event_id="event-1",
        model_version="test-v3",
        prediction=0.61,
        directory=tmp_path,
    )
    original = path.read_text(encoding="utf-8")
    persist_evidence_bundle(
        context=context,
        vector=vector,
        event_id="event-1",
        model_version="test-v3",
        prediction=0.12,
        directory=tmp_path,
    )
    assert path.read_text(encoding="utf-8") == original
    payload = json.loads(original)
    assert payload["prediction"] == 0.61
    assert payload["feature_spec_version"] == FEATURE_SPEC_VERSION_V3
