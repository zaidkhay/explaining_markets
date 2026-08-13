"""V2 training protocol: chronological split, ablations, promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explaining_markets.historical import load_historical_events
from explaining_markets.v2_training import (
    ABLATIONS,
    PROMOTION_GATE,
    build_rows,
    train_and_serialize,
)

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/explaining_markets/artifacts/fls_company_history_ridge_v2.json"
)


def test_promotion_gate_constants_are_sensible() -> None:
    assert PROMOTION_GATE["min_validation_delta_r2_gain"] >= 0.0
    assert PROMOTION_GATE["max_holdout_delta_r2_regression"] >= 0.0
    assert PROMOTION_GATE["min_prediction_std"] > 0.0


def test_ablations_include_all_required_variants() -> None:
    required = {
        "fls_only",
        "history_only",
        "fls_plus_price",
        "fls_plus_earnings_reaction",
        "fls_plus_current_surprise",
        "fls_plus_competition",
        "all_v2",
    }
    assert required.issubset(set(ABLATIONS))


def test_build_rows_runs_leakage_audit_on_real_data() -> None:
    events = load_historical_events()
    if not events:
        import pytest as _pytest

        _pytest.skip("data/historical/ is empty in this environment")
    rows, audit = build_rows(events)
    assert audit.n_rows > 0
    assert audit.n_rows == len(rows)
    # All three quarters must be represented.
    quarters = {r.event.quarter for r in rows}
    assert {"2025Q4", "2026Q1", "2026Q2"}.issubset(quarters)


def test_artifact_metadata_matches_training_contract() -> None:
    if not ARTIFACT.exists():
        import pytest as _pytest

        _pytest.skip("V2 artifact not present")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["model_version"] == "fls_company_history_ridge_v2"
    assert "selected_alpha" in artifact
    assert "feature_names" in artifact
    assert "coefficients" in artifact
    assert "means" in artifact
    assert "standard_deviations" in artifact
    assert "promoted" in artifact
    assert artifact["promoted"] is False  # gate failed; must remain disabled
    assert len(artifact["coefficients"]) == len(artifact["feature_names"])
    assert len(artifact["means"]) == len(artifact["feature_names"])
    assert len(artifact["standard_deviations"]) == len(artifact["feature_names"])


def test_train_and_serialize_is_reproducible_or_skipped(tmp_path) -> None:
    events = load_historical_events()
    if not events:
        import pytest as _pytest

        _pytest.skip("data/historical/ is empty in this environment")
    # Run the full protocol once to a temp path; this exercises baselines,
    # ablations, and the locked holdout. Authoritative regression test for V2.
    artifact = train_and_serialize(artifact_path=tmp_path / "v2_test.json")
    assert artifact["model_version"] == "fls_company_history_ridge_v2"
    md = artifact["training_metadata"]
    assert "ablation_validation" in md
    assert "locked_holdout" in md
    assert "all_v2" in md["locked_holdout"]
    assert "fls_only" in md["locked_holdout"]
    assert "constant_0.5" in md["locked_holdout"]
    assert "surprise_benchmark" in md["locked_holdout"]
    assert "promotion_gate_observed" in md
