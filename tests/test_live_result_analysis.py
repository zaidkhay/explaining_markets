from __future__ import annotations

import json

from explaining_markets.live_result_analysis import analyze_recent_results, metric_block, shrink_prediction


def test_metric_block_and_shrinkage():
    block = metric_block([0.9, 0.5, 0.1], [0.8, 0.5, 0.2])
    assert block["n"] == 3
    assert block["spearman"] == 1.0
    assert abs(block["mae"] - (0.1 + 0.0 + 0.1) / 3) < 1e-12
    assert shrink_prediction(0.9, 0.5) == 0.7
    assert shrink_prediction(0.1, 0.0) == 0.5


def test_recent_analysis_joins_prediction_diagnostics(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    diag = {
        "event": {"event_id": "e1", "ticker": "AAA"},
        "score": {
            "submitted_percentile": 0.9,
            "features": [
                {
                    "feature": "guidance_direction",
                    "family": "forward_looking",
                    "raw_score_contribution": 0.04,
                },
                {
                    "feature": "revenue_surprise_percent",
                    "family": "revenue_results",
                    "raw_score_contribution": 0.02,
                },
            ],
        },
        "claims": [
            {"interpretation_confidence": 0.9, "model_relevance": 0.8},
            {"interpretation_confidence": 0.8, "model_relevance": 0.7},
        ],
        "diagnostic_signal": {
            "mean_claim_interpretation_confidence": 0.85,
            "mean_claim_model_relevance": 0.75,
            "raw_signal_z_vs_validation": 1.2,
        },
        "external_context": {
            "provider_errors": 0,
            "nonzero_deployed_features": 8,
            "family_availability": {"revenue": 1.0, "eps": 1.0, "guidance": 1.0},
        },
    }
    (diagnostics / "e1__AAA.json").write_text(json.dumps(diag), encoding="utf-8")

    report = analyze_recent_results(
        [
            {"event_id": "e1", "ticker": "AAA", "realized_percentile": 0.2},
            {"event_id": "e2", "ticker": "BBB", "predicted_percentile": 0.3, "realized_percentile": 0.4},
            {"event_id": "e3", "ticker": "CCC", "predicted_percentile": 0.6, "realized_percentile": 0.7},
        ],
        diagnostics_dir=diagnostics,
        validation_metrics={"n": 100, "spearman": 0.1, "mae": 0.25},
    )
    assert report["summary"]["n"] == 3
    row = next(row for row in report["rows"] if row["ticker"] == "AAA")
    assert row["predicted_percentile"] == 0.9
    assert row["revenue_available"] is True
    assert row["model_relevance"] == 0.75
    assert row["family_contributions"]["forward_looking"] == 0.04
    assert report["counterfactual_comparisons"]["shrink_0.00"]["spearman"] is None
    assert report["feature_family_signal"]["revenue_results"]["n"] == 3
    assert report["recommendations"]


def test_analysis_refuses_missing_realized_targets(tmp_path):
    try:
        analyze_recent_results(
            [{"ticker": "AAA", "predicted_percentile": 0.8}],
            diagnostics_dir=tmp_path,
        )
    except ValueError as exc:
        assert "predicted_percentile and realized_percentile" in str(exc)
    else:
        raise AssertionError("expected ValueError")
