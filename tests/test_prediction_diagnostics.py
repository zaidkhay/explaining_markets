from datetime import datetime, timezone

from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.model_v3_lite import V3LiteCandidateModel
from explaining_markets.prediction_dashboard import render_prediction_dashboard
from explaining_markets.prediction_diagnostics import build_prediction_diagnostics, feature_contributions
from explaining_markets.v3_records import V3Context


def _context():
    return V3Context(ticker="TEST", cutoff=datetime(2026, 8, 20, 13, 0, tzinfo=timezone.utc))


def test_feature_contributions_reconstruct_raw_linear_score():
    disclosure = [
        "Revenue beat consensus by 8%.",
        "Management raised full-year guidance and expects stronger margins.",
    ]
    context = _context()
    vector = build_feature_vector_v3(disclosure=disclosure, context=context)
    model = V3LiteCandidateModel()
    block = feature_contributions(model, vector)

    reconstructed = block["intercept"] + block["sum_feature_contributions"]
    assert abs(reconstructed - block["raw_unclipped"]) < 1e-12
    assert len(block["features"]) == len(model.feature_names)
    assert 0.0 <= block["submitted_percentile"] <= 1.0


def test_claim_diagnostics_distinguish_interpretation_from_truth_confidence():
    disclosure = [
        "Revenue beat consensus by 8%.",
        "Management raised full-year guidance and expects stronger margins.",
        "The company opened a new office in Chicago.",
    ]
    context = _context()
    vector = build_feature_vector_v3(disclosure=disclosure, context=context)
    model = V3LiteCandidateModel()
    report = build_prediction_diagnostics(model=model, vector=vector, disclosure=disclosure, context=context)

    claims = report["claims"]
    assert len(claims) == 3
    assert claims[0]["interpretation_confidence"] >= 0.9
    assert claims[0]["model_relevance"] > claims[2]["model_relevance"]
    assert all(claim["truth_confidence"] is None for claim in claims)
    assert "not calibrated probability" in report["diagnostic_signal"]["note"]


def test_prediction_dashboard_renders_auditable_sections(tmp_path):
    disclosure = ["Revenue missed consensus by 5%."]
    context = _context()
    vector = build_feature_vector_v3(disclosure=disclosure, context=context)
    model = V3LiteCandidateModel()
    report = build_prediction_diagnostics(model=model, vector=vector, disclosure=disclosure, context=context)
    report["event"] = {"event_id": "evt", "ticker": "TEST", "cutoff": context.cutoff.isoformat()}

    output = render_prediction_dashboard(report, tmp_path / "dashboard.html")
    text = output.read_text(encoding="utf-8")
    assert "Claim-by-claim interpretation" in text
    assert "Top exact feature contributions" in text
    assert "Historical validation performance" in text
    assert "Truth confidence: not assessed" in text
