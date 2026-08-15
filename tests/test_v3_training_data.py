from __future__ import annotations

from pathlib import Path

from explaining_markets.historical import HistoricalEvent
from explaining_markets.model_v3 import MultiSignalV3Model
from explaining_markets.v3_research_training import serialize_research_linear_artifact
from explaining_markets.v3_training_data import (
    build_archive_seed_rows,
    load_training_rows,
    target_percentiles,
    training_data_report,
    write_training_rows,
)


def _event(event_id, ticker, quarter, when, car1, surprise, disclosure=None):
    return HistoricalEvent(
        event_id=event_id,
        ticker=ticker,
        event_type="EARNINGS_RELEASE",
        event_datetime=when,
        disclosure=list(disclosure or []),
        car1=car1,
        earnings_surprise=surprise,
        quarter=quarter,
    )


def _events():
    return [
        _event("e1", "AAPL", "2025Q4", "2025-10-30T20:00:00Z", -0.10, -0.05, ["Management expects demand to improve."]),
        _event("e2", "MSFT", "2025Q4", "2025-11-01T20:00:00Z", 0.20, 0.08, ["Company raised full-year guidance."]),
        _event("e3", "AAPL", "2026Q1", "2026-01-30T20:00:00Z", 0.10, 0.03, ["Management sees strong demand next quarter."]),
        _event("e4", "MSFT", "2026Q1", "2026-02-01T20:00:00Z", -0.05, -0.02, ["Company expects margins to decline."]),
    ]


def test_target_percentiles_match_competition_min_max_semantics():
    ranks = target_percentiles(_events())
    assert ranks[("e1", "AAPL")] == 0.0
    assert ranks[("e2", "MSFT")] == 1.0
    assert ranks[("e3", "AAPL")] == 1.0
    assert ranks[("e4", "MSFT")] == 0.0


def test_archive_seed_rows_use_prior_reactions_not_target_outcomes():
    rows = build_archive_seed_rows(_events())
    by_key = {(row.event_id, row.ticker): row for row in rows}

    first_aapl = by_key[("e1", "AAPL")]
    second_aapl = by_key[("e3", "AAPL")]
    assert first_aapl.values["has_company_earnings_history"] == 0.0
    assert second_aapl.values["has_company_earnings_history"] == 1.0
    assert second_aapl.values["prior_earnings_count"] == 1.0
    assert second_aapl.values["mean_prior_earnings_abnormal_return"] == -0.10
    assert "car1" not in second_aapl.values
    assert "target_percentile" not in second_aapl.values
    assert second_aapl.leakage_violations == 0


def test_training_row_round_trip_and_report(tmp_path: Path):
    rows = build_archive_seed_rows(_events())
    path = tmp_path / "rows.jsonl.gz"
    write_training_rows(rows, path)
    loaded = load_training_rows(path)
    assert loaded == rows

    report = training_data_report(loaded, archive_seed_only=True)
    assert report.rows == 4
    assert report.quarter_counts == {"2025Q4": 2, "2026Q1": 2}
    assert report.family_coverage["company_history"] == 0.5
    assert report.active_non_fls_features > 0
    assert report.archive_seed_only is True


def test_research_artifact_is_loadable_and_never_promoted(tmp_path: Path):
    rows = build_archive_seed_rows(_events())
    evaluation = {
        "promoted": False,
        "ablations": {
            "full_v3": {
                "candidates": [
                    {
                        "kind": "ridge",
                        "params": {"alpha": 1.0},
                        "metrics": {"pearson": 0.1, "mae": 0.2},
                    }
                ]
            }
        },
    }
    path = tmp_path / "v3_research.json"
    artifact = serialize_research_linear_artifact(rows, evaluation, path)
    assert artifact["promoted"] is False
    assert artifact["research_only"] is True
    assert "2026Q3" not in artifact["training_quarters"]

    model = MultiSignalV3Model(path)
    assert model.promoted is False
    assert model.model_version == "multi_signal_v3_research"
