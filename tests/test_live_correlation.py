from __future__ import annotations

import pandas as pd
import pytest

from explaining_markets.live_correlation import (
    correlation_summary,
    expanding_correlation,
    load_joined_rows,
    plot_correlation,
)


def _frame(predicted, realized):
    return pd.DataFrame({"predicted_percentile": predicted, "realized_percentile": realized})


def test_correlation_summary_perfect_positive():
    df = _frame([0.1, 0.4, 0.9], [0.2, 0.5, 0.95])
    summary = correlation_summary(df)
    assert summary["n"] == 3
    assert summary["pearson"] > 0.99
    assert summary["spearman"] == 1.0


def test_correlation_summary_too_few_rows_is_none():
    df = _frame([0.5], [0.5])
    assert correlation_summary(df) == {"n": 1, "pearson": None, "spearman": None}


def test_correlation_summary_constant_column_is_none():
    df = _frame([0.5, 0.5, 0.5], [0.1, 0.4, 0.9])
    summary = correlation_summary(df)
    assert summary["n"] == 3
    assert summary["pearson"] is None
    assert summary["spearman"] is None


def test_load_joined_rows_drops_missing_and_sorts_chronologically(tmp_path):
    csv_path = tmp_path / "joined.csv"
    csv_path.write_text(
        "event_id,ticker,event_datetime,predicted_percentile,realized_percentile\n"
        "e2,BBB,2026-08-02T00:00:00+00:00,0.6,0.7\n"
        "e1,AAA,2026-08-01T00:00:00+00:00,0.2,0.3\n"
        "e3,CCC,2026-08-03T00:00:00+00:00,,0.9\n",
        encoding="utf-8",
    )
    df = load_joined_rows(csv_path)
    assert list(df["event_id"]) == ["e1", "e2"]
    assert len(df) == 2


def test_load_joined_rows_requires_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("ticker,predicted_percentile\nAAA,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_joined_rows(csv_path)


def test_expanding_correlation_needs_min_periods():
    df = _frame([0.1, 0.2], [0.9, 0.8])
    trend = expanding_correlation(df, min_periods=3)
    assert trend.dropna().empty


def test_expanding_correlation_tracks_growing_sample():
    df = _frame([0.1, 0.3, 0.5, 0.9], [0.15, 0.28, 0.55, 0.85])
    trend = expanding_correlation(df, min_periods=3)
    assert trend.isna().sum() == 2
    valid = trend.dropna()
    assert len(valid) == 2
    assert all(v > 0.9 for v in valid)


def test_plot_correlation_writes_png(tmp_path):
    df = _frame([0.1, 0.4, 0.6, 0.9], [0.2, 0.35, 0.65, 0.85])
    out = plot_correlation(df, tmp_path / "correlation.png")
    assert out.exists()
    assert out.stat().st_size > 0
