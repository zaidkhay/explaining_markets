from __future__ import annotations

from explaining_markets.competition_scoring import (
    percentile_ranks,
    score_complete_predictions,
    score_with_missing_predictions,
)


def test_percentile_ranks_match_competition_semantics():
    assert percentile_ranks([]) == []
    assert percentile_ranks([5.0]) == [0.5]
    assert percentile_ranks([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert percentile_ranks([1.0, 1.0, 3.0]) == [0.25, 0.25, 1.0]


def test_constant_prediction_adds_zero_delta_r2():
    block = score_complete_predictions(
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.33, 0.66, 1.0],
        [0.0, 0.33, 0.66, 1.0],
    )
    assert block["delta_r_squared"] == 0.0
    assert block["beta"] == 0.0


def test_affine_prediction_remap_preserves_delta_r2():
    y = [0.0, 0.2, 0.6, 0.8, 1.0]
    surprise = [0.2, 0.0, 0.8, 0.4, 1.0]
    pred = [0.1, 0.7, 0.4, 0.9, 0.3]
    affine = [0.5 + 0.25 * (p - 0.5) for p in pred]
    a = score_complete_predictions(pred, y, surprise)
    b = score_complete_predictions(affine, y, surprise)
    assert a["delta_r_squared"] is not None
    assert b["delta_r_squared"] is not None
    assert abs(a["delta_r_squared"] - b["delta_r_squared"]) < 1e-12


def test_missing_predictions_are_mean_imputed():
    block = score_with_missing_predictions(
        [0.2, None, 0.8, None],
        [0.0, 0.33, 0.66, 1.0],
        [0.0, 0.66, 0.33, 1.0],
    )
    assert block["n_obs"] == 2
    assert block["imputed_event_count"] == 2
    assert block["imputed_mean"] == 0.5
