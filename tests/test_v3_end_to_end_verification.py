from explaining_markets.v3_verification import run_synthetic_suite, summarize_scores


def test_v3_synthetic_outcome_matrix_has_real_dispersion():
    rows = run_synthetic_suite()
    summary = summarize_scores(rows)
    assert len(rows) >= 10
    assert summary["spread"] >= 0.25
    assert summary["std"] >= 0.07
    assert summary["fraction_048_052"] < 0.5


def test_v3_synthetic_scores_are_not_all_049():
    rows = run_synthetic_suite()
    rounded = {round(row.score, 2) for row in rows}
    assert len(rounded) >= 5
    assert any(score > 0.60 for score in (row.score for row in rows))
    assert any(score < 0.40 for score in (row.score for row in rows))


def test_v3_economic_ordering_and_contradiction():
    rows = {row.name: row for row in run_synthetic_suite()}
    assert rows["strong_double_beat_raise"].score > rows["neutral_inline"].score
    assert rows["neutral_inline"].score > rows["strong_double_miss_cut"].score
    assert rows["positive_news_only"].score > rows["negative_news_only"].score
    assert rows["miss_but_raise_after_selloff"].score > rows["beat_but_guidance_cut"].score
    assert rows["beat_but_guidance_cut"].contradiction_score > 0


def test_v3_reasoning_signal_moves_with_inputs():
    rows = {row.name: row for row in run_synthetic_suite()}
    assert rows["strong_double_beat_raise"].overall_event_signal > 0
    assert rows["strong_double_miss_cut"].overall_event_signal < 0
    assert rows["positive_news_only"].company_news_signal > 0
    assert rows["negative_news_only"].company_news_signal < 0
