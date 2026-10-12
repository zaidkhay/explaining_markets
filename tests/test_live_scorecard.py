"""Live scorecard: schema-tolerant evidence loading, join reasons, scoring, alarms."""

from __future__ import annotations

import json
import random

import pytest

from explaining_markets.competition_scoring import percentile_ranks
from explaining_markets.live_scorecard import (
    EvidenceRow,
    alarms,
    contest_style_quarter_score,
    disclosure_looks_broken,
    event_table,
    join,
    load_evidence,
    null_expectation,
    parse_evidence,
    score_block,
    scorecard_summary,
)


def _ev(event_id="ea_AAA_Q3_2026", ticker="AAA", submitted=0.5, raw=None, created="2026-09-01T00:00:00",
        families=None, disclosure=None, nonzero_fls=None, schema="summary") -> EvidenceRow:
    return EvidenceRow(
        event_id=event_id, ticker=ticker, submitted=submitted, raw=raw, created_at=created,
        cutoff=created, schema=schema, path="",
        feature_availability=families if families is not None else {"reasoning": 1.0},
        disclosure=disclosure, nonzero_fls=nonzero_fls,
    )


def _record(event_id, ticker, car1, surprise=0.1, status="ok") -> dict:
    return {
        "event_id": event_id,
        "event_datetime": "2026-09-01T21:00:00Z",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
        "event_returns": {} if car1 is None else {ticker: {"car1": car1}},
        "metrics": {"earnings_surprise": {"surprise": surprise, "surprise_status": status}},
    }


# --- parsing ----------------------------------------------------------------------

def test_legacy_bundle_recovers_ticker_and_submitted_value() -> None:
    row = parse_evidence({"event_id": "ea_STEX_Q2_2026", "submitted_percentile": 0.31, "cutoff": "2026-08-17"})
    assert row is not None
    assert (row.ticker, row.submitted, row.schema) == ("STEX", 0.31, "legacy")


def test_dotted_ticker_parsed_from_event_id() -> None:
    row = parse_evidence({"event_id": "ea_BRK.B_Q3_2026", "submitted_percentile": 0.4})
    assert row is not None and row.ticker == "BRK.B"


def test_full_bundle_exposes_raw_disclosure_fls_and_contributions() -> None:
    row = parse_evidence({
        "event_id": "ea_AAA_Q3_2026", "ticker": "aaa", "prediction": 0.7, "raw_prediction": 0.6,
        "disclosure": ["Revenue rose 9% year over year to $1.2 billion."],
        "feature_values": {"fls_count": 2.0, "fls_ratio": 0.0, "is_revenue_beat": 1.0},
        "prediction_diagnostics": {"score": {"features": [
            {"feature": "fls_count", "raw_score_contribution": 0.05},
            {"feature": "is_revenue_beat", "raw_score_contribution": -0.01},
        ]}},
    })
    assert row is not None
    assert (row.ticker, row.schema, row.raw) == ("AAA", "full", 0.6)
    assert row.nonzero_fls == 1
    assert row.disclosure == ("Revenue rose 9% year over year to $1.2 billion.",)
    assert row.top_contributions[0] == ("fls_count", 0.05)


def test_bundle_without_any_prediction_is_skipped() -> None:
    assert parse_evidence({"event_id": "ea_AAA_Q3_2026", "ticker": "AAA"}) is None


def test_load_evidence_reads_nested_dirs_in_time_order(tmp_path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "b.json").write_text(json.dumps(
        {"event_id": "ea_BBB_Q3_2026", "ticker": "BBB", "prediction": 0.2, "created_at": "2026-09-02"}))
    (tmp_path / "a.json").write_text(json.dumps(
        {"event_id": "ea_AAA_Q3_2026", "submitted_percentile": 0.9, "cutoff": "2026-08-01"}))
    (tmp_path / "junk.json").write_text("{not json")
    rows = load_evidence(tmp_path)
    assert [r.ticker for r in rows] == ["AAA", "BBB"]


# --- join ---------------------------------------------------------------------------

def test_join_assigns_a_reason_to_every_row() -> None:
    archives = {"2026Q3": [
        _record("ea_AAA_Q3_2026", "AAA", 0.02),
        _record("ea_BBB_Q3_2026", "BBB", -0.01, surprise=None, status="missing_eps"),
        _record("ea_CCC_Q3_2026", "CCC", None),
        _record("ea_DDD_Q3_2026", "DDD", 0.03),
    ]}
    evidence = [
        _ev("ea_AAA_Q3_2026", "AAA"),
        _ev("ea_BBB_Q3_2026", "BBB"),
        _ev("ea_CCC_Q3_2026", "CCC"),
        _ev("ea_DDD_Q3_2026", "ZZZ"),
        _ev("modal-diagnostic-AAPL", "AAPL"),
    ]
    statuses = [row["join_status"] for row in join(evidence, archives)]
    assert statuses == [
        "joined", "no_surprise_benchmark", "outcome_pending", "ticker_not_in_event", "event_not_in_archive",
    ]


def test_percentiles_are_ranked_within_each_quarter_separately() -> None:
    archives = {
        "2026Q3": [_record("ea_A_Q3_2026", "A", 0.10), _record("ea_B_Q3_2026", "B", 0.20)],
        "2026Q4": [_record("ea_C_Q4_2026", "C", -0.50)],
    }
    a, c = join([_ev("ea_A_Q3_2026", "A"), _ev("ea_C_Q4_2026", "C")], archives)
    assert (a["outcome"]["quarter"], c["outcome"]["quarter"]) == ("2026Q3", "2026Q4")
    # A is the lower of Q3's two outcomes. C has the lowest CAR1 overall but is
    # alone in Q4, so it must get the single-element rank, not the bottom rank
    # it would get if the quarters were pooled.
    assert a["outcome"]["realized_percentile"] < 0.5
    assert c["outcome"]["realized_percentile"] == percentile_ranks([-0.50])[0]


# --- scoring ------------------------------------------------------------------------

def test_null_expectation_matches_claude_md_worked_example() -> None:
    assert null_expectation(0.2905, 18) == pytest.approx(0.0473, abs=1e-4)


def _synthetic(n, signal, seed=1):
    rng = random.Random(seed)
    surprise = [rng.random() for _ in range(n)]
    hidden = [rng.random() for _ in range(n)]
    realized = [0.5 * s + signal * h + 0.3 * rng.random() for s, h in zip(surprise, hidden)]
    return hidden, realized, surprise


def test_real_signal_beats_null_and_permutation() -> None:
    preds, realized, surprise = _synthetic(200, signal=0.6)
    block = score_block(preds, realized, surprise, permutations=200)
    assert block["delta_r_squared"] > 10 * block["null_expectation"]
    assert block["permutation_p"] < 0.01


def test_noise_is_indistinguishable_from_null() -> None:
    _, realized, surprise = _synthetic(200, signal=0.0)
    rng = random.Random(99)
    noise = [rng.random() for _ in range(200)]
    block = score_block(noise, realized, surprise, permutations=200)
    assert block["delta_r_squared"] >= 0.0  # non-negative by construction (§2.1)
    assert block["permutation_p"] > 0.05


def test_too_few_rows_is_not_scored() -> None:
    assert score_block([0.1, 0.2], [0.3, 0.4], [0.5, 0.6])["delta_r_squared"] is None


def test_summary_reports_raw_and_submitted_separately() -> None:
    preds, realized, surprise = _synthetic(40, signal=0.6)
    records = [
        _record(f"ea_T{i}_Q3_2026", f"T{i}", realized[i], surprise=surprise[i]) for i in range(40)
    ]
    evidence = [
        _ev(f"ea_T{i}_Q3_2026", f"T{i}", submitted=preds[i], raw=preds[i] if i < 20 else None)
        for i in range(40)
    ]
    summary = scorecard_summary(join(evidence, {"2026Q3": records}), permutations=50)
    block = summary["quarters"]["2026Q3"]
    assert block["submitted"]["n"] == 40
    assert block["raw"]["n"] == 20
    assert summary["join_status_counts"] == {"joined": 40}


def test_contest_style_score_mean_imputes_missing_events() -> None:
    records = [_record(f"ea_T{i}_Q3_2026", f"T{i}", i / 10, surprise=(i * 7 % 10) / 10) for i in range(10)]
    joined = join([_ev("ea_T1_Q3_2026", "T1", 0.2), _ev("ea_T8_Q3_2026", "T8", 0.9)], {"2026Q3": records})
    block = contest_style_quarter_score(joined, records, "2026Q3")
    assert block["n_obs"] == 2
    assert block["imputed_event_count"] == 8
    assert block["imputed_mean"] == pytest.approx(0.55)


def test_event_table_computes_signed_error() -> None:
    records = [_record("ea_AAA_Q3_2026", "AAA", 0.05), _record("ea_BBB_Q3_2026", "BBB", 0.01)]
    table = event_table(join([_ev(submitted=0.9)], {"2026Q3": records}))
    assert table[0]["error"] == pytest.approx(0.9 - table[0]["realized_percentile"])


# --- alarms -------------------------------------------------------------------------

def _codes(found):
    return {(a.level, a.code) for a in found}


def test_identical_predictions_trigger_low_spread_and_repeat() -> None:
    rows = [_ev(f"ea_T{i}_Q3_2026", f"T{i}", submitted=0.7046, created=f"2026-09-{i + 1:02d}") for i in range(12)]
    codes = _codes(alarms(rows))
    assert ("critical", "low_spread") in codes
    assert ("warn", "repeated_value") in codes


def test_healthy_spread_raises_nothing_critical() -> None:
    rng = random.Random(3)
    rows = [_ev(f"ea_T{i}_Q3_2026", f"T{i}", submitted=rng.random(), created=f"2026-09-{i + 1:02d}",
                families={"reasoning": 1.0}, schema="full") for i in range(25)]
    assert not [a for a in alarms(rows) if a.level in ("critical", "warn")]


def test_zero_fls_on_every_recent_event_is_critical() -> None:
    rows = [_ev(f"ea_T{i}_Q3_2026", f"T{i}", submitted=i / 10, nonzero_fls=0) for i in range(10)]
    assert ("critical", "zero_fls") in _codes(alarms(rows))


def test_family_dropout_detected() -> None:
    before = [_ev(f"ea_A{i}_Q3_2026", f"A{i}", submitted=i / 30, created=f"2026-09-{i + 1:02d}",
                  families={"company_news": 1.0, "reasoning": 1.0}) for i in range(20)]
    after = [_ev(f"ea_B{i}_Q4_2026", f"B{i}", submitted=i / 10, created=f"2026-10-{i + 1:02d}",
                 families={"company_news": 0.0, "reasoning": 1.0}) for i in range(10)]
    found = alarms(before + after)
    assert any(a.code == "family_dropout" and "company_news" in a.message for a in found)


@pytest.mark.parametrize(
    "facts, broken",
    [
        ((), True),
        (("1.0", "ea_ADI_Q3_2026", "2026-08-19T21:00:00Z"), True),  # the §9.1 failure exactly
        (("Revenue increased 9% year over year to $1.2 billion.",), False),
        (("1.0", "Company raised full-year EPS guidance to $4.10-$4.20."), False),
    ],
)
def test_disclosure_sanity(facts, broken) -> None:
    assert disclosure_looks_broken(facts) is broken


def test_broken_disclosure_raises_critical_alarm() -> None:
    rows = [_ev(disclosure=("1.0", "ea_AAA_Q3_2026"), submitted=0.3)]
    assert ("critical", "disclosure_sanity") in _codes(alarms(rows))


def test_normal_raw_scale_dispersion_does_not_trip_low_spread() -> None:
    """Raw clipped submissions have sd ~0.044; that is healthy, not a failure."""
    rng = random.Random(5)
    rows = [_ev(f"ea_T{i}_Q4_2026", f"T{i}", submitted=0.49 + rng.gauss(0, 0.044),
                created=f"2026-10-{i + 1:02d}", schema="full") for i in range(12)]
    assert "low_spread" not in {a.code for a in alarms(rows)}
