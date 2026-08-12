from __future__ import annotations

import math

from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES,
    classify_statement,
    extract_forward_looking_features,
    is_earnings_related,
    is_forward_looking,
    is_quantitative,
)


def test_forward_looking_inflections() -> None:
    for text in (
        "We expect demand to improve.",
        "Management projected stronger margins.",
        "We are planning additional capacity.",
        "Revenue may accelerate.",
        "The company will seek growth.",
    ):
        assert is_forward_looking(text)
    assert not is_forward_looking("Revenue increased 8% last quarter.")


def test_quantitative_requires_financial_evidence_not_arbitrary_date() -> None:
    for text in (
        "We expect revenue of $4.2 billion.",
        "Margins should reach 28%.",
        "We project sales of 900M.",
        "EPS is expected at 2.15 dollars.",
    ):
        assert is_quantitative(text)
    assert not is_quantitative("The meeting is expected on August 14, 2026.")


def test_earnings_classification() -> None:
    for term in ("earnings", "EPS", "income", "losses", "profit"):
        assert is_earnings_related(f"We expect {term} to improve.")
    assert not is_earnings_related("We expect revenue growth.")


def test_all_four_forward_statement_categories() -> None:
    cases = {
        "We expect EPS of $2.50.": "quantitative_earnings",
        "We expect earnings to improve.": "nonquantitative_earnings",
        "We expect revenue growth of 12%.": "quantitative_non_earnings",
        "We expect customer demand to strengthen.": "nonquantitative_non_earnings",
    }
    for text, category in cases.items():
        assert classify_statement(text).category == category


def test_feature_set_is_complete_finite_and_uses_fact_denominator() -> None:
    disclosure = [
        "We expect EPS of $2.50 and stronger profit growth.",
        "We expect revenue to grow 12%.",
        "Results for 2025 were reported today.",
        "We believe demand will improve.",
    ]
    f = extract_forward_looking_features(disclosure)
    assert tuple(f.values) == MODEL_FEATURE_NAMES
    assert f.values["fls_count"] == 3
    assert f.values["fls_ratio"] == 0.75
    assert f.values["quant_earnings_fls_count"] == 1
    assert f.values["other_fls_count"] == 2
    assert all(math.isfinite(v) for v in f.values.values())


def test_negation_reverses_obvious_direction_and_tone_is_bounded() -> None:
    positive = extract_forward_looking_features(["We expect margins to improve and demand to be stronger."])
    negative = extract_forward_looking_features(["We do not expect margins to improve; demand could be weaker."])
    assert positive.values["signed_forward_tone"] > negative.values["signed_forward_tone"]
    assert -1.0 <= negative.values["signed_forward_tone"] <= 1.0


def test_guidance_direction_only_when_supported_by_text() -> None:
    raised = extract_forward_looking_features(["We raised full-year guidance and expect stronger profit."])
    lowered = extract_forward_looking_features(["We lowered our outlook and expect weaker demand."])
    maintained = extract_forward_looking_features(["We reaffirmed guidance and expect stable earnings."])
    no_guidance = extract_forward_looking_features(["We expect revenue growth."])
    assert raised.values["guidance_direction"] == 1.0
    assert lowered.values["guidance_direction"] == -1.0
    assert maintained.values["guidance_direction"] == 0.0
    assert no_guidance.values["guidance_direction"] == 0.0


def test_extractor_api_cannot_receive_post_event_target_fields() -> None:
    # There is intentionally no event object / **kwargs input through which
    # car1, surprise, event_returns, or baseline_predictions can enter.
    f = extract_forward_looking_features(["We expect earnings to improve."])
    forbidden = {"car1", "earnings_surprise", "surprise", "event_returns", "baseline_predictions", "y"}
    assert forbidden.isdisjoint(f.values)
