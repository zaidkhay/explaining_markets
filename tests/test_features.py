"""Feature extraction: transparent, disclosure-only, leakage-guarded."""

from __future__ import annotations

import pytest

from explaining_markets.features import FeatureVector, assert_no_leakage, extract_features


def test_neutral_text_has_zero_net_sentiment() -> None:
    features = extract_features(
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        disclosure=["Quarterly results in line with expectations."],
    )
    assert features.positive_hits == 0
    assert features.negative_hits == 0
    assert features.net_sentiment == 0
    assert features.has_guidance_mention is False


def test_positive_terms_are_counted() -> None:
    features = extract_features(
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        disclosure=["Revenue beat expectations.", "Guidance raised for the full year."],
    )
    assert features.positive_hits >= 2  # "beat", "raised"
    assert features.negative_hits == 0
    assert features.net_sentiment > 0
    assert features.has_guidance_mention is True


def test_negative_terms_are_counted() -> None:
    features = extract_features(
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        disclosure=["Revenue missed consensus.", "Guidance lowered."],
    )
    assert features.negative_hits >= 2  # "missed", "lowered"
    assert features.net_sentiment < 0


def test_empty_disclosure_is_neutral() -> None:
    features = extract_features(ticker="AAPL", event_type="EARNINGS_RELEASE", disclosure=[])
    assert features.n_facts == 0
    assert features.text_length == 0
    assert features.net_sentiment == 0


def test_n_facts_and_text_length_reflect_input() -> None:
    disclosure = ["Fact one.", "Fact two.", "Fact three."]
    features = extract_features(ticker="AAPL", event_type="EARNINGS_RELEASE", disclosure=disclosure)
    assert features.n_facts == 3
    assert features.text_length == len(" ".join(disclosure))


def test_as_dict_contains_no_forbidden_keys() -> None:
    features = extract_features(
        ticker="AAPL", event_type="EARNINGS_RELEASE", disclosure=["Revenue beat."]
    )
    d = features.as_dict()
    assert_no_leakage(d)  # must not raise
    assert "car1" not in d
    assert "earnings_surprise" not in d
    assert "surprise" not in d


@pytest.mark.parametrize("leaked_key", ["car1", "earnings_surprise", "surprise", "y"])
def test_assert_no_leakage_raises_on_forbidden_key(leaked_key: str) -> None:
    with pytest.raises(ValueError, match="leaked"):
        assert_no_leakage({"ticker": "AAPL", leaked_key: 0.5})


def test_assert_no_leakage_allows_clean_dict() -> None:
    assert_no_leakage({"ticker": "AAPL", "net_sentiment": 2})  # must not raise


def test_extract_features_has_no_realized_field_parameter() -> None:
    # Structural guard: extract_features's signature must not accept a
    # realized/label field at all - not just "we didn't pass one."
    import inspect

    params = set(inspect.signature(extract_features).parameters)
    assert params == {"ticker", "event_type", "disclosure"}


def test_feature_vector_is_frozen() -> None:
    features = FeatureVector(
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        n_facts=0,
        text_length=0,
        positive_hits=0,
        negative_hits=0,
        net_sentiment=0,
        has_guidance_mention=False,
    )
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        features.ticker = "MSFT"  # type: ignore[misc]
