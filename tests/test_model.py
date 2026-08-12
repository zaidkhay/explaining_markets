"""Model interface: baseline/heuristic fallbacks and trained FLS default."""

from __future__ import annotations

from explaining_markets.features import FeatureVector
from explaining_markets.model import (
    BaselineModel,
    ForwardLookingRidgeModel,
    HeuristicFactModel,
    get_default_model,
)


def _features(net_sentiment: int) -> FeatureVector:
    positive = max(net_sentiment, 0)
    negative = max(-net_sentiment, 0)
    return FeatureVector(
        ticker="AAPL",
        event_type="EARNINGS_RELEASE",
        n_facts=1,
        text_length=10,
        positive_hits=positive,
        negative_hits=negative,
        net_sentiment=net_sentiment,
        has_guidance_mention=False,
    )


def test_baseline_model_always_returns_half() -> None:
    model = BaselineModel()
    assert model.predict_percentile(_features(0)) == 0.5
    assert model.predict_percentile(_features(5)) == 0.5
    assert model.predict_percentile(_features(-5)) == 0.5


def test_baseline_model_fit_is_a_noop() -> None:
    model = BaselineModel()
    model.fit([(_features(3), 0.9)])
    assert model.predict_percentile(_features(3)) == 0.5


def test_heuristic_model_neutral_sentiment_is_half() -> None:
    assert HeuristicFactModel().predict_percentile(_features(0)) == 0.5


def test_heuristic_model_positive_sentiment_raises_percentile() -> None:
    p = HeuristicFactModel().predict_percentile(_features(3))
    assert 0.5 < p <= 0.90


def test_heuristic_model_negative_sentiment_lowers_percentile() -> None:
    p = HeuristicFactModel().predict_percentile(_features(-3))
    assert 0.10 <= p < 0.5


def test_heuristic_model_is_monotonic_in_sentiment() -> None:
    model = HeuristicFactModel()
    scores = [model.predict_percentile(_features(s)) for s in range(-6, 7)]
    assert scores == sorted(scores)


def test_heuristic_model_never_exceeds_bounds_even_for_extreme_input() -> None:
    model = HeuristicFactModel()
    assert model.predict_percentile(_features(1000)) <= 0.90
    assert model.predict_percentile(_features(-1000)) >= 0.10


def test_heuristic_model_fit_is_a_noop() -> None:
    model = HeuristicFactModel()
    model.fit([(_features(3), 0.9)])
    assert model.predict_percentile(_features(0)) == 0.5


def test_get_default_model_returns_trained_fls_ridge() -> None:
    model = get_default_model()
    assert isinstance(model, ForwardLookingRidgeModel)
    assert model.model_version == "fls_ridge_v1"


def test_get_default_model_needs_no_arguments_and_predicts_finite_percentile() -> None:
    model = get_default_model()
    assert isinstance(model, ForwardLookingRidgeModel)
    p = model.predict_disclosure(["We expect revenue growth of 12% and stronger demand."])
    assert 0.0 <= p <= 1.0
