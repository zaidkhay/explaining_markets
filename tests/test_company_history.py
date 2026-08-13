"""Company-history feature layer: price stats, reactions, z-scores, cutoffs."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from explaining_markets.company_history import (
    COMPANY_HISTORY_FEATURE_NAMES,
    SESSIONS_1Y,
    SESSIONS_3M,
    HistoricalEarningsEvent,
    HistoricalPriceStats,
    compute_company_history_features,
    compute_price_stats,
    empty_company_history,
)

UTC = timezone.utc
CUTOFF = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _prior(days_before: int, *, surprise=None, reaction=None, event_id=None) -> HistoricalEarningsEvent:
    return HistoricalEarningsEvent(
        event_timestamp=CUTOFF - timedelta(days=days_before),
        eps_surprise=surprise,
        abnormal_return=reaction,
        source="fixture",
        source_event_id=event_id,
    )


# ----- price stats -------------------------------------------------------------


def test_price_returns_require_the_full_window() -> None:
    closes = [100.0] * (SESSIONS_3M)  # one session short of a 3m return
    stats = compute_price_stats(closes)
    assert stats.return_3m is None

    closes = [100.0] * (SESSIONS_3M + 1)
    stats = compute_price_stats(closes)
    assert stats.return_3m == pytest.approx(0.0)


def test_price_return_computes_simple_window_return() -> None:
    closes = [50.0] + [50.0] * (SESSIONS_1Y - 1) + [100.0]  # 252 sessions ago -> now
    stats = compute_price_stats(closes)
    assert stats.return_1y == pytest.approx(1.0)  # doubled


def test_volatility_zero_for_constant_prices_and_positive_for_moves() -> None:
    flat = compute_price_stats([100.0] * (SESSIONS_3M + 1))
    assert flat.volatility_3m == pytest.approx(0.0)
    wiggly = compute_price_stats([100.0 + (i % 2) for i in range(SESSIONS_3M + 1)])
    assert wiggly.volatility_3m > 0.0


def test_max_drawdown_measures_peak_to_trough() -> None:
    # Rise to 200, fall to 100 -> 50% drawdown within the window.
    n = SESSIONS_1Y + 1
    closes = [100.0] * (n - 3) + [200.0, 100.0, 150.0]
    stats = compute_price_stats(closes)
    assert stats.max_drawdown_1y == pytest.approx(0.5)


def test_new_ipo_has_no_price_features() -> None:
    stats = compute_price_stats([100.0, 101.0])  # 2 sessions of history
    assert stats.return_3m is None
    assert stats.volatility_3m is None
    assert stats.max_drawdown_1y is None


# ----- earnings/surprise history --------------------------------------------------


def test_reaction_statistics() -> None:
    priors = [
        _prior(300, surprise=0.02, reaction=0.05),
        _prior(200, surprise=-0.01, reaction=-0.03),
        _prior(100, surprise=0.03, reaction=0.04),
    ]
    f = compute_company_history_features(ticker="AAPL", cutoff=CUTOFF, prior_events=priors)
    v = f.values
    assert v["prior_earnings_count"] == 3.0
    assert v["mean_prior_earnings_abnormal_return"] == pytest.approx((0.05 - 0.03 + 0.04) / 3)
    assert v["median_prior_earnings_abnormal_return"] == pytest.approx(0.04)
    assert v["positive_prior_earnings_rate"] == pytest.approx(2 / 3)
    assert v["mean_prior_eps_surprise"] == pytest.approx((0.02 - 0.01 + 0.03) / 3)
    assert v["positive_eps_surprise_rate"] == pytest.approx(2 / 3)
    assert v["negative_eps_surprise_rate"] == pytest.approx(1 / 3)
    assert v["mean_reaction_after_positive_surprise"] == pytest.approx((0.05 + 0.04) / 2)
    assert v["mean_reaction_after_negative_surprise"] == pytest.approx(-0.03)
    assert v["has_eps_surprise_history"] == 1.0


def test_std_requires_two_observations() -> None:
    f = compute_company_history_features(
        ticker="AAPL", cutoff=CUTOFF, prior_events=[_prior(100, surprise=0.01, reaction=0.02)]
    )
    assert f.values["std_prior_earnings_abnormal_return"] is None
    assert f.values["std_prior_eps_surprise"] is None


def test_recency_weighting_prefers_recent_reactions() -> None:
    # Old reaction -0.10, recent reaction +0.10: the weighted mean must be
    # positive and closer to the recent value; the unweighted mean is 0.
    priors = [_prior(1000, reaction=-0.10), _prior(10, reaction=0.10)]
    f = compute_company_history_features(ticker="AAPL", cutoff=CUTOFF, prior_events=priors)
    assert f.values["recency_weighted_earnings_reaction"] > 0.0


# ----- current surprise vs history -------------------------------------------------


def test_zscore_requires_availability_flag() -> None:
    with pytest.raises(ValueError, match="point-in-time source"):
        compute_company_history_features(
            ticker="AAPL", cutoff=CUTOFF, prior_events=[], current_eps_surprise=0.05
        )


def test_zscore_and_percentile_with_sufficient_history() -> None:
    priors = [
        _prior(400, surprise=0.00, reaction=0.01),
        _prior(300, surprise=0.01, reaction=0.02),
        _prior(200, surprise=0.02, reaction=0.03),
        _prior(100, surprise=0.03, reaction=0.04),
    ]
    f = compute_company_history_features(
        ticker="AAPL",
        cutoff=CUTOFF,
        prior_events=priors,
        current_eps_surprise=0.04,
        current_surprise_available=True,
    )
    v = f.values
    assert v["current_eps_surprise"] == pytest.approx(0.04)
    assert v["current_eps_surprise_zscore"] > 0.0
    assert v["current_eps_surprise_percentile_company"] == pytest.approx(1.0)
    assert v["has_similar_surprise_history"] == 1.0
    # Nearest-3 by |surprise diff| are 0.03, 0.02, 0.01 -> reactions 0.04, 0.03, 0.02.
    assert v["similar_surprise_mean_reaction"] == pytest.approx((0.04 + 0.03 + 0.02) / 3)
    assert v["similar_surprise_median_reaction"] == pytest.approx(0.03)
    assert v["similar_surprise_count"] == 3.0


def test_zscore_handles_low_sample_and_zero_variance() -> None:
    # Only two prior surprises -> below MIN_SURPRISE_HISTORY_FOR_ZSCORE.
    few = [_prior(300, surprise=0.01, reaction=0.02), _prior(100, surprise=0.02, reaction=0.01)]
    f = compute_company_history_features(
        ticker="AAPL", cutoff=CUTOFF, prior_events=few,
        current_eps_surprise=0.05, current_surprise_available=True,
    )
    assert f.values["current_eps_surprise_zscore"] is None

    # Zero-variance history -> undefined z-score, never a division blowup.
    flat = [_prior(d, surprise=0.01, reaction=0.0) for d in (400, 300, 200, 100)]
    f = compute_company_history_features(
        ticker="AAPL", cutoff=CUTOFF, prior_events=flat,
        current_eps_surprise=0.05, current_surprise_available=True,
    )
    assert f.values["current_eps_surprise_zscore"] is None


def test_zscore_outliers_are_clipped() -> None:
    priors = [_prior(d, surprise=s, reaction=0.0) for d, s in ((400, 0.010), (300, 0.011), (200, 0.012), (100, 0.013))]
    f = compute_company_history_features(
        ticker="AAPL", cutoff=CUTOFF, prior_events=priors,
        current_eps_surprise=10.0, current_surprise_available=True,
    )
    assert f.values["current_eps_surprise_zscore"] == pytest.approx(5.0)


# ----- cutoff enforcement / missing history --------------------------------------


def test_prior_event_at_or_after_cutoff_is_rejected() -> None:
    at_cutoff = HistoricalEarningsEvent(event_timestamp=CUTOFF, source="fixture")
    with pytest.raises(ValueError, match="strictly"):
        compute_company_history_features(ticker="AAPL", cutoff=CUTOFF, prior_events=[at_cutoff])


def test_naive_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_company_history_features(
            ticker="AAPL", cutoff=datetime(2026, 6, 1), prior_events=[]
        )


def test_empty_history_yields_indicators_not_fabricated_values() -> None:
    f = empty_company_history("NEWIPO", CUTOFF)
    v = f.values
    assert v["prior_earnings_count"] == 0.0
    assert v["has_1y_price_history"] == 0.0
    assert v["has_5y_price_history"] == 0.0
    assert v["has_eps_surprise_history"] == 0.0
    assert v["has_similar_surprise_history"] == 0.0
    assert v["mean_prior_earnings_abnormal_return"] is None  # missing, not zero
    assert v["current_eps_surprise"] is None
    assert set(v) == set(COMPANY_HISTORY_FEATURE_NAMES)


def test_all_populated_values_are_finite() -> None:
    priors = [_prior(d, surprise=0.01 * d, reaction=0.001 * d) for d in (400, 300, 200, 100)]
    f = compute_company_history_features(ticker="AAPL", cutoff=CUTOFF, prior_events=priors)
    for name, value in f.values.items():
        if value is not None:
            assert math.isfinite(value), name


def test_price_stats_flow_into_feature_block() -> None:
    stats = HistoricalPriceStats(
        n_sessions=1300, return_1y=0.25, return_5y=1.5, volatility_1y=0.02,
        max_drawdown_1y=0.3, max_drawdown_5y=0.6, return_3m=0.05, return_6m=0.1,
        return_3y=0.9, volatility_3m=0.015,
    )
    f = compute_company_history_features(
        ticker="AAPL", cutoff=CUTOFF, prior_events=[], price_stats=stats
    )
    assert f.values["return_1y"] == 0.25
    assert f.values["has_1y_price_history"] == 1.0
    assert f.values["has_5y_price_history"] == 1.0
