from datetime import datetime, timezone

from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.reasoning.event_reasoner import EventReasoner
from explaining_markets.v3_records import V3Context

CUTOFF = datetime(2026, 8, 13, 20, tzinfo=timezone.utc)


def _base(**overrides):
    values = {
        "has_eps_surprise": 1.0,
        "has_revenue_surprise": 1.0,
        "has_numeric_guidance": 1.0,
        "eps_surprise_percent": 0.04,
        "revenue_surprise_percent": 0.01,
        "guidance_surprise_percent": 0.01,
        "numeric_guidance_raised": 0.0,
        "numeric_guidance_lowered": 0.0,
        "guidance_raised": 0.0,
        "guidance_lowered": 0.0,
        "guidance_direction": 0.0,
        "return_20d": 0.0,
        "stock_minus_sector_20d": 0.0,
        "peer_abnormal_return_1d": 0.0,
        "recent_peer_eps_surprise_mean": 0.0,
        "recent_peer_revenue_surprise_mean": 0.0,
        "sector_return_5d": 0.0,
        "similar_earnings_event_mean_reaction": 0.0,
        "has_company_earnings_history": 0.0,
    }
    values.update(overrides)
    return values


def _vector(reasoning):
    return build_feature_vector_v3(disclosure=[], context=V3Context(ticker="XYZ", cutoff=CUTOFF, event_reasoning=reasoning))


def test_guidance_raise_vs_cut_changes_reasoning_features():
    r = EventReasoner(use_openai=False)
    raised = r.reason(values=_base(guidance_surprise_percent=0.06, numeric_guidance_raised=1.0), cutoff=CUTOFF)
    cut = r.reason(values=_base(guidance_surprise_percent=-0.06, numeric_guidance_lowered=1.0), cutoff=CUTOFF)
    assert raised.guidance_quality > cut.guidance_quality
    assert raised.overall_event_signal > cut.overall_event_signal
    assert _vector(raised).vector() != _vector(cut).vector()


def test_strong_vs_weak_peer_signal_changes_reasoning_features():
    r = EventReasoner(use_openai=False)
    strong = r.reason(values=_base(peer_abnormal_return_1d=0.08, recent_peer_eps_surprise_mean=0.08), cutoff=CUTOFF)
    weak = r.reason(values=_base(peer_abnormal_return_1d=-0.08, recent_peer_eps_surprise_mean=-0.08), cutoff=CUTOFF)
    assert strong.peer_signal > weak.peer_signal
    assert strong.overall_event_signal > weak.overall_event_signal
    assert _vector(strong).vector() != _vector(weak).vector()


def test_pre_event_runup_changes_priced_in_score():
    r = EventReasoner(use_openai=False)
    runup = r.reason(values=_base(return_20d=0.25, stock_minus_sector_20d=0.12), cutoff=CUTOFF)
    selloff = r.reason(values=_base(return_20d=-0.25, stock_minus_sector_20d=-0.12), cutoff=CUTOFF)
    assert runup.priced_in_score < selloff.priced_in_score
    assert _vector(runup).vector() != _vector(selloff).vector()


def test_eps_revenue_beat_vs_miss_changes_direction():
    r = EventReasoner(use_openai=False)
    beat = r.reason(values=_base(eps_surprise_percent=0.10, revenue_surprise_percent=0.05), cutoff=CUTOFF)
    miss = r.reason(values=_base(eps_surprise_percent=-0.10, revenue_surprise_percent=-0.05), cutoff=CUTOFF)
    assert beat.earnings_quality > miss.earnings_quality
    assert beat.revenue_quality > miss.revenue_quality
    assert beat.overall_event_signal > miss.overall_event_signal
