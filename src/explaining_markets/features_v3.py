"""Frozen multi-signal V3 feature specification and assembler."""
from __future__ import annotations

from dataclasses import dataclass

from explaining_markets.disclosure_results_v3 import (
    merge_earnings_records,
    merge_guidance_records,
    parse_disclosure_records,
)
from explaining_markets.feature_families.company_history_v3 import COMPANY_HISTORY_V3_FEATURE_NAMES, company_history_features_v3
from explaining_markets.feature_families.earnings_surprise import EARNINGS_SURPRISE_FEATURE_NAMES, earnings_surprise_features
from explaining_markets.feature_families.guidance_expectations import GUIDANCE_FEATURE_NAMES, guidance_expectation_features
from explaining_markets.feature_families.market_sector import MARKET_SECTOR_FEATURE_NAMES, market_sector_features
from explaining_markets.feature_families.news import NEWS_FEATURE_NAMES, news_features
from explaining_markets.feature_families.peer_sympathy import PEER_FEATURE_NAMES, peer_sympathy_features
from explaining_markets.feature_families.price_context import PRICE_CONTEXT_FEATURE_NAMES, price_context_features
from explaining_markets.feature_families.reasoning import REASONING_FEATURE_NAMES, reasoning_features
from explaining_markets.feature_families.revenue_results import REVENUE_SURPRISE_FEATURE_NAMES, revenue_surprise_features
from explaining_markets.forward_looking_features import MODEL_FEATURE_NAMES, ForwardLookingFeatures, extract_forward_looking_features
from explaining_markets.v3_records import V3Context

FEATURE_SPEC_VERSION_V3 = "v3.2-disclosure-results"

GUIDANCE_INTERACTION_FEATURE_NAMES = (
    "eps_beat_but_guidance_cut", "eps_miss_but_guidance_raised",
    "revenue_beat_and_guidance_above", "double_beat_and_raise", "double_miss_and_cut",
)

MODEL_FEATURE_NAMES_V3: tuple[str, ...] = (
    *MODEL_FEATURE_NAMES,
    *EARNINGS_SURPRISE_FEATURE_NAMES,
    *REVENUE_SURPRISE_FEATURE_NAMES,
    *GUIDANCE_FEATURE_NAMES,
    *GUIDANCE_INTERACTION_FEATURE_NAMES,
    *COMPANY_HISTORY_V3_FEATURE_NAMES,
    *PRICE_CONTEXT_FEATURE_NAMES,
    *MARKET_SECTOR_FEATURE_NAMES,
    *PEER_FEATURE_NAMES,
    *NEWS_FEATURE_NAMES,
    *REASONING_FEATURE_NAMES,
)

FORBIDDEN_V3_FEATURE_NAMES = frozenset({
    "car1", "current_event_car1", "post_event_return", "future_price", "future_news",
    "leaderboard_outcome", "realized_percentile", "predicted_percentile", "y",
})
if FORBIDDEN_V3_FEATURE_NAMES.intersection(MODEL_FEATURE_NAMES_V3):
    raise RuntimeError("V3 feature list contains a forbidden realized/future field")
if len(MODEL_FEATURE_NAMES_V3) != len(set(MODEL_FEATURE_NAMES_V3)):
    raise RuntimeError("MODEL_FEATURE_NAMES_V3 contains duplicate feature names")


@dataclass(frozen=True)
class FeatureVectorV3:
    values: dict[str, float]
    fls: ForwardLookingFeatures

    def vector(self, names: tuple[str, ...] = MODEL_FEATURE_NAMES_V3) -> list[float]:
        return [float(self.values[name]) for name in names]


def _guidance_interactions(eps, rev, guide, fls_values):
    raised = bool(guide["numeric_guidance_raised"] or fls_values.get("guidance_raised", 0.0))
    lowered = bool(guide["numeric_guidance_lowered"] or fls_values.get("guidance_lowered", 0.0))
    return {
        "eps_beat_but_guidance_cut": float(eps["is_eps_beat"] and lowered),
        "eps_miss_but_guidance_raised": float(eps["is_eps_miss"] and raised),
        "revenue_beat_and_guidance_above": float(rev["is_revenue_beat"] and guide["guidance_above_consensus"]),
        "double_beat_and_raise": float(rev["eps_beat_and_revenue_beat"] and raised),
        "double_miss_and_cut": float(rev["eps_miss_and_revenue_miss"] and lowered),
    }


def build_feature_vector_v3(*, disclosure: list[str], context: V3Context) -> FeatureVectorV3:
    """Build V3 features using vendor records plus focal-disclosure fallbacks.

    Complete provider actual/consensus pairs win.  When they are absent, the
    competition disclosure may supply the same information directly (for
    example "EPS beat consensus by 12%").  This same function is used by the
    historical enrichment pipeline and live inference, keeping feature
    semantics aligned.
    """
    fls = extract_forward_looking_features(disclosure)
    parsed = parse_disclosure_records(disclosure, ticker=context.ticker, cutoff=context.cutoff)
    current_earnings = merge_earnings_records(
        context.earnings, parsed.earnings, cutoff=context.cutoff
    )
    current_guidance = merge_guidance_records(
        context.guidance, parsed.guidance, cutoff=context.cutoff
    )

    eps = earnings_surprise_features(current_earnings, context.company_history, context.cutoff)
    rev = revenue_surprise_features(current_earnings, context.company_history, context.cutoff)
    guide = guidance_expectation_features(current_guidance, context.cutoff)
    price = price_context_features(context.stock_prices, context.cutoff)
    market = market_sector_features(context.stock_prices, context.market_prices, context.sector_prices, context.cutoff)
    peers = peer_sympathy_features(context.peers, context.peer_prices, context.peer_earnings, market["market_return_1d"], context.cutoff)
    news = news_features(context.company_news, context.peer_news, context.sector_news, context.cutoff)
    history = company_history_features_v3(
        context.company_history,
        context.cutoff,
        current_eps_surprise=eps["eps_surprise_percent"] if eps["has_eps_surprise"] else None,
        current_revenue_surprise=rev["revenue_surprise_percent"] if rev["has_revenue_surprise"] else None,
        shrinkage_priors=context.extras.get("history_shrinkage_priors"),
    )
    reasoning = reasoning_features(context.event_reasoning)
    values: dict[str, float] = {name: float(fls.values[name]) for name in MODEL_FEATURE_NAMES}
    blocks = (
        eps, rev, guide, _guidance_interactions(eps, rev, guide, fls.values), history,
        price, market, peers, news, reasoning,
    )
    for block in blocks:
        values.update({name: float(value) for name, value in block.items()})
    if tuple(values) != MODEL_FEATURE_NAMES_V3:
        raise ValueError("V3 feature assembly order does not match MODEL_FEATURE_NAMES_V3")
    return FeatureVectorV3(values=values, fls=fls)


def family_availability(vector: FeatureVectorV3) -> dict[str, float]:
    v = vector.values
    return {
        "eps": v["has_eps_surprise"], "revenue": v["has_revenue_surprise"],
        "guidance": v["has_numeric_guidance"], "guidance_consensus": v["has_guidance_consensus"],
        "price_5y": v["has_5y_price_history"], "company_history": v["has_company_earnings_history"],
        "peers": v["has_peer_data"], "company_news": v["has_company_news"],
        "peer_news": v["has_peer_news"], "sector_news": v["has_sector_news"],
        "reasoning": v["has_reasoning"],
    }
