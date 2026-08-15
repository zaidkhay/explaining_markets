"""Build one audited live V3 context from cache + bounded fresh provider calls."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

from explaining_markets.cached_v3_context import context_from_existing_cache
from explaining_markets.features_v3 import build_feature_vector_v3
from explaining_markets.news_ranking import rank_news
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.v3_providers import V3ProviderBundle
from explaining_markets.v3_records import CompanyMetadataRecord, PeerRecord, V3Context


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe(receipts: list[dict], name: str, fn: Callable, default):
    started = _utcnow()
    try:
        value = fn()
        count = len(value) if isinstance(value, (tuple, list, dict)) else int(value is not None)
        receipts.append({"provider_call": name, "status": "ok", "count": count, "retrieved_at": started.isoformat()})
        return value
    except Exception as exc:
        receipts.append({"provider_call": name, "status": "error", "count": 0, "error": type(exc).__name__, "retrieved_at": started.isoformat()})
        return default


def _event_metadata(ticker: str, event: dict, cutoff) -> CompanyMetadataRecord | None:
    sector = event.get("sector") or event.get("focal_sector")
    industry = event.get("industry")
    sub_industry = event.get("sub_industry")
    if not any((sector, industry, sub_industry)):
        return None
    return CompanyMetadataRecord(
        value_timestamp=cutoff,
        available_at=cutoff,
        retrieved_at=_utcnow(),
        source="event_payload",
        ticker=ticker,
        sector=str(sector) if sector else None,
        industry=str(industry) if industry else None,
        sub_industry=str(sub_industry) if sub_industry else None,
    )


def _event_peers(ticker: str, event: dict, cutoff) -> tuple[PeerRecord, ...]:
    raw = event.get("peer_tickers") or ()
    out = []
    for index, peer in enumerate(raw[:10] if isinstance(raw, list) else ()):
        name = str(peer).strip().upper()
        if not name or name == ticker.upper():
            continue
        out.append(PeerRecord(
            value_timestamp=cutoff,
            available_at=cutoff,
            retrieved_at=_utcnow(),
            source="event_payload",
            ticker=ticker,
            peer_ticker=name,
            score=max(0.1, 1.0 - 0.05 * index),
            reason="explicit event peer",
        ))
    return tuple(out)


def build_live_v3_context(*, ticker: str, event: dict, cutoff, providers: V3ProviderBundle) -> V3Context:
    """Assemble live context while failing individual provider families closed."""
    receipts: list[dict] = []
    base = context_from_existing_cache(ticker, cutoff)

    earnings = _safe(receipts, "earnings.current", lambda: providers.earnings.current(ticker, cutoff), None)
    if earnings is not None and not earnings.eligible(cutoff):
        earnings = None
    guidance = _safe(receipts, "guidance.current", lambda: providers.guidance.current(ticker, cutoff), None)
    if guidance is not None and not guidance.eligible(cutoff):
        guidance = None
    metadata = _safe(receipts, "metadata", lambda: providers.metadata.metadata(ticker, cutoff), None)
    if metadata is not None and not metadata.eligible(cutoff):
        metadata = None
    if metadata is None:
        metadata = _event_metadata(ticker, event, cutoff)

    peers = _safe(receipts, "peers", lambda: providers.peers.peers(ticker, cutoff, limit=10), ())
    peers = tuple(row for row in peers if row.eligible(cutoff))
    if not peers:
        peers = _event_peers(ticker, event, cutoff)
    peer_names = tuple(row.peer_ticker for row in peers[:10])

    peer_prices: dict[str, tuple] = {}
    peer_earnings: list = []
    for peer in peer_names:
        cached = context_from_existing_cache(peer, cutoff)
        if cached.stock_prices:
            peer_prices[peer] = cached.stock_prices
        peer_earnings.extend(cached.company_history[-8:])

    sector_prices = base.sector_prices
    sector_ticker = event.get("sector_ticker")
    if sector_ticker:
        sector_prices = context_from_existing_cache(str(sector_ticker), cutoff).stock_prices

    company_raw = _safe(receipts, "news.company", lambda: providers.news.company_news(ticker, cutoff, days=7), ())
    peer_raw = _safe(receipts, "news.peers", lambda: providers.news.peer_news(peer_names, cutoff, days=7), ()) if peer_names else ()
    sector_name = metadata.sector if metadata is not None else event.get("sector")
    sector_raw = _safe(receipts, "news.sector", lambda: providers.news.sector_news(sector_name, cutoff, days=7), ()) if sector_name else ()

    company_ranked = rank_news(company_raw, cutoff, targets={ticker}, top_n=10, require_target=True)
    peer_ranked = rank_news(peer_raw, cutoff, targets=set(peer_names), top_n=10, require_target=True) if peer_names else ()
    sector_ranked = rank_news(sector_raw, cutoff, targets=set(), top_n=8, require_target=False) if sector_raw else ()

    reasoned_company = providers.article_reasoner.reason_many(company_ranked, relation="company")
    reasoned_peers = providers.article_reasoner.reason_many(peer_ranked, relation="peer")
    reasoned_sector = providers.article_reasoner.reason_many(sector_ranked, relation="sector")

    extras = dict(base.extras)
    vendor_receipts = list(getattr(providers.news, "receipts", ()) or ())
    extras.update({
        "event_id": event.get("event_id") or event.get("id"),
        "provider_receipts": tuple(receipts + vendor_receipts),
        "news_priority_scores": {
            "company": tuple(item.priority_score for item in company_ranked),
            "peer": tuple(item.priority_score for item in peer_ranked),
            "sector": tuple(item.priority_score for item in sector_ranked),
        },
    })
    preliminary = V3Context(
        ticker=ticker,
        cutoff=cutoff,
        earnings=earnings,
        guidance=guidance,
        company_history=base.company_history,
        stock_prices=base.stock_prices,
        market_prices=base.market_prices,
        sector_prices=sector_prices,
        peers=peers,
        peer_prices=peer_prices,
        peer_earnings=tuple(peer_earnings),
        company_news=tuple(item.record for item in company_ranked),
        peer_news=tuple(item.record for item in peer_ranked),
        sector_news=tuple(item.record for item in sector_ranked),
        reasoned_company_news=reasoned_company,
        reasoned_peer_news=reasoned_peers,
        reasoned_sector_news=reasoned_sector,
        metadata=metadata,
        extras=extras,
    )
    audit_context(preliminary)
    raw_vector = build_feature_vector_v3(disclosure=list(event.get("disclosure") or ()), context=preliminary)
    event_reasoning = providers.event_reasoner.reason(
        values=raw_vector.values,
        cutoff=cutoff,
        company_news=reasoned_company,
        peer_news=reasoned_peers,
        sector_news=reasoned_sector,
    )
    final = replace(preliminary, event_reasoning=event_reasoning)
    audit_context(final)
    return final


def feed_diagnostics(context: V3Context) -> dict[str, object]:
    reasoning_count = len(context.reasoned_company_news) + len(context.reasoned_peer_news) + len(context.reasoned_sector_news)
    return {
        "ticker": context.ticker,
        "cutoff": context.cutoff.isoformat(),
        "earnings_received": int(context.earnings is not None),
        "revenue_received": int(context.earnings is not None and context.earnings.reported_revenue is not None),
        "guidance_received": int(context.guidance is not None),
        "price_rows": len(context.stock_prices),
        "peer_count": len(context.peers),
        "company_news_count": len(context.company_news),
        "peer_news_count": len(context.peer_news),
        "sector_news_count": len(context.sector_news),
        "reasoned_news_count": reasoning_count,
        "event_reasoning_available": int(context.event_reasoning is not None),
        "provider_receipts": context.extras.get("provider_receipts", ()),
    }
