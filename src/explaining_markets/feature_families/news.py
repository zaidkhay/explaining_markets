"""Company, peer, and sector news features with cutoff filtering and deduplication."""
from __future__ import annotations

from datetime import timedelta
from difflib import SequenceMatcher

from explaining_markets.v3_records import NewsRecord

NEWS_FEATURE_NAMES = (
    "company_news_count_24h", "company_news_count_7d", "peer_news_count_24h",
    "sector_news_count_24h", "company_positive_news_score", "company_negative_news_score",
    "peer_positive_news_score", "peer_negative_news_score", "sector_positive_news_score",
    "sector_negative_news_score", "company_material_news_flag", "peer_material_news_flag",
    "sector_material_news_flag", "has_company_news", "has_peer_news", "has_sector_news",
)


def _canonical(record: NewsRecord) -> str:
    if record.url:
        return record.url.split("?", 1)[0].rstrip("/").lower()
    return " ".join(record.headline.lower().split())


def deduplicate_news(records: tuple[NewsRecord, ...], cutoff) -> tuple[NewsRecord, ...]:
    eligible = sorted(
        (r for r in records if r.eligible(cutoff) and r.published_at <= cutoff),
        key=lambda r: r.published_at,
    )
    kept: list[NewsRecord] = []
    keys: set[str] = set()
    for row in eligible:
        key = _canonical(row)
        if key in keys:
            continue
        title = " ".join(row.headline.lower().split())
        duplicate = False
        for prior in kept[-25:]:
            if prior.source != row.source:
                continue
            prior_title = " ".join(prior.headline.lower().split())
            if abs((row.published_at - prior.published_at).total_seconds()) <= 3600 and SequenceMatcher(None, title, prior_title).ratio() >= 0.92:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
            keys.add(key)
    return tuple(kept)


def _block(records: tuple[NewsRecord, ...], cutoff, prefix: str) -> dict[str, float]:
    rows = deduplicate_news(records, cutoff)
    day = [r for r in rows if r.published_at >= cutoff - timedelta(hours=24)]
    week = [r for r in rows if r.published_at >= cutoff - timedelta(days=7)]
    scores = [max(-1.0, min(1.0, float(r.sentiment))) for r in week if r.sentiment is not None]
    pos = sum(max(x, 0.0) for x in scores) / max(len(scores), 1)
    neg = sum(max(-x, 0.0) for x in scores) / max(len(scores), 1)
    return {
        f"{prefix}_news_count_24h": float(len(day)),
        f"{prefix}_positive_news_score": pos,
        f"{prefix}_negative_news_score": neg,
        f"{prefix}_material_news_flag": float(any(r.material for r in week)),
        f"has_{prefix}_news": float(bool(week)),
        f"{prefix}_news_count_7d": float(len(week)),
    }


def news_features(company: tuple[NewsRecord, ...], peers: tuple[NewsRecord, ...], sector: tuple[NewsRecord, ...], cutoff) -> dict[str, float]:
    c = _block(company, cutoff, "company")
    p = _block(peers, cutoff, "peer")
    s = _block(sector, cutoff, "sector")
    return {
        "company_news_count_24h": c["company_news_count_24h"],
        "company_news_count_7d": c["company_news_count_7d"],
        "peer_news_count_24h": p["peer_news_count_24h"],
        "sector_news_count_24h": s["sector_news_count_24h"],
        "company_positive_news_score": c["company_positive_news_score"],
        "company_negative_news_score": c["company_negative_news_score"],
        "peer_positive_news_score": p["peer_positive_news_score"],
        "peer_negative_news_score": p["peer_negative_news_score"],
        "sector_positive_news_score": s["sector_positive_news_score"],
        "sector_negative_news_score": s["sector_negative_news_score"],
        "company_material_news_flag": c["company_material_news_flag"],
        "peer_material_news_flag": p["peer_material_news_flag"],
        "sector_material_news_flag": s["sector_material_news_flag"],
        "has_company_news": c["has_company_news"],
        "has_peer_news": p["has_peer_news"],
        "has_sector_news": s["has_sector_news"],
    }
