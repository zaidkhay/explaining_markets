"""Point-in-time news eligibility, deduplication, relevance and ranking."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher

from explaining_markets.v3_records import NewsRecord

_MATERIAL_TERMS = {
    "guidance", "forecast", "outlook", "earnings", "revenue", "margin", "layoff", "layoffs",
    "fda", "approval", "regulatory", "lawsuit", "litigation", "acquisition", "merger", "m&a",
    "offering", "capital raise", "customer", "contract", "supply", "shortage", "pricing",
    "ceo", "cfo", "management", "demand", "credit", "default", "recall",
}
_HIGH_QUALITY = ("reuters", "associated press", "ap news", "sec", "business wire", "globe newswire")
_MEDIUM_QUALITY = ("bloomberg", "wsj", "wall street journal", "financial times", "cnbc", "marketwatch", "barron's")


@dataclass(frozen=True)
class RankedNewsRecord:
    record: NewsRecord
    relevance: float
    materiality: float
    novelty: float
    source_quality: float
    recency_weight: float
    priority_score: float


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def canonical_url(record: NewsRecord) -> str:
    if not record.url:
        return ""
    return record.url.split("?", 1)[0].rstrip("/").lower()


def normalized_title(record: NewsRecord) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", record.headline.lower()).split())


def eligible_news(records, cutoff, *, days: int = 7) -> tuple[NewsRecord, ...]:
    floor = cutoff - timedelta(days=days)
    return tuple(
        row for row in records
        if row.published_at <= cutoff and row.available_at <= cutoff and row.published_at >= floor
    )


def deduplicate_news(records, cutoff, *, days: int = 7) -> tuple[NewsRecord, ...]:
    rows = sorted(eligible_news(records, cutoff, days=days), key=lambda r: (r.published_at, r.headline))
    kept: list[NewsRecord] = []
    urls: set[str] = set()
    source_ids: set[tuple[str, str]] = set()
    for row in rows:
        url = canonical_url(row)
        sid = (row.source.lower(), row.source_id or "")
        if url and url in urls:
            continue
        if row.source_id and sid in source_ids:
            continue
        title = normalized_title(row)
        duplicate = False
        for prior in kept[-40:]:
            hours = abs((row.published_at - prior.published_at).total_seconds()) / 3600.0
            if hours <= 8 and SequenceMatcher(None, title, normalized_title(prior)).ratio() >= 0.90:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(row)
        if url:
            urls.add(url)
        if row.source_id:
            source_ids.add(sid)
    return tuple(kept)


def source_quality(record: NewsRecord) -> float:
    source = record.source.lower()
    url = (record.url or "").lower()
    combined = f"{source} {url}"
    if any(token in combined for token in _HIGH_QUALITY):
        return 0.95
    if any(token in combined for token in _MEDIUM_QUALITY):
        return 0.85
    if any(token in combined for token in ("investor", "ir.", "sec.gov")):
        return 0.90
    return 0.60


def materiality_score(record: NewsRecord) -> float:
    text = f"{record.headline} {getattr(record, 'summary', '') or ''}".lower()
    hits = sum(term in text for term in _MATERIAL_TERMS)
    if record.material:
        hits += 2
    return _clip(0.30 + 0.16 * hits)


def relevance_score(record: NewsRecord, *, targets: set[str], broad_topic: bool = False) -> float:
    entities = {entity.upper() for entity in record.entities}
    normalized_targets = {x.upper() for x in targets}
    target_hits = len(entities.intersection(normalized_targets))
    vendor = getattr(record, "vendor_relevance", None)
    if broad_topic:
        base = float(vendor) if vendor is not None else 0.60
    else:
        base = float(vendor) if vendor is not None else (0.85 if target_hits else 0.10)
    if target_hits:
        base = max(base, min(1.0, 0.75 + 0.08 * target_hits))
    return _clip(base)


def rank_news(
    records,
    cutoff,
    *,
    targets: set[str],
    days: int = 7,
    top_n: int = 12,
    require_target: bool = True,
) -> tuple[RankedNewsRecord, ...]:
    rows = deduplicate_news(records, cutoff, days=days)
    ranked: list[RankedNewsRecord] = []
    seen_titles: list[str] = []
    normalized_targets = {x.upper() for x in targets}
    for row in sorted(rows, key=lambda r: r.published_at, reverse=True):
        entities = {x.upper() for x in row.entities}
        if require_target and normalized_targets and not entities.intersection(normalized_targets):
            continue
        title = normalized_title(row)
        max_similarity = max((SequenceMatcher(None, title, prior).ratio() for prior in seen_titles), default=0.0)
        novelty = _clip(1.0 - 0.75 * max_similarity)
        age_hours = max(0.0, (cutoff - row.published_at).total_seconds() / 3600.0)
        recency = math.exp(-age_hours / 72.0)
        relevance = relevance_score(row, targets=targets, broad_topic=not require_target)
        materiality = materiality_score(row)
        quality = source_quality(row)
        priority = relevance * materiality * novelty * quality * recency
        ranked.append(RankedNewsRecord(row, relevance, materiality, novelty, quality, recency, priority))
        seen_titles.append(title)
    ranked.sort(key=lambda x: (-x.priority_score, -x.record.published_at.timestamp(), x.record.headline))
    return tuple(ranked[:top_n])
