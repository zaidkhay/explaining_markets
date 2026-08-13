"""Point-in-time records consumed by the V3 multi-signal feature system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TimedRecord:
    value_timestamp: datetime
    available_at: datetime
    retrieved_at: datetime
    source: str

    def eligible(self, cutoff: datetime) -> bool:
        return self.available_at <= cutoff


@dataclass(frozen=True)
class EarningsRecord(TimedRecord):
    ticker: str
    reported_eps: float | None = None
    consensus_eps: float | None = None
    reported_revenue: float | None = None
    consensus_revenue: float | None = None
    abnormal_return: float | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class GuidanceRecord(TimedRecord):
    ticker: str
    revenue_low: float | None = None
    revenue_high: float | None = None
    eps_low: float | None = None
    eps_high: float | None = None
    ebitda: float | None = None
    margin: float | None = None
    revenue_consensus: float | None = None
    eps_consensus: float | None = None
    direction: str | None = None
    material_kpis: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PriceRecord(TimedRecord):
    ticker: str
    close: float
    volume: float | None = None


@dataclass(frozen=True)
class CompanyMetadataRecord(TimedRecord):
    ticker: str
    sector: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    market_cap: float | None = None


@dataclass(frozen=True)
class PeerRecord(TimedRecord):
    ticker: str
    peer_ticker: str
    score: float
    reason: str


@dataclass(frozen=True)
class NewsRecord(TimedRecord):
    headline: str
    published_at: datetime
    entities: tuple[str, ...] = ()
    url: str | None = None
    source_id: str | None = None
    sentiment: float | None = None
    material: bool = False
    topic: str | None = None


@dataclass(frozen=True)
class V3Context:
    ticker: str
    cutoff: datetime
    earnings: EarningsRecord | None = None
    guidance: GuidanceRecord | None = None
    company_history: tuple[EarningsRecord, ...] = ()
    stock_prices: tuple[PriceRecord, ...] = ()
    market_prices: tuple[PriceRecord, ...] = ()
    sector_prices: tuple[PriceRecord, ...] = ()
    peers: tuple[PeerRecord, ...] = ()
    peer_prices: dict[str, tuple[PriceRecord, ...]] = field(default_factory=dict)
    peer_earnings: tuple[EarningsRecord, ...] = ()
    company_news: tuple[NewsRecord, ...] = ()
    peer_news: tuple[NewsRecord, ...] = ()
    sector_news: tuple[NewsRecord, ...] = ()
    metadata: CompanyMetadataRecord | None = None
    extras: dict[str, Any] = field(default_factory=dict)
