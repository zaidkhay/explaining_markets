"""Alpha Vantage Market News & Sentiment adapter for V3.

The vendor supplies an explicit ``time_published`` timestamp. We treat that
published timestamp as ``available_at`` and exclude any record published after
the focal cutoff. Retrieval time is provenance only and may be after cutoff.
If a row lacks a parseable publication timestamp it is excluded (fail closed).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import httpx

from explaining_markets.v3_records import NewsRecord

_API_URL = "https://www.alphavantage.co/query"
_SECTOR_TOPICS = {
    "technology": "technology",
    "information technology": "technology",
    "health care": "life_sciences",
    "healthcare": "life_sciences",
    "biotechnology": "life_sciences",
    "financials": "finance",
    "financial services": "finance",
    "industrials": "manufacturing",
    "manufacturing": "manufacturing",
    "energy": "energy_transportation",
    "transportation": "energy_transportation",
    "real estate": "real_estate",
    "consumer discretionary": "retail_wholesale",
    "consumer staples": "retail_wholesale",
    "retail": "retail_wholesale",
}
_MATERIAL_WORDS = (
    "guidance", "outlook", "earnings", "revenue", "margin", "fda", "approval",
    "lawsuit", "acquisition", "merger", "offering", "layoff", "contract", "customer",
    "supply", "pricing", "ceo", "cfo", "demand", "credit", "recall",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_published(value: object) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _source_id(item: dict) -> str:
    stable = f"{item.get('url','')}|{item.get('time_published','')}|{item.get('title','')}"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


class AlphaVantageNewsProvider:
    """Bounded live news provider. Provider failures return empty tuples."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 8.0,
        limit: int = 100,
        max_peer_queries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.limit = max(1, min(int(limit), 1000))
        self.max_peer_queries = max(1, min(int(max_peer_queries), 5))
        self.client = client
        self.receipts: list[dict] = []

    def _request(self, *, kind: str, cutoff: datetime, days: int, tickers: str | None = None, topic: str | None = None) -> tuple[NewsRecord, ...]:
        retrieved_at = _utcnow()
        params = {
            "function": "NEWS_SENTIMENT",
            "time_from": (cutoff - timedelta(days=days)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M"),
            "time_to": cutoff.astimezone(timezone.utc).strftime("%Y%m%dT%H%M"),
            "sort": "LATEST",
            "limit": str(self.limit),
            "apikey": self.api_key,
        }
        if tickers:
            params["tickers"] = tickers
        if topic:
            params["topics"] = topic
        try:
            if self.client is None:
                response = httpx.get(_API_URL, params=params, timeout=self.timeout_seconds)
            else:
                response = self.client.get(_API_URL, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            feed = payload.get("feed") if isinstance(payload, dict) else None
            if not isinstance(feed, list):
                note = payload.get("Note") or payload.get("Information") or payload.get("Error Message") if isinstance(payload, dict) else None
                raise RuntimeError(str(note or "Alpha Vantage response did not contain a feed"))
            rows = tuple(self._normalize(item, cutoff=cutoff, retrieved_at=retrieved_at) for item in feed if isinstance(item, dict))
            rows = tuple(row for row in rows if row is not None)
            self.receipts.append({"kind": kind, "status": "ok", "count": len(rows), "retrieved_at": retrieved_at.isoformat()})
            return rows
        except Exception as exc:
            self.receipts.append({"kind": kind, "status": "error", "count": 0, "error": type(exc).__name__, "retrieved_at": retrieved_at.isoformat()})
            return ()

    def _normalize(self, item: dict, *, cutoff: datetime, retrieved_at: datetime) -> NewsRecord | None:
        published = _parse_published(item.get("time_published"))
        if published is None or published > cutoff:
            return None
        title = str(item.get("title") or "").strip()
        if not title:
            return None
        ticker_sentiment = item.get("ticker_sentiment") or []
        entities: list[str] = []
        ticker_scores: list[float] = []
        relevance_scores: list[float] = []
        for row in ticker_sentiment:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker:
                entities.append(ticker)
            try:
                ticker_scores.append(float(row.get("ticker_sentiment_score")))
            except (TypeError, ValueError):
                pass
            try:
                relevance_scores.append(float(row.get("relevance_score")))
            except (TypeError, ValueError):
                pass
        overall = item.get("overall_sentiment_score")
        try:
            sentiment = float(overall)
        except (TypeError, ValueError):
            sentiment = sum(ticker_scores) / len(ticker_scores) if ticker_scores else None
        topics = item.get("topics") or []
        topic = None
        best_topic_score = -1.0
        for row in topics:
            if not isinstance(row, dict):
                continue
            try:
                score = float(row.get("relevance_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score > best_topic_score:
                best_topic_score = score
                topic = str(row.get("topic") or "").strip() or None
        summary = str(item.get("summary") or "").strip() or None
        text = f"{title} {summary or ''}".lower()
        return NewsRecord(
            value_timestamp=published,
            published_at=published,
            available_at=published,
            retrieved_at=retrieved_at,
            source=str(item.get("source") or "Alpha Vantage"),
            source_id=_source_id(item),
            headline=title,
            url=str(item.get("url") or "").strip() or None,
            entities=tuple(dict.fromkeys(entities)),
            sentiment=sentiment,
            material=any(word in text for word in _MATERIAL_WORDS),
            topic=topic,
            summary=summary,
            excerpt=None,
            vendor_relevance=max(relevance_scores) if relevance_scores else None,
        )

    def company_news(self, ticker, cutoff, days: int = 7) -> tuple[NewsRecord, ...]:
        return self._request(kind="company_news", tickers=str(ticker).upper(), cutoff=cutoff, days=days)

    def peer_news(self, tickers, cutoff, days: int = 7) -> tuple[NewsRecord, ...]:
        out: list[NewsRecord] = []
        for ticker in tuple(tickers)[: self.max_peer_queries]:
            out.extend(self._request(kind=f"peer_news:{ticker}", tickers=str(ticker).upper(), cutoff=cutoff, days=days))
        return tuple(out)

    def sector_news(self, sector, cutoff, days: int = 7) -> tuple[NewsRecord, ...]:
        key = str(sector or "").strip().lower()
        topic = _SECTOR_TOPICS.get(key)
        if not topic:
            for candidate, mapped in _SECTOR_TOPICS.items():
                if candidate in key:
                    topic = mapped
                    break
        if not topic:
            return ()
        return self._request(kind=f"sector_news:{topic}", topic=topic, cutoff=cutoff, days=days)
