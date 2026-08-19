"""Point-in-time parsing of realized disclosure facts for V3.

The competition information URL often supplies concise realized facts such as
"EPS beat consensus by 12%" rather than vendor-style actual/estimate fields.
V3 already has trained EPS/revenue/guidance feature families, so this module
maps those disclosure facts into the same record schema instead of inventing a
second set of live-only features.

Only the focal disclosure is used.  Parsed records are timestamped at the
focal cutoff and therefore cannot introduce post-cutoff information.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

from explaining_markets.v3_records import EarningsRecord, GuidanceRecord

PARSER_VERSION = "disclosure_results_v1"

_EPS_LABEL = re.compile(r"\b(?:eps|earnings\s+per\s+share|diluted\s+earnings\s+per\s+share)\b", re.I)
_REVENUE_LABEL = re.compile(r"\b(?:revenue|revenues|sales|net\s+sales)\b", re.I)
_CONSENSUS = re.compile(r"\b(?:consensus|estimate|estimates|estimated|expected|expectations)\b", re.I)
_INLINE = re.compile(r"\b(?:in[ -]?line\s+with|matched|matches|matching)\s+(?:the\s+)?(?:consensus|estimate|estimates|expectations)\b", re.I)

_BEAT_WORDS = r"(?:beat|beats|beating|above|exceeded|exceeds|topped|tops)"
_MISS_WORDS = r"(?:miss|missed|misses|missing|below|fell\s+short\s+of)"
_PERCENT_AFTER_CONSENSUS = re.compile(
    rf"(?P<direction>{_BEAT_WORDS}|{_MISS_WORDS}).*?"
    r"(?:consensus|estimate|estimates|expectations).*?\bby\s+"
    r"(?P<pct>\d+(?:\.\d+)?)\s*%",
    re.I,
)
_PERCENT_BEFORE_CONSENSUS = re.compile(
    rf"(?P<direction>{_BEAT_WORDS}|{_MISS_WORDS}).*?\bby\s+"
    r"(?P<pct>\d+(?:\.\d+)?)\s*%.*?"
    r"(?:consensus|estimate|estimates|expectations)",
    re.I,
)

_AMOUNT_TOKEN = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<number>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>trillion|billion|million|thousand|[TtBbMmKk])?\b"
)
_VS_SPLIT = re.compile(r"\b(?:vs\.?|versus|compared\s+(?:with|to))\b", re.I)

_RAISED_GUIDANCE = re.compile(r"\b(?:raise[sd]?|raising|increase[sd]?|boost(?:ed|s)?|upped)\b.*?\b(?:guidance|outlook|forecast)\b|\b(?:guidance|outlook|forecast)\b.*?\b(?:raise[sd]?|raising|increase[sd]?|boost(?:ed|s)?|upped)\b", re.I)
_LOWERED_GUIDANCE = re.compile(r"\b(?:lower(?:ed|s)?|cut|cuts|cutting|reduce[sd]?|slash(?:ed|es)?)\b.*?\b(?:guidance|outlook|forecast)\b|\b(?:guidance|outlook|forecast)\b.*?\b(?:lower(?:ed|s)?|cut|cuts|cutting|reduce[sd]?|slash(?:ed|es)?)\b", re.I)
_REAFFIRMED_GUIDANCE = re.compile(r"\b(?:reaffirm(?:ed|s)?|maintain(?:ed|s)?|reiterate[sd]?)\b.*?\b(?:guidance|outlook|forecast)\b|\b(?:guidance|outlook|forecast)\b.*?\b(?:reaffirm(?:ed|s)?|maintain(?:ed|s)?|reiterate[sd]?)\b", re.I)


@dataclass(frozen=True)
class DisclosureResultRecords:
    earnings: EarningsRecord | None
    guidance: GuidanceRecord | None
    matched_fields: tuple[str, ...] = ()
    parser_version: str = PARSER_VERSION


def _scale_value(number: str, scale: str | None) -> float:
    value = float(number.replace(",", ""))
    key = (scale or "").lower()
    multiplier = {
        "k": 1e3,
        "thousand": 1e3,
        "m": 1e6,
        "million": 1e6,
        "b": 1e9,
        "billion": 1e9,
        "t": 1e12,
        "trillion": 1e12,
    }.get(key, 1.0)
    return value * multiplier


def _amounts(text: str) -> list[float]:
    values: list[float] = []
    for match in _AMOUNT_TOKEN.finditer(text):
        raw = match.group("number")
        # Avoid treating years as financial values when no currency/scale exists.
        plain = not match.group("currency") and not match.group("scale")
        try:
            number = float(raw.replace(",", ""))
        except ValueError:
            continue
        if plain and number.is_integer() and 1900 <= abs(number) <= 2100:
            continue
        values.append(_scale_value(raw, match.group("scale")))
    return values


def _exact_pair(line: str) -> tuple[float, float] | None:
    """Return reported/consensus when a fact contains two explicit values."""
    if not _CONSENSUS.search(line):
        return None
    parts = _VS_SPLIT.split(line, maxsplit=1)
    if len(parts) == 2:
        left, right = _amounts(parts[0]), _amounts(parts[1])
        if left and right:
            return left[-1], right[0]
    values = _amounts(line)
    if len(values) >= 2:
        return values[0], values[-1]
    return None


def _relative_pair(line: str) -> tuple[float, float] | None:
    """Encode an explicitly stated surprise percentage as a normalized pair."""
    if not _CONSENSUS.search(line):
        return None
    if _INLINE.search(line):
        return 1.0, 1.0
    match = _PERCENT_AFTER_CONSENSUS.search(line) or _PERCENT_BEFORE_CONSENSUS.search(line)
    if match is None:
        return None
    pct = float(match.group("pct")) / 100.0
    direction = match.group("direction").lower()
    negative = bool(re.search(_MISS_WORDS, direction, re.I))
    signed = -pct if negative else pct
    return 1.0 + signed, 1.0


def _metric_pair(line: str, label: re.Pattern[str]) -> tuple[float, float] | None:
    if not label.search(line):
        return None
    # Prefer exact actual/consensus values when present.  Percentage language is
    # a fallback for disclosure facts that only state the surprise magnitude.
    exact = _exact_pair(line)
    if exact is not None:
        return exact
    return _relative_pair(line)


def parse_disclosure_records(
    disclosure: list[str] | tuple[str, ...],
    *,
    ticker: str,
    cutoff: datetime,
) -> DisclosureResultRecords:
    eps_pair: tuple[float, float] | None = None
    revenue_pair: tuple[float, float] | None = None
    guidance_direction: str | None = None
    matched: list[str] = []

    for raw in disclosure:
        line = " ".join(str(raw).split())
        if not line:
            continue
        if eps_pair is None:
            pair = _metric_pair(line, _EPS_LABEL)
            if pair is not None:
                eps_pair = pair
                matched.append("eps")
        if revenue_pair is None:
            pair = _metric_pair(line, _REVENUE_LABEL)
            if pair is not None:
                revenue_pair = pair
                matched.append("revenue")
        if guidance_direction is None:
            if _RAISED_GUIDANCE.search(line):
                guidance_direction = "raised"
                matched.append("guidance_raised")
            elif _LOWERED_GUIDANCE.search(line):
                guidance_direction = "lowered"
                matched.append("guidance_lowered")
            elif _REAFFIRMED_GUIDANCE.search(line):
                guidance_direction = "reaffirmed"
                matched.append("guidance_reaffirmed")

    earnings = None
    if eps_pair is not None or revenue_pair is not None:
        earnings = EarningsRecord(
            value_timestamp=cutoff,
            available_at=cutoff,
            retrieved_at=cutoff,
            source=PARSER_VERSION,
            ticker=ticker.upper(),
            reported_eps=eps_pair[0] if eps_pair else None,
            consensus_eps=eps_pair[1] if eps_pair else None,
            reported_revenue=revenue_pair[0] if revenue_pair else None,
            consensus_revenue=revenue_pair[1] if revenue_pair else None,
        )

    guidance = None
    if guidance_direction is not None:
        guidance = GuidanceRecord(
            value_timestamp=cutoff,
            available_at=cutoff,
            retrieved_at=cutoff,
            source=PARSER_VERSION,
            ticker=ticker.upper(),
            direction=guidance_direction,
        )

    return DisclosureResultRecords(
        earnings=earnings,
        guidance=guidance,
        matched_fields=tuple(dict.fromkeys(matched)),
    )


def merge_earnings_records(
    provider: EarningsRecord | None,
    disclosure: EarningsRecord | None,
    *,
    cutoff: datetime,
) -> EarningsRecord | None:
    """Prefer complete provider pairs; fill otherwise from focal disclosure."""
    provider = provider if provider is not None and provider.eligible(cutoff) else None
    disclosure = disclosure if disclosure is not None and disclosure.eligible(cutoff) else None
    if provider is None:
        return disclosure
    if disclosure is None:
        return provider

    provider_eps_complete = provider.reported_eps is not None and provider.consensus_eps is not None
    disclosure_eps_complete = disclosure.reported_eps is not None and disclosure.consensus_eps is not None
    provider_rev_complete = provider.reported_revenue is not None and provider.consensus_revenue is not None
    disclosure_rev_complete = disclosure.reported_revenue is not None and disclosure.consensus_revenue is not None

    eps_source = provider if provider_eps_complete or not disclosure_eps_complete else disclosure
    rev_source = provider if provider_rev_complete or not disclosure_rev_complete else disclosure
    return EarningsRecord(
        value_timestamp=max(provider.value_timestamp, disclosure.value_timestamp),
        available_at=max(provider.available_at, disclosure.available_at),
        retrieved_at=max(provider.retrieved_at, disclosure.retrieved_at),
        source=f"{provider.source}+{PARSER_VERSION}",
        ticker=provider.ticker,
        reported_eps=eps_source.reported_eps,
        consensus_eps=eps_source.consensus_eps,
        reported_revenue=rev_source.reported_revenue,
        consensus_revenue=rev_source.consensus_revenue,
        abnormal_return=provider.abnormal_return,
        event_id=provider.event_id,
    )


def merge_guidance_records(
    provider: GuidanceRecord | None,
    disclosure: GuidanceRecord | None,
    *,
    cutoff: datetime,
) -> GuidanceRecord | None:
    provider = provider if provider is not None and provider.eligible(cutoff) else None
    disclosure = disclosure if disclosure is not None and disclosure.eligible(cutoff) else None
    if provider is None:
        return disclosure
    if disclosure is None:
        return provider
    if provider.direction:
        return provider
    return GuidanceRecord(
        value_timestamp=max(provider.value_timestamp, disclosure.value_timestamp),
        available_at=max(provider.available_at, disclosure.available_at),
        retrieved_at=max(provider.retrieved_at, disclosure.retrieved_at),
        source=f"{provider.source}+{PARSER_VERSION}",
        ticker=provider.ticker,
        revenue_low=provider.revenue_low,
        revenue_high=provider.revenue_high,
        eps_low=provider.eps_low,
        eps_high=provider.eps_high,
        ebitda=provider.ebitda,
        margin=provider.margin,
        revenue_consensus=provider.revenue_consensus,
        eps_consensus=provider.eps_consensus,
        direction=disclosure.direction,
        material_kpis=dict(provider.material_kpis),
    )
