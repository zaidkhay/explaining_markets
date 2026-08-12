"""Turn pre-event / event-time disclosure information into model features.

Hard rule, enforced by both the function signature below and
:func:`assert_no_leakage`: nothing in this module may read a realized,
post-event field (``car1``, ``earnings_surprise``, or any percentile/rank
derived from them). Only ``event_type``, ``ticker``, and disclosure/summary
TEXT are legitimate inputs — that text describes the event itself and is
"fair game" regardless of the competition's knowledge cutoff, whereas the
realized return/surprise are only known after the fact.

``predict.py`` and ``backtest.py`` are the only callers, and both are
structured so a realized field never reaches :func:`extract_features`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Small, literal keyword lexicons behind the transparent MVP heuristic in
# `model.py`. Kept here (not in model.py) so feature extraction is fully
# auditable/testable in isolation. Deliberately not an NLP pipeline — see
# `docs`/the historical-data investigation notes for why a more ambitious
# approach needs data we don't yet have.
_POSITIVE_TERMS = (
    "raised",
    "raise",
    "record",
    "beat",
    "above",
    "exceeded",
    "strong",
    "accelerat",
    "expand",
    "outperform",
    "upgrade",
    "buyback",
    "authorization",
    "reaccelerat",
)
_NEGATIVE_TERMS = (
    "missed",
    "miss",
    "below",
    "cut",
    "lowered",
    "decline",
    "weak",
    "delay",
    "shortfall",
    "downgrade",
    "compress",
    "warn",
)

# Fields that must never appear on a features dict — a defensive trip-wire
# against accidentally smuggling a realized/label field into model input.
FORBIDDEN_KEYS = ("car1", "earnings_surprise", "surprise", "predicted_percentile", "y")


@dataclass(frozen=True)
class FeatureVector:
    """A transparent, auditable feature set for one ``(event, ticker)``.

    Every field here is derived only from information available at or before
    the event (disclosure text + event metadata) — never from a realized
    outcome.
    """

    ticker: str
    event_type: str
    n_facts: int
    text_length: int
    positive_hits: int
    negative_hits: int
    net_sentiment: int  # positive_hits - negative_hits
    has_guidance_mention: bool

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "event_type": self.event_type,
            "n_facts": self.n_facts,
            "text_length": self.text_length,
            "positive_hits": self.positive_hits,
            "negative_hits": self.negative_hits,
            "net_sentiment": self.net_sentiment,
            "has_guidance_mention": self.has_guidance_mention,
        }


def extract_features(*, ticker: str, event_type: str, disclosure: list[str]) -> FeatureVector:
    """Build a :class:`FeatureVector` from disclosure facts only.

    ``disclosure`` must be a list of fact/summary sentences describing the
    event itself — e.g. the live webhook's ``information_url`` payload, or a
    historical archive record's ``disclosure.items[].content``. There is
    deliberately no parameter for realized return/surprise data; do not add
    one.
    """
    text = " ".join(disclosure)
    lowered = text.lower()
    positive_hits = sum(lowered.count(term) for term in _POSITIVE_TERMS)
    negative_hits = sum(lowered.count(term) for term in _NEGATIVE_TERMS)
    return FeatureVector(
        ticker=ticker,
        event_type=event_type,
        n_facts=len(disclosure),
        text_length=len(text),
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        net_sentiment=positive_hits - negative_hits,
        has_guidance_mention="guidance" in lowered,
    )


def assert_no_leakage(features: dict) -> None:
    """Raise if a features dict smuggles a realized/label field.

    Called by ``backtest.py`` before features ever reach a model, and safe to
    call anywhere else that builds a features dict. This is a cheap, explicit
    trip-wire against accidental look-ahead leakage — it does not replace the
    design-level separation already enforced by :func:`extract_features`'s
    signature, it backs it up.
    """
    leaked = [k for k in features if k in FORBIDDEN_KEYS]
    if leaked:
        raise ValueError(f"leaked realized field(s) into features: {leaked}")
