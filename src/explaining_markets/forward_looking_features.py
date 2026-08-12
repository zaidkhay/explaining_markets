"""Deterministic forward-looking-disclosure features for production and backtests.

The taxonomy follows Bozanic, Roulstone, and Van Buskirk: prospective
statements are separated along earnings/non-earnings and
quantitative/non-quantitative dimensions.  The competition gives us roughly
10 extracted disclosure facts instead of a full earnings announcement, so all
ratios below deliberately use the number of *available disclosure facts* as
the denominator.  This keeps historical and live extraction identical.

Only disclosure text enters this module.  Realized CAR1, earnings surprise,
event_returns, baseline_predictions, and other post-event outcome fields are
not accepted by the public API and cannot enter any feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Forward-looking constructions required by the paper/task, with ordinary
# English inflections. Word boundaries avoid accidental substring matches.
_FLS_RE = re.compile(
    r"\b(?:anticipat(?:e|es|ed|ing)|expect(?:s|ed|ing)?|forecast(?:s|ed|ing)?|"
    r"hop(?:e|es|ed|ing)|intend(?:s|ed|ing)?|plan(?:s|ned|ning)?|"
    r"project(?:s|ed|ing)?|seek(?:s|ing)?|sought|believ(?:e|es|ed|ing)|"
    r"goals?|objectives?|may|might|can|could|should|will)\b",
    re.IGNORECASE,
)
_EARNINGS_RE = re.compile(r"\b(?:earnings|eps|income|loss(?:es)?|profit(?:s)?)\b", re.I)

# Quantitative evidence must look financially meaningful. Bare years, dates,
# head counts, quarter numbers, etc. do not qualify merely for containing a digit.
_CURRENCY_RE = re.compile(
    r"(?:[$€£¥]\s*[-+]?\d[\d,]*(?:\.\d+)?|\b(?:usd|dollars?|euros?|yen|gbp)\s*[-+]?\d[\d,]*(?:\.\d+)?)",
    re.I,
)
_PERCENT_RE = re.compile(r"(?:[-+]?\d+(?:\.\d+)?\s*%|[-+]?\d+(?:\.\d+)?\s*percent\b)", re.I)
_SCALE_RE = re.compile(
    r"\b[-+]?\d[\d,]*(?:\.\d+)?\s*(?:thousand|million|billion|trillion|mn|mm|bn|[mbt])\b",
    re.I,
)
_FINANCIAL_NUMERIC_CONTEXT_RE = re.compile(
    r"\b(?:revenue|sales|margin|margins|eps|earnings|income|profit|loss|cash flow|"
    r"capex|capital expenditures?|ebitda|ebit|free cash flow|fcf|bookings|arr|"
    r"gross margin|operating margin)\b[^.!?]{0,35}\b[-+]?\d+(?:\.\d+)?\b",
    re.I,
)

_POSITIVE_PATTERNS = (
    r"\brais(?:e|ed|es|ing)\b", r"\bincreas(?:e|ed|es|ing)\b",
    r"\baccelerat(?:e|ed|es|ing)\b", r"\bstrong(?:er)?\b",
    r"\bimprov(?:e|ed|es|ing|ement)\b", r"\bexpand(?:s|ed|ing|ion)?\b",
    r"\bgrowth\b", r"\boutperform(?:s|ed|ing)?\b", r"\brecord\b",
    r"\bhigher\b", r"\bconfiden(?:ce|t)\b", r"\bdemand strength\b",
)
_NEGATIVE_PATTERNS = (
    r"\blower(?:ed|ing)?\b", r"\bdecreas(?:e|ed|es|ing)\b", r"\bdeclin(?:e|ed|es|ing)\b",
    r"\bweaker\b", r"\bslow(?:er|ed|ing)?\b", r"\bpressure(?:s|d)?\b",
    r"\bcompression\b", r"\bheadwinds?\b", r"\brisks?\b", r"\buncertain(?:ty)?\b",
    r"\bsoftness\b", r"\bdeteriorat(?:e|ed|es|ing|ion)\b", r"\bcut(?:s|ting)?\b",
    r"\breduc(?:e|ed|es|ing|tion)\b", r"\bdemand weakness\b",
)
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|isn't|wasn't|won't|wouldn't|doesn't|didn't)\b", re.I)

_GUIDANCE_CONTEXT = r"(?:guidance|outlook|forecast|expectations?)"
_GUIDANCE_RAISED_RE = re.compile(
    rf"(?:{_GUIDANCE_CONTEXT})[^.!?]{{0,45}}\b(?:rais(?:e|ed|ing)|increas(?:e|ed|ing)|higher|above)\b|"
    rf"\b(?:rais(?:e|ed|ing)|increas(?:e|ed|ing))\b[^.!?]{{0,45}}{_GUIDANCE_CONTEXT}", re.I
)
_GUIDANCE_LOWERED_RE = re.compile(
    rf"(?:{_GUIDANCE_CONTEXT})[^.!?]{{0,45}}\b(?:lower(?:ed|ing)?|cut|reduc(?:e|ed|ing)|decreas(?:e|ed|ing)|below)\b|"
    rf"\b(?:lower(?:ed|ing)?|cut|reduc(?:e|ed|ing))\b[^.!?]{{0,45}}{_GUIDANCE_CONTEXT}", re.I
)
_GUIDANCE_MAINTAINED_RE = re.compile(
    rf"(?:{_GUIDANCE_CONTEXT})[^.!?]{{0,45}}\b(?:maintain(?:ed|s|ing)?|reaffirm(?:ed|s|ing)?|unchanged|repeat(?:ed|s|ing)?)\b|"
    rf"\b(?:maintain(?:ed|s|ing)?|reaffirm(?:ed|s|ing)?|unchanged)\b[^.!?]{{0,45}}{_GUIDANCE_CONTEXT}", re.I
)

MODEL_FEATURE_NAMES: tuple[str, ...] = (
    "fls_count", "fls_ratio",
    "earnings_fls_count", "earnings_fls_ratio",
    "non_earnings_fls_count", "non_earnings_fls_ratio",
    "quantitative_fls_count", "quantitative_fls_ratio",
    "non_quantitative_fls_count", "non_quantitative_fls_ratio",
    "quant_earnings_fls_count", "quant_earnings_fls_ratio",
    "other_fls_count", "other_fls_ratio",
    "quant_non_earnings_ratio", "nonquant_earnings_ratio",
    "other_to_quant_earnings_ratio",
    "positive_forward_count", "negative_forward_count", "signed_forward_tone",
    "guidance_raised", "guidance_lowered", "guidance_maintained", "guidance_direction",
    "signed_fls_intensity", "signed_quant_earnings_intensity", "signed_other_fls_intensity",
    "guidance_fls_interaction", "guidance_quant_earnings_interaction",
    "other_minus_quant_earnings",
)

@dataclass(frozen=True)
class StatementClassification:
    text: str
    forward_looking: bool
    earnings_related: bool
    quantitative: bool
    category: str | None
    directional_score: int

@dataclass(frozen=True)
class ForwardLookingFeatures:
    values: dict[str, float]
    statements: tuple[StatementClassification, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def vector(self, names: tuple[str, ...] = MODEL_FEATURE_NAMES) -> list[float]:
        return [float(self.values[name]) for name in names]


def is_forward_looking(text: str) -> bool:
    return bool(_FLS_RE.search(text or ""))


def is_earnings_related(text: str) -> bool:
    return bool(_EARNINGS_RE.search(text or ""))


def is_quantitative(text: str) -> bool:
    text = text or ""
    return bool(
        _CURRENCY_RE.search(text)
        or _PERCENT_RE.search(text)
        or _SCALE_RE.search(text)
        or _FINANCIAL_NUMERIC_CONTEXT_RE.search(text)
    )


def classify_statement(text: str) -> StatementClassification:
    text = str(text or "")
    fls = is_forward_looking(text)
    earnings = is_earnings_related(text)
    quantitative = is_quantitative(text)
    category = None
    if fls:
        if quantitative and earnings:
            category = "quantitative_earnings"
        elif (not quantitative) and earnings:
            category = "nonquantitative_earnings"
        elif quantitative and (not earnings):
            category = "quantitative_non_earnings"
        else:
            category = "nonquantitative_non_earnings"
    return StatementClassification(
        text=text,
        forward_looking=fls,
        earnings_related=earnings,
        quantitative=quantitative,
        category=category,
        directional_score=_directional_statement_score(text) if fls else 0,
    )


def extract_forward_looking_features(disclosure: list[str] | tuple[str, ...]) -> ForwardLookingFeatures:
    facts = [str(x) for x in disclosure if x is not None]
    classified = tuple(classify_statement(f) for f in facts)
    denom = max(len(facts), 1)
    fls = [s for s in classified if s.forward_looking]

    def count(*, earnings: bool | None = None, quantitative: bool | None = None) -> int:
        rows = fls
        if earnings is not None:
            rows = [s for s in rows if s.earnings_related is earnings]
        if quantitative is not None:
            rows = [s for s in rows if s.quantitative is quantitative]
        return len(rows)

    fls_count = len(fls)
    earnings_count = count(earnings=True)
    non_earnings_count = count(earnings=False)
    quantitative_count = count(quantitative=True)
    nonquant_count = count(quantitative=False)
    quant_earnings_count = count(earnings=True, quantitative=True)
    quant_non_earnings_count = count(earnings=False, quantitative=True)
    nonquant_earnings_count = count(earnings=True, quantitative=False)
    other_count = fls_count - quant_earnings_count

    positive_count = sum(1 for s in fls if s.directional_score > 0)
    negative_count = sum(1 for s in fls if s.directional_score < 0)
    signed_tone = (positive_count - negative_count) / max(fls_count, 1)

    full_text = " ".join(facts)
    raised = int(bool(_GUIDANCE_RAISED_RE.search(full_text)))
    lowered = int(bool(_GUIDANCE_LOWERED_RE.search(full_text)))
    maintained = int(bool(_GUIDANCE_MAINTAINED_RE.search(full_text)))
    if raised and not lowered:
        guidance_direction = 1.0
    elif lowered and not raised:
        guidance_direction = -1.0
    else:
        # maintained, no guidance change, or conflicting change language -> neutral
        guidance_direction = 0.0

    ratio = lambda n: n / denom
    fls_ratio = ratio(fls_count)
    quant_earnings_ratio = ratio(quant_earnings_count)
    other_ratio = ratio(other_count)
    values = {
        "fls_count": float(fls_count), "fls_ratio": fls_ratio,
        "earnings_fls_count": float(earnings_count), "earnings_fls_ratio": ratio(earnings_count),
        "non_earnings_fls_count": float(non_earnings_count), "non_earnings_fls_ratio": ratio(non_earnings_count),
        "quantitative_fls_count": float(quantitative_count), "quantitative_fls_ratio": ratio(quantitative_count),
        "non_quantitative_fls_count": float(nonquant_count), "non_quantitative_fls_ratio": ratio(nonquant_count),
        "quant_earnings_fls_count": float(quant_earnings_count), "quant_earnings_fls_ratio": quant_earnings_ratio,
        "other_fls_count": float(other_count), "other_fls_ratio": other_ratio,
        "quant_non_earnings_ratio": ratio(quant_non_earnings_count),
        "nonquant_earnings_ratio": ratio(nonquant_earnings_count),
        # A +1 count pseudo-denominator prevents singular/infinite values when no
        # quantitative earnings forecast appears in the ~10-fact disclosure.
        "other_to_quant_earnings_ratio": other_count / (quant_earnings_count + 1.0),
        "positive_forward_count": float(positive_count),
        "negative_forward_count": float(negative_count),
        "signed_forward_tone": float(max(-1.0, min(1.0, signed_tone))),
        "guidance_raised": float(raised), "guidance_lowered": float(lowered),
        "guidance_maintained": float(maintained), "guidance_direction": guidance_direction,
        "signed_fls_intensity": fls_ratio * signed_tone,
        "signed_quant_earnings_intensity": quant_earnings_ratio * signed_tone,
        "signed_other_fls_intensity": other_ratio * signed_tone,
        "guidance_fls_interaction": fls_ratio * guidance_direction,
        "guidance_quant_earnings_interaction": quant_earnings_ratio * guidance_direction,
        "other_minus_quant_earnings": other_ratio - quant_earnings_ratio,
    }
    return ForwardLookingFeatures(values=values, statements=classified)


def _directional_statement_score(text: str) -> int:
    positive = _count_directional_hits(text, _POSITIVE_PATTERNS, base_sign=1)
    negative = _count_directional_hits(text, _NEGATIVE_PATTERNS, base_sign=-1)
    score = positive + negative
    return 1 if score > 0 else -1 if score < 0 else 0


def _count_directional_hits(text: str, patterns: tuple[str, ...], *, base_sign: int) -> int:
    score = 0
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            # Obvious nearby negation reverses the concept instead of blindly
            # counting the dictionary word in its original direction.
            prefix = text[max(0, match.start() - 28):match.start()]
            words = re.findall(r"[A-Za-z']+", prefix)[-3:]
            negated = bool(_NEGATION_RE.search(" ".join(words)))
            score += -base_sign if negated else base_sign
    return score
