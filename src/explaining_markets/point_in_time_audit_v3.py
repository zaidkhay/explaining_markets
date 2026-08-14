"""Point-in-time audit for V3 external records and structured reasoning."""
from __future__ import annotations

from dataclasses import dataclass

from explaining_markets.features_v3 import FORBIDDEN_V3_FEATURE_NAMES, MODEL_FEATURE_NAMES_V3
from explaining_markets.v3_records import V3Context


class PointInTimeViolation(AssertionError):
    pass


@dataclass(frozen=True)
class AuditSummary:
    records_checked: int
    violations: int = 0


def audit_feature_names(names=MODEL_FEATURE_NAMES_V3) -> None:
    bad = FORBIDDEN_V3_FEATURE_NAMES.intersection(names)
    if bad:
        raise PointInTimeViolation(f"forbidden feature names: {sorted(bad)}")


def audit_context(context: V3Context) -> AuditSummary:
    audit_feature_names()
    checked = 0

    def check(record, label):
        nonlocal checked
        if record is None:
            return
        if record.available_at > context.cutoff or record.value_timestamp > context.cutoff:
            raise PointInTimeViolation(f"{label} is not available by focal cutoff")
        if hasattr(record, "published_at") and record.published_at > context.cutoff:
            raise PointInTimeViolation(f"{label} was published after focal cutoff")
        checked += 1

    check(context.earnings, "earnings")
    check(context.guidance, "guidance")
    check(context.metadata, "metadata")
    for label, rows in (
        ("company_history", context.company_history),
        ("stock_prices", context.stock_prices),
        ("market_prices", context.market_prices),
        ("sector_prices", context.sector_prices),
        ("peers", context.peers),
        ("peer_earnings", context.peer_earnings),
        ("company_news", context.company_news),
        ("peer_news", context.peer_news),
        ("sector_news", context.sector_news),
    ):
        for row in rows:
            check(row, label)
    for ticker, rows in context.peer_prices.items():
        for row in rows:
            check(row, f"peer_price:{ticker}")

    selected_source_ids: set[str] = set()
    for label, rows in (
        ("reasoned_company_news", context.reasoned_company_news),
        ("reasoned_peer_news", context.reasoned_peer_news),
        ("reasoned_sector_news", context.reasoned_sector_news),
    ):
        for row in rows:
            if row.available_at > context.cutoff or row.published_at > context.cutoff:
                raise PointInTimeViolation(f"{label} contains post-cutoff reasoning input")
            if row.source_id:
                selected_source_ids.add(row.source_id)
            checked += 1

    if context.event_reasoning is not None:
        if context.event_reasoning.cutoff != context.cutoff:
            raise PointInTimeViolation("event reasoning cutoff does not match focal cutoff")
        unknown = set(context.event_reasoning.source_ids).difference(selected_source_ids)
        if unknown:
            raise PointInTimeViolation(f"event reasoning references unselected news sources: {sorted(unknown)}")
        checked += 1
    return AuditSummary(records_checked=checked)
