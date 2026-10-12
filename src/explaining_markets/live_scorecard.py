"""Per-event live scorecard: evidence loading, archive join, scoring, alarms.

Diagnostic only (CLAUDE.md §7). Nothing here fits, tunes or selects anything.
It scores already-frozen live predictions against archive outcomes, which is
legitimate on every quarter including 2026Q3 (§1.2): the predictions were
fixed before the outcomes existed.

Which rows touched the fit that produced the evaluated predictions? The
deployed artifact was fit on 2025Q4 (coefficients, scaler) and 2026Q1
(calibration_v1). Live events are 2026Q3 onward, disjoint from both.

Evidence schemas are not uniform (§7.4):
  legacy   -- earliest bundles: ``submitted_percentile``, no ``ticker`` field
  summary  -- ``prediction`` (submitted) but no raw score or disclosure
  full     -- adds ``raw_prediction``, ``disclosure``, ``feature_values`` and
              ``prediction_diagnostics`` (commit 817787e onward)
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from explaining_markets.competition_scoring import (
    percentile_ranks,
    score_complete_predictions,
    score_with_missing_predictions,
)

# Fitted parameters in the scoring regression: intercept, prediction, surprise.
SCORING_K = 3

_EVENT_ID_TICKER = re.compile(r"^ea_(?P<ticker>.+)_Q[1-4]_\d{4}$")


@dataclass(frozen=True)
class EvidenceRow:
    event_id: str
    ticker: str
    submitted: float
    raw: float | None
    created_at: str
    cutoff: str
    schema: str
    path: str
    model_version: str | None = None
    feature_availability: dict[str, float] = field(default_factory=dict)
    disclosure: tuple[str, ...] | None = None
    nonzero_fls: int | None = None
    top_contributions: tuple[tuple[str, float], ...] = ()


def _ticker_from_event_id(event_id: str) -> str:
    match = _EVENT_ID_TICKER.match(event_id)
    return match.group("ticker").upper() if match else ""


def parse_evidence(payload: dict[str, Any], path: str = "") -> EvidenceRow | None:
    """Normalise one evidence bundle of any schema. None if it carries no prediction."""
    event_id = str(payload.get("event_id") or "")
    if payload.get("prediction") is not None:
        submitted = float(payload["prediction"])
        schema = "full" if payload.get("raw_prediction") is not None else "summary"
    elif payload.get("submitted_percentile") is not None:
        submitted = float(payload["submitted_percentile"])
        schema = "legacy"
    else:
        return None
    ticker = str(payload.get("ticker") or "").upper() or _ticker_from_event_id(event_id)
    if not event_id or not ticker:
        return None

    raw = payload.get("raw_prediction")
    disclosure = payload.get("disclosure")
    values = payload.get("feature_values")
    nonzero_fls = None
    if isinstance(values, dict) and values:
        nonzero_fls = sum(
            1 for name, value in values.items() if "fls" in name and abs(float(value)) > 1e-12
        )
    top: tuple[tuple[str, float], ...] = ()
    features = ((payload.get("prediction_diagnostics") or {}).get("score") or {}).get("features")
    if isinstance(features, list):
        top = tuple(
            (str(row["feature"]), float(row["raw_score_contribution"]))
            for row in features[:3]
            if isinstance(row, dict) and "feature" in row
        )
    return EvidenceRow(
        event_id=event_id,
        ticker=ticker,
        submitted=submitted,
        raw=None if raw is None else float(raw),
        created_at=str(payload.get("created_at") or payload.get("cutoff") or ""),
        cutoff=str(payload.get("cutoff") or ""),
        schema=schema,
        path=path,
        model_version=payload.get("model_version"),
        feature_availability={
            str(k): float(v) for k, v in (payload.get("feature_availability") or {}).items()
        },
        disclosure=tuple(str(x) for x in disclosure) if isinstance(disclosure, list) else None,
        nonzero_fls=nonzero_fls,
        top_contributions=top,
    )


def load_evidence(directory: str | Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    root = Path(directory)
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            row = parse_evidence(payload, str(path))
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (r.created_at, r.event_id, r.ticker))
    return rows


# --- Archive outcomes -----------------------------------------------------------

def outcome_rows(records: list[dict], quarter: str = "") -> list[dict]:
    """One row per (event, focal ticker) with CAR1 and surprise percentiles.

    Percentiles are ranked within this quarter's available outcomes only: the
    target is quarter-relative (§1.4), so quarters must never be ranked together.
    """
    rows: list[dict] = []
    for record in records:
        es = (record.get("metrics") or {}).get("earnings_surprise") or {}
        surprise = es.get("surprise") if es.get("surprise_status") == "ok" else None
        returns = record.get("event_returns") or {}
        for asset in record.get("focal_assets") or []:
            ticker = str(asset.get("identifier_value") or "").upper()
            car1 = (returns.get(ticker) or {}).get("car1")
            if not ticker or car1 is None:
                continue
            rows.append({
                "event_id": str(record.get("event_id") or ""),
                "ticker": ticker,
                "quarter": quarter,
                "event_datetime": record.get("event_datetime") or record.get("knowledge_cutoff"),
                "car1": float(car1),
                "earnings_surprise": None if surprise is None else float(surprise),
                "realized_percentile": None,
                "surprise_percentile": None,
            })
    if rows:
        for row, value in zip(rows, percentile_ranks([r["car1"] for r in rows]), strict=True):
            row["realized_percentile"] = value
        have = [i for i, r in enumerate(rows) if r["earnings_surprise"] is not None]
        ranks = percentile_ranks([rows[i]["earnings_surprise"] for i in have])
        for i, value in zip(have, ranks, strict=True):
            rows[i]["surprise_percentile"] = value
    return rows


JOIN_REASONS = {
    "joined": "scored: prediction, outcome and surprise benchmark all present",
    "event_not_in_archive": "event id absent from the downloaded quarters (test/diagnostic id, or quarter not downloaded)",
    "ticker_not_in_event": "event found but the predicted ticker is not one of its focal assets",
    "outcome_pending": "event found but CAR1 not yet published for this ticker",
    "no_surprise_benchmark": "outcome present but the archive has no earnings surprise; excluded from the benchmark regression",
}


def join(evidence: Iterable[EvidenceRow], archives: dict[str, list[dict]]) -> list[dict]:
    """Every evidence row with its outcome (if any) and a join_status from JOIN_REASONS."""
    records: dict[str, dict] = {}
    outcomes: dict[tuple[str, str], dict] = {}
    for quarter, recs in archives.items():
        for rec in recs:
            records[str(rec.get("event_id") or "")] = rec
        for row in outcome_rows(recs, quarter):
            outcomes[(row["event_id"], row["ticker"])] = row

    out: list[dict] = []
    for ev in evidence:
        outcome = outcomes.get((ev.event_id, ev.ticker))
        record = records.get(ev.event_id)
        if outcome is not None:
            status = "joined" if outcome["surprise_percentile"] is not None else "no_surprise_benchmark"
        elif record is None:
            status = "event_not_in_archive"
        elif ev.ticker not in {
            str(a.get("identifier_value") or "").upper() for a in record.get("focal_assets") or []
        }:
            status = "ticker_not_in_event"
        else:
            status = "outcome_pending"
        out.append({"evidence": ev, "outcome": outcome, "join_status": status})
    return out


# --- Scoring --------------------------------------------------------------------

def null_expectation(r2_benchmark: float, n: int, k: int = SCORING_K) -> float | None:
    """E[Delta R2] for a pure-noise predictor (§2.2)."""
    return None if n <= k else (1.0 - r2_benchmark) / (n - k)


def score_block(
    predictions: list[float],
    realized: list[float],
    surprise: list[float],
    *,
    permutations: int = 2000,
    seed: int = 7,
) -> dict[str, Any]:
    """Delta R2 with the context §2 and §13 require: null expectation, F and a
    permutation p-value (the correct small-sample reference, §2.5)."""
    n = len(predictions)
    base = score_complete_predictions(predictions, realized, surprise) if n > SCORING_K else None
    if not base or base["delta_r_squared"] is None:
        return {"n": n, "delta_r_squared": None, "note": "too few rows to score"}
    delta = float(base["delta_r_squared"])
    r2_s = float(base["r_squared_surprise"])
    r2_f = float(base["r_squared"])
    null = null_expectation(r2_s, n)
    f_stat = None if r2_f >= 1.0 else delta / ((1.0 - r2_f) / (n - SCORING_K))

    p_value = None
    if permutations > 0:
        rng = random.Random(seed)
        shuffled = list(predictions)
        at_least = 0
        for _ in range(permutations):
            rng.shuffle(shuffled)
            d = score_complete_predictions(shuffled, realized, surprise)["delta_r_squared"]
            if d is not None and d >= delta:
                at_least += 1
        p_value = (at_least + 1) / (permutations + 1)
    return {
        "n": n,
        "r_squared_surprise": r2_s,
        "r_squared": r2_f,
        "delta_r_squared": delta,
        "null_expectation": null,
        "ratio_to_null": None if not null else delta / null,
        "f_statistic": f_stat,
        "permutation_p": p_value,
        "permutations": permutations,
    }


def scorecard_summary(joined: list[dict], *, permutations: int = 2000) -> dict[str, Any]:
    """Raw and submitted Delta R2 reported separately (§5.4, §13), per quarter."""
    scored = [row for row in joined if row["join_status"] == "joined"]
    by_quarter: dict[str, list[dict]] = {}
    for row in scored:
        by_quarter.setdefault(row["outcome"]["quarter"], []).append(row)

    quarters: dict[str, Any] = {}
    for quarter, rows in sorted(by_quarter.items()):
        realized = [r["outcome"]["realized_percentile"] for r in rows]
        surprise = [r["outcome"]["surprise_percentile"] for r in rows]
        block: dict[str, Any] = {
            "partition": "live (frozen predictions; never used for fitting or selection)",
            "submitted": score_block(
                [r["evidence"].submitted for r in rows], realized, surprise, permutations=permutations
            ),
        }
        with_raw = [r for r in rows if r["evidence"].raw is not None]
        if len(with_raw) > SCORING_K:
            block["raw"] = score_block(
                [r["evidence"].raw for r in with_raw],
                [r["outcome"]["realized_percentile"] for r in with_raw],
                [r["outcome"]["surprise_percentile"] for r in with_raw],
                permutations=permutations,
            )
            block["raw"]["note"] = f"only the {len(with_raw)} rows whose evidence stores a raw score"
        else:
            block["raw"] = {
                "n": len(with_raw),
                "delta_r_squared": None,
                "note": "raw scores are only persisted from commit 817787e onward",
            }
        quarters[quarter] = block

    statuses = Counter(row["join_status"] for row in joined)
    return {
        "evidence_rows": len(joined),
        "join_status_counts": dict(sorted(statuses.items())),
        "join_status_meaning": {k: JOIN_REASONS[k] for k in statuses},
        "schema_counts": dict(Counter(row["evidence"].schema for row in joined)),
        "quarters": quarters,
    }


def contest_style_quarter_score(
    joined: list[dict], archive: list[dict], quarter: str
) -> dict[str, Any]:
    """Approximate the organiser's common-sample score for one quarter: every
    archive outcome with a surprise, our prediction where we have one, the
    official per-submission mean imputation elsewhere (competition.md §9)."""
    ours = {
        (row["evidence"].event_id, row["evidence"].ticker): row["evidence"].submitted
        for row in joined
    }
    rows = [r for r in outcome_rows(archive, quarter) if r["surprise_percentile"] is not None]
    block = score_with_missing_predictions(
        [ours.get((r["event_id"], r["ticker"])) for r in rows],
        [r["realized_percentile"] for r in rows],
        [r["surprise_percentile"] for r in rows],
    )
    r2_s = block.get("r_squared_surprise_imputed")
    block["null_expectation"] = None if r2_s is None else null_expectation(r2_s, len(rows))
    block["quarter"] = quarter
    block["note"] = (
        "Approximation of the leaderboard common sample; the organiser's scorer is authoritative. "
        "Submitted scores only."
    )
    return block


# --- Per-event table ------------------------------------------------------------

def event_table(joined: list[dict]) -> list[dict]:
    table = []
    for row in joined:
        ev: EvidenceRow = row["evidence"]
        out = row["outcome"] or {}
        realized = out.get("realized_percentile")
        table.append({
            "event_id": ev.event_id,
            "ticker": ev.ticker,
            "quarter": out.get("quarter", ""),
            "created_at": ev.created_at,
            "schema": ev.schema,
            "submitted": ev.submitted,
            "raw": ev.raw,
            "realized_percentile": realized,
            "surprise_percentile": out.get("surprise_percentile"),
            "error": None if realized is None else ev.submitted - realized,
            "abs_error": None if realized is None else abs(ev.submitted - realized),
            "join_status": row["join_status"],
            "disclosure_facts": None if ev.disclosure is None else len(ev.disclosure),
            "nonzero_fls": ev.nonzero_fls,
            "families_available": ",".join(
                sorted(k for k, v in ev.feature_availability.items() if v > 0)
            ),
            "top_contributions": "; ".join(f"{name}={value:+.4f}" for name, value in ev.top_contributions),
            "evidence_path": ev.path,
        })
    return table


# --- Silent-failure alarms (§9.2) --------------------------------------------------

@dataclass(frozen=True)
class Alarm:
    level: str  # "critical" | "warn" | "info"
    code: str
    message: str


_ID_LIKE = re.compile(
    r"^\s*(ea_[A-Za-z0-9.]+_Q[1-4]_\d{4}|\d+(\.\d+)?|\d{4}-\d{2}-\d{2}([T ][\d:.+Z-]+)?|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\s*$"
)


def disclosure_looks_broken(facts: tuple[str, ...]) -> bool:
    """The §9.1 failure: identifiers/numbers/timestamps instead of prose facts."""
    if not facts:
        return True
    prose = [f for f in facts if not _ID_LIKE.match(f) and len(f.strip()) >= 20]
    return not prose


def alarms(
    evidence: list[EvidenceRow],
    *,
    window: int = 10,
    min_spread: float = 0.01,
    repeat_window: int = 20,
    repeat_share: float = 0.25,
) -> list[Alarm]:
    """Checks for the failure class that does not raise (§9). Rows must be in
    chronological order (load_evidence sorts them).

    ``min_spread`` is on the submitted scale. Since 2026-10-12 production
    submits clipped raw scores, whose historical OOS sd is ~0.044 (vs ~0.3
    calibrated), so the floor is 0.01: well below normal raw dispersion, far
    above the ~0 sd of the §9.1 constant-score failure."""
    found: list[Alarm] = []
    if not evidence:
        return [Alarm("warn", "no_evidence", "no evidence bundles found")]

    recent = evidence[-window:]
    if len(recent) >= 5:
        sd = statistics.pstdev([r.submitted for r in recent])
        if sd < min_spread:
            found.append(Alarm(
                "critical", "low_spread",
                f"last {len(recent)} submitted predictions have sd {sd:.4f} < {min_spread} "
                "(identical predictions across unrelated tickers is the §9.1 signature)",
            ))

    tail = evidence[-repeat_window:]
    if len(tail) >= 5:
        value, count = Counter(round(r.submitted, 4) for r in tail).most_common(1)[0]
        if count >= 5 and count / len(tail) >= repeat_share:
            found.append(Alarm(
                "warn", "repeated_value",
                f"{count}/{len(tail)} recent predictions equal {value}; check these events' "
                "disclosures parsed (a no-signal output repeated across tickers)",
            ))

    fls_known = [r for r in recent if r.nonzero_fls is not None]
    if len(fls_known) >= 5:
        zero = sum(1 for r in fls_known if r.nonzero_fls == 0)
        if zero == len(fls_known):
            found.append(Alarm("critical", "zero_fls", f"all {zero} recent events have zero nonzero FLS features"))
        elif zero / len(fls_known) >= 0.5:
            found.append(Alarm("warn", "zero_fls", f"{zero}/{len(fls_known)} recent events have zero FLS features"))

    for r in recent:
        if r.disclosure is not None and disclosure_looks_broken(r.disclosure):
            found.append(Alarm(
                "critical", "disclosure_sanity",
                f"{r.event_id}/{r.ticker}: disclosure has no prose facts ({len(r.disclosure)} items)",
            ))

    families = sorted({k for r in evidence for k in r.feature_availability})
    with_fams = [r for r in evidence if r.feature_availability]
    recent_fams = with_fams[-window:]
    earlier = with_fams[:-window]
    for fam in families:
        if len(with_fams) >= 20 and all(r.feature_availability.get(fam, 0) <= 0 for r in with_fams):
            found.append(Alarm(
                "info", "family_never_available",
                f"'{fam}' unavailable on all {len(with_fams)} events (check the data source / mount)",
            ))
            continue
        if len(earlier) >= window and len(recent_fams) >= 5:
            before = sum(r.feature_availability.get(fam, 0) > 0 for r in earlier) / len(earlier)
            now = sum(r.feature_availability.get(fam, 0) > 0 for r in recent_fams) / len(recent_fams)
            if before >= 0.5 and now == 0.0:
                found.append(Alarm(
                    "warn", "family_dropout",
                    f"'{fam}' was available on {before:.0%} of earlier events and 0% of the last "
                    f"{len(recent_fams)} (provider failing or returning empty?)",
                ))

    legacy_recent = [r for r in recent if r.schema != "full"]
    if legacy_recent:
        found.append(Alarm(
            "info", "thin_evidence",
            f"{len(legacy_recent)}/{len(recent)} recent bundles lack raw score/disclosure "
            "(deployed code predates 817787e for those events)",
        ))
    order = {"critical": 0, "warn": 1, "info": 2}
    return sorted(found, key=lambda a: order[a.level])
