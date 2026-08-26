"""Join Modal evidence predictions to official archive outcomes and score recent events.

Workflow:
  1. Download the Modal evidence directory locally:
       uv run modal volume get em-v3-data evidence data/live_eval/evidence
  2. Run this script. It fetches the current quarter archive with EM_API_KEY,
     computes CAR1 and surprise percentiles over the WHOLE available quarter,
     joins exact persisted predictions by (event_id, ticker), and evaluates the
     recent joined sample with the frozen competition Delta-R^2 formula.

Because a recent live slice can be small, the report also includes a
conditional nonparametric bootstrap and leave-one-out sensitivity diagnostic.
These are stability diagnostics, not the organizer's official confidence
interval; the organizer's live/final scorer remains authoritative.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from explaining_markets.competition_scoring import percentile_ranks, score_complete_predictions
from explaining_markets.config import Config


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _download_archive(cfg: Config, event_type: str, quarter: str) -> list[dict]:
    headers = {"X-API-Key": cfg.api_key}
    endpoint = f"{cfg.api_base_url}/archive/{event_type}/{quarter}"
    meta = httpx.get(endpoint, headers=headers, timeout=30.0)
    meta.raise_for_status()
    payload = meta.json()
    url = payload.get("url")
    if not url:
        raise RuntimeError(f"archive response had no signed url: {payload}")
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    raw = gzip.decompress(response.content).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _outcome_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        es = ((record.get("metrics") or {}).get("earnings_surprise") or {})
        surprise = es.get("surprise") if es.get("surprise_status") == "ok" else None
        returns = record.get("event_returns") or {}
        event_dt = record.get("event_datetime") or record.get("knowledge_cutoff")
        for asset in record.get("focal_assets") or []:
            ticker = str(asset.get("identifier_value") or "").upper()
            if not ticker:
                continue
            leg = returns.get(ticker) or {}
            car1 = leg.get("car1")
            if car1 is None:
                continue
            rows.append({
                "event_id": str(record.get("event_id") or ""),
                "ticker": ticker,
                "event_datetime": event_dt,
                "car1": float(car1),
                "earnings_surprise": None if surprise is None else float(surprise),
            })
    if not rows:
        return rows

    y = percentile_ranks([row["car1"] for row in rows])
    for row, value in zip(rows, y, strict=True):
        row["realized_percentile"] = value

    have = [i for i, row in enumerate(rows) if row["earnings_surprise"] is not None]
    surprise_pct = percentile_ranks([rows[i]["earnings_surprise"] for i in have])
    for row in rows:
        row["surprise_percentile"] = None
    for i, value in zip(have, surprise_pct, strict=True):
        rows[i]["surprise_percentile"] = value
    return rows


def _load_evidence(directory: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not directory.exists():
        return out
    for path in directory.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        event_id = str(payload.get("event_id") or "")
        ticker = str(payload.get("ticker") or "").upper()
        prediction = payload.get("prediction")
        if not event_id or not ticker or prediction is None:
            continue
        out[(event_id, ticker)] = {
            "predicted_percentile": float(prediction),
            "evidence_path": str(path),
            "model_version": payload.get("model_version"),
            "cutoff": payload.get("cutoff"),
            "feature_availability": payload.get("feature_availability") or {},
        }
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id", "ticker", "event_datetime", "predicted_percentile",
        "realized_percentile", "car1", "earnings_surprise", "surprise_percentile",
        "model_version", "evidence_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _score(rows: list[dict]) -> dict:
    if len(rows) < 3:
        return {
            "n": len(rows),
            "r_squared_surprise": None,
            "r_squared": None,
            "delta_r_squared": None,
            "beta": None,
            "beta_surprise": None,
            "alpha": None,
            "mse": None,
        }
    return score_complete_predictions(
        [row["predicted_percentile"] for row in rows],
        [row["realized_percentile"] for row in rows],
        [row["surprise_percentile"] for row in rows],
    )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _uncertainty(rows: list[dict], *, repetitions: int = 2000, seed: int = 7) -> dict:
    base = _score(rows)
    base_delta = base.get("delta_r_squared")
    if len(rows) < 4 or base_delta is None:
        return {
            "bootstrap_repetitions": repetitions,
            "bootstrap_valid": 0,
            "delta_r2_p025": None,
            "delta_r2_median": None,
            "delta_r2_p975": None,
            "bootstrap_fraction_positive": None,
            "leave_one_out_min": None,
            "leave_one_out_max": None,
            "most_score_supportive_event": None,
            "most_score_harmful_event": None,
            "note": "insufficient rows for stability diagnostics",
        }

    rng = random.Random(seed)
    boot: list[float] = []
    n = len(rows)
    for _ in range(max(0, int(repetitions))):
        sample = [rows[rng.randrange(n)] for _j in range(n)]
        delta = _score(sample).get("delta_r_squared")
        if delta is not None:
            boot.append(float(delta))

    loo: list[dict] = []
    for index, row in enumerate(rows):
        subset = rows[:index] + rows[index + 1 :]
        delta = _score(subset).get("delta_r_squared")
        if delta is None:
            continue
        change_without = float(delta) - float(base_delta)
        loo.append({
            "event_id": row.get("event_id"),
            "ticker": row.get("ticker"),
            "delta_r2_without_event": float(delta),
            # Positive => removing this event improves the score, so this event
            # was harmful to the observed recent Delta-R2. Negative => removing
            # it lowers the score, so it was supportive.
            "change_without_event": change_without,
        })

    supportive = min(loo, key=lambda x: x["change_without_event"]) if loo else None
    harmful = max(loo, key=lambda x: x["change_without_event"]) if loo else None
    loo_values = [x["delta_r2_without_event"] for x in loo]
    return {
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "bootstrap_valid": len(boot),
        "delta_r2_p025": _percentile(boot, 0.025),
        "delta_r2_median": _percentile(boot, 0.50),
        "delta_r2_p975": _percentile(boot, 0.975),
        "bootstrap_fraction_positive": None if not boot else sum(x > 0 for x in boot) / len(boot),
        "leave_one_out_min": min(loo_values) if loo_values else None,
        "leave_one_out_max": max(loo_values) if loo_values else None,
        "most_score_supportive_event": supportive,
        "most_score_harmful_event": harmful,
        "note": (
            "Conditional stability diagnostic over the currently joined recent slice. "
            "Quarter-wide CAR1/surprise percentile ranks are held fixed; this is not an organizer-issued confidence interval."
        ),
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarter", default="2026Q3")
    parser.add_argument("--event-type", default="EARNINGS_RELEASE")
    parser.add_argument("--evidence-dir", default="data/live_eval/evidence")
    parser.add_argument("--output-dir", default="data/diagnostics/recent_eval")
    parser.add_argument("--since", help="Optional ISO date/datetime; filters joined events only after quarter-wide ranking")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap repetitions for recent-score stability")
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    args = parser.parse_args()

    load_dotenv(".env")
    cfg = Config.from_env()
    records = _download_archive(cfg, args.event_type, args.quarter)
    outcomes = _outcome_rows(records)
    evidence = _load_evidence(Path(args.evidence_dir))

    since = _parse_dt(args.since) if args.since else None
    joined: list[dict] = []
    for row in outcomes:
        pred = evidence.get((row["event_id"], row["ticker"]))
        if pred is None or row["surprise_percentile"] is None:
            continue
        event_dt = _parse_dt(row.get("event_datetime"))
        if since is not None and event_dt is not None:
            compare_since = since
            if compare_since.tzinfo is None and event_dt.tzinfo is not None:
                compare_since = compare_since.replace(tzinfo=event_dt.tzinfo)
            if event_dt < compare_since:
                continue
        joined.append({**row, **pred})

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "recent_joined.csv"
    _write_csv(csv_path, joined)

    score = _score(joined)
    uncertainty = _uncertainty(
        joined,
        repetitions=max(0, int(args.bootstrap)),
        seed=int(args.bootstrap_seed),
    )

    report = {
        "quarter": args.quarter,
        "event_type": args.event_type,
        "archive_records": len(records),
        "quarter_outcome_rows": len(outcomes),
        "evidence_predictions": len(evidence),
        "joined_recent_rows": len(joined),
        "since": args.since,
        "recent_subset_official_formula": score,
        "stability": uncertainty,
        "note": (
            "CAR1 and surprise percentiles are ranked over every currently available archive outcome in the quarter. "
            "The Delta-R^2 fit is then reported on the joined recent subset; the official leaderboard/full contest common sample remains authoritative."
        ),
    }
    json_path = root / "archive_score.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== ARCHIVE-DERIVED RECENT SCORE ===")
    print(f"quarter outcome rows: {len(outcomes)}")
    print(f"evidence predictions: {len(evidence)}")
    print(f"joined recent rows: {len(joined)}")
    print(f"R2 surprise: {score['r_squared_surprise']}")
    print(f"R2 full: {score['r_squared']}")
    print(f"Delta R2: {score['delta_r_squared']}")
    print(f"prediction beta: {score['beta']}")
    print("\nSTABILITY")
    print(
        "bootstrap Delta R2 95% interval: "
        f"[{_fmt(uncertainty['delta_r2_p025'])}, {_fmt(uncertainty['delta_r2_p975'])}]"
    )
    print(f"bootstrap median: {_fmt(uncertainty['delta_r2_median'])}")
    frac = uncertainty.get("bootstrap_fraction_positive")
    print(f"bootstrap fraction Delta R2 > 0: {'n/a' if frac is None else f'{100*float(frac):.1f}%'}")
    print(
        "leave-one-out Delta R2 range: "
        f"[{_fmt(uncertainty['leave_one_out_min'])}, {_fmt(uncertainty['leave_one_out_max'])}]"
    )
    if uncertainty.get("most_score_harmful_event"):
        row = uncertainty["most_score_harmful_event"]
        print(
            "most score-harmful event: "
            f"{row.get('ticker')} {row.get('event_id')} "
            f"change_if_removed={float(row['change_without_event']):+.6f}"
        )
    if uncertainty.get("most_score_supportive_event"):
        row = uncertainty["most_score_supportive_event"]
        print(
            "most score-supportive event: "
            f"{row.get('ticker')} {row.get('event_id')} "
            f"change_if_removed={float(row['change_without_event']):+.6f}"
        )
    print(f"csv: {csv_path}")
    print(f"report: {json_path}")
    print("\nNext:")
    print(
        "uv run python scripts/analyze_recent_live_results.py "
        f"--input {csv_path} --output-dir {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
