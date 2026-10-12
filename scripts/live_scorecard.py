"""Per-event live scorecard + silent-failure alarms. Diagnostic only (CLAUDE.md §7).

    uv run modal volume get --force em-v3-data evidence data/live_eval   # refresh evidence
    uv run python scripts/live_scorecard.py --refresh-archive            # score + alarms

Writes data/diagnostics/scorecard/{events.csv,summary.json}. Exit code 2 when a
critical alarm fires, so this can gate a cron job or a pre-deploy check.

Scores frozen live predictions against archive outcomes; every Delta R2 is
reported with n, its null expectation (1-R2_bench)/(n-3), F, and a permutation
p-value, raw and submitted separately (§13). It never fits or selects anything.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

from explaining_markets.live_scorecard import (
    alarms,
    contest_style_quarter_score,
    event_table,
    join,
    load_evidence,
    scorecard_summary,
)


def _archive_path(directory: Path, event_type: str, quarter: str) -> Path:
    return directory / f"{event_type}_{quarter}.jsonl.gz"


def _download(event_type: str, quarter: str, path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dotenv import load_dotenv

    from evaluate_recent_from_archive import _download_archive
    from explaining_markets.config import Config

    load_dotenv(".env")
    records = _download_archive(Config.from_env(), event_type, quarter)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_archive(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fmt(value, spec: str = "+.6f") -> str:
    return "n/a" if value is None else format(float(value), spec)


def _print_block(label: str, block: dict) -> None:
    if block.get("delta_r_squared") is None:
        print(f"  {label:9s} n={block.get('n')}  {block.get('note', '')}")
        return
    print(
        f"  {label:9s} n={block['n']:4d}  dR2={_fmt(block['delta_r_squared'])}  "
        f"null={_fmt(block['null_expectation'], '.6f')}  "
        f"x_null={_fmt(block['ratio_to_null'], '.2f')}  F={_fmt(block['f_statistic'], '.2f')}  "
        f"perm_p={_fmt(block['permutation_p'], '.3f')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarters", default="2026Q3,2026Q4")
    parser.add_argument("--event-type", default="EARNINGS_RELEASE")
    parser.add_argument("--evidence-dir", default="data/live_eval/evidence")
    parser.add_argument("--archive-dir", default="data/live_eval/archive")
    parser.add_argument("--output-dir", default="data/diagnostics/scorecard")
    parser.add_argument("--refresh-archive", action="store_true", help="re-download archives (needs EM_API_KEY)")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--worst", type=int, default=10, help="largest-error events to list")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    archives: dict[str, list[dict]] = {}
    for quarter in [q.strip() for q in args.quarters.split(",") if q.strip()]:
        path = _archive_path(archive_dir, args.event_type, quarter)
        if args.refresh_archive or not path.exists():
            _download(args.event_type, quarter, path)
        archives[quarter] = _read_archive(path)

    evidence = load_evidence(args.evidence_dir)
    joined = join(evidence, archives)
    summary = scorecard_summary(joined, permutations=args.permutations)
    summary["contest_style"] = {
        quarter: contest_style_quarter_score(joined, records, quarter)
        for quarter, records in archives.items()
    }
    found = alarms(evidence)
    summary["alarms"] = [alarm.__dict__ for alarm in found]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = event_table(joined)
    with (out / "events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()) if table else ["event_id"])
        writer.writeheader()
        writer.writerows(table)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== LIVE SCORECARD (diagnostic; organiser scorer is authoritative) ===")
    print(f"evidence rows: {summary['evidence_rows']}  schemas: {summary['schema_counts']}")
    print("join status:")
    for status, count in summary["join_status_counts"].items():
        print(f"  {count:4d}  {status:22s} {summary['join_status_meaning'][status]}")
    for quarter, block in summary["quarters"].items():
        print(f"\n{quarter}  joined live subset")
        _print_block("submitted", block["submitted"])
        _print_block("raw", block["raw"])
    for quarter, block in summary["contest_style"].items():
        print(
            f"\n{quarter}  contest-style common sample (mean-imputed, submitted): "
            f"n_obs={block['n_obs']}  imputed={block['imputed_event_count']}  "
            f"dR2={_fmt(block['delta_r_squared_imputed'])}  null={_fmt(block['null_expectation'], '.6f')}"
        )

    scored = [row for row in table if row["abs_error"] is not None]
    scored.sort(key=lambda row: row["abs_error"], reverse=True)
    if scored:
        print(f"\nlargest errors (submitted - realized), top {args.worst}:")
        for row in scored[: args.worst]:
            print(
                f"  {row['event_id']:24s} sub={row['submitted']:.3f} real={row['realized_percentile']:.3f} "
                f"err={row['error']:+.3f}  families={row['families_available'] or '-'}"
            )

    print("\nalarms:")
    for alarm in found or []:
        print(f"  [{alarm.level.upper():8s}] {alarm.code}: {alarm.message}")
    if not found:
        print("  none")
    print(f"\nwrote {out / 'events.csv'} and {out / 'summary.json'}")
    return 2 if any(alarm.level == "critical" for alarm in found) else 0


if __name__ == "__main__":
    raise SystemExit(main())
