#!/usr/bin/env python3
"""Build leakage-safe V3 archive seed rows from local competition history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explaining_markets.historical import DEFAULT_HISTORICAL_DIR
from explaining_markets.v3_training_data import (
    DEFAULT_ROWS_PATH,
    build_archive_seed_rows,
    training_data_report,
    write_training_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build point-in-time-safe V3 training rows")
    parser.add_argument("--historical-dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument(
        "--event-types",
        default="EARNINGS_RELEASE",
        help="comma-separated event types to create feature rows for; labels still rank against all events in-quarter",
    )
    args = parser.parse_args()

    event_types = tuple(value.strip() for value in args.event_types.split(",") if value.strip())
    rows = build_archive_seed_rows(source=args.historical_dir, event_types=event_types)
    if not rows:
        print(f"No V3 rows were built from {args.historical_dir}")
        print("Place competition archive *.jsonl or *.jsonl.gz files there first.")
        return 2

    output = write_training_rows(rows, args.output)
    report = training_data_report(rows, archive_seed_only=True)

    print("=== V3 TRAINING ROW BUILD ===")
    print(f"historical_dir: {args.historical_dir}")
    print(f"output: {output}")
    print(f"rows: {report.rows}")
    print("quarter_counts:")
    for quarter, count in report.quarter_counts.items():
        print(f"  {quarter}: {count}")
    print("family_coverage:")
    for family, coverage in report.family_coverage.items():
        print(f"  {family}: {coverage:.3f}")
    print(f"active_non_fls_features: {report.active_non_fls_features}")
    print(f"target_range: {report.target_min} .. {report.target_max}")
    print(f"target_std: {report.target_std}")
    print()
    print("NOTE: this is an archive SEED dataset. It does not fabricate missing")
    print("historical EPS/revenue/guidance/price/news/reasoning families.")
    print("A research model may be trained from it, but production promotion must")
    print("remain fail-closed until richer point-in-time V3 history is available.")
    print()
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
