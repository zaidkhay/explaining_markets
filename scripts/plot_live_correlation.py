"""Plot predicted-vs-realized correlation for joined live events.

Diagnostic only (CLAUDE.md §7.1, §13): does not fit, select, or promote
anything, and correlation is not the scored objective (Delta R2).

Input is the join produced by scripts/evaluate_recent_from_archive.py:

    uv run modal volume get em-v3-data evidence data/live_eval/evidence
    uv run python scripts/evaluate_recent_from_archive.py
    uv run python scripts/plot_live_correlation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from explaining_markets.live_correlation import (
    correlation_summary,
    load_joined_rows,
    plot_correlation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/diagnostics/recent_eval/recent_joined.csv")
    parser.add_argument("--output", default="data/diagnostics/recent_eval/correlation.png")
    parser.add_argument("--title", default="Live events: prediction vs realized")
    parser.add_argument("--min-periods", type=int, default=3, help="Minimum events before the trend panel starts")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"{input_path} not found. Run scripts/evaluate_recent_from_archive.py first "
            "to join Modal evidence predictions to archive outcomes."
        )

    df = load_joined_rows(input_path)
    summary = correlation_summary(df)
    output_path = plot_correlation(df, args.output, title=args.title, min_periods=args.min_periods)

    print("=== LIVE EVENT CORRELATION (diagnostic only -- not Delta R2) ===")
    print(f"n: {summary['n']}")
    print(f"pearson r: {summary['pearson']}")
    print(f"spearman rho: {summary['spearman']}")
    print(
        "Reminder (CLAUDE.md §2, §7.1): small n makes this noisy and these predictions "
        "were never fit on this sample. For a formal significance check against the null "
        "expectation, see scripts/evaluate_recent_from_archive.py's archive_score.json."
    )
    print(f"plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
