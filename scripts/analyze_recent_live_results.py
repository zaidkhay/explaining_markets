"""Quantify recent live predictions, compare counterfactuals, and propose testable improvements.

Input CSV requires either:
    ticker,predicted_percentile,realized_percentile
or, when per-event diagnostic JSON already contains prediction/realized data:
    event_id,ticker

Optional columns:
    event_id,date,car1,diagnostic_json

Example:
    uv run python scripts/analyze_recent_live_results.py \
      --input data/live_eval/recent.csv \
      --output-dir data/diagnostics/recent_eval
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from explaining_markets.live_result_analysis import analyze_recent_results
from explaining_markets.model_v3_lite import V3LiteCandidateModel


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        return [dict(row) for row in rows if isinstance(row, dict)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _markdown(report: dict) -> str:
    summary = report["summary"]
    validation = report.get("validation_reference") or {}
    comparisons = report["counterfactual_comparisons"]
    lines = [
        "# Recent live model evaluation",
        "",
        "## Current live performance",
        "",
        "| Metric | Recent live | Stored validation |",
        "| --- | ---: | ---: |",
        f"| N | {summary['n']} | {validation.get('n', 'n/a')} |",
        f"| Spearman | {_fmt(summary.get('spearman'))} | {_fmt(validation.get('spearman'))} |",
        f"| Pearson | {_fmt(summary.get('pearson'))} | {_fmt(validation.get('pearson'))} |",
        f"| MAE | {_fmt(summary.get('mae'))} | {_fmt(validation.get('mae'))} |",
        f"| RMSE | {_fmt(summary.get('rmse'))} | {_fmt(validation.get('rmse'))} |",
        f"| Direction accuracy | {_fmt(summary.get('direction_accuracy'))} | n/a |",
        "",
        "## Calibration/extremeness counterfactuals",
        "",
        "These are retrospective diagnostics only. They are not automatically promoted.",
        "",
        "| Shrink factor | Spearman | MAE | RMSE | Mean predicted extremeness |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in comparisons.items():
        factor = name.split("_", 1)[1]
        lines.append(
            f"| {factor} | {_fmt(block.get('spearman'))} | {_fmt(block.get('mae'))} | "
            f"{_fmt(block.get('rmse'))} | {_fmt(block.get('mean_predicted_extremeness'))} |"
        )

    lines += ["", "## Failure-mode groups", ""]
    lines += [
        "| Group | N | Spearman | MAE | Signed error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, block in report["groups"].items():
        lines.append(
            f"| {name} | {block['n']} | {_fmt(block.get('spearman'))} | {_fmt(block.get('mae'))} | {_fmt(block.get('mean_signed_error'))} |"
        )

    lines += ["", "## Feature-family live signal", ""]
    lines += [
        "| Family | Mean contribution | Mean abs contribution | Spearman vs realized | Pearson vs realized |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family, block in report["feature_family_signal"].items():
        lines.append(
            f"| {family} | {_fmt(block.get('mean_contribution'))} | {_fmt(block.get('mean_abs_contribution'))} | "
            f"{_fmt(block.get('spearman'))} | {_fmt(block.get('pearson'))} |"
        )

    lines += ["", "## Largest misses", ""]
    lines += [
        "| Ticker | Predicted | Realized | Abs error | Revenue available | Model relevance | Top features |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in report["largest_misses"]:
        top = ", ".join(
            f"{x.get('feature')} ({float(x.get('contribution') or 0):+.3f})"
            for x in (row.get("top_features") or [])[:3]
        ) or "n/a"
        lines.append(
            f"| {row['ticker']} | {row['predicted_percentile']:.3f} | {row['realized_percentile']:.3f} | "
            f"{row['absolute_error']:.3f} | {row['revenue_available']} | {_fmt(row.get('model_relevance'))} | {top} |"
        )

    lines += ["", "## Evidence-based improvement hypotheses", ""]
    for index, rec in enumerate(report["recommendations"], 1):
        lines.append(f"{index}. **{rec['priority'].upper()} — {rec['action']}**")
        lines.append(f"   - Evidence: {rec['evidence']}")
    lines += [
        "",
        "## Promotion rule",
        "",
        "Do not change production weights from this live sample alone. Implement each high-priority hypothesis as an ablation/candidate and require chronological historical validation improvement plus the existing live gate before promotion.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--diagnostics-dir", default="data/diagnostics")
    parser.add_argument("--output-dir", default="data/diagnostics/recent_eval")
    parser.add_argument("--title", default="Recent V3-lite live performance")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _load_rows(input_path)
    model = V3LiteCandidateModel()
    validation = dict(model.training_metadata.get("validation_metrics") or {})
    report = analyze_recent_results(
        rows,
        diagnostics_dir=args.diagnostics_dir,
        validation_metrics=validation,
    )

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "analysis.json"
    md_path = root / "analysis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")

    # Reuse the existing visual predicted-vs-realized dashboard for the same rows.
    from render_model_performance_dashboard import render
    html_path = root / "performance.html"
    render(report["rows"], html_path, args.title)

    s = report["summary"]
    print("=== RECENT LIVE MODEL EVALUATION ===")
    print(f"events: {s['n']}")
    print(f"spearman: {_fmt(s.get('spearman'))}")
    print(f"pearson: {_fmt(s.get('pearson'))}")
    print(f"mae: {_fmt(s.get('mae'))}")
    print(f"rmse: {_fmt(s.get('rmse'))}")
    print(f"direction_accuracy: {_fmt(s.get('direction_accuracy'))}")
    print("\nrecommendations:")
    for rec in report["recommendations"]:
        print(f"  [{rec['priority']}] {rec['action']}")
        print(f"    {rec['evidence']}")
    print(f"\njson: {json_path}")
    print(f"markdown: {md_path}")
    print(f"html: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
