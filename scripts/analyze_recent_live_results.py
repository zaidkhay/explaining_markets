"""Quantify recent live predictions, compare counterfactuals, and propose testable improvements.

Input CSV requires either:
    ticker,predicted_percentile,realized_percentile
or, when per-event diagnostic JSON already contains prediction/realized data:
    event_id,ticker

Optional columns:
    event_id,date,car1,diagnostic_json,evidence_path

When ``evidence_path`` points at a persisted Modal V3 evidence bundle, this
script adapts both the old evidence schema and the richer current schema into
the prediction-diagnostics shape consumed by the analyzer. That lets old live
predictions contribute provider/family-availability diagnostics even when they
predate exact feature-contribution persistence.
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


def _adapt_evidence(rows: list[dict], *, root: Path, model: V3LiteCandidateModel) -> list[dict]:
    """Materialize a diagnostic-compatible view of old/new Modal evidence."""
    adapted_root = root / "adapted_evidence"
    adapted_root.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []

    for source in rows:
        row = dict(source)
        explicit_diag = str(row.get("diagnostic_json") or "").strip()
        if explicit_diag and Path(explicit_diag).exists():
            out.append(row)
            continue

        evidence_raw = str(row.get("evidence_path") or "").strip()
        evidence_path = Path(evidence_raw) if evidence_raw else None
        if evidence_path is None or not evidence_path.exists():
            out.append(row)
            continue
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append(row)
            continue
        if not isinstance(evidence, dict):
            out.append(row)
            continue

        nested = evidence.get("prediction_diagnostics")
        diagnostic = dict(nested) if isinstance(nested, dict) else {}

        external = dict(diagnostic.get("external_context") or {})
        external.setdefault("family_availability", evidence.get("feature_availability") or {})
        receipts = evidence.get("provider_receipts") or []
        if isinstance(receipts, list):
            external.setdefault(
                "provider_successes",
                sum(1 for item in receipts if isinstance(item, dict) and item.get("status") == "ok"),
            )
            external.setdefault(
                "provider_errors",
                sum(1 for item in receipts if isinstance(item, dict) and item.get("status") == "error"),
            )
            external.setdefault("provider_receipts", receipts)
        values = evidence.get("feature_values") or {}
        if isinstance(values, dict):
            external.setdefault(
                "nonzero_deployed_features",
                sum(abs(float(values.get(name, 0.0) or 0.0)) > 1e-12 for name in model.feature_names),
            )
            external.setdefault("deployed_feature_count", len(model.feature_names))
        diagnostic["external_context"] = external

        # New evidence already carries exact score/claim contributions inside
        # prediction_diagnostics. Old evidence cannot recreate those after the
        # fact, but preserving the submitted/raw numbers is still useful.
        score = dict(diagnostic.get("score") or {})
        if "submitted_percentile" not in score and evidence.get("prediction") is not None:
            score["submitted_percentile"] = float(evidence["prediction"])
        if "raw_score" not in score and evidence.get("raw_prediction") is not None:
            score["raw_score"] = float(evidence["raw_prediction"])
        diagnostic["score"] = score
        diagnostic.setdefault("claims", [])
        diagnostic.setdefault("diagnostic_signal", {})
        diagnostic["event"] = {
            "event_id": row.get("event_id") or evidence.get("event_id"),
            "ticker": row.get("ticker") or evidence.get("ticker"),
            "cutoff": evidence.get("cutoff"),
        }
        diagnostic["realized"] = {
            "car1": None if row.get("car1") in (None, "") else float(row["car1"]),
            "realized_percentile": (
                None if row.get("realized_percentile") in (None, "") else float(row["realized_percentile"])
            ),
        }

        event_id = str(row.get("event_id") or evidence.get("event_id") or "unknown")
        ticker = str(row.get("ticker") or evidence.get("ticker") or "UNKNOWN").upper()
        safe_event = "".join(c for c in event_id if c.isalnum() or c in "-_") or "unknown"
        safe_ticker = "".join(c for c in ticker if c.isalnum() or c in ".-_") or "UNKNOWN"
        target = adapted_root / f"{safe_event}__{safe_ticker}.json"
        target.write_text(json.dumps(diagnostic, indent=2, sort_keys=True), encoding="utf-8")
        row["diagnostic_json"] = str(target)
        out.append(row)

    return out


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
        "These are retrospective diagnostics only. They are not automatically promoted. Note that uniform affine shrinkage does not change the official Delta-R2 objective; this section diagnoses absolute percentile error only.",
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
        "Do not change production weights from this live sample alone. Implement each high-priority hypothesis as an ablation/candidate and require chronological historical validation improvement on the official Delta-R2 objective plus the existing live gate before promotion.",
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
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = _adapt_evidence(rows, root=root, model=model)

    validation = dict(model.training_metadata.get("validation_metrics") or {})
    report = analyze_recent_results(
        rows,
        diagnostics_dir=args.diagnostics_dir,
        validation_metrics=validation,
    )

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
