"""Dependency-free HTML/SVG dashboards for V3-lite diagnostics."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _bar(value: float, *, max_abs: float, positive: bool) -> str:
    max_abs = max(max_abs, 1e-12)
    width = min(100.0, 100.0 * abs(float(value)) / max_abs)
    cls = "pos" if positive else "neg"
    return f'<div class="bar-track"><div class="bar {cls}" style="width:{width:.2f}%"></div></div>'


def _meter(value: float, *, label: str) -> str:
    value = max(0.0, min(1.0, float(value)))
    return (
        f'<div class="meter-row"><span>{html.escape(label)}</span>'
        f'<div class="meter"><div class="meter-fill" style="width:{100*value:.1f}%"></div></div>'
        f'<strong>{100*value:.0f}%</strong></div>'
    )


def render_prediction_dashboard(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = report["model"]
    score = report["score"]
    claims = report["claims"]
    ext = report["external_context"]
    hist = report["historical_validation"]
    diag = report["diagnostic_signal"]
    event = report.get("event") or {}

    features = list(score["features"])
    max_contrib = max((abs(float(row["raw_score_contribution"])) for row in features), default=1.0)
    feature_rows = []
    for row in features[:20]:
        contribution = float(row["raw_score_contribution"])
        feature_rows.append(
            "<tr>"
            f'<td><code>{html.escape(str(row["feature"]))}</code><div class="muted">{html.escape(str(row["family"]))}</div></td>'
            f'<td>{_num(row["value"], 4)}</td>'
            f'<td>{_num(row["standardized_value"], 3)}</td>'
            f'<td>{_num(row["coefficient"], 6)}</td>'
            f'<td class="signed">{float(contribution):+.6f}{_bar(contribution, max_abs=max_contrib, positive=contribution >= 0)}</td>'
            "</tr>"
        )

    claim_cards = []
    for row in claims:
        direction = str(row["direction"])
        tags = []
        if row["forward_looking"]:
            tags.append("forward-looking")
        if row["earnings_related"]:
            tags.append("earnings")
        if row["quantitative"]:
            tags.append("quantitative")
        tags.extend(str(x) for x in row["parser_matched_fields"])
        claim_cards.append(
            '<div class="claim">'
            f'<div class="claim-head"><strong>Claim {int(row["claim_index"]):02d}</strong><span class="pill {html.escape(direction)}">{html.escape(direction)}</span></div>'
            f'<p>{html.escape(str(row["text"]))}</p>'
            + _meter(float(row["interpretation_confidence"]), label="Interpretation confidence")
            + _meter(float(row["model_relevance"]), label="Model relevance")
            + f'<div class="muted">Why: {html.escape(str(row["interpretation_confidence_reason"]))}</div>'
            + f'<div class="tags">{"".join(f"<span>{html.escape(tag)}</span>" for tag in tags) if tags else "<span>unmapped</span>"}</div>'
            + '<div class="truth-note">Truth confidence: not assessed. The model treats supplied competition facts as evidence.</div>'
            + "</div>"
        )

    availability_cards = []
    for name, value in (ext.get("family_availability") or {}).items():
        available = bool(value)
        availability_cards.append(
            f'<div class="availability {"on" if available else "off"}"><span>{html.escape(str(name))}</span><strong>{"AVAILABLE" if available else "MISSING"}</strong></div>'
        )

    raw = float(score["raw_score"])
    submitted = float(score["submitted_percentile"])
    score_position = max(0.0, min(1.0, submitted))
    signed_signal = raw - 0.5

    performance_rows = [
        ("Validation N", hist.get("n"), "integer"),
        ("Spearman", hist.get("spearman"), "number"),
        ("Pearson", hist.get("pearson"), "number"),
        ("MAE", hist.get("mae"), "number"),
        ("RMSE", hist.get("rmse"), "number"),
        ("Raw prediction std", hist.get("prediction_std"), "number"),
    ]
    perf_html = "".join(
        f'<div class="stat"><span>{html.escape(label)}</span><strong>{int(value) if kind == "integer" and value is not None else _num(value, 4)}</strong></div>'
        for label, value, kind in performance_rows
    )

    realized = report.get("realized") or {}
    realized_html = ""
    if realized:
        realized_percentile = realized.get("realized_percentile")
        realized_car1 = realized.get("car1")
        error = None if realized_percentile is None else abs(float(realized_percentile) - submitted)
        realized_html = (
            '<section><h2>Realized outcome</h2><div class="stats-grid">'
            f'<div class="stat"><span>Realized CAR1</span><strong>{_num(realized_car1, 4)}</strong></div>'
            f'<div class="stat"><span>Realized percentile</span><strong>{_pct(realized_percentile)}</strong></div>'
            f'<div class="stat"><span>Absolute percentile error</span><strong>{_pct(error)}</strong></div>'
            '</div></section>'
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3-lite prediction dashboard</title>
<style>
:root {{ --bg:#0b1020; --panel:#121a2d; --panel2:#18233b; --text:#edf2ff; --muted:#99a8c7; --line:#263553; --good:#31c48d; --bad:#f05252; --accent:#7aa2ff; --warn:#f6ad55; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}} .wrap{{max-width:1400px;margin:0 auto;padding:28px}} h1{{font-size:28px;margin:0 0 6px}} h2{{font-size:18px;margin:0 0 14px}} section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}} .muted{{color:var(--muted);font-size:12px}} code{{color:#c8d7ff}} .hero{{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}} .score-card{{background:var(--panel2);border-radius:12px;padding:18px}} .big{{font-size:44px;font-weight:800;letter-spacing:-1px}} .score-track{{height:20px;background:#202d47;border-radius:999px;overflow:hidden;margin-top:10px}} .score-fill{{height:100%;background:linear-gradient(90deg,var(--bad),var(--warn),var(--good));width:{100*score_position:.2f}%}} .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}} .stat{{background:var(--panel2);border-radius:10px;padding:12px}} .stat span{{display:block;color:var(--muted);font-size:12px}} .stat strong{{display:block;font-size:20px;margin-top:3px}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:middle}} th{{color:var(--muted);font-size:12px}} .signed{{min-width:260px}} .bar-track{{height:7px;background:#202d47;border-radius:999px;overflow:hidden;margin-top:5px}} .bar{{height:100%}} .bar.pos{{background:var(--good)}} .bar.neg{{background:var(--bad)}} .claim-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}} .claim{{background:var(--panel2);border-radius:12px;padding:14px}} .claim-head{{display:flex;justify-content:space-between;align-items:center}} .claim p{{min-height:58px}} .pill{{padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase}} .pill.positive{{background:rgba(49,196,141,.16);color:#6ce0b5}} .pill.negative{{background:rgba(240,82,82,.16);color:#ff8e8e}} .pill.neutral{{background:rgba(122,162,255,.14);color:#aac2ff}} .meter-row{{display:grid;grid-template-columns:145px 1fr 40px;gap:8px;align-items:center;margin:7px 0;font-size:12px}} .meter{{height:8px;background:#202d47;border-radius:999px;overflow:hidden}} .meter-fill{{height:100%;background:var(--accent)}} .tags{{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}} .tags span{{background:#22314f;color:#c7d6f5;padding:3px 7px;border-radius:6px;font-size:11px}} .truth-note{{margin-top:10px;color:var(--muted);font-size:11px;border-top:1px solid var(--line);padding-top:8px}} .availability-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}} .availability{{padding:10px;border-radius:9px;background:var(--panel2)}} .availability span{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}} .availability strong{{font-size:12px}} .availability.on strong{{color:var(--good)}} .availability.off strong{{color:var(--bad)}} .warning{{border-left:4px solid var(--warn);padding-left:12px;color:#ffd5a0}} @media(max-width:800px){{.hero{{grid-template-columns:1fr}}.wrap{{padding:14px}}.signed{{min-width:180px}}}}
</style>
</head>
<body><div class="wrap">
<header><h1>{html.escape(str(event.get('ticker') or report.get('ticker') or 'Event'))} — V3-lite interpretation dashboard</h1>
<div class="muted">{html.escape(str(event.get('event_id') or ''))} · cutoff {html.escape(str(event.get('cutoff') or ''))} · model {html.escape(str(model['version']))}</div></header>
<section class="hero">
<div class="score-card"><div class="muted">Submitted percentile</div><div class="big">{submitted:.4f}</div><div class="score-track"><div class="score-fill"></div></div><p>Raw model score: <strong>{raw:.4f}</strong> · raw displacement from 0.5: <strong>{signed_signal:+.4f}</strong></p><p class="muted">Calibration maps the raw score to its empirical percentile among {int(model['calibration_n_fitted'])} historical out-of-sample validation predictions.</p></div>
<div class="score-card"><div class="stats-grid"><div class="stat"><span>Raw signal z vs validation</span><strong>{_num(diag['raw_signal_z_vs_validation'],2)}</strong></div><div class="stat"><span>Submitted extremeness</span><strong>{_pct(diag['submitted_extremeness'])}</strong></div><div class="stat"><span>Mean claim interpretation confidence</span><strong>{_pct(diag['mean_claim_interpretation_confidence'])}</strong></div><div class="stat"><span>Mean claim model relevance</span><strong>{_pct(diag['mean_claim_model_relevance'])}</strong></div></div><p class="warning">These are support/strength diagnostics, not a calibrated probability that the prediction is correct.</p></div>
</section>
<section><h2>External variables and data availability</h2><div class="availability-grid">{''.join(availability_cards)}</div><div class="stats-grid" style="margin-top:10px"><div class="stat"><span>Provider successes</span><strong>{int(ext.get('provider_successes') or 0)}</strong></div><div class="stat"><span>Provider errors</span><strong>{int(ext.get('provider_errors') or 0)}</strong></div><div class="stat"><span>Non-zero deployed features</span><strong>{int(ext.get('nonzero_deployed_features') or 0)}/{int(ext.get('deployed_feature_count') or 0)}</strong></div></div><p class="muted">A family can be available in V3 context without being consumed by the deployed {html.escape(str(model['ablation']))} artifact. Model-used families: {html.escape(', '.join(model.get('used_families') or []))}.</p></section>
<section><h2>Claim-by-claim interpretation</h2><div class="claim-grid">{''.join(claim_cards)}</div></section>
<section><h2>Top exact feature contributions to raw score</h2><table><thead><tr><th>Feature</th><th>Value</th><th>Z</th><th>Coefficient</th><th>Raw-score contribution</th></tr></thead><tbody>{''.join(feature_rows)}</tbody></table><p class="muted">Linear identity: intercept {float(score['intercept']):.6f} + feature contributions {float(score['sum_feature_contributions']):+.6f} = raw unclipped {float(score['raw_unclipped']):.6f}.</p></section>
<section><h2>Historical validation performance</h2><div class="stats-grid">{perf_html}</div><p class="warning">Historical correlation is modest. Interpretation confidence and extreme submitted percentiles must not be mistaken for high forecast accuracy.</p></section>
{realized_html}
<footer class="muted">Generated from the same V3-lite artifact, deterministic disclosure parser, and feature assembler used by production.</footer>
</div></body></html>"""

    output.write_text(document, encoding="utf-8")
    return output


def write_report_json(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
