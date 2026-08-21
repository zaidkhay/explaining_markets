"""Graph live prediction performance once realized CAR1 percentiles are known.

Input CSV columns:
    ticker,predicted_percentile,realized_percentile
Optional:
    event_id,car1,date

Example:
    uv run python scripts/render_model_performance_dashboard.py \
      --input data/live_eval/2026-08-20.csv \
      --output data/diagnostics/performance_2026-08-20.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean

from explaining_markets.calibration import spearman


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    ma, mb = mean(a), mean(b)
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    if da <= 1e-12 or db <= 1e-12:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(da * db)


def _load(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    out = []
    for raw in rows:
        if raw.get("predicted_percentile") in (None, "") or raw.get("realized_percentile") in (None, ""):
            continue
        out.append({
            **raw,
            "ticker": str(raw.get("ticker") or raw.get("identifier_value") or "?").upper(),
            "predicted_percentile": float(raw["predicted_percentile"]),
            "realized_percentile": float(raw["realized_percentile"]),
            "car1": None if raw.get("car1") in (None, "") else float(raw["car1"]),
        })
    return out


def _scatter_svg(rows: list[dict], width: int = 620, height: int = 500) -> str:
    pad = 54
    inner_w, inner_h = width - 2 * pad, height - 2 * pad
    def x(v): return pad + max(0.0, min(1.0, v)) * inner_w
    def y(v): return pad + (1.0 - max(0.0, min(1.0, v))) * inner_h
    grid = []
    for step in range(0, 11, 2):
        v = step / 10
        grid.append(f'<line x1="{x(v):.1f}" y1="{pad}" x2="{x(v):.1f}" y2="{height-pad}" class="grid"/>')
        grid.append(f'<line x1="{pad}" y1="{y(v):.1f}" x2="{width-pad}" y2="{y(v):.1f}" class="grid"/>')
        grid.append(f'<text x="{x(v):.1f}" y="{height-pad+22}" text-anchor="middle">{v:.1f}</text>')
        grid.append(f'<text x="{pad-12}" y="{y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>')
    dots = []
    for row in rows:
        pred, real = row["predicted_percentile"], row["realized_percentile"]
        err = abs(pred - real)
        radius = 5 + min(6, 8 * err)
        title = html.escape(f"{row['ticker']} predicted={pred:.3f} realized={real:.3f} error={err:.3f}")
        dots.append(f'<circle cx="{x(pred):.1f}" cy="{y(real):.1f}" r="{radius:.1f}" class="dot"><title>{title}</title></circle>')
        dots.append(f'<text x="{x(pred)+7:.1f}" y="{y(real)-7:.1f}" class="ticker">{html.escape(row["ticker"])}</text>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Predicted versus realized percentile scatter">{''.join(grid)}<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" class="diag"/>{''.join(dots)}<text x="{width/2}" y="{height-8}" text-anchor="middle" class="axis-label">Predicted percentile</text><text transform="translate(16 {height/2}) rotate(-90)" text-anchor="middle" class="axis-label">Realized percentile</text></svg>'''


def _error_bars(rows: list[dict], limit: int = 15) -> str:
    ranked = sorted(rows, key=lambda r: abs(r["predicted_percentile"] - r["realized_percentile"]), reverse=True)[:limit]
    max_error = max((abs(r["predicted_percentile"] - r["realized_percentile"]) for r in ranked), default=1.0)
    parts = []
    for row in ranked:
        error = abs(row["predicted_percentile"] - row["realized_percentile"])
        width = 100 * error / max(max_error, 1e-12)
        parts.append(f'<div class="error-row"><span>{html.escape(row["ticker"])}</span><div class="bar-track"><div class="error-bar" style="width:{width:.1f}%"></div></div><strong>{error:.3f}</strong></div>')
    return "".join(parts)


def _bucket_rows(rows: list[dict]) -> str:
    parts = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        bucket = [r for r in rows if lo <= r["predicted_percentile"] < hi or (hi >= 1.0 and r["predicted_percentile"] == 1.0)]
        if bucket:
            avg_pred = mean(r["predicted_percentile"] for r in bucket)
            avg_real = mean(r["realized_percentile"] for r in bucket)
            gap = avg_real - avg_pred
        else:
            avg_pred = avg_real = gap = None
        parts.append(
            "<tr>"
            f"<td>{lo:.1f}–{hi:.1f}</td><td>{len(bucket)}</td>"
            f"<td>{'n/a' if avg_pred is None else f'{avg_pred:.3f}'}</td>"
            f"<td>{'n/a' if avg_real is None else f'{avg_real:.3f}'}</td>"
            f"<td>{'n/a' if gap is None else f'{gap:+.3f}'}</td></tr>"
        )
    return "".join(parts)


def render(rows: list[dict], output: Path, title: str) -> None:
    if not rows:
        raise ValueError("no rows with predicted_percentile and realized_percentile")
    predicted = [r["predicted_percentile"] for r in rows]
    realized = [r["realized_percentile"] for r in rows]
    errors = [abs(a - b) for a, b in zip(predicted, realized, strict=True)]
    squared = [(a - b) ** 2 for a, b in zip(predicted, realized, strict=True)]
    spear = spearman(predicted, realized)
    pear = _pearson(predicted, realized)
    mae = mean(errors)
    rmse = math.sqrt(mean(squared))
    direction_hit = mean(1.0 if (p - 0.5) * (r - 0.5) > 0 else 0.0 if p != 0.5 and r != 0.5 else 0.5 for p, r in zip(predicted, realized, strict=True))
    best = sorted(rows, key=lambda r: abs(r["predicted_percentile"] - r["realized_percentile"]))[:5]
    worst = sorted(rows, key=lambda r: abs(r["predicted_percentile"] - r["realized_percentile"]), reverse=True)[:5]

    def cards(items):
        return "".join(f'<div class="mini"><strong>{html.escape(r["ticker"])}</strong><span>pred {r["predicted_percentile"]:.3f} · real {r["realized_percentile"]:.3f} · err {abs(r["predicted_percentile"]-r["realized_percentile"]):.3f}</span></div>' for r in items)

    doc = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
:root{{--bg:#0b1020;--panel:#121a2d;--panel2:#18233b;--text:#edf2ff;--muted:#9ba9c7;--line:#263553;--accent:#7aa2ff;--bad:#f05252;--good:#31c48d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}.wrap{{max-width:1350px;margin:auto;padding:28px}}h1{{margin:0 0 4px}}h2{{font-size:18px}}section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.metric,.mini{{background:var(--panel2);padding:12px;border-radius:10px}}.metric span,.mini span{{display:block;color:var(--muted);font-size:12px}}.metric strong{{font-size:24px}}.two{{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}}svg{{width:100%;height:auto;background:var(--panel2);border-radius:10px}}svg text{{fill:var(--muted);font-size:11px}}.grid{{stroke:#293958;stroke-width:1}}.diag{{stroke:var(--good);stroke-width:2;stroke-dasharray:6 5}}.dot{{fill:var(--accent);opacity:.85}}.ticker{{fill:var(--text);font-size:10px}}.axis-label{{fill:var(--text);font-size:12px}}.error-row{{display:grid;grid-template-columns:65px 1fr 55px;gap:8px;align-items:center;margin:9px 0}}.bar-track{{height:9px;background:#202d47;border-radius:999px;overflow:hidden}}.error-bar{{height:100%;background:var(--bad)}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}.list{{display:grid;gap:7px}}.note{{color:var(--muted)}}@media(max-width:850px){{.two{{grid-template-columns:1fr}}.wrap{{padding:14px}}}}</style></head><body><div class="wrap"><h1>{html.escape(title)}</h1><div class="note">Realized-percentile evaluation of {len(rows)} events. Ranking metrics matter most for the competition target.</div>
<section><div class="metrics"><div class="metric"><span>Events</span><strong>{len(rows)}</strong></div><div class="metric"><span>Spearman</span><strong>{'n/a' if spear is None else f'{spear:.3f}'}</strong></div><div class="metric"><span>Pearson</span><strong>{'n/a' if pear is None else f'{pear:.3f}'}</strong></div><div class="metric"><span>MAE</span><strong>{mae:.3f}</strong></div><div class="metric"><span>RMSE</span><strong>{rmse:.3f}</strong></div><div class="metric"><span>Above/below 0.5 agreement</span><strong>{100*direction_hit:.0f}%</strong></div></div></section>
<section class="two"><div><h2>Predicted vs realized percentile</h2>{_scatter_svg(rows)}</div><div><h2>Largest absolute misses</h2>{_error_bars(rows)}</div></section>
<section><h2>Calibration by submitted-score bucket</h2><table><thead><tr><th>Predicted bucket</th><th>N</th><th>Avg predicted</th><th>Avg realized</th><th>Realized − predicted</th></tr></thead><tbody>{_bucket_rows(rows)}</tbody></table></section>
<section class="two"><div><h2>Best calls</h2><div class="list">{cards(best)}</div></div><div><h2>Worst calls</h2><div class="list">{cards(worst)}</div></div></section>
</div></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/diagnostics/model_performance.html")
    parser.add_argument("--title", default="V3-lite realized prediction performance")
    args = parser.parse_args()
    rows = _load(Path(args.input))
    output = Path(args.output)
    render(rows, output, args.title)
    print(f"rows: {len(rows)}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
