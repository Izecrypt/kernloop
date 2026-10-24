"""Render a run log to a single self-contained HTML trajectory page.

Inline SVG, no JavaScript, no external assets: iterations on the x-axis, kernel
mean time (ms) on the y-axis, a champion step line, per-iteration markers
(green = promoted, orange = rejected on merit, red = failed compile or
correctness), and decision annotations. Everything is HTML-escaped.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .runlog import read_log

WIDTH, HEIGHT = 960, 460
PAD_L, PAD_R, PAD_T, PAD_B = 80, 30, 40, 70


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_html(records: list[dict], run_id: str, backend: str) -> str:
    timed = [(r["iteration"], r["timing"]["mean_ms"], r) for r in records if r.get("timing", {}).get("mean_ms") is not None]
    if not timed:
        body = "<p>No timed iterations in this run log.</p>"
        return _page(run_id, backend, body, records)

    iters = [t[0] for t in timed]
    means = [t[1] for t in timed]
    t_min, t_max = min(means), max(means)
    if t_max == t_min:
        t_max = t_min + 1.0
    pad_y = (t_max - t_min) * 0.15
    t_min, t_max = t_min - pad_y, t_max + pad_y
    x_min, x_max = min(iters) - 0.5, max(iters) + 0.5

    def sx(i: float) -> float:
        return PAD_L + (i - x_min) / (x_max - x_min) * (WIDTH - PAD_L - PAD_R)

    def sy(t: float) -> float:
        return PAD_T + (t_max - t) / (t_max - t_min) * (HEIGHT - PAD_T - PAD_B)

    # Champion step line: best champion mean up to and including each iteration.
    champ_steps: list[tuple[float, float]] = []
    best = None
    recs_by_iter = {r["iteration"]: r for r in records}
    for i in range(min(iters), max(iters) + 1):
        rec = recs_by_iter.get(i)
        if rec and rec.get("timing", {}).get("mean_ms") is not None:
            if rec["decision"]["promoted"]:
                best = rec["timing"]["mean_ms"]
        if best is not None:
            champ_steps.append((i, best))

    parts: list[str] = []
    parts.append(f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="KernLoop trajectory">')
    # axes
    parts.append(f'<line x1="{PAD_L}" y1="{HEIGHT-PAD_B}" x2="{WIDTH-PAD_R}" y2="{HEIGHT-PAD_B}" stroke="#444"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{HEIGHT-PAD_B}" stroke="#444"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = t_min + frac * (t_max - t_min)
        y = sy(val)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{WIDTH-PAD_R}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#555">{val:.1f}</text>')
    for i in range(min(iters), max(iters) + 1):
        x = sx(i)
        parts.append(f'<text x="{x:.1f}" y="{HEIGHT-PAD_B+18}" text-anchor="middle" font-size="11" fill="#555">{i}</text>')
    parts.append(f'<text x="{WIDTH/2}" y="{HEIGHT-18}" text-anchor="middle" font-size="13" fill="#333">iteration</text>')
    parts.append(f'<text x="18" y="{HEIGHT/2}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 18 {HEIGHT/2})">kernel mean time (ms)</text>')

    # champion step line
    if champ_steps:
        d = " ".join(
            f"{'M' if k == 0 else 'L'}{sx(i):.1f},{sy(t):.1f} " + (f"L{sx(i+1):.1f},{sy(t):.1f}" if k < len(champ_steps) - 1 else "")
            for k, (i, t) in enumerate(champ_steps)
        )
        parts.append(f'<path d="{d.strip()}" fill="none" stroke="#1a73e8" stroke-width="2"/>')
        parts.append(f'<text x="{WIDTH-PAD_R-4}" y="{sy(champ_steps[-1][1])-8:.1f}" text-anchor="end" font-size="11" fill="#1a73e8">champion</text>')

    # per-iteration markers
    for i, mean, rec in timed:
        promoted = rec["decision"]["promoted"]
        failed = not (rec.get("compile", {}).get("ok") and rec.get("correctness", {}).get("ok"))
        color = "#188038" if promoted else ("#d93025" if failed else "#e8710a")
        x, y = sx(i), sy(mean)
        reason = html.escape(str(rec["decision"]["reason"]))
        model = html.escape(str(rec.get("proposal", {}).get("model", "?")))
        title = html.escape(f"iter {i}: mean {mean:.4f} ms | {'PROMOTED' if promoted else 'rejected'}: {reason} | {model}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{title}</title></circle>')

    parts.append("</svg>")
    body = "\n".join(parts)
    return _page(run_id, backend, body, records)


def _page(run_id: str, backend: str, chart: str, records: list[dict]) -> str:
    timed = [r for r in records if r.get("timing", {}).get("mean_ms") is not None]
    best = min((r["timing"]["mean_ms"] for r in timed), default=None)
    promotions = sum(1 for r in records if r.get("decision", {}).get("promoted"))
    rows = []
    for r in records:
        decision = r.get("decision", {})
        cls = "promoted" if decision.get("promoted") else ("failed" if not r.get("correctness", {}).get("ok", True) else "rejected")
        rows.append(
            "<tr class='{cls}'><td>{it}</td><td>{mean}</td><td>{minv}</td><td>{dec}</td><td>{reason}</td><td>{model}</td></tr>".format(
                cls=cls,
                it=r.get("iteration"),
                mean=_fmt(r.get("timing", {}).get("mean_ms")),
                minv=_fmt(r.get("timing", {}).get("min_ms")),
                dec="PROMOTED" if decision.get("promoted") else "rejected",
                reason=html.escape(str(decision.get("reason", ""))),
                model=html.escape(str(r.get("proposal", {}).get("model", "?"))),
            )
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KernLoop trajectory {html.escape(run_id)}</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 1000px; color: #222; }}
h1 {{ font-size: 1.4rem; }} .meta {{ color: #555; font-size: 0.95rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 1.5rem; }}
td, th {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
tr.promoted td {{ background: #e6f4ea; }} tr.failed td {{ background: #fce8e6; }}
.legend span {{ margin-right: 1rem; font-size: 0.85rem; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 5px; margin-right: 4px; }}
</style>
</head>
<body>
<h1>KernLoop trajectory</h1>
<p class="meta">run <code>{html.escape(run_id)}</code> | backend <b>{html.escape(backend)}</b> | iterations {len(records)} | promotions {promotions} | best mean {_fmt(best)} ms</p>
<p class="legend"><span><span class="dot" style="background:#188038"></span>promoted</span><span><span class="dot" style="background:#e8710a"></span>rejected (no real improvement)</span><span><span class="dot" style="background:#d93025"></span>failed compile or correctness</span><span style="color:#1a73e8">champion line</span></p>
{chart}
<table><thead><tr><th>iter</th><th>mean ms</th><th>min ms</th><th>decision</th><th>reason</th><th>model</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body>
</html>
"""


def render_log_to_html(log_path: str | Path, out_path: str | Path) -> Path:
    path = Path(log_path)
    with path.open("r", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    run_id = first.get("run_id", path.stem)
    backend = first.get("backend", "unknown")
    records = [r for r in read_log(path) if "iteration" in r]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(records, run_id, backend), encoding="utf-8")
    return out
