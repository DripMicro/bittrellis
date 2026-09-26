"""Draw the public progress chart from the score record: what merged pull requests have added.

    python evaluator/progress_chart.py <ledger>/<epoch> > progress.svg

One column per evaluated pull request, in first-seen order. The line is the frontier gain (FG-2)
credited to merged pull requests so far; it steps up at each merge. Pull requests that were
rejected or added nothing sit on the line where they were scored. The evaluator redraws it after
every pass (evaluator/ledger.py), so it only ever shows what the published records say.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
# The README figures' palette (scripts/render_readme_assets.py), readable on light and dark backgrounds.
STYLE = """<style>
.bg{fill:#ffffff;stroke:#d0d7de}.t{fill:#1f2328}.m{fill:#59636e}.rule{stroke:#d0d7de}.hole{fill:#ffffff}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117;stroke:#30363d}.t{fill:#e6edf3}.m{fill:#9198a1}.rule{stroke:#30363d}.hole{fill:#0d1117}}
</style>"""
ACCENT, REJECT, NONE = "#8b5cf6", "#cf222e", "#8c959f"
MEASURED = {"frontier", "provisional", "dominated", "gate", "audit", "same-encoder", "nondeterministic",
            "invalid", "build", "memory", "duplicate"}


def load(epoch_dir: Path) -> list[dict]:
    """The latest record for each PR's latest head, in first-seen order, with `merged` set."""
    epoch_dir = Path(epoch_dir)
    accepted = {p.name for p in (epoch_dir / "accepted").iterdir()} if (epoch_dir / "accepted").exists() else set()
    latest: dict[tuple[int, str], dict] = {}
    for path in sorted((epoch_dir / "results").glob("pr-*.json")):  # .remeasured-N sorts after its original
        rec = json.loads(path.read_text())
        latest[(rec["pr"], rec["head"])] = rec
    by_pr: dict[int, dict] = {}
    for rec in latest.values():
        if rec["pr"] not in by_pr or rec["first_seen"] > by_pr[rec["pr"]]["first_seen"]:
            by_pr[rec["pr"]] = rec
    out = [r for r in by_pr.values() if r.get("status") in MEASURED]
    for r in out:
        r["merged"] = r.get("status") == "frontier" and r.get("name") in accepted
    return sorted(out, key=lambda r: (r["first_seen"], r["pr"]))


def _text(x: float, y: float, s: str, cls: str = "t", size: int = 12, weight: int = 400, anchor: str = "middle",
          fill: str | None = None) -> str:
    paint = f'fill="{fill}"' if fill else f'class="{cls}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" {paint} font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def _marker(x: float, y: float, rec: dict) -> str:
    if rec["merged"]:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{ACCENT}"/>'
    if rec.get("tier") == "REJECT":
        d = 5
        return (f'<path d="M{x - d:.1f},{y - d:.1f}L{x + d:.1f},{y + d:.1f}M{x - d:.1f},{y + d:.1f}L{x + d:.1f},{y - d:.1f}" '
                f'stroke="{REJECT}" stroke-width="2.5" stroke-linecap="round"/>')
    color = ACCENT if rec.get("tier") not in (None, "none", "REJECT") else NONE  # open with a paid tier, or no gain
    return f'<circle class="hole" cx="{x:.1f}" cy="{y:.1f}" r="5.5" stroke="{color}" stroke-width="2.5"/>'


def render(records: list[dict], epoch: str, window: int = 16) -> str:
    w, h = 900, 380
    left, right, top, bottom = 70, 30, 96, 300
    merged = [r for r in records if r["merged"]]
    authors = {r["author"].lower() for r in records}
    earlier, shown = records[:-window], records[-window:]  # the latest `window` columns stay readable
    start = sum(100 * (r.get("gain") or 0) for r in earlier if r["merged"])
    n = max(len(shown), 1)
    col = (w - left - right) / n
    xs = [left + col * (i + 0.5) for i in range(len(shown))]

    total, levels = start, []
    for r in shown:
        if r["merged"]:
            total += 100 * (r.get("gain") or 0)
        levels.append(total)
    ymax = max(total * 1.25, 0.05)
    y = lambda v: bottom - (bottom - top) * v / ymax  # noqa: E731

    b = [_text(24, 36, "What pull requests have added", size=18, weight=600, anchor="start"),
         _text(24, 58, f"Frontier gain (FG-2) credited to merged pull requests · epoch {epoch}"
               + (f" · latest {len(shown)} of {len(records)}" if earlier else ""), cls="m", size=12, anchor="start"),
         _text(w - 24, 36, f"+{total:.3f}%", size=18, weight=600, anchor="end", fill=ACCENT),
         _text(w - 24, 58, f"{len(merged)} merged · {len(records)} scored · {len(authors)} authors", cls="m", size=12,
               anchor="end")]

    for v in (0, ymax / 2, ymax):
        b.append(f'<line class="rule" x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke-width="1"/>')
        b.append(_text(left - 8, y(v) + 4, f"{v:.2f}%", cls="m", size=11, anchor="end"))

    if shown:  # step line: flat across each column, rising where a PR merged
        pts, prev = [f"{left:.1f},{y(start):.1f}"], start
        for x, lv in zip(xs, levels, strict=True):
            if lv != prev:
                pts += [f"{x:.1f},{y(prev):.1f}", f"{x:.1f},{y(lv):.1f}"]
            prev = lv
        pts.append(f"{w - right:.1f},{y(prev):.1f}")
        b.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ACCENT}" stroke-width="2.5" '
                 'stroke-linejoin="round"/>')

    for x, lv, r in zip(xs, levels, shown, strict=True):
        b.append(_marker(x, y(lv), r))
        if r["merged"]:
            b.append(_text(x, y(lv) - 12, r.get("tier") or "", size=11, weight=600, fill=ACCENT))
        b.append(_text(x, bottom + 22, f"#{r['pr']}", size=12, weight=600))
        author = r["author"] if len(r["author"]) <= 14 else r["author"][:13] + "…"
        b.append(_text(x, bottom + 38, author, cls="m", size=11))

    if not records:
        b.append(_text(w / 2, (top + bottom) / 2, "No pull requests scored yet in this epoch", cls="m", size=13))

    ly, lx = h - 14, 24  # legend
    for kind, label in (({"merged": True}, "merged, with its tier"), ({"merged": False, "tier": "XS"}, "scored, open"),
                        ({"merged": False, "tier": "none"}, "no new frontier space"),
                        ({"merged": False, "tier": "REJECT"}, "rejected")):
        b.append(_marker(lx + 6, ly - 4, kind))
        b.append(_text(lx + 18, ly, label, cls="m", size=11, anchor="start"))
        lx += 36 + 6.2 * len(label)

    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                      f'font-family="{FONT}">', STYLE,
                      f'<rect class="bg" x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12"/>', *b, "</svg>"]) + "\n"


def main() -> int:
    epoch_dir = Path(sys.argv[1])
    sys.stdout.write(render(load(epoch_dir), epoch_dir.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
