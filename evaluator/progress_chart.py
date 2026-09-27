"""Draw the public progress chart from the score record: what merged pull requests have added.

    python evaluator/progress_chart.py <ledger>/<epoch> > progress.svg

Built to stay readable at hundreds of pull requests, so nothing is drawn per pull request:

* headline numbers: frontier gain (FG-2) credited to merged PRs, merged PRs, authors, PRs scored;
* credited FG-2 over time, a step line on a date axis that climbs at each merge (the three largest
  steps are labelled);
* PRs scored per day (per week or longer on a long epoch), stacked by outcome;
* the authors with the most credited FG-2, the rest folded into "others".

Dates are first-seen dates, and the axis ends at the latest record rather than at "now", so a pass
that publishes nothing new redraws the same bytes. The evaluator redraws it after every pass
(evaluator/ledger.py), from the published records only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import floor, log10
from pathlib import Path
from xml.sax.saxutils import escape

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
# The README figures' palette, one hue plus gray and red, validated for both themes
# (lightness band, chroma, colour-blind separation, contrast against the surface).
STYLE = """<style>
.bg{fill:#ffffff;stroke:#d0d7de}.t{fill:#1f2328}.m{fill:#59636e}.rule{stroke:#d0d7de}.ring{stroke:#ffffff}
.acc{fill:#8b5cf6}.accl{stroke:#8b5cf6}.wash{fill:#8b5cf6;fill-opacity:.1}.gry{fill:#8c959f}.rej{fill:#d03b3b}.track{fill:#eaeef2}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117;stroke:#30363d}.t{fill:#e6edf3}.m{fill:#9198a1}.rule{stroke:#30363d}.ring{stroke:#0d1117}
.acc{fill:#9a75f8}.accl{stroke:#9a75f8}.wash{fill:#9a75f8;fill-opacity:.14}.gry{fill:#9198a1}.rej{fill:#e5534b}.track{fill:#21262d}}
</style>"""
MEASURED = {"frontier", "provisional", "dominated", "gate", "audit", "same-encoder", "nondeterministic",
            "invalid", "build", "memory", "duplicate"}
PAID = {"XL", "L", "M", "S", "XS"}
OUTCOMES = (("credited", "acc", "merged or earning a tier"), ("none", "gry", "no new frontier space"),
            ("rejected", "rej", "rejected"))
W, H = 900, 470


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


def outcome(r: dict) -> str:
    if r.get("tier") == "REJECT":
        return "rejected"
    return "credited" if r["merged"] or r.get("tier") in PAID else "none"


def _when(r: dict) -> datetime:
    return datetime.fromisoformat(r["first_seen"].replace("Z", "+00:00")).astimezone(timezone.utc)


def _nice(x: float) -> float:
    """The smallest 1, 2, 2.5 or 5 × 10^k at or above x."""
    if x <= 0:
        return 1.0
    k = 10 ** floor(log10(x))
    return next(m * k for m in (1, 2, 2.5, 5, 10) if m * k >= x)


def _text(x: float, y: float, s: str, cls: str = "t", size: int = 12, weight: int = 400, anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def _pct(v: float) -> str:
    return f"{v:.3f}%" if v < 1 else f"{v:.2f}%"


def _column(x: float, y: float, w: float, h: float, cls: str, round_top: bool) -> str:
    """A column segment: 4px rounded at the data end, square at the baseline."""
    if not round_top or h < 5:
        return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>'
    r = min(4.0, w / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}'
            f'H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z"/>')


def render(records: list[dict], epoch: str) -> str:
    merged = [r for r in records if r["merged"]]
    credit = [100 * (r.get("gain") or 0) for r in merged]
    total = sum(credit)
    authors = {r["author"].lower() for r in records}
    b = [_text(24, 36, "What pull requests have added", size=18, weight=600),
         _text(24, 57, f"Epoch {epoch} · redrawn by the evaluator after every pass · first-seen dates, UTC", "m")]

    # ---- headline numbers ----
    tiles = (("Frontier gain credited", f"+{_pct(total)}"), ("Merged pull requests", f"{len(merged)}"),
             ("Authors", f"{len(authors)}"), ("Pull requests scored", f"{len(records)}"))
    for i, (label, value) in enumerate(tiles):
        x = 24 + i * 213
        b += [_text(x, 88, label, "m", 12), _text(x, 120, value, size=28, weight=600)]
    b.append(f'<line class="rule" x1="24" x2="{W - 24}" y1="140" y2="140" stroke-width="1"/>')

    left, right = 64, 596            # the two time plots share this x range
    g_top, g_bot = 196, 318          # credited FG-2 over time
    c_top, c_bot = 362, 412          # PRs scored per bin
    b += [_text(24, 170, "Credited frontier gain over time", weight=600),
          _text(24, 346, "Pull requests scored", weight=600)]

    if not records:
        b.append(_text((left + right) / 2, (g_top + c_bot) / 2, "No pull requests scored yet in this epoch", "m", 13,
                       anchor="middle"))
    else:
        day = timedelta(days=1)
        t1 = datetime.combine(_when(records[-1]).date(), datetime.min.time(), timezone.utc) + day
        t0 = datetime.combine(_when(records[0]).date(), datetime.min.time(), timezone.utc)
        t0 = min(t0, t1 - 7 * day)                       # never narrower than a week
        span = (t1 - t0).days
        bin_days = next((d for d in (1, 7, 14, 30, 91) if span / d <= 90), 365)
        nbins = -(-span // bin_days)
        t1 = t0 + nbins * bin_days * day
        x = lambda t: left + (right - left) * (t - t0) / (t1 - t0)  # noqa: E731

        # credited FG-2: step line over a 10% wash
        top = _nice(total * 1.15) if total else 0.1
        gy = lambda v: g_bot - (g_bot - g_top) * v / top  # noqa: E731
        for v in (0, top / 2, top):
            b += [f'<line class="rule" x1="{left}" x2="{right}" y1="{gy(v):.1f}" y2="{gy(v):.1f}" stroke-width="1"/>',
                  _text(left - 8, gy(v) + 4, f"{v:g}%", "m", 11, anchor="end")]
        pts, level, steps = [(x(t0), gy(0))], 0.0, []
        for r, c in zip(merged, credit, strict=True):
            xm = x(_when(r))
            pts += [(xm, gy(level)), (xm, gy(level + c))]
            level += c
            steps.append((c, xm, gy(level), r))
        pts.append((x(t1), gy(level)))
        line = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        b += [f'<polygon class="wash" points="{line} {x(t1):.1f},{gy(0):.1f}"/>',
              f'<polyline class="accl" points="{line}" fill="none" stroke-width="2" stroke-linejoin="round" '
              'stroke-linecap="round"/>']
        largest = sorted(steps, key=lambda s: -s[0])[:3]
        for _, xm, ym, _ in steps if len(steps) <= 24 else largest:   # past 24 merges, dots would bury the line
            b.append(f'<circle class="acc ring" cx="{xm:.1f}" cy="{ym:.1f}" r="4.5" stroke-width="2"/>')
        placed: list[float] = []
        for _, xm, ym, r in largest:                                   # label only the three largest steps
            if all(abs(xm - p) > 72 for p in placed):
                placed.append(xm)
                b.append(_text(xm, ym - 11, f"#{r['pr']} · {r.get('tier')}", size=11, weight=600, anchor="middle"))

        # PRs scored per bin, stacked by outcome, 2px surface gap between segments
        bins: list[dict[str, int]] = [defaultdict(int) for _ in range(nbins)]
        for r in records:
            bins[min(nbins - 1, (_when(r) - t0).days // bin_days)][outcome(r)] += 1
        cmax = max(sum(bn.values()) for bn in bins)
        unit = (c_bot - c_top) / cmax
        slot = (right - left) / nbins
        cw = min(24.0, max(2.0, slot - 2))
        for i, bn in enumerate(bins):
            cx, y0 = left + slot * i + (slot - cw) / 2, float(c_bot)
            present = [k for k, _, _ in OUTCOMES if bn.get(k)]
            for k, cls, _ in OUTCOMES:
                if bn.get(k):
                    h, gap = bn[k] * unit, 2 if y0 < c_bot else 0
                    b.append(_column(cx, y0 - h, cw, h - gap, cls, k == present[-1]))
                    y0 -= h
        b += [f'<line class="rule" x1="{left}" x2="{right}" y1="{c_bot}" y2="{c_bot}" stroke-width="1"/>',
              _text(left - 8, c_top + 4, f"{cmax}", "m", 11, anchor="end"),
              _text(left - 8, c_bot, "0", "m", 11, anchor="end"),
              _text(right, 346, f"per {'day' if bin_days == 1 else f'{bin_days} days'}", "m", 11, anchor="end")]

        # shared date axis: at most seven labels
        every = max(1, -(-nbins // 6))
        for i in range(0, nbins + 1, every):
            t = t0 + i * bin_days * day
            b.append(_text(x(t), c_bot + 17, f"{t:%b} {t.day}", "m", 11, anchor="middle"))

        lx = left  # legend for the columns: words, not colour alone
        for _, cls, label in OUTCOMES:
            b += [f'<rect class="{cls}" x="{lx}" y="{H - 24}" width="10" height="10" rx="2"/>',
                  _text(lx + 15, H - 15, label, "m", 11)]
            lx += 30 + 6.2 * len(label)

    # ---- authors with the most credited FG-2 ----
    px, pw = 640, W - 24 - 640
    b.append(_text(px, 170, "Top authors by credited gain", weight=600))
    by_author: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for r, c in zip(merged, credit, strict=True):
        by_author[r["author"]][0] += c
        by_author[r["author"]][1] += 1
    ranked = sorted(by_author.items(), key=lambda kv: (-kv[1][0], kv[0].lower()))
    top, rest = ranked[:7], ranked[7:]
    best = max((v[0] for _, v in top), default=0) or 1        # the bars compare individual authors only
    if not ranked:
        b.append(_text(px, 200, "No merged pull requests yet", "m"))
    for i, (name, (gain, n)) in enumerate(top):
        y = 196 + i * 34
        name = name if len(name) <= 22 else name[:21] + "…"
        b += [_text(px, y, name, size=12),
              _text(px + pw, y, f"+{_pct(gain)} · {n} merged", "m", 11, anchor="end"),
              f'<rect class="track" x="{px}" y="{y + 6}" width="{pw}" height="8" rx="4"/>',
              f'<rect class="acc" x="{px}" y="{y + 6}" width="{max(8.0, pw * gain / best):.1f}" height="8" rx="4"/>']
    if rest:
        y = 196 + len(top) * 34
        b += [_text(px, y, f"{len(rest)} more authors", "m", 12),
              _text(px + pw, y, f"+{_pct(sum(v[0] for _, v in rest))} · {sum(v[1] for _, v in rest)} merged", "m", 11,
                    anchor="end")]

    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                      f'font-family="{FONT}" role="img" aria-label="Frontier gain credited to merged pull requests: '
                      f'+{_pct(total)} from {len(merged)} merges, {len(records)} pull requests scored">', STYLE,
                      f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12"/>', *b, "</svg>"]) + "\n"


def main() -> int:
    epoch_dir = Path(sys.argv[1])
    sys.stdout.write(render(load(epoch_dir), epoch_dir.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
