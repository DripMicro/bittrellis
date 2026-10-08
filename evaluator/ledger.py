"""The public score record: what was measured, for which commit, and what it earned.

The evaluator runs on a rented GPU box. Its state -- who submitted what first, which results are
accepted, what each PR scored -- lived only on that disk, so returning the box would take the
frontier and the "who was first" evidence with it. After every pass the evaluator writes this
directory and pushes it to a repository of its own (evaluator/publish_ledger.py).

    <ledger>/
      README.md                        current frontier, regenerated each pass
      progress.svg                     what merged pull requests have added (evaluator/progress_chart.py)
      tradeoffs.svg                    the trade-off each merged recipe makes, for choosing one
      <epoch>/frontier.json            the ranking after the pass
      <epoch>/results/<pr>-<head>.json one record per evaluated PR head, never rewritten
                                       (a re-measurement is published beside it as .remeasured-N)
      <epoch>/observations/*.json      first-seen records, copied from the evaluator's own store
      <epoch>/accepted/<name>/*        artifacts of merged, frontier-moving results

Anyone can re-derive every score from `accepted/` and a result record:

    bittrellis frontier <ledger>/<epoch>/accepted <your artifact>

What never enters the ledger: the private holdout (only PASS/FAIL reaches a record), the evaluator's
secret, tokens, and the sandbox's contributed code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import progress_chart

RECORD_FIELDS = ("pr", "head", "author", "first_seen", "kind", "status", "tier", "candidate", "name",
                 "gain", "references", "skipped", "screen")
ARTIFACT_FILES = ("candidate.json", "quality.json", "performance.json", "tasks.json", "correctness.json",
                  "audit.json", "environment.json", "holdout.json", "kl_positions.npz")


class Ledger:
    def __init__(self, root: Path, epoch: str):
        self.root, self.epoch = Path(root), epoch
        self.dir = self.root / epoch
        for sub in ("results", "observations", "accepted"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict, row: dict | None) -> Path:
        """Write one PR head's outcome. The first record for a head is never rewritten.

        A head can legitimately be measured more than once: a dominated result resumes when the PR
        above it closes, and a replacement box re-measures an open PR it has no artifact for. Speed
        is measured, not derived, so the second row is never byte-identical to the first. Overwriting
        would let a re-run quietly restate what a contributor was told they had earned, so the
        original stands and the new measurement is published beside it as a numbered revision.
        """
        results = self.dir / "results"
        stem = f"pr-{entry['pr']:06d}-{entry['head'][:12]}"
        doc = {k: entry.get(k) for k in RECORD_FIELDS if entry.get(k) is not None}
        doc["row"] = row
        path = results / f"{stem}.json"
        if not path.exists():
            path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            return path
        for existing in [path, *sorted(results.glob(f"{stem}.remeasured-*.json"))]:
            if json.loads(existing.read_text()) == doc:
                return existing                      # already published, unchanged
        n = 1 + len(list(results.glob(f"{stem}.remeasured-*.json")))
        revision = results / f"{stem}.remeasured-{n}.json"
        revision.write_text(json.dumps({**doc, "supersedes": path.name}, indent=1, sort_keys=True) + "\n")
        return revision

    def observations(self, source: Path) -> int:
        """Copy the first-seen records: the evidence for who submitted a recipe first."""
        n = 0
        for src in sorted(Path(source).glob("pr-*.json")):
            dst = self.dir / "observations" / src.name
            if not dst.exists() or dst.read_text() != src.read_text():
                shutil.copyfile(src, dst)
                n += 1
        return n

    def accept(self, name: str, artifact: Path) -> None:
        """Publish a merged, frontier-moving result so anyone can re-rank against it."""
        dst = self.dir / "accepted" / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ARTIFACT_FILES:
            src = Path(artifact) / f
            if src.exists():
                shutil.copyfile(src, dst / f)

    def speeds(self, source: Path) -> None:
        """Publish the reference speeds measured on this machine (hpc01-e5), so a ranking can be re-derived."""
        machine = Path(source) / "machine.json"
        if not machine.exists():
            return
        dst = self.dir / "speeds" / json.loads(machine.read_text())["measured_utc"].replace(":", "")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(machine, dst / "machine.json")
        for perf in Path(source).glob("*/performance.json"):
            (dst / perf.parent.name).mkdir(exist_ok=True)
            shutil.copyfile(perf, dst / perf.parent.name / "performance.json")

    def frontier(self, doc: dict) -> None:
        records = progress_chart.load(self.dir)
        merged = [r for r in records if r["merged"]]
        (self.dir / "frontier.json").write_text(json.dumps(doc, indent=1) + "\n")
        # One evaluator per track writes here: HPC-01 keeps the record's front page (and the links to it), every
        # other track writes its page into its own epoch folder, and the front page links to those.
        primary = self.epoch.startswith("hpc01")
        page = self.root if primary else self.dir
        others = sorted(p.parent.name for p in self.root.glob("*/README.md") if not p.parent.name.startswith("hpc01")) if primary else []
        (page / "README.md").write_text(render_readme(doc, self.epoch, merged, others))
        (page / "progress.svg").write_text(progress_chart.render(records, self.epoch))
        (page / "tradeoffs.svg").write_text(progress_chart.render_tradeoffs(doc, merged))


TIER_COLORS = {"XL": "0e8a16", "L": "2da44e", "M": "4ac26b", "S": "8ddb8c", "XS": "c6efce"}  # as evaluator/pr_bot.py
PICKS = (("Closest to the original model", "rp_kl", min), ("Fastest prompt reading", "prefill_tps", max),
         ("Least GPU memory", "peak_gpu_gib", min))


def _versus(r: dict, inc: dict) -> tuple[str, str, str]:
    """Drift, prompt speed and memory of `r`, each with its change against the shipped checkpoint in words."""
    drift = (inc["rp_kl"] - r["rp_kl"]) / inc["rp_kl"]
    speed = r["prefill_tps"] / inc["prefill_tps"] - 1
    mem = inc["peak_gpu_gib"] - r["peak_gpu_gib"]
    return (f"{r['rp_kl']:.4f} · {abs(drift):.1%} {'closer' if drift >= 0 else 'further'}",
            f"{r['prefill_tps']:,.0f} tok/s · {abs(speed):.0%} {'faster' if speed >= 0 else 'slower'}",
            f"{r['peak_gpu_gib']:.2f} GiB · {abs(mem):.2f} GiB {'less' if mem >= 0 else 'more'}")


def recommend(frontier: dict, merged: list[dict]) -> list[str]:
    """'Which checkpoint to use': the best merged recipe on each axis that still stands on the frontier.

    Only results that passed the private holdout and are not dominated by a later result are offered.
    """
    rows = {r["name"]: r for r in frontier.get("internal", [])}
    inc = rows.get(frontier.get("incumbent", ""))
    cands = [(m, rows[m["name"]]) for m in merged if m["name"] in rows and rows[m["name"]].get("valid")
             and rows[m["name"]].get("frontier") and rows[m["name"]].get("holdout") == "PASS"]
    if not inc or not cands:
        return []
    lines = ["## Which checkpoint to use", "",
             "Every merged recipe trades a little of one thing for another. Pick by what you need; each change is "
             "against today's shipped checkpoint (V0).", "",
             "![The trade-off each merged recipe makes: prompt speed against closeness to the original model, "
             "with peak GPU memory](tradeoffs.svg)", "",
             "| Best for | Recipe | Closeness to the original (RP-KL) | Prompt reading, 4K | Peak GPU memory |",
             "|---|---|---|---|---|"]
    picks: dict[str, tuple[dict, dict, list[str], set[int]]] = {}   # one row per recipe, however many picks it wins
    for i, (label, key, pick) in enumerate(PICKS):
        m, r = pick(cands, key=lambda c: c[1][key])
        entry = picks.setdefault(m["name"], (m, r, [], set()))
        entry[2].append(label)
        entry[3].add(i)
    for m, r, labels, won in picks.values():
        tier = m.get("tier") or ""
        badge = (f"![eval:{tier}](https://img.shields.io/badge/eval%3A{tier}-{TIER_COLORS[tier]}?style=flat-square)"
                 if tier in TIER_COLORS else "")
        cells = [f"**{c}**" if i in won else c for i, c in enumerate(_versus(r, inc))]   # bold what it was picked for
        lines.append(f"| **{' · '.join(labels)}** | `{m['name']}`<br>#{m['pr']} by @{m['author']} {badge} | "
                     + " | ".join(cells) + " |")
    lines += [f"| *for reference* | V0, today's shipped checkpoint | {inc['rp_kl']:.4f} | {inc['prefill_tps']:,.0f} tok/s | "
              f"{inc['peak_gpu_gib']:.2f} GiB |",
              "", "Build one yourself (after `scripts/setup_models.sh` in "
              "[bittrellis](https://github.com/coderbench/bittrellis)):", "", "```bash",
              *[f"bittrellis build manifests/{n}.yaml --out models/{n}" for n in picks], "```", ""]
    return lines


def render_readme(frontier: dict, epoch: str, merged: list[dict] | None = None, other_tracks: list[str] | None = None) -> str:
    rows = sorted((r for r in frontier.get("internal", [])), key=lambda r: r["rp_kl"])
    track = "HPC-" + epoch.split("-")[0][3:]                       # hpc02-e1 -> HPC-02
    flag = "" if track == "HPC-01" else f"--track {track} "
    others = ["Other tracks: " + " · ".join(f"[{d}]({d}/README.md)" for d in other_tracks), ""] if other_tracks else []
    lines = [f"# BitTrellis score records ({epoch})", "",
             "> Every evaluated pull request, the frontier it was ranked against, and the artifacts behind both.",
             "", *others, *recommend(frontier, merged or []),
             "## Progress", "", "![Frontier gain credited to merged pull requests over time, pull requests scored per day by outcome, and the authors with the most credited gain.](progress.svg)",
             "", "## Every measured result", "", "Written by the evaluator after each pass. Re-derive any score yourself:", "",
             "```bash", f"bittrellis {flag}frontier {epoch}/accepted <your artifact>", "```", "",
             "| | Checkpoint | RP-KL ↓ | tasks ↑ | decode tok/s ↑ | prefill 4K tok/s ↑ | peak GPU GiB ↓ | holdout | FG-2 |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        mark = "★" if r.get("frontier") else " "
        tasks = f"{r['tasks_passed']}/{r['tasks_n']}" if r.get("tasks_n") else "—"
        gain = f"{100 * (r['frontier_gain'] or 0):.3f}%" if r.get("frontier_gain") is not None else "—"
        gates = "; ".join(r.get("gate_failures") or [])
        lines.append(f"| {mark} | {r['name']} | {r['rp_kl']:.4f} | {tasks} | {r['decode_tps']:.1f} | "
                     f"{r['prefill_tps']:,.0f} | {r['peak_gpu_gib']:.2f} | {r.get('holdout') or '—'} | {gain} |")
        if gates:
            lines.append(f"| | ↳ *not credited: {gates}* | | | | | | | |")
    lines += ["", f"Epoch `{epoch}` · rules: [bittrellis](https://github.com/coderbench/bittrellis) "
              "([specification](https://github.com/coderbench/bittrellis/blob/main/docs/specification.md)).",
              "", "`results/` holds one record per evaluated pull-request head -- never rewritten, with any "
              "re-measurement published beside it as `.remeasured-N` -- `observations/` the "
              "first-seen record that decides who submitted a recipe first, and `accepted/` the artifacts of "
              "merged results. The private holdout never appears here: records carry PASS or FAIL only."]
    return "\n".join(lines) + "\n"
