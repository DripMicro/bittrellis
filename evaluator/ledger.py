"""The public score record: what was measured, for which commit, and what it earned.

The evaluator runs on a rented GPU box. Its state -- who submitted what first, which results are
accepted, what each PR scored -- lived only on that disk, so returning the box would take the
frontier and the "who was first" evidence with it. After every pass the evaluator writes this
directory and pushes it to a repository of its own (evaluator/publish_ledger.py).

    <ledger>/
      README.md                        current frontier, regenerated each pass
      progress.svg                     what merged pull requests have added (evaluator/progress_chart.py)
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

    def frontier(self, doc: dict) -> None:
        (self.dir / "frontier.json").write_text(json.dumps(doc, indent=1) + "\n")
        (self.root / "README.md").write_text(render_readme(doc, self.epoch))
        (self.root / "progress.svg").write_text(progress_chart.render(progress_chart.load(self.dir), self.epoch))


def render_readme(frontier: dict, epoch: str) -> str:
    rows = sorted((r for r in frontier.get("internal", [])), key=lambda r: r["rp_kl"])
    lines = [f"# BitTrellis score records ({epoch})", "",
             "> Every evaluated pull request, the frontier it was ranked against, and the artifacts behind both.",
             "", "![Frontier gain credited to merged pull requests over time, pull requests scored per day by outcome, and the authors with the most credited gain.](progress.svg)",
             "", "Written by the evaluator after each pass. Re-derive any score yourself:", "",
             "```bash", f"bittrellis frontier {epoch}/accepted <your artifact>", "```", "",
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
