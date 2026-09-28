"""BitTrellis PR evaluator: scores miner pull requests on the pinned RTX 5090 host.

Run it on the evaluation host (cron or a loop):

    GITHUB_TOKEN=... python evaluator/pr_bot.py --repo coderbench/bittrellis --root /workspace/bt-eval \
        --private /secure/holdout-epoch [--once]

Every pass:

0. OBSERVE every open PR head before evaluating anything, in an append-only record under <root>
   (evaluator/guards.py). The queue runs in first-seen order, so an original is always measured
   before a later near-copy of it.

For each PR, in that order:

1. CLASSIFY from its changed files.
   * manifest PR     exactly one added/changed `manifests/*.yaml`, nothing else
                     -> evaluated with the trusted `main` code; only the YAML is read from the PR
   * code PR         changes under bittrellis/quantizers/ or bittrellis/search.py (+ at most one manifest)
                     -> evaluated only after a maintainer adds `eval-approved`, from a worktree of the PR head
   * evaluator paths -> never evaluated automatically; labelled `bt:touches-evaluator`
2. SCREEN, no GPU (seconds):
   queue share per author · manifest valid · exact duplicate (seeds, accepted, earlier PRs) ·
   near-copy of an earlier PR by another author · predicted peak memory ·
   code PRs: new quantizers are deterministic and do not reproduce an existing encoder (probe bytes)
3. BUILD and AUDIT on the CPU. Code PRs: the new encoder's REAL stored bytes are compared with every
   existing encoder's bytes for the same tensors (sketches), still before any GPU time.
4. MEASURE in stages, stopping as soon as the outcome is known:
   quality -> quality gates -> performance (2 runs) -> dominated? stop (tasks and holdout can only fail
   a result, never lift a dominated one) -> tasks -> private holdout -> rank.
5. RANK against seeds + accepted results + earlier open PRs by other authors, so a later submission
   earns only what it adds over work already on the table. Comment once, label the status.

After the queue, RE-RANK evaluated open PRs whose reference set changed (an earlier PR merged or
closed): labels stay current, and a PR that becomes non-dominated is resumed for tasks and holdout.

6. MERGE the top-ranked open result, with --auto-merge. Merging is the payment event and a merged
   tier is final, so every condition is re-checked against a fresh read of the PR at merge time, and
   the merge names the exact evaluated head SHA: GitHub refuses it if the branch moved since.
7. CLOSE open PRs whose head was rejected or is a duplicate: they cannot earn, and an open PR takes one
   of the author's open-PR slots on Gittensor. Dominated PRs stay open (a re-rank can revive them).

State lives in <root>/state.json; artifacts in <root>/prs/<number>-<sha>/; accepted artifacts in
<root>/accepted/ (copied when a PR that the bot evaluated is merged).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import guards as G  # noqa: E402
from ledger import Ledger  # noqa: E402

MANIFEST_GLOB = "manifests/*.yaml"
CODE_GLOBS = ("bittrellis/quantizers/*", "bittrellis/search.py", "tests/*")
EVALUATOR_GLOBS = ("configs/*", "data/*", "bittrellis/eval/*", "bittrellis/frontier/*", "bittrellis/validate.py",
                   "bittrellis/lineage.py", "bittrellis/holdout.py", "bittrellis/runtime.py", "bittrellis/build.py",
                   "bittrellis/fingerprint.py", "bittrellis/synthetic.py", "bittrellis/manifest.py", "bittrellis/precision.py",
                   "bittrellis/cli.py", "evaluator/*", "tools/*", ".github/*", "scripts/*", "pyproject.toml")
# Colours carry the meaning, so a label reads at a glance:
#   green       credited              pale yellow  measured, partial credit     grey   measured or seen, nothing new
#   pale blue   waiting in line       yellow/amber waiting for a maintainer     blue   maintainer review of evaluator changes
#   light red   fix your submission   red          failed a measured gate       dark red  integrity: audit or copied encoder
#   orange      would not fit the GPU purple       the evaluator's own fault (retried)
LABELS = {
    "frontier": ("bt:frontier", "0e8a16", "moves the internal frontier; credited"),
    "dominated": ("bt:dominated", "bfc5cc", "measured and valid, but another result is at least as good on every objective"),
    "duplicate": ("bt:duplicate", "e1e4e8", "same recipe as a seed, an accepted result or an earlier PR; not measured"),
    "queued": ("bt:queued", "c5def5", "waiting: the author's earlier PRs are ahead in the queue"),
    "needs_approval": ("bt:needs-approval", "fbca04", "runs contributed code; waiting for a maintainer's eval-approved"),
    "copy-review": ("bt:copy-review", "e99a1c", "repeated near-copies of other authors' PRs; waiting for a maintainer"),
    "provisional": ("bt:provisional", "bfd4f2", "measured without a private holdout PASS; no paid tier"),
    "evaluator": ("bt:touches-evaluator", "1d76db", "changes evaluator paths; maintainer review, not evaluated"),
    "invalid": ("bt:invalid-manifest", "f4a6a6", "the manifest does not validate; fix and push"),
    "build": ("bt:build-fail", "f4a6a6", "the checkpoint or quantizer probe did not build; fix and push"),
    "memory": ("bt:memory", "e8590c", "predicted to exceed the GPU's memory; not measured"),
    "gate": ("bt:gate-fail", "d73a4a", "failed a quality, task, runtime or holdout gate"),
    "nondeterministic": ("bt:nondeterministic", "d73a4a", "the new quantizer produced different bytes on identical runs"),
    "audit": ("bt:audit-fail", "b60205", "the checkpoint is not a legal encoding of the pinned weights"),
    "same-encoder": ("bt:same-encoder", "b60205", "the new quantizer reproduces an existing encoder's bytes"),
    "error": ("bt:eval-error", "8250df", "the evaluator failed, not the submission; retried automatically"),
}
TIERS = ("XL", "L", "M", "S", "XS")
TIER_COLORS = {"XL": "0e8a16", "L": "2da44e", "M": "4ac26b", "S": "8ddb8c", "XS": "c6efce", "none": "bfc5cc", "REJECT": "b60205"}
TIER_TEXT = {"XL": "very large frontier gain", "L": "large frontier gain", "M": "medium frontier gain",
             "S": "small frontier gain", "XS": "minimum credited frontier gain",
             "none": "evaluated, no new frontier space", "REJECT": "failed a gate, the audit or a screen"}
REJECTED = {"gate", "audit", "same-encoder", "nondeterministic", "invalid", "build", "memory"}
EXTRA_LABELS = {
    "merge-first": ("bt:merge-first", "2da44e", "the highest-scoring open result; merged next"),
    "derivative": ("bt:derivative", "fff3b0", "close to an earlier PR by another author; credited only for what it adds"),
    "approved": ("eval-approved", "0e8a16", "maintainer: evaluate this PR's contributed code in the sandbox"),
    "copy-cleared": ("copy-cleared", "0e8a16", "maintainer: measure this PR despite repeated near-copies"),
}
APPROVED, COPY_CLEARED = "eval-approved", "copy-cleared"
RESCREEN = {"queued", "needs-approval", "copy-review", "unsafe-host"}   # re-checked every pass
RANKED = {"frontier", "dominated", "gate"}
MAX_ERRORS = 3


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo, self.token = repo, token

    def api(self, method: str, path: str, body: dict | None = None):
        url = path if path.startswith("https://") else f"https://api.github.com/repos/{self.repo}{path}"
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return json.loads(data) if data else None

    def paged(self, path: str) -> list:
        out, page = [], 1
        while True:
            sep = "&" if "?" in path else "?"
            batch = self.api("GET", f"{path}{sep}per_page=100&page={page}")
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def ensure_labels(self) -> None:
        existing = {lab["name"]: lab for lab in self.paged("/labels")}
        tiers = [(f"{REWARDS['label_family']}:{t}", c, TIER_TEXT[t]) for t, c in TIER_COLORS.items()]
        for name, color, desc in [*LABELS.values(), *EXTRA_LABELS.values(), *tiers]:
            if name not in existing:
                self.api("POST", "/labels", {"name": name, "color": color, "description": desc})
            elif (existing[name]["color"].lower(), existing[name].get("description") or "") != (color, desc):
                self.api("PATCH", f"/labels/{urllib.request.quote(name)}", {"color": color, "description": desc})

    def set_status_label(self, number: int, key: str) -> None:
        current = {lab["name"] for lab in self.api("GET", f"/issues/{number}")["labels"]}
        for k, (name, _, _) in LABELS.items():
            if name in current and k != key:
                try:
                    self.api("DELETE", f"/issues/{number}/labels/{urllib.request.quote(name)}")
                except urllib.error.HTTPError:
                    pass
        if LABELS[key][0] not in current:
            self.api("POST", f"/issues/{number}/labels", {"labels": [LABELS[key][0]]})

    def set_tier_label(self, number: int, tier: str | None) -> None:
        """Exactly one eval:* label (or none while pending) — the label Gittensor pays."""
        family = REWARDS["label_family"]
        want = f"{family}:{tier}" if tier else None
        current = {lab["name"] for lab in self.api("GET", f"/issues/{number}")["labels"]}
        for name in current:
            if name.startswith(f"{family}:") and name != want:
                try:
                    self.api("DELETE", f"/issues/{number}/labels/{urllib.request.quote(name)}")
                except urllib.error.HTTPError:
                    pass
        if want and want not in current:
            self.api("POST", f"/issues/{number}/labels", {"labels": [want]})

    def remove_label(self, number: int, name: str) -> None:
        try:
            self.api("DELETE", f"/issues/{number}/labels/{urllib.request.quote(name)}")
        except urllib.error.HTTPError:
            pass

    def add_label(self, number: int, name: str) -> None:
        self.api("POST", f"/issues/{number}/labels", {"labels": [name]})

    def comment(self, number: int, body: str) -> None:
        self.api("POST", f"/issues/{number}/comments", {"body": body})

    def close(self, number: int) -> None:
        self.api("PATCH", f"/pulls/{number}", {"state": "closed"})

    def pull(self, number: int) -> dict:
        """A fresh read of one PR: mergeability, draft state and labels as they are right now."""
        return self.api("GET", f"/pulls/{number}")

    def merge(self, number: int, sha: str, method: str, title: str, message: str) -> tuple[bool, str]:
        """Merge exactly `sha`. GitHub refuses with 409 if the head moved after it was evaluated."""
        try:
            r = self.api("PUT", f"/pulls/{number}/merge",
                         {"sha": sha, "merge_method": method, "commit_title": title, "commit_message": message}) or {}
            return bool(r.get("merged")), r.get("sha", "")
        except urllib.error.HTTPError as e:
            try:
                reason = json.loads(e.read().decode(errors="replace")).get("message", "")
            except (json.JSONDecodeError, OSError):
                reason = ""
            return False, f"HTTP {e.code} {reason}".strip()


def classify(files: list[str]) -> tuple[str, list[str]]:
    manifests = [f for f in files if fnmatch.fnmatch(f, MANIFEST_GLOB)]
    if any(any(fnmatch.fnmatch(f, g) for g in EVALUATOR_GLOBS) for f in files):
        return "evaluator", manifests
    others = [f for f in files if f not in manifests]
    if not others and len(manifests) == 1:
        return "manifest", manifests
    code = [f for f in others if any(fnmatch.fnmatch(f, g) for g in CODE_GLOBS)]
    if code and all(f in code or f.endswith(".md") for f in others) and len(manifests) <= 1:
        return "code", manifests
    return "other", manifests


def run(cmd: list[str], cwd: Path, log: Path, timeout: int = 6 * 3600) -> int:
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode


# ── pure decisions (unit-tested) ───────────────────────────────────────────────────────────────


def queue_order(prs: list[dict], first_seen: dict[tuple[int, str], str]) -> list[dict]:
    """Oldest observed head first; PR number only breaks exact ties."""
    return sorted(prs, key=lambda p: (first_seen[(p["number"], p["head"]["sha"])], p["number"]))


def quality_gate_failures(quality: dict, gates: dict) -> list[str]:
    """Gates decidable from the quality stage alone; failing one makes the speed runs pointless."""
    fails = []
    if quality.get("nonfinite_logprobs"):
        fails.append(f"{quality['nonfinite_logprobs']} non-finite log-probabilities")
    if quality["rp_kl"] > gates["rp_kl_max"]:
        fails.append(f"RP-KL {quality['rp_kl']:.4f} > {gates['rp_kl_max']}")
    if quality.get("top1") is not None and quality["top1"] < gates["top1_min"]:
        fails.append(f"top-1 {quality['top1']:.3f} < {gates['top1_min']}")
    for stream, need in gates["long_context_guard"]["required_success"].items():
        got = (quality.get("needles_by_length") or {}).get(stream)
        if got and got["required"] and got["retrieved"] / got["required"] < need:
            fails.append(f"long-context guard: {stream} {got['retrieved']}/{got['required']} needles")
    return fails


def reference_entries(me: dict, state: dict, live_heads: dict[int, str]) -> list[str]:
    """State keys of earlier, measured PRs by other authors that are still open or merged.

    `live_heads`: {PR number: head sha} for open PRs and merged PRs. A later PR is ranked with these
    already on the frontier, so it earns only what it adds over work already on the table.
    """
    out = []
    for key, e in state.items():
        if not isinstance(e, dict) or e.get("status") not in RANKED or not e.get("artifact") or "first_seen" not in e:
            continue
        if e["pr"] == me["pr"] or e["author"].lower() == me["author"].lower() or e["first_seen"] >= me["first_seen"]:
            continue
        if live_heads.get(e["pr"]) != e["head"]:
            continue
        out.append(key)
    return sorted(out)


def status_from_row(row: dict) -> str:
    if not row["valid"]:
        return "gate"
    if not (row["frontier"] and (row["frontier_gain"] or 0) > 0):
        return "dominated"
    # A paid tier needs an explicit private-holdout PASS; without one the result is only provisional.
    return "frontier" if row.get("holdout") == "PASS" else "provisional"


# ── comment ────────────────────────────────────────────────────────────────────────────────────


PENDING = {"queued", "needs_approval", "copy-review", "evaluator", "error", "provisional"}
NOT_MEASURED = {"memory", "invalid", "build", "same-encoder", "nondeterministic", "audit"}
def _rewards() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / "configs/hpc01.yaml").read_text())["rewards"]


REWARDS = _rewards()
FG_EXPLAINED = ("FG-2 is the share of the quality × speed × memory space (normalized to the track's box) that this "
                "result adds on top of every earlier result. It is the PR's score.")


def tier_for(label: str, gain: float | None, thresholds: dict) -> str | None:
    """The eval:* tier Gittensor pays for a status: a bucket of FG-2, none, REJECT, or None while pending."""
    if label in PENDING:
        return None
    if label in REJECTED:
        return "REJECT"
    if label != "frontier" or not gain or gain <= 0:
        return "none"
    return next((t for t in TIERS if gain >= thresholds[t]), "none")


def pick_merge_first(candidates: list[dict]) -> dict | None:
    """Among open results with a paid tier: highest tier, then largest FG-2, then observed first."""
    rank = {t: i for i, t in enumerate(TIERS)}
    paid = [c for c in candidates if c.get("tier") in rank]
    return min(paid, key=lambda c: (rank[c["tier"]], -(c.get("gain") or 0), c["first_seen"])) if paid else None


# Closed automatically: this head can no longer earn, and an open PR takes one of the author's open-PR
# slots on Gittensor. Dominated results stay open, since a closing reference can revive them (rerank).
CLOSED_STATUSES = REJECTED | {"duplicate"}


def to_close(open_prs: list[dict], state: dict) -> list[int]:
    """Open PRs whose current head was rejected or duplicates a known recipe."""
    return [p["number"] for p in open_prs
            if state.get(f"{p['number']}-{p['head']['sha'][:12]}", {}).get("status") in CLOSED_STATUSES]


# GitHub's mergeable_state. "clean" alone: "unstable" means a check is failing or still running, and
# "blocked", "behind" and "dirty" mean a review, an update or a conflict is outstanding.
MERGEABLE_STATES = {"clean"}


def merge_blockers(entry: dict, detail: dict) -> list[str]:
    """Why the top-ranked result must not be merged automatically; empty means merge it.

    Merging pays, and a merged tier is final, so this repeats what the pass already decided against a
    fresh read of the PR: the measured head is still the head, the result still earns, contributed
    code still carries its approval, and GitHub itself considers the branch clean.
    """
    out = []
    if entry.get("status") != "frontier":
        out.append(f"status is {entry.get('status') or 'unevaluated'}, not frontier")
    if entry.get("tier") not in TIERS:
        out.append(f"tier {entry.get('tier') or 'none'} is not paid")
    if entry.get("head") != detail.get("head", {}).get("sha"):
        out.append("the head moved after it was evaluated")
    if detail.get("draft"):
        out.append("the PR is a draft")
    if detail.get("merged"):
        out.append("already merged")
    if entry.get("kind") == "code" and APPROVED not in {lab["name"] for lab in detail.get("labels", [])}:
        out.append(f"contributed code without {APPROVED}")
    state = detail.get("mergeable_state")
    if detail.get("mergeable") is None:
        out.append("GitHub has not finished computing mergeability")
    elif not detail["mergeable"]:
        out.append(f"GitHub says it is not mergeable ({state})")
    elif state not in MERGEABLE_STATES:
        out.append(f"mergeable_state is {state}, not clean")
    return out


def score_header(label: str, row: dict | None = None) -> str:
    """The first line of every bot comment: the PR's score and whether it is credited."""
    if label == "provisional":
        return "**Score: pending** · no private holdout PASS, so no paid tier"
    if label in PENDING:
        return "**Score: pending** · not evaluated yet"
    if label == "duplicate":
        return f"**Score: `{REWARDS['label_family']}:none` · ×0** · not measured, duplicate"
    if label in NOT_MEASURED:
        return f"**Score: `{REWARDS['label_family']}:REJECT` · ×0** · not measured"
    gain = 100 * ((row or {}).get("frontier_gain") or 0)
    if label == "frontier":
        tier = tier_for(label, gain / 100, REWARDS["tiers_fg2"])
        mult = REWARDS["multipliers"][tier]
        return f"**Score: `{REWARDS['label_family']}:{tier}` · ×{mult:g} on Gittensor when merged** · FG-2 +{gain:.3f}%"
    if label == "gate":
        return f"**Score: `{REWARDS['label_family']}:REJECT` · ×0** · failed a gate"
    if label == "duplicate":
        return f"**Score: `{REWARDS['label_family']}:none` · ×0** · not measured, duplicate"
    return f"**Score: `{REWARDS['label_family']}:none` · ×0** · no new frontier space"


OBJECTIVE_SENSE = {"rp_kl": min, "decode_tps": max, "prefill_tps": max, "peak_gpu_gib": min}


def nearest_on_frontier(row: dict, frontier: dict, k: int = 2) -> list[dict]:
    """The `k` valid frontier results closest to `row` in the track's normalized box, V0 aside.

    FG-2 is the space a result adds beyond these, so they are what explains its score.
    """
    box = frontier.get("box")
    if not box:
        return []

    def point(r: dict) -> list[float]:
        return [(r[o] - box[o][0]) / (box[o][1] - box[o][0]) for o in OBJECTIVE_SENSE]

    here = point(row)
    others = [r for r in frontier["internal"] if r.get("valid") and r.get("frontier")
              and r["name"] not in (row["name"], frontier.get("incumbent"))]
    return sorted(others, key=lambda r: sum((a - b) ** 2 for a, b in zip(here, point(r), strict=True)))[:k]


def render_comment(name: str, cid: str, frontier: dict | None, cmp: dict | None, label: str, notes: list[str],
                   screen: dict | None = None, timings: dict | None = None, epoch: str | None = None,
                   pr_of: dict[str, int] | None = None) -> str:
    epoch = frontier["evaluator_epoch"] if frontier else epoch or "—"
    row = next((r for r in frontier["internal"] if r["name"] == name), None) if frontier else None
    lines = [f"### BitTrellis evaluation · `{name}` · `{cid}`", "", score_header(label, row), "",
             f"Epoch `{epoch}` · status **{LABELS[label][0]}**", ""]
    if row:
        def tasks(r: dict) -> str:
            return f"{r['tasks_passed']}/{r['tasks_n']}" if r.get("tasks_n") else "not run"

        def cells(r: dict, bold: set[str] = frozenset()) -> str:
            v = {"rp_kl": f"{r['rp_kl']:.4f}", "decode_tps": f"{r['decode_tps']:.1f}", "prefill_tps": f"{r['prefill_tps']:,.0f}",
                 "peak_gpu_gib": f"{r['peak_gpu_gib']:.2f}"}
            v = {k: f"**{x}**" if k in bold else x for k, x in v.items()}
            return f"{v['rp_kl']} | {tasks(r)} | {v['decode_tps']} | {v['prefill_tps']} | {v['peak_gpu_gib']}"

        inc = next((r for r in frontier["internal"] if r["name"] == frontier["incumbent"]), None)
        near = nearest_on_frontier(row, frontier) if row["valid"] else []
        shown = [r for r in [inc, *near] if r]
        wins = {k for k, best in OBJECTIVE_SENSE.items()                      # this PR beats every row shown
                if shown and all(best(row[k], r[k]) == row[k] and row[k] != r[k] for r in shown)}
        lines += ["| | RP-KL ↓ | tasks passed ↑ | decode tok/s ↑ | prefill 4K tok/s ↑ | peak GPU GiB ↓ | holdout | FG-2 |",
                  "|---|---:|---:|---:|---:|---:|---|---:|",
                  f"| **this PR** | {cells(row, wins)} | {row['holdout'] or 'not run'} | {100 * (row['frontier_gain'] or 0):.3f}% |"]
        if inc:
            lines.append(f"| V0 incumbent | {cells(inc)} | — | — |")
        for r in near:
            who = f" (#{pr_of[r['name']]})" if pr_of and r["name"] in pr_of else ""
            lines.append(f"| ↳ `{r['name']}`{who} | {cells(r)} | {r.get('holdout') or '—'} | — |")
        if near:
            lines += ["", "↳ the nearest results already on the frontier: FG-2 is the space this PR adds beyond them and "
                      "every other result." + (" **Bold** marks where this PR beats every row shown." if wins else "")]
        lines += ["", "RP-KL measures how closely the model keeps the original's predictions (fidelity), not task accuracy; "
                  "tasks are a guard, compared with V0 question by question."]
        if cmp:
            k = cmp["rp_kl"]
            lines += ["", f"Paired RP-KL vs V0: **{k['delta']:+.4f}** nats/token, 95% CI [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]"
                      f"{' (significant)' if k['significant'] else ' (not significant)'}; decode {100 * cmp['decode_tps']['rel']:+.1f}%, "
                      f"prefill {100 * cmp['prefill_tps']['rel']:+.1f}%, peak GPU {cmp['peak_gpu_gib']['delta']:+.2f} GiB."]
        if row["gate_failures"]:
            lines += ["", "Gate failures:", *[f"- {g}" for g in row["gate_failures"]]]
        if row.get("dominated_by"):
            lines += ["", "Dominated by " + ", ".join(f"`{n}`" for n in row["dominated_by"]) + ": at least as good on every objective "
                      "within noise, and better on one."]
        elif row["valid"] and not (row["frontier_gain"] or 0):
            lines += ["", "On the frontier, but it adds no new frontier space: another result covers the same trade-off within noise."]
    if notes:
        lines += ["", *[f"- {n}" for n in notes]]
    if row:
        lines += ["", f"<sub>{FG_EXPLAINED}</sub>"]
    if timings:
        lines += ["", "GPU time: " + " · ".join(f"{k.removesuffix('_seconds')} {v / 60:.1f} min" for k, v in timings.items())]
    lines += ["", f"<!-- bittrellis-result {json.dumps({'name': name, 'id': cid, 'status': LABELS[label][0], 'row': row, 'screen': screen})} -->"]
    return "\n".join(lines)


# ── evaluator ──────────────────────────────────────────────────────────────────────────────────


class Evaluator:
    def __init__(self, gh: GitHub, args):
        from bittrellis.track import load_track

        self.gh, self.args = gh, args
        self.root = Path(args.root)
        self.root.mkdir(parents=True, exist_ok=True)
        # The sandbox account must reach only <root>/prs/<pr>/untrusted: traversable root, private state.
        os.chmod(self.root, 0o711)
        (self.root / "prs").mkdir(exist_ok=True)
        os.chmod(self.root / "prs", 0o711)
        self.track = load_track("HPC-01")
        self.cfg = self.track["evaluation"]["screen"]
        self.epoch = self.track["evaluation"]["epoch"]
        self.obs = G.Observations(self.root)
        self.accepted = self.root / "accepted"
        self.accepted.mkdir(exist_ok=True)
        os.chmod(self.accepted, 0o700)
        os.chmod(self.obs.dir, 0o700)
        self.state_path = self.root / "state.json"
        self.state: dict = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if self.state_path.exists():
            os.chmod(self.state_path, 0o600)
        self.state.setdefault("_merged", {})
        secret = self.secret_path = self.root / "secret.txt"
        if not secret.exists():
            secret.write_text(secrets.token_hex(32) + "\n")
        secret.chmod(0o600)
        self.secret = secret.read_text().strip()
        self.sandbox = None
        self.probe_seed = int(hashlib.sha256(f"{self.secret}:probe".encode()).hexdigest()[:8], 16)
        self.ledger = None
        if args.ledger:
            self.restore(Path(args.ledger))
            self.ledger = Ledger(Path(args.ledger), self.epoch)
            os.chmod(self.ledger.root, 0o700)
        self.py = [sys.executable, "-m", "bittrellis.cli"]
        self.env_args = ["--base", args.base, "--shipped", args.shipped, "--unsloth", args.unsloth]
        self.sources = {"base": Path(args.base), "gittensor_nvfp4": Path(args.shipped), "unsloth_nvfp4": Path(args.unsloth)}

    def restore(self, ledger_dir: Path) -> None:
        """Rebuild what a replacement box would otherwise lose, from the published record.

        A rented GPU box is replaced, not repaired, and everything that decides what a contributor is
        owed -- who submitted a recipe first, and which results are already on the table -- lived only
        on its disk. Before anything is written this pass: fetch the published history, then take back
        the first-seen records (submission priority and copy credit) and the accepted artifacts
        (the references later PRs are ranked against). Both stores are write-once, so restoring is a
        copy of what is missing; nothing already here is overwritten.
        """
        if self.args.ledger_remote:
            try:
                import publish_ledger as P

                if P.adopt(ledger_dir, self.args.ledger_remote, os.environ.get(P.TOKEN_ENV)):
                    print(f"[restore] fetched the published record into {ledger_dir}")
            except Exception as e:  # noqa: BLE001 - a box with no network still evaluates
                print(f"[restore] could not fetch the published record: {e!r}")
        # A new epoch changes the rules, not the history: who submitted first, and which PRs are merged, carry
        # over from earlier epochs. Seeds are not carried: they come from the repository, judged by this epoch.
        src = ledger_dir / self.epoch
        earlier = sorted((d for d in ledger_dir.iterdir() if d.is_dir() and d.name != self.epoch
                          and (d / "observations").exists()), reverse=True) if ledger_dir.exists() else []
        seeds = {p.name for p in self._artifacts(Path(self.args.seeds))}
        obs = 0
        for epoch_dir in [src, *earlier]:
            for f in sorted((epoch_dir / "observations").glob("pr-*.json")) if (epoch_dir / "observations").exists() else []:
                if not (self.obs.dir / f.name).exists():
                    shutil.copyfile(f, self.obs.dir / f.name)
                    obs += 1
        acc = 0
        for epoch_dir in [src, *earlier]:
            for d in sorted((epoch_dir / "accepted").iterdir()) if (epoch_dir / "accepted").exists() else []:
                carried = epoch_dir != src
                if d.is_dir() and not (self.accepted / d.name).exists() and not (carried and d.name in seeds):
                    shutil.copytree(d, self.accepted / d.name)
                    acc += 1
        if obs or acc:
            print(f"[restore] {obs} first-seen record(s), {acc} accepted result(s) recovered")

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")
        os.chmod(self.state_path, 0o600)

    # ---- known results --------------------------------------------------------------------

    def _artifacts(self, base: Path) -> list[Path]:
        return sorted(p for p in base.iterdir() if (p / "candidate.json").exists()) if base.exists() else []

    def known_ids(self) -> dict[str, str]:
        out = {}
        for p in self._artifacts(Path(self.args.seeds)) + self._artifacts(self.accepted):
            c = json.loads((p / "candidate.json").read_text())
            if c.get("kind") in ("internal", "candidate"):
                out[c["id"]] = c["name"]
        return out

    def seed_memory(self) -> list[tuple[dict[str, str], float]]:
        out = []
        for p in self._artifacts(Path(self.args.seeds)):
            c = json.loads((p / "candidate.json").read_text())
            perf = p / "performance.json"
            if c.get("kind") in ("internal", "candidate") and perf.exists():
                out.append((G.keys_from_expanded(c["manifest"]["expanded"]), json.loads(perf.read_text())["peak_gpu_gib"]))
        return out

    def live_heads(self, open_prs: list[dict]) -> dict[int, str]:
        heads = {p["number"]: p["head"]["sha"] for p in open_prs}
        heads.update({int(k): v for k, v in self.state["_merged"].items()})
        return heads

    # ---- one pass -------------------------------------------------------------------------

    def run_once(self) -> None:
        self.sync_merged()
        open_prs = self.gh.paged("/pulls?state=open&sort=created&direction=asc")
        first_seen = {}
        for pr in open_prs:  # observe everything before evaluating anything
            o = self.obs.observe(pr["number"], pr["user"]["login"], pr["head"]["sha"])
            first_seen[(pr["number"], pr["head"]["sha"])] = o["first_seen"]
        self.rerank(open_prs)  # first, so a PR lifted by a closed reference resumes in this pass
        for pr in queue_order(open_prs, first_seen):
            key = f"{pr['number']}-{pr['head']['sha'][:12]}"
            entry = self.state.get(key, {})
            status = entry.get("status")
            if status and status not in RESCREEN and status != "resume" and not (status == "error" and entry.get("errors", 0) < MAX_ERRORS):
                continue
            try:
                self.evaluate(pr, open_prs, resume=status == "resume")
            except Exception as e:  # noqa: BLE001 - one broken PR must not stop the queue
                self.state[key] = {**self.state.get(key, {}), "status": "error", "error": repr(e),
                                   "errors": entry.get("errors", 0) + 1}
                try:
                    self.gh.set_status_label(pr["number"], "error")
                except urllib.error.URLError:
                    pass
                traceback.print_exc()
            self.save()
        self.rerank(open_prs)
        self.auto_merge(self.mark_merge_first(open_prs))
        self.close_rejected(open_prs)
        self.publish_records()
        self.save()

    def publish_records(self) -> None:
        """Write the public score record and push it. A failed push never fails the pass."""
        if not self.ledger:
            return
        self.ledger.observations(self.obs.dir)
        for name in self._accepted_names():
            self.ledger.accept(name, self.accepted / name)
        frontier_json = self.root / "frontier.json"
        paths = [self.args.seeds, str(self.accepted)]
        if subprocess.run(self.py + ["frontier", *paths, "--out", str(frontier_json)], cwd=REPO_ROOT,
                          capture_output=True, text=True).returncode == 0:
            self.ledger.frontier(json.loads(frontier_json.read_text()))
        if not self.args.ledger_remote:
            return
        try:
            import publish_ledger as P

            commit = P.publish(Path(self.args.ledger), self.args.ledger_remote, os.environ.get(P.TOKEN_ENV),
                               f"records: {self.epoch} pass")
            print(f"[ledger] published {commit}" if commit else "[ledger] nothing new")
        except Exception as e:  # noqa: BLE001 - the records stay local and go out with the next pass
            print(f"[ledger] not published: {e!r}")

    def _accepted_names(self) -> list[str]:
        return sorted(p.name for p in self.accepted.iterdir()) if self.accepted.exists() else []

    def mark_merge_first(self, open_prs: list[dict]) -> int | None:
        """One `bt:merge-first` per pass: the result that is merged next, and the number it is on.

        With merging off nothing is merged next, so the label is removed everywhere rather than promised.
        """
        entries = {pr["number"]: self.state.get(f"{pr['number']}-{pr['head']['sha'][:12]}", {}) for pr in open_prs}
        best = pick_merge_first([{**e, "pr": n} for n, e in entries.items() if e.get("status") == "frontier" and "first_seen" in e])
        name = EXTRA_LABELS["merge-first"][0]
        merging = getattr(self.args, "auto_merge", False)
        for pr in open_prs:
            has = name in {lab["name"] for lab in pr["labels"]}
            if merging and best and pr["number"] == best["pr"]:
                if not has:
                    self.gh.add_label(pr["number"], name)
            elif has:
                self.gh.remove_label(pr["number"], name)
        return best["pr"] if best else None

    def sync_merged(self) -> None:
        """Copy artifacts of merged, frontier-moving PRs into accepted/ so later PRs are ranked against them."""
        for pr in self.gh.paged("/pulls?state=closed&sort=updated&direction=desc")[:100]:
            if not pr.get("merged_at"):
                continue
            # Merged is final: the tier on a merged PR is what Gittensor pays, so the bot never relabels it.
            self.accept_merged(pr["number"], pr["head"]["sha"])

    def accept_merged(self, number: int, sha: str) -> None:
        """Record a merge and make its artifact a ranking reference for every later PR."""
        self.state["_merged"][str(number)] = sha
        entry = self.state.get(f"{number}-{sha[:12]}")
        if entry and entry.get("status") == "frontier" and not (self.accepted / entry["name"]).exists():
            shutil.copytree(entry["artifact"], self.accepted / entry["name"])

    def auto_merge(self, number: int | None) -> None:
        """Merge the top-ranked open result. Refusing is always safe: it is merged on a later pass."""
        if not getattr(self.args, "auto_merge", False) or number is None:
            return
        try:
            detail = self.gh.pull(number)
        except urllib.error.URLError as e:
            print(f"[merge] #{number}: cannot read the PR ({e!r})")
            return
        sha = detail.get("head", {}).get("sha", "")
        entry = self.state.get(f"{number}-{sha[:12]}", {})
        blockers = merge_blockers(entry, detail)
        if blockers:
            print(f"[merge] #{number} not merged: {'; '.join(blockers)}")
            return
        tier, gain = entry["tier"], 100 * (entry.get("gain") or 0)
        family = REWARDS["label_family"]
        merged, why = self.gh.merge(
            number, sha, self.args.merge_method, f"{detail['title']} (#{number})",
            f"{family}:{tier} \u00b7 FG-2 +{gain:.3f}% \u00b7 evaluator epoch {self.epoch}\n"
            f"Merged by the evaluator at {sha}.")
        if not merged:
            print(f"[merge] #{number} refused by GitHub: {why}")
            return
        self.gh.comment(number, f"Merged automatically as the top-ranked open result: `{family}:{tier}`, "
                                f"FG-2 +{gain:.3f}%. A merged tier is final and is what Gittensor pays.")
        self.accept_merged(number, sha)
        print(f"[merge] #{number} merged as {why}")

    def close_rejected(self, open_prs: list[dict]) -> None:
        """Close rejected and duplicate PRs, so they stop taking the author's open-PR slots on Gittensor."""
        for number in to_close(open_prs, self.state):
            try:
                self.gh.close(number)
            except urllib.error.URLError as e:
                print(f"[close] #{number}: not closed ({e!r})")
                continue
            try:
                self.gh.comment(number, "Closed automatically: this head cannot earn, and an open PR counts against "
                                        "your open-PR limit on Gittensor. Closing does not affect your credibility on this "
                                        "repository (`min_credibility` is 0). A revised recipe is welcome as a new PR.")
            except urllib.error.URLError:
                pass
            print(f"[close] #{number} closed")

    def evaluate(self, pr: dict, open_prs: list[dict], resume: bool = False) -> None:
        number, sha, author = pr["number"], pr["head"]["sha"], pr["user"]["login"]
        key = f"{number}-{sha[:12]}"
        me = self.obs.observe(number, author, sha)
        work = self.root / "prs" / key
        work.mkdir(parents=True, exist_ok=True)
        log = work / "eval.log"
        labels = {lab["name"] for lab in pr["labels"]}
        files = [f["filename"] for f in self.gh.paged(f"/pulls/{number}/files")]
        kind, manifests = classify(files)
        base_entry = {"pr": number, "head": sha, "author": author, "first_seen": me["first_seen"], "kind": kind}

        def finish(status: str, label: str | None = None, body: str | None = None, **extra) -> None:
            tier = tier_for(label or status, extra.get("gain"), REWARDS["tiers_fg2"])
            self.state[key] = {**self.state.get(key, {}), **base_entry, "status": status, "tier": tier, **extra}
            if self.ledger and status not in RESCREEN:
                self.ledger.record(self.state[key], extra.get("row"))
            if label:
                self.gh.set_status_label(number, label)
                self.gh.set_tier_label(number, tier)
            if body:
                if not body.startswith("### BitTrellis evaluation"):
                    body = f"{score_header(label or status)}\n\n{body}"
                self.gh.comment(number, body)

        if kind in ("evaluator", "other") or not manifests:
            return finish(kind, "evaluator", "BitTrellis evaluator: this PR changes evaluator or other protected paths "
                                             "(or adds no manifest), so it is not evaluated automatically. A maintainer will review it.")
        if kind == "code" and APPROVED not in labels:
            if self.state.get(key, {}).get("status") != "needs-approval":
                finish("needs-approval", "needs_approval")
            return None

        # queue share: an author's PRs wait their turn; nothing is rejected
        pending = [self.obs.observe(p["number"], p["user"]["login"], p["head"]["sha"]) for p in open_prs
                   if self.state.get(f"{p['number']}-{p['head']['sha'][:12]}", {}).get("status") in (None, "queued")]
        q = G.judge_queue(me, pending, self.cfg)
        if q.stop:
            if self.state.get(key, {}).get("status") != "queued":
                finish("queued", "queued")
            return None

        untrusted = kind == "code"
        if untrusted:
            problems = self.sandbox_problems()
            if problems:
                if self.state.get(key, {}).get("status") != "unsafe-host":
                    finish("unsafe-host", "error", "BitTrellis evaluator: this PR runs contributed code, and this evaluation "
                           "host is not isolated yet, so it is not evaluated. A maintainer has been notified.\n\n"
                           + "\n".join(f"- {p}" for p in problems))
                return None
        code = (work / "untrusted" / "code") if untrusted else (work / "code")
        code.parent.mkdir(parents=True, exist_ok=True)
        if code.exists():
            run(["git", "worktree", "remove", "--force", str(code)], REPO_ROOT, log)
            shutil.rmtree(code, ignore_errors=True)
        run(["git", "fetch", "--quiet", "origin", f"pull/{number}/head", "main"], REPO_ROOT, log)
        run(["git", "worktree", "add", "--force", "--detach", str(code), sha if kind == "code" else "origin/main"], REPO_ROOT, log)
        if untrusted:
            os.chmod(work, 0o711)
            self.sandbox.own(code.parent)
        try:
            return self._evaluate_in(pr, me, kind, manifests, labels, work, code, log, finish, open_prs, resume)
        finally:
            run(["git", "worktree", "remove", "--force", str(code)], REPO_ROOT, log)

    def _evaluate_in(self, pr, me, kind, manifests, labels, work, code, log, finish, open_prs, resume):
        from bittrellis import fingerprint as F
        from bittrellis.manifest import candidate_hash_keys
        from bittrellis.model.qwen38 import Qwen38Arch

        number, sha = pr["number"], pr["head"]["sha"]
        untrusted = kind == "code"
        utr = work / "untrusted" if untrusted else work   # where contributed code may write
        xrun = self._runner(untrusted)
        notes: list[str] = []
        manifest = utr / "manifest.yaml"
        manifest.write_text(subprocess.run(["git", "show", f"{sha}:{manifests[0]}"], cwd=REPO_ROOT,
                                           capture_output=True, text=True).stdout)
        if untrusted:
            self.sandbox.own(utr)
        ids = utr / "ids.json"
        if xrun(self.py + ["manifest", str(manifest), "--ids-out", str(ids), "--shipped", self.args.shipped], code, log) != 0 or not ids.exists():
            return finish("invalid", "invalid", "BitTrellis evaluator: the manifest does not validate:\n\n```\n"
                          + log.read_text()[-3000:] + "\n```")
        ident = json.loads(ids.read_text())
        cid, keys = ident["id"], ident["keys"]
        if candidate_hash_keys(self.track.id, keys) != cid:
            return finish("invalid", "invalid", "BitTrellis evaluator: the candidate id reported by this PR's code does not "
                          "match its recipe.")
        units = Qwen38Arch().units()
        numel = {u.id: u.numel for u in units}

        # ---- screen: duplicates and near-copies (by expanded recipe) ----
        live = self.live_heads(open_prs)
        earlier = [o for o in self.obs.all() if live.get(o["pr"]) == o["head"]]
        verdict = G.judge_manifest(me, cid, keys, numel, earlier, self.known_ids(), self.cfg, self.epoch,
                                   cleared=COPY_CLEARED in labels)
        self.obs.annotate(number, sha, candidate_id=cid, keys=keys, epoch=self.epoch, derivative_of=verdict.derivative_of)
        screen = {"manifest": verdict.to_dict()}
        if verdict.outcome == "duplicate":
            return finish("duplicate", "duplicate", f"BitTrellis evaluator: **not measured**. This recipe (`{cid}`) is "
                          f"{verdict.reason}. Identical recipes earn nothing, however their rules are written.", screen=screen)
        if verdict.outcome == "copy-review":
            if self.state.get(f"{number}-{sha[:12]}", {}).get("status") != "copy-review":
                finish("copy-review", "copy-review", f"BitTrellis evaluator: **waiting for a maintainer**. This is "
                       f"{verdict.reason}. A maintainer adds `{COPY_CLEARED}` to measure it.", screen=screen)
            return None
        if verdict.derivative_of:
            self.gh.add_label(number, EXTRA_LABELS["derivative"][0])
            notes.append(f"Close to #{verdict.derivative_of['pr']} by @{verdict.derivative_of['author']}, observed earlier "
                         f"({verdict.details['share_different']:.2%} of weights differ). Ranked with that result on the frontier, "
                         "so this PR is credited only for what it adds.")

        # ---- screen: memory ----
        mem = G.judge_memory(keys, units, self.seed_memory(), self.cfg)
        screen["memory"] = mem.to_dict()
        if mem.stop:
            return finish("memory", "memory", f"BitTrellis evaluator: **not measured**. {mem.reason}; it would run out of "
                          "memory on the 32 GB card.", screen=screen)

        # ---- screen: new quantizers, by probe bytes ----
        new_refs: list[str] = []
        if kind == "code":
            pr_probe, pr_repeat = utr / "probe.npz", utr / "probe-repeat.npz"
            if xrun(self.py + ["fingerprint", "--seed", str(self.probe_seed), "--out", str(pr_probe), "--repeat-out", str(pr_repeat)],
                   code, log, timeout=1800) != 0:
                return finish("build", "build", "BitTrellis evaluator: the quantizer probe failed to run:\n\n```\n"
                              + log.read_text()[-3000:] + "\n```", screen=screen)
            main_fp = F.by_quantizer(F.probe(None, self.probe_seed))
            mine, again = F.by_quantizer(F.load_probe(pr_probe)), F.by_quantizer(F.load_probe(pr_repeat))
            new = {ref: fp for ref, fp in mine.items() if ref not in main_fp}
            new_refs = sorted(new)
            known = {f"{ref} (main)": fp for ref, fp in main_fp.items()}
            probes = self.root / "probes"
            probes.mkdir(exist_ok=True)
            for o in earlier:
                path = probes / f"pr-{o['pr']:06d}-{o['head'][:12]}-{self.probe_seed}.npz"
                if o["pr"] != number and o["author"].lower() != me["author"].lower() and o["first_seen"] < me["first_seen"] and path.exists():
                    for ref, fp in F.by_quantizer(F.load_probe(path)).items():
                        known[f"{ref} (#{o['pr']})"] = fp
            qv = G.judge_quantizers(new, known, self.cfg, repeat={r: again.get(r) for r in new})
            screen["quantizers"] = {**qv.to_dict(), "new": new_refs}
            if qv.stop:
                return finish(qv.outcome, qv.outcome, f"BitTrellis evaluator: **not measured**. {qv.reason}.", screen=screen)
            if new:
                F.save_probe({f"{ref}|{k}": v for ref, fp in new.items() for k, v in fp.items()},
                             probes / f"pr-{number:06d}-{sha[:12]}-{self.probe_seed}.npz")
            self.obs.annotate(number, sha, quantizers=new_refs)

        # ---- build and audit (CPU) ----
        ckpt, art = (work / "sealed" / "checkpoint") if untrusted else (work / "checkpoint"), work / "artifact"
        if not (ckpt / "bittrellis_build.json").exists():
            shutil.rmtree(ckpt, ignore_errors=True)
            out = utr / "checkpoint"
            shutil.rmtree(out, ignore_errors=True)
            # the sandbox cannot write the sources' verification cache; the trusted audit verifies every source
            if xrun(self.py + ["build", str(manifest), "--out", str(out)] + (["--no-verify"] if untrusted else []) + self.env_args,
                    code, log) != 0:
                return finish("build", "build", "BitTrellis evaluator: the checkpoint did not build:\n\n```\n"
                              + log.read_text()[-3000:] + "\n```", screen=screen)
            if untrusted:
                self.sandbox.seal(out, ckpt)

        # ---- screen: the new encoder's real stored bytes ----
        if new_refs:
            sv = self._sketch_guard(me, number, sha, keys, new_refs, units, ckpt, earlier)
            screen["stored_bytes"] = sv.to_dict()
            if sv.stop:
                return finish(sv.outcome, sv.outcome, f"BitTrellis evaluator: **not measured**. {sv.reason}.", screen=screen)

        ev = self.py + ["evaluate", str(ckpt), "--out", str(art), "--sparkinfer", self.args.sparkinfer,
                        "--reference", self.args.reference, "--sample-secret-file", str(self.secret_path)] + self.env_args
        reuse = ["--audit-json", str(art / "audit.json")]
        if untrusted and not resume:
            manifest_text = subprocess.run(["git", "show", f"{sha}:{manifests[0]}"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
            self._fresh_untrusted(utr, code, sha, manifest, manifest_text, log)  # nothing the build left behind survives
            failed = self._isolated_audit(cid, keys, units, manifest, code, utr, work, ckpt, art, xrun, log)
            if failed is not None:
                return finish("audit", "audit", "BitTrellis evaluator: **audit failed**.\n\n" + "\n".join(f"- {e}" for e in failed[:20]),
                              screen=screen)
        gates = self.track["gates"]
        skipped: list[str] = []
        if not resume:
            # ---- stage 1: quality ----
            rc = run(ev + ["--stages", "quality"] + (reuse if untrusted else []), REPO_ROOT, log)
            cand = json.loads((art / "candidate.json").read_text()) if (art / "candidate.json").exists() else {}
            if cand.get("audit_ok") is False:
                audit = json.loads((art / "audit.json").read_text())
                return finish("audit", "audit", "BitTrellis evaluator: **audit failed**.\n\n" + "\n".join(f"- {e}" for e in audit["errors"][:20]),
                              screen=screen)
            if rc != 0:
                raise RuntimeError("quality stage failed")
            fails = quality_gate_failures(json.loads((art / "quality.json").read_text()), gates)
            if fails:
                self._delete(ckpt)
                return finish("gate", "gate", render_comment(cand["name"], cand["id"], None, None, "gate",
                              notes + ["Quality gates failed; speed runs, tasks and holdout skipped:", *fails], screen,
                              self._timings(art), epoch=self.epoch), screen=screen, artifact=str(art), name=cand["name"])
            # ---- stage 2: performance ----
            if run(ev + ["--stages", "performance"] + reuse, REPO_ROOT, log) != 0:
                raise RuntimeError("performance stage failed")
            frontier, row, refs = self._rank(me, art, open_prs, work)
            if status_from_row(row) not in ("frontier", "provisional"):  # the holdout has not run yet at this stage
                skipped = ["tasks", "holdout"]
                label = status_from_row(row)
                notes.append("Tasks and private holdout skipped: they can only fail a result, and this one is already "
                             f"{'dominated' if label == 'dominated' else 'invalid'} on the measured objectives.")
                return self._report(pr, cand, art, ckpt, frontier, label, notes, screen, refs, skipped, finish, work)
        # ---- stage 3: tasks and holdout ----
        cand = json.loads((art / "candidate.json").read_text())
        if run(ev + ["--stages", "tasks"] + reuse, REPO_ROOT, log) != 0:
            raise RuntimeError("tasks stage failed")
        if self.args.private:
            incumbent = Path(self.args.seeds) / self.track["frontier"]["incumbent"]
            (art / "holdout.json").unlink(missing_ok=True)
            t0 = time.time()
            run(self.py + ["holdout", "check", str(ckpt), "--private", self.args.private, "--artifact", str(art),
                           "--incumbent-artifact", str(incumbent), "--shipped", self.args.shipped, "--sparkinfer", self.args.sparkinfer],
                REPO_ROOT, log)  # always trusted code: contributed code must never see the private holdout
            if not (art / "holdout.json").exists():  # a crash must never read as "no holdout failure"
                raise RuntimeError("holdout check did not produce a verdict")
            timings = self._timings(art)
            timings["holdout_seconds"] = round(time.time() - t0, 1)
            (art / "timings.json").write_text(json.dumps(timings) + "\n")
        else:
            notes.append("No private holdout on this evaluator: the result is provisional and gets no paid tier.")
        frontier, row, refs = self._rank(me, art, open_prs, work)
        return self._report(pr, cand, art, ckpt, frontier, status_from_row(row), notes, screen, refs, skipped, finish, work)

    # ---- helpers --------------------------------------------------------------------------

    def sandbox_problems(self) -> list[str]:
        if self.args.no_sandbox:
            return []
        try:
            from sandbox import Sandbox

            self.sandbox = Sandbox(self.args.sandbox_user)
        except KeyError:
            return [f"sandbox account {self.args.sandbox_user} does not exist (run evaluator/setup_sandbox.sh)"]
        secrets_ = [self.secret_path, self.state_path, Path(self.args.token_file)] if self.args.token_file else [self.secret_path, self.state_path]
        protected = [self.accepted, self.obs.dir] + ([Path(self.args.private)] if self.args.private else [])
        return self.sandbox.problems(secrets_, protected, [Path(self.args.base), Path(self.args.shipped), Path(self.args.unsloth)])

    def _runner(self, untrusted: bool):
        if not untrusted or self.args.no_sandbox:
            return run
        python_bin = str(Path(sys.executable).parent)
        return lambda cmd, cwd, log, timeout=6 * 3600: self.sandbox.run(cmd, cwd, log, python_bin, timeout)

    def _fresh_untrusted(self, utr: Path, code: Path, sha: str, manifest: Path, manifest_text: str, log: Path) -> None:
        run(["git", "worktree", "remove", "--force", str(code)], REPO_ROOT, log)
        shutil.rmtree(utr, ignore_errors=True)
        utr.mkdir(parents=True)
        run(["git", "worktree", "add", "--force", "--detach", str(code), sha], REPO_ROOT, log)
        manifest.write_text(manifest_text)
        self.sandbox.own(utr)

    def _isolated_audit(self, cid, keys, units, manifest, code, utr, work, ckpt, art, xrun, log) -> list[str] | None:
        """Contributed code regenerates secret samples without the checkpoint; trusted code compares."""
        from bittrellis import quantizers as Q
        from bittrellis.manifest import Assignment
        from bittrellis.validate import regenerable_samples

        assignments = {uid: Assignment(k.split("@", 1)[0], k.split("@", 2)[1]) for uid, k in keys.items()}
        foreign = sorted({a.quantizer for a in assignments.values()} - set(Q.REGISTRY))
        targets = sorted({u.id for q in foreign for u in regenerable_samples(units, assignments, q, f"{cid}:{self.secret}")})
        regen, sealed = utr / "regen", work / "sealed" / "regen"
        shutil.rmtree(regen, ignore_errors=True)
        shutil.rmtree(sealed, ignore_errors=True)
        if targets:
            if xrun(self.py + ["regenerate", str(manifest), "--units", ",".join(targets), "--out", str(regen)] + self.env_args,
                    code, log) != 0:
                return ["the contributed quantizer failed to regenerate the audit samples"]
            self.sandbox.seal(regen, sealed)
        else:
            sealed.mkdir(parents=True, exist_ok=True)
        art.mkdir(parents=True, exist_ok=True)
        run(self.py + ["audit", str(ckpt), "--out", str(art / "audit.json"), "--sample-secret-file", str(self.secret_path),
                       "--regenerated", str(sealed)] + self.env_args, REPO_ROOT, log)
        doc = json.loads((art / "audit.json").read_text()) if (art / "audit.json").exists() else {"ok": False, "errors": ["audit crashed"]}
        return None if doc["ok"] else doc["errors"]

    def _sketch_guard(self, me, number, sha, keys, new_refs, units, ckpt, earlier) -> G.Verdict:
        from bittrellis import fingerprint as F

        mine_units = [u for u in units if keys[u.id].split("@", 1)[1].split("+")[0] in new_refs]
        mine_units.sort(key=lambda u: hashlib.sha256(f"{self.secret}:{u.id}".encode()).hexdigest())
        sample = mine_units[:4]
        prefixes = [lin.prefix for u in sample for lin in u.linears]
        pr_sketch = F.sketch(ckpt, prefixes, self.secret)
        known = {f"{ref} (main)": sk for ref, sk in
                 F.reference_sketches(self.sources, sample, {u.id: keys[u.id].split("@", 1)[0] for u in sample}, self.secret).items()}
        sketches = self.root / "sketches"
        sketches.mkdir(exist_ok=True)
        for o in earlier:
            path = sketches / f"pr-{o['pr']:06d}-{o['head'][:12]}.npz"
            if o["pr"] != number and o["author"].lower() != me["author"].lower() and o["first_seen"] < me["first_seen"] and path.exists():
                known[f"#{o['pr']}"] = F.load_sketches(path)
        verdict = G.judge_quantizers({"the stored bytes of " + ", ".join(new_refs): pr_sketch}, known, self.cfg)
        F.save_sketches(F.sketch(ckpt, [lin.prefix for u in mine_units for lin in u.linears][:64], self.secret),
                        sketches / f"pr-{number:06d}-{sha[:12]}.npz")
        return verdict

    def _rank(self, me: dict, art: Path, open_prs: list[dict], work: Path) -> tuple[dict, dict, list[str]]:
        refs = reference_entries(me, self.state, self.live_heads(open_prs))
        frontier_json = work / "frontier.json"
        paths = [self.args.seeds, str(self.accepted)] + [self.state[k]["artifact"] for k in refs] + [str(art)]
        if subprocess.run(self.py + ["frontier", *paths, "--out", str(frontier_json)], cwd=REPO_ROOT,
                          capture_output=True, text=True).returncode != 0:
            raise RuntimeError("frontier ranking failed")
        frontier = json.loads(frontier_json.read_text())
        cand = json.loads((art / "candidate.json").read_text())
        row = next(r for r in frontier["internal"] if r["id"] == cand["id"])
        return frontier, row, refs

    def _report(self, pr, cand, art, ckpt, frontier, label, notes, screen, refs, skipped, finish, work):
        cmp_out = subprocess.run(self.py + ["compare", str(Path(self.args.seeds) / self.track["frontier"]["incumbent"]), str(art)],
                                 cwd=REPO_ROOT, capture_output=True, text=True)
        cmp = json.loads(cmp_out.stdout) if cmp_out.returncode == 0 else None
        if refs:
            notes.append("Ranked with earlier open PRs on the frontier: " + ", ".join(f"#{self.state[k]['pr']}" for k in refs) + ".")
        pr_of = {e["name"]: e["pr"] for e in self.state.values() if isinstance(e, dict) and e.get("name") and e.get("pr")}
        if self.ledger:  # a replacement box starts with empty state; the published records still name every PR
            for path in (self.ledger.dir / "results").glob("pr-*.json"):
                rec = json.loads(path.read_text())
                if rec.get("name") and rec.get("pr"):
                    pr_of.setdefault(rec["name"], rec["pr"])
        body = render_comment(cand["name"], cand["id"], frontier, cmp, label, notes, screen, self._timings(art), pr_of=pr_of)
        self._delete(ckpt)  # a skipped result that a later re-rank lifts is rebuilt (deterministic, CPU)
        row = next((r for r in frontier["internal"] if r["id"] == cand["id"]), {})
        return finish(label, label, body, artifact=str(art), candidate=cand["id"], name=cand["name"], references=refs,
                      accepted=self._accepted_names(), skipped=skipped, screen=screen, row=row,
                      gain=row.get("frontier_gain") or 0.0)

    def _timings(self, art: Path) -> dict:
        p = art / "timings.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _delete(self, ckpt: Path) -> None:
        if not self.args.keep_checkpoints:
            shutil.rmtree(ckpt, ignore_errors=True)

    def rerank(self, open_prs: list[dict]) -> None:
        """Re-rank measured open PRs whose set of earlier references changed (CPU only)."""
        live = self.live_heads(open_prs)
        for pr in open_prs:
            key = f"{pr['number']}-{pr['head']['sha'][:12]}"
            e = self.state.get(key)
            if (not e or e.get("status") not in RANKED or not e.get("artifact") or "first_seen" not in e
                    or not (Path(e["artifact"]) / "performance.json").exists()):
                continue
            me = {"pr": e["pr"], "author": e["author"], "first_seen": e["first_seen"]}
            refs = reference_entries(me, self.state, live)
            if refs == e.get("references", []) and self._accepted_names() == e.get("accepted", []):
                continue
            work = self.root / "prs" / key
            try:
                frontier, row, refs = self._rank(me, Path(e["artifact"]), open_prs, work)
            except (RuntimeError, StopIteration, OSError):
                traceback.print_exc()
                continue
            label = status_from_row(row)
            e["references"], e["accepted"] = refs, self._accepted_names()
            tier = tier_for(label, row.get("frontier_gain"), REWARDS["tiers_fg2"])
            if label == e["status"] and tier == e.get("tier"):
                continue
            if label in ("frontier", "provisional") and e.get("skipped"):
                e["status"] = "resume"  # measured tasks and holdout never ran; the next pass rebuilds and finishes it
                self.gh.comment(pr["number"], f"{score_header('queued')}\n\nBitTrellis evaluator: an earlier PR this result was ranked against has "
                                "closed, so it is no longer dominated. Resuming: tasks and private holdout.")
                continue
            e["status"], e["tier"], e["gain"] = label, tier, row.get("frontier_gain") or 0.0
            self.gh.set_status_label(pr["number"], label)
            self.gh.set_tier_label(pr["number"], tier)
            self.gh.comment(pr["number"], f"{score_header(label, row)}\n\nBitTrellis evaluator: re-ranked after the set of "
                            f"earlier or merged results changed. Status is now **{LABELS[label][0]}**.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True, help="evaluator working directory (state, records, artifacts)")
    ap.add_argument("--token-env", default="GITHUB_TOKEN")
    ap.add_argument("--base", default=str(REPO_ROOT / "models/Qwen3.8-27B"))
    ap.add_argument("--shipped", default=str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"))
    ap.add_argument("--unsloth", default=str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-unsloth"))
    ap.add_argument("--sparkinfer", default=str(REPO_ROOT / "third_party/sparkinfer"))
    ap.add_argument("--reference", default=str(REPO_ROOT / "data/reference/hpc01-public-v2-k256"))
    ap.add_argument("--seeds", default=str(REPO_ROOT / "results/feasibility/artifacts"))
    ap.add_argument("--private", help="private holdout directory (see bittrellis/holdout.py)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--ledger", default=os.environ.get("BT_LEDGER"), help="public score-record directory (evaluator/ledger.py)")
    ap.add_argument("--ledger-remote", default=os.environ.get("BT_LEDGER_REMOTE"), help="repository the record is pushed to")
    ap.add_argument("--sandbox-user", default=os.environ.get("BT_SANDBOX_USER", "bt-sandbox"),
                    help="unprivileged account that runs contributed code (evaluator/setup_sandbox.sh)")
    ap.add_argument("--token-file", default=os.environ.get("BT_TOKEN_FILE"), help="checked to be unreadable by the sandbox")
    ap.add_argument("--no-sandbox", action="store_true", help="run contributed code as the evaluator (never on a host with secrets)")
    ap.add_argument("--auto-merge", action="store_true",
                    help="merge the top-ranked open result each pass; needs a token with contents: write")
    ap.add_argument("--merge-method", default="squash", choices=("squash", "merge", "rebase"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()

    gh = GitHub(args.repo, os.environ[args.token_env])
    gh.ensure_labels()
    ev = Evaluator(gh, args)
    while True:
        ev.run_once()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
