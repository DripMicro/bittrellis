"""The PR bot end to end with a fake GitHub and fake GPU stages that replay measured seed artifacts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bittrellis.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "results/feasibility/artifacts"
spec = importlib.util.spec_from_file_location("pr_bot", ROOT / "evaluator/pr_bot.py")
pr_bot = importlib.util.module_from_spec(spec)
sys.modules["pr_bot"] = pr_bot
spec.loader.exec_module(pr_bot)
EPOCH = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/hpc01.yaml").read_text())["evaluation"]["epoch"]


def manifest_yaml(name, rules):
    return json.dumps({"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": name, "default": "NVFP4", "rules": rules})


class FakeGitHub(pr_bot.GitHub):
    def __init__(self, prs):
        super().__init__("o/r", "t")
        self.prs, self.labels, self.comments, self.closed = prs, {}, {}, []

    def paged(self, path):
        if path.startswith("/pulls?state=open"):
            return self.prs
        if path.startswith("/pulls?state=closed") or path == "/labels":
            return []
        number = int(path.split("/")[2])
        return [{"filename": f"manifests/{number}.yaml"}]

    def api(self, method, path, body=None):
        if method == "PATCH" and path.startswith("/pulls/") and (body or {}).get("state") == "closed":
            number = int(path.split("/")[2])
            self.closed.append(number)
            self.prs = [p for p in self.prs if p["number"] != number]
            return None
        number = int(path.split("/")[2]) if path.startswith("/issues/") else None
        if method == "GET":
            return {"labels": [{"name": n} for n in self.labels.get(number, [])]}
        if method == "DELETE":
            self.labels[number].remove(urllib_unquote(path.rsplit("/", 1)[1]))
        elif path.endswith("/labels"):
            self.labels.setdefault(number, []).extend(body["labels"])
        elif path.endswith("/comments"):
            self.comments.setdefault(number, []).append(body["body"])
        return None


def urllib_unquote(s):
    from urllib.parse import unquote

    return unquote(s)


@pytest.fixture()
def bot(tmp_path, monkeypatch):
    stages_run: list[tuple[int, str]] = []
    recipes: dict[str, tuple[str, Path, dict]] = {}   # sha -> (manifest yaml, seed artifact to replay, perf overrides)

    holdout = {"result": "PASS"}

    def fake_run(cmd, cwd, log, timeout=0):
        if cmd[0] == "git":
            return 0
        sub = cmd[3]
        if sub == "holdout":
            art = Path(cmd[cmd.index("--artifact") + 1])
            stages_run.append((int(art.parent.name.split("-")[0]), "holdout"))
            (art / "holdout.json").write_text(json.dumps({"epoch": "test", "result": holdout["result"]}))
            return 0
        work = Path(cmd[4]).parent if sub in ("manifest", "build") else Path(cmd[cmd.index("--out") + 1]).parent
        sha = next(s for s in recipes if work.name.endswith(s[:12]))
        if sub == "manifest":
            return cli_main(cmd[3:])
        if sub == "build":
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "bittrellis_build.json").write_text("{}")
            return 0
        if sub == "evaluate":
            art = Path(cmd[cmd.index("--out") + 1])
            art.mkdir(parents=True, exist_ok=True)
            ids = json.loads((work / "ids.json").read_text())
            (art / "candidate.json").write_text(json.dumps({"id": ids["id"], "name": ids["name"], "kind": "internal", "audit_ok": True}))
            (art / "audit.json").write_text(json.dumps({"ok": True, "errors": []}))
            stage = cmd[cmd.index("--stages") + 1]
            stages_run.append((int(work.name.split("-")[0]), stage))
            _, src, perf = recipes[sha]
            files = {"quality": ["quality.json", "correctness.json", "kl_positions.npz"], "performance": ["performance.json"],
                     "tasks": ["tasks.json"]}[stage]
            for f in files:
                shutil.copy(src / f, art / f)
            if stage == "performance" and perf:
                p = json.loads((art / "performance.json").read_text())
                p.update(perf)
                (art / "performance.json").write_text(json.dumps(p))
            return 0
        raise AssertionError(cmd)

    real_run = subprocess.run

    def fake_subprocess_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "show"]:
            sha = cmd[2].split(":")[0]
            return subprocess.CompletedProcess(cmd, 0, stdout=recipes[sha][0], stderr="")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(pr_bot, "run", fake_run)
    monkeypatch.setattr(pr_bot.subprocess, "run", fake_subprocess_run)
    args = SimpleNamespace(root=str(tmp_path / "eval"), base="/nonexistent", shipped="/nonexistent", unsloth="/nonexistent",
                           sparkinfer="/nonexistent", reference="/nonexistent", seeds=str(SEEDS), private="/nonexistent-private",
                           keep_checkpoints=False, ledger=str(tmp_path / "ledger"), ledger_remote=None)
    return SimpleNamespace(args=args, recipes=recipes, stages=stages_run, holdout=holdout)


def pr(number, author, sha):
    return {"number": number, "user": {"login": author}, "head": {"sha": sha}, "labels": []}


RECIPE = [{"match": "L*.mlp", "layers": "0-55", "format": "NVFP4", "quantizer": "unsloth"},
          {"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}]


def test_frontier_duplicate_near_copy_and_staged_skip(bot):
    a, b, c, d = "a" * 40, "b" * 40, "c" * 40, "d" * 40
    faster = {"prefill_tps": 16000.0}
    bot.recipes[a] = (manifest_yaml("alice-calibrated", RECIPE), SEEDS / "V13-mlp-unsloth-bytes", faster)
    # same recipe, rules written differently
    split = [{**RECIPE[0], "layers": "0-27"}, {**RECIPE[0], "layers": "28-55"}, RECIPE[1]]
    bot.recipes[b] = (manifest_yaml("bob-same", split), SEEDS / "V13-mlp-unsloth-bytes", faster)
    # one attention projection different: a near-copy, measuring exactly like the original
    bot.recipes[c] = (json.dumps({**json.loads(manifest_yaml("carol-tweak", RECIPE)), "modules": {"L3.attn.o": "Q4_K"}}),
                      SEEDS / "V13-mlp-unsloth-bytes", faster)
    # a clearly dominated recipe
    bot.recipes[d] = (manifest_yaml("dave-q4k", [{"match": "L*.mlp", "layers": "0-62", "format": "Q4_K"}, {"match": "L*.gdn.*", "format": "Q4_K"}]),
                      SEEDS / "V9-gdn-q4k-mlp-q4k", {})
    gh = FakeGitHub([pr(4, "dave", d), pr(3, "carol", c), pr(2, "bob", b), pr(1, "alice", a)])
    ev = pr_bot.Evaluator(gh, bot.args)
    for i, p in enumerate(sorted(gh.prs, key=lambda p: p["number"])):  # observed in PR order, one second apart
        ev.obs.observe(p["number"], p["user"]["login"], p["head"]["sha"], now=f"2026-09-17T10:00:0{i}Z")
    ev.run_once()

    paid = {f"eval:{t}" for t in pr_bot.TIERS}
    assert "bt:frontier" in gh.labels[1] and len(paid & set(gh.labels[1])) == 1
    assert "bt:merge-first" not in gh.labels[1]                                # merging is off in this run
    assert set(gh.labels[2]) == {"bt:duplicate", "eval:none"}                  # not measured at all
    assert set(gh.labels[3]) == {"bt:derivative", "bt:dominated", "eval:none"}
    assert set(gh.labels[4]) == {"bt:dominated", "eval:none"}
    assert (2, "quality") not in bot.stages
    assert (4, "tasks") not in bot.stages and (3, "tasks") not in bot.stages  # dominated: tasks and holdout skipped
    assert (1, "tasks") in bot.stages
    assert any("Ranked with earlier open PRs on the frontier: #1" in c for c in gh.comments[3])
    assert sorted(gh.closed) == [2, 3, 4] and gh.comments[3][-1] == pr_bot.CLOSE_COMMENT   # none of them can be merged
    assert "Claude" not in json.dumps(gh.comments)

    # the public score record: one write-once record per evaluated head, plus the frontier and README
    ledger = Path(bot.args.ledger) / EPOCH
    records = {json.loads(p.read_text())["pr"]: json.loads(p.read_text()) for p in (ledger / "results").glob("*.json")}
    assert set(records) == {1, 2, 3, 4}
    assert records[1]["tier"] in {f"{t}" for t in pr_bot.TIERS} and records[1]["row"]["rp_kl"] > 0
    assert records[2]["status"] == "duplicate" and records[2]["screen"]["manifest"]["original"]["pr"] == 1
    assert (ledger / "observations" / "pr-000001-aaaaaaaaaaaa.json").exists()
    readme = (Path(bot.args.ledger) / "README.md").read_text()
    assert "V0-baseline-rebuild" in readme and f"{EPOCH}/accepted" in readme
    assert "holdout" not in json.dumps(records[1]).lower() or records[1]["row"]["holdout"] in ("PASS", "FAIL", None)

    # alice closes #1 unmerged and carol reopens #3: it is re-ranked, becomes non-dominated, and resumes for tasks
    gh.prs = [p for p in gh.prs if p["number"] != 1] + [pr(3, "carol", c)]
    ev.run_once()
    assert ev.state[f"3-{c[:12]}"]["status"] == "frontier"
    assert (3, "tasks") in bot.stages
    assert "bt:frontier" in gh.labels[3] and len(paid & set(gh.labels[3])) == 1
    assert "bt:merge-first" not in gh.labels[3]                                # merging is off in this run
    assert "eval:none" not in gh.labels[3]


class MergingGitHub(FakeGitHub):
    """FakeGitHub that can also report mergeability and record merges."""

    def __init__(self, prs, mergeable_state="clean"):
        super().__init__(prs)
        self.merged, self.mergeable_state = [], mergeable_state

    def pull(self, number):
        pr = next(p for p in self.prs if p["number"] == number)
        return {"head": pr["head"], "draft": False, "merged": False, "title": f"candidate #{number}",
                "labels": [{"name": n} for n in self.labels.get(number, [])],
                "mergeable": True, "mergeable_state": self.mergeable_state}

    def merge(self, number, sha, method, title, message):
        self.merged.append((number, sha, method))
        return True, f"merge-commit-{number}"


def _one_frontier_pr(bot):
    sha = "a" * 40
    bot.recipes[sha] = (manifest_yaml("alice-calibrated", RECIPE), SEEDS / "V13-mlp-unsloth-bytes",
                        {"prefill_tps": 16000.0})
    bot.args.auto_merge, bot.args.merge_method = True, "squash"
    return sha


def test_the_top_result_is_merged_at_the_sha_it_was_measured_at(bot):
    sha = _one_frontier_pr(bot)
    gh = MergingGitHub([pr(1, "alice", sha)])
    ev = pr_bot.Evaluator(gh, bot.args)
    ev.run_once()

    assert ev.state[f"1-{sha[:12]}"]["status"] == "frontier"
    assert gh.merged == [(1, sha, "squash")]                      # exactly the evaluated head
    assert (Path(bot.args.root) / "accepted" / "alice-calibrated").exists()   # ranks later PRs at once
    assert "Merged automatically" in gh.comments[1][-1]


def test_a_failing_check_stops_the_merge_and_it_is_retried_later(bot):
    sha = _one_frontier_pr(bot)
    gh = MergingGitHub([pr(1, "alice", sha)], mergeable_state="unstable")
    ev = pr_bot.Evaluator(gh, bot.args)
    ev.run_once()

    assert gh.merged == []                                        # CI red: nothing merged
    assert pr_bot.EXTRA_LABELS["merge-first"][0] in gh.labels[1]  # still first in line
    gh.mergeable_state = "clean"                                  # the check goes green
    ev.run_once()
    assert gh.merged == [(1, sha, "squash")]


def test_nothing_is_merged_unless_auto_merge_is_on(bot):
    sha = _one_frontier_pr(bot)
    bot.args.auto_merge = False
    gh = MergingGitHub([pr(1, "alice", sha)])
    pr_bot.Evaluator(gh, bot.args).run_once()
    assert gh.merged == []


def test_evaluator_root_is_created_private(bot, tmp_path):
    ev = pr_bot.Evaluator(FakeGitHub([]), bot.args)
    ev.save()
    mode = lambda p: oct(Path(p).stat().st_mode & 0o777)  # noqa: E731
    root = Path(bot.args.root)
    assert mode(root) == "0o711" and mode(root / "prs") == "0o711"      # traversable to each PR's untrusted tree
    assert mode(root / "accepted") == "0o700" and mode(ev.obs.dir) == "0o700"
    assert mode(root / "state.json") == "0o600" and mode(root / "secret.txt") == "0o600"
    assert mode(bot.args.ledger) == "0o700"


def test_a_replacement_box_recovers_priority_and_accepted_results(bot, tmp_path):
    """Return the GPU box and everything that decides what a contributor is owed must come back."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    sha = _one_frontier_pr(bot)
    bot.args.ledger_remote = str(remote)

    gh = MergingGitHub([pr(1, "alice", sha)])
    ev = pr_bot.Evaluator(gh, bot.args)
    ev.obs.observe(1, "alice", sha, now="2026-09-17T10:00:00Z")
    ev.run_once()

    first_seen = ev.state[f"1-{sha[:12]}"]["first_seen"]
    tier = ev.state[f"1-{sha[:12]}"]["tier"]
    epoch = ev.epoch
    published = Path(bot.args.ledger) / epoch
    assert list((published / "observations").glob("pr-*.json")), "first-seen records must be published"
    assert list((published / "results").glob("pr-*.json")), "per-PR result records must be published"
    assert (published / "accepted" / "alice-calibrated").is_dir()

    # the box is returned: a different machine, nothing local, same published record
    bot.args.root = str(tmp_path / "eval-on-the-new-box")
    bot.args.ledger = str(tmp_path / "ledger-on-the-new-box")
    fresh = pr_bot.Evaluator(MergingGitHub([pr(1, "alice", sha)]), bot.args)

    # priority: the same submission keeps the moment it was first seen, so it cannot lose its place
    assert fresh.obs.observe(1, "alice", sha)["first_seen"] == first_seen
    # references: the merged result is back on the table, so a later PR earns only what it adds
    assert (Path(bot.args.root) / "accepted" / "alice-calibrated").is_dir()
    assert fresh._accepted_names() == ["alice-calibrated"]
    # score: the published record is intact and still says what it earned
    record = json.loads(next((Path(bot.args.ledger) / epoch / "results").glob("pr-000001-*.json")).read_text())
    assert record["tier"] == tier and record["first_seen"] == first_seen


def test_a_holdout_failure_is_known_before_the_task_suite_runs(bot):
    sha = _one_frontier_pr(bot)
    bot.holdout["result"] = "FAIL"
    gh = MergingGitHub([pr(1, "alice", sha)])
    pr_bot.Evaluator(gh, bot.args).run_once()
    assert (1, "holdout") in bot.stages and (1, "tasks") not in bot.stages     # 784 questions not spent on a reject
    assert "eval:REJECT" in gh.labels[1] and gh.merged == []
