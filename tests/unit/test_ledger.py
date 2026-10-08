import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"evaluator/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("progress_chart")  # imported by ledger
L, P = _load("ledger"), _load("publish_ledger")
FRONTIER = {"internal": [{"name": "V0", "rp_kl": 0.1357, "decode_tps": 94.9, "prefill_tps": 14760.0,
                          "peak_gpu_gib": 22.02, "tasks_passed": 570, "tasks_n": 784, "frontier": True,
                          "frontier_gain": 0.00193}]}


def test_records_are_write_once_and_carry_no_private_data(tmp_path):
    led = L.Ledger(tmp_path, "hpc01-e3")
    entry = {"pr": 7, "head": "a" * 40, "author": "alice", "first_seen": "t0", "kind": "manifest",
             "status": "frontier", "tier": "L", "candidate": "cid", "name": "mine", "gain": 0.003,
             "artifact": "/workspace/bt-eval/prs/7/artifact"}
    path = led.record(entry, {"rp_kl": 0.12, "holdout": "PASS"})
    doc = json.loads(path.read_text())
    assert doc["tier"] == "L" and doc["row"]["holdout"] == "PASS"
    assert "artifact" not in doc and "/workspace" not in path.read_text()   # no evaluator paths
    before = path.read_text()
    led.record(entry, {"rp_kl": 0.12, "holdout": "PASS"})
    assert path.read_text() == before


def test_frontier_readme_lists_results_and_how_to_check(tmp_path):
    led = L.Ledger(tmp_path, "hpc01-e3")
    led.frontier(FRONTIER)
    readme = (tmp_path / "README.md").read_text()
    assert "570/784" in readme and "0.193%" in readme and "bittrellis frontier hpc01-e3/accepted" in readme
    assert json.loads((tmp_path / "hpc01-e3/frontier.json").read_text()) == FRONTIER


def test_publish_pushes_without_force_and_hides_the_token(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    led = L.Ledger(tmp_path / "ledger", "hpc01-e3")
    led.frontier(FRONTIER)
    seen = []
    real = subprocess.run

    def spy(args, **kw):
        seen.append(args)
        return real(args, **kw)

    monkeypatch.setattr(P.subprocess, "run", spy)
    commit = P.publish(tmp_path / "ledger", str(remote), "ghp_secret", "records: test")
    assert commit and P.publish(tmp_path / "ledger", str(remote), "ghp_secret", "records: test") is None
    assert not any("--force" in a for a in seen for a in a)
    assert not any("ghp_secret" in " ".join(a) for a in seen)                      # never on a command line
    assert "ghp_secret" not in (tmp_path / "ledger/.git/config").read_text()       # nor in a file
    out = real(["git", "--no-pager", "log", "--oneline", "-1", "--name-only", "main"], cwd=remote,
               capture_output=True, text=True).stdout
    assert "hpc01-e3/frontier.json" in out and "records: test" in out


def test_publish_refuses_a_remote_with_credentials(tmp_path):
    with pytest.raises(P.PublishError):
        P.publish(tmp_path, "https://user:token@github.com/o/r.git", None, "m")


def _bare(path):
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(path)], check=True)
    return str(path)


def test_a_failed_push_is_retried_on_a_pass_that_changes_nothing(tmp_path):
    """One network failure must not strand the record: the next pass pushes it anyway."""
    remote = _bare(tmp_path / "remote.git")
    ledger = tmp_path / "ledger"
    led = L.Ledger(ledger, "hpc01-e3")
    led.frontier(FRONTIER)

    with pytest.raises(P.PublishError):                     # the remote is unreachable this pass
        P.publish(ledger, str(tmp_path / "gone.git"), None, "records: pass 1")
    assert subprocess.run(["git", "log", "--oneline", "main"], cwd=remote, capture_output=True,
                          text=True).stdout.strip() == ""   # nothing published yet

    commit = P.publish(ledger, remote, None, "records: pass 2")   # nothing new was written
    assert commit, "an unpushed commit must go out even when the pass wrote nothing"
    assert "hpc01-e3/frontier.json" in subprocess.run(
        ["git", "--no-pager", "log", "-1", "--name-only", "main"], cwd=remote, capture_output=True,
        text=True).stdout
    assert P.publish(ledger, remote, None, "records: pass 3") is None   # now it is up to date


def test_a_replacement_box_continues_the_published_history(tmp_path):
    """A fresh box must extend the record, not fork a second one that can never be pushed."""
    remote = _bare(tmp_path / "remote.git")
    first = tmp_path / "box1"
    L.Ledger(first, "hpc01-e3").frontier(FRONTIER)
    P.publish(first, remote, None, "records: box 1")

    fresh = tmp_path / "box2"                                # the box was returned; nothing local
    assert P.adopt(fresh, remote, None) is True
    assert (fresh / "hpc01-e3/frontier.json").exists()        # the record came back
    L.Ledger(fresh, "hpc01-e3").record({"pr": 4, "head": "b" * 40, "author": "bob", "status": "frontier"}, None)
    assert P.publish(fresh, remote, None, "records: box 2")   # accepted: same history, fast-forward

    log = subprocess.run(["git", "log", "--oneline", "main"], cwd=remote, capture_output=True, text=True).stdout
    assert "records: box 1" in log and "records: box 2" in log


def test_adopt_keeps_records_already_written_on_this_box(tmp_path):
    remote = _bare(tmp_path / "remote.git")
    first = tmp_path / "box1"
    L.Ledger(first, "hpc01-e3").frontier(FRONTIER)
    P.publish(first, remote, None, "records: box 1")

    fresh = tmp_path / "box2"                                 # a pass wrote before the history arrived
    L.Ledger(fresh, "hpc01-e3").record({"pr": 9, "head": "c" * 40, "author": "carol", "status": "frontier"}, None)
    P.adopt(fresh, remote, None)
    assert (fresh / "hpc01-e3/results/pr-000009-cccccccccccc.json").exists()   # local record survived
    assert (fresh / "hpc01-e3/frontier.json").exists()                          # published record arrived
    assert P.publish(fresh, remote, None, "records: box 2")


def test_adopt_on_an_empty_remote_starts_a_history(tmp_path):
    remote = _bare(tmp_path / "remote.git")
    fresh = tmp_path / "box"
    assert P.adopt(fresh, remote, None) is False              # nothing to fetch, but usable
    L.Ledger(fresh, "hpc01-e3").frontier(FRONTIER)
    assert P.publish(fresh, remote, None, "records: first")


def test_a_re_measurement_never_rewrites_what_was_published(tmp_path):
    """A replacement box re-measures open PRs. Speed is measured, so the row differs -- and the
    original verdict must still be there, or a re-run could restate what a contributor earned."""
    led = L.Ledger(tmp_path, "hpc01-e3")
    entry = {"pr": 5, "head": "d" * 40, "author": "dave", "first_seen": "t0", "status": "frontier", "tier": "M"}
    first = led.record(entry, {"decode_tps": 96.0, "rp_kl": 0.1318})

    assert led.record(entry, {"decode_tps": 96.0, "rp_kl": 0.1318}) == first      # unchanged: no revision
    again = led.record(entry, {"decode_tps": 95.7, "rp_kl": 0.1318})              # re-measured on a new box
    assert again != first
    assert json.loads(first.read_text())["row"]["decode_tps"] == 96.0             # the original stands
    assert json.loads(again.read_text())["supersedes"] == first.name
    assert again.name.endswith(".remeasured-1.json")

    third = led.record({**entry, "tier": "S"}, {"decode_tps": 95.1, "rp_kl": 0.1319})
    assert third.name.endswith(".remeasured-2.json")
    assert json.loads(first.read_text())["tier"] == "M"                           # still untouched


def _result(led, pr, author, first_seen, status, tier, gain, name):
    led.record({"pr": pr, "head": f"{pr}" * 40, "author": author, "first_seen": first_seen, "status": status,
                "tier": tier, "gain": gain, "name": name}, None)


def test_progress_chart_climbs_only_at_merges(tmp_path):
    led = L.Ledger(tmp_path, "hpc01-e3")
    _result(led, 4, "maint", "2026-09-24T00:00:00Z", "frontier", "M", 0.0012, "four")
    _result(led, 6, "alice", "2026-09-26T00:00:01Z", "gate", "REJECT", 0.0, "six")
    _result(led, 7, "alice", "2026-09-26T00:00:02Z", "frontier", "XS", 0.0001, "seven")   # scored, still open
    _result(led, 8, "bob", "2026-09-26T00:00:03Z", "queued", None, None, "eight")          # not measured yet
    (led.dir / "accepted" / "four").mkdir()
    records = C.load(led.dir)
    assert [(r["pr"], r["merged"]) for r in records] == [(4, True), (6, False), (7, False)]
    assert [C.outcome(r) for r in records] == ["credited", "rejected", "credited"]
    led.frontier(FRONTIER)
    svg = (tmp_path / "progress.svg").read_text()
    assert "+0.120%" in svg and ">#4 · M<" in svg and ">maint<" in svg and ">alice<" not in svg   # alice has no merge
    assert "progress.svg" in (tmp_path / "README.md").read_text()
    assert led.frontier(FRONTIER) is None and (tmp_path / "progress.svg").read_text() == svg      # same records, same bytes


def test_progress_chart_stays_readable_with_many_pull_requests():
    recs = [{"pr": n, "author": f"a{n % 12}", "first_seen": f"2026-{9 + n // 150:02d}-{1 + n % 28:02d}T00:00:00Z",
             "status": "frontier", "tier": "XS", "gain": 0.0001, "merged": True} for n in range(300)]
    svg = C.render(sorted(recs, key=lambda r: r["first_seen"]), "e")
    assert ">+3.00%<" in svg and ">300<" in svg and "5 more authors" in svg   # 12 authors: 7 shown, 5 folded
    assert svg.count("<circle") == 3                                             # past 24 merges only the labelled steps
    assert "per day" in svg


def test_which_checkpoint_offers_only_standing_holdout_passes_one_row_per_recipe():
    def row(name, kl, pre, mem, holdout="PASS", frontier=True):
        return {"name": name, "rp_kl": kl, "decode_tps": 95.0, "prefill_tps": pre, "peak_gpu_gib": mem,
                "holdout": holdout, "valid": True, "frontier": frontier}
    doc = {"incumbent": "V0", "internal": [row("V0", 0.136, 14760, 22.0, None), row("lean", 0.124, 9000, 20.8),
                                           row("quick", 0.129, 14300, 21.8), row("failed", 0.110, 15000, 20.0, "FAIL"),
                                           row("passed-by", 0.100, 16000, 19.0, frontier=False)]}
    merged = [{"name": n, "pr": i, "author": "a", "tier": "XS"} for i, n in enumerate(["lean", "quick", "failed", "passed-by"])]
    text = "\n".join(L.recommend(doc, merged))
    assert "**Closest to the original model · Least GPU memory**" in text and "**Fastest prompt reading**" in text
    assert "`failed`" not in text and "`passed-by`" not in text        # holdout FAIL, or no longer on the frontier
    assert "**0.1240 · 8.8% closer**" in text and "9,000 tok/s · 39% slower" in text   # bold only what it was picked for
    assert "bittrellis build manifests/lean.yaml" in text and "tradeoffs.svg" in text
    assert L.recommend({"incumbent": "V0", "internal": []}, merged) == []


def test_two_evaluators_publish_to_one_record(tmp_path, monkeypatch):
    """One evaluator per track, each with its own clone, push records to the same remote: neither is rejected."""
    import subprocess

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "hpc01-e6").mkdir(parents=True)
    (a / "hpc01-e6" / "r1.json").write_text("{}")
    assert P.publish(a, str(remote), None, "records: a")
    P.adopt(b, str(remote), None)
    (a / "hpc01-e6" / "r2.json").write_text("{}")
    assert P.publish(a, str(remote), None, "records: a2")                 # a moves the remote on
    (b / "hpc02-e1").mkdir()
    (b / "hpc02-e1" / "s1.json").write_text("{}")
    assert P.publish(b, str(remote), None, "records: b")                  # b is behind, but still publishes
    log = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"], capture_output=True, text=True).stdout.split()
    assert sorted(log) == ["hpc01-e6/r1.json", "hpc01-e6/r2.json", "hpc02-e1/s1.json"]
