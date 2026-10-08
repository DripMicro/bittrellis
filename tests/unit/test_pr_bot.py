import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pr_bot", Path(__file__).resolve().parents[2] / "evaluator/pr_bot.py")
pr_bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_bot)


def test_classify_manifest_code_and_evaluator_prs():
    assert pr_bot.classify(["manifests/gdn-deep.yaml"]) == ("manifest", ["manifests/gdn-deep.yaml"])
    assert pr_bot.classify(["manifests/a.yaml", "manifests/b.yaml"])[0] == "other"
    assert pr_bot.classify(["bittrellis/quantizers/gptq.py", "manifests/a.yaml", "docs/quantizers.md"])[0] == "code"
    assert pr_bot.classify(["manifests/a.yaml", "configs/hpc01.yaml"])[0] == "evaluator"
    assert pr_bot.classify(["bittrellis/eval/logits.py"])[0] == "evaluator"
    assert pr_bot.classify(["README.md"])[0] == "other"


GATES = {"rp_kl_max": 0.30, "top1_min": 0.80, "long_context_guard": {"required_success": {"long-8k": 1.0, "long-16k": 1.0}}}


def test_queue_runs_in_first_seen_order_not_pr_number():
    prs = [{"number": 5, "head": {"sha": "a"}}, {"number": 9, "head": {"sha": "b"}}]
    seen = {(5, "a"): "2026-09-17T12:00:00Z", (9, "b"): "2026-09-17T11:00:00Z"}  # #5 force-pushed later
    assert [p["number"] for p in pr_bot.queue_order(prs, seen)] == [9, 5]


def test_quality_gates_stop_before_speed_runs():
    ok = {"rp_kl": 0.12, "top1": 0.91, "nonfinite_logprobs": 0,
          "needles_by_length": {"long-8k": {"required": 3, "retrieved": 3}, "long-16k": {"required": 3, "retrieved": 3}}}
    assert pr_bot.quality_gate_failures(ok, GATES) == []
    bad = {**ok, "rp_kl": 0.4, "needles_by_length": {**ok["needles_by_length"], "long-16k": {"required": 3, "retrieved": 2}}}
    fails = pr_bot.quality_gate_failures(bad, GATES)
    assert len(fails) == 2 and "long-16k 2/3" in fails[1]


def test_references_are_earlier_live_prs_by_other_authors():
    def entry(pr, author, t, status="frontier", head="h"):
        return {"pr": pr, "author": author, "first_seen": t, "status": status, "artifact": f"/a/{pr}", "head": head}

    state = {"_merged": {}, "1-h": entry(1, "alice", "t1"), "2-h": entry(2, "bob", "t2"), "3-h": entry(3, "carol", "t4"),
             "4-h": entry(4, "dave", "t0", status="duplicate"), "5-h": entry(5, "erin", "t1", head="old")}
    me = {"pr": 6, "author": "bob", "first_seen": "t3"}
    live = {1: "h", 2: "h", 3: "h", 4: "h", 5: "new"}
    assert pr_bot.reference_entries(me, state, live) == ["1-h"]  # not own, not later, not unmeasured, not a stale head
    assert pr_bot.reference_entries(me, state, {2: "h", 3: "h"}) == []  # #1 closed unmerged
    state["_merged"]["1"] = "h"
    assert pr_bot.reference_entries(me, state, live) == []        # merged: ranked from accepted/, not listed twice


def test_status_and_comment():
    row = {"name": "x", "valid": True, "frontier": True, "frontier_gain": 0.004, "rp_kl": 0.12, "decode_tps": 94.0,
           "prefill_tps": 14000.0, "peak_gpu_gib": 22.0, "holdout": "PASS", "gate_failures": []}
    assert pr_bot.status_from_row(row) == "frontier"
    assert pr_bot.status_from_row({**row, "frontier_gain": 0.0}) == "dominated"
    assert pr_bot.status_from_row({**row, "valid": False}) == "gate"
    frontier = {"evaluator_epoch": "hpc01-e2", "incumbent": "V0", "internal": [row]}
    body = pr_bot.render_comment("x", "id", frontier, None, "frontier", ["note"], {"manifest": {"outcome": "pass"}},
                                 {"quality_seconds": 360.0})
    assert "bt:frontier" in body and "quality 6.0 min" in body and "Claude" not in body


def test_comment_names_what_dominates():
    row = {"name": "x", "valid": True, "frontier": False, "frontier_gain": 0.0, "rp_kl": 0.12, "decode_tps": 93.0,
           "prefill_tps": 13700.0, "peak_gpu_gib": 22.2, "holdout": None, "gate_failures": [], "dominated_by": ["V13-mlp-unsloth-bytes"]}
    body = pr_bot.render_comment("x", "id", {"evaluator_epoch": "e", "incumbent": "V0", "internal": [row]}, None, "dominated", [])
    assert "Dominated by `V13-mlp-unsloth-bytes`" in body


def test_label_colours_follow_meaning():
    color = {k: v[1] for k, v in pr_bot.LABELS.items()}
    assert color["frontier"] == pr_bot.EXTRA_LABELS["approved"][1]                      # green: go / credited
    assert color["audit"] == color["same-encoder"]                                       # integrity failures share dark red
    assert color["gate"] == color["nondeterministic"] and color["invalid"] == color["build"]
    assert len({color["frontier"], color["dominated"], color["gate"], color["audit"], color["error"], color["queued"]}) == 6
    assert all(len(c) == 6 and c == c.lower() for _, c, _ in [*pr_bot.LABELS.values(), *pr_bot.EXTRA_LABELS.values()])


def test_every_comment_leads_with_the_score():
    row = {"name": "x", "valid": True, "frontier": True, "frontier_gain": 0.00435, "rp_kl": 0.12, "decode_tps": 94.0,
           "prefill_tps": 14000.0, "peak_gpu_gib": 22.0, "holdout": "PASS", "gate_failures": []}
    body = pr_bot.render_comment("x", "id", {"evaluator_epoch": "e", "incumbent": "V0", "internal": [row]}, None, "frontier", [])
    assert body.splitlines()[2] == "**Score: `eval:L` · ×2.5 on Gittensor when merged** · FG-2 +0.435%"
    assert pr_bot.score_header("dominated", {**row, "frontier_gain": 0.0}) == "**Score: `eval:none` · ×0** · no new frontier space"
    assert pr_bot.score_header("duplicate").startswith("**Score: `eval:none` · ×0**")
    assert pr_bot.score_header("audit").startswith("**Score: `eval:REJECT` · ×0**")
    assert pr_bot.score_header("queued") == "**Score: pending** · not evaluated yet"
    for key in pr_bot.LABELS:
        assert pr_bot.score_header(key, row).startswith("**Score:")


def test_tiers_are_calibrated_to_the_seed_gains():
    t = pr_bot.REWARDS["tiers_fg2"]
    seeds = {"V13": 0.003013, "V0": 0.001934, "V4": 0.00052, "V1": 0.000397, "V6": 0.000238, "V5": 0.000073, "noise": 0.00004}
    assert {k: pr_bot.tier_for("frontier", g, t) for k, g in seeds.items()} == \
        {"V13": "L", "V0": "M", "V4": "S", "V1": "S", "V6": "XS", "V5": "XS", "noise": "none"}
    assert pr_bot.tier_for("frontier", 0.0, t) == "none"
    assert pr_bot.tier_for("dominated", 0.004, t) == "none"
    assert pr_bot.tier_for("gate", None, t) == pr_bot.tier_for("same-encoder", None, t) == "REJECT"
    assert pr_bot.tier_for("queued", None, t) is None and pr_bot.tier_for("needs_approval", None, t) is None
    assert set(pr_bot.REWARDS["multipliers"]) == {*pr_bot.TIERS, "none", "REJECT"}


def test_no_holdout_pass_no_paid_tier():
    row = {"valid": True, "frontier": True, "frontier_gain": 0.003}
    assert pr_bot.status_from_row({**row, "holdout": "PASS"}) == "frontier"
    assert pr_bot.status_from_row({**row, "holdout": None}) == "provisional"
    assert pr_bot.tier_for("provisional", 0.003, pr_bot.REWARDS["tiers_fg2"]) is None
    assert "no paid tier" in pr_bot.score_header("provisional")


def test_merge_first_prefers_tier_then_gain_then_first_seen():
    c = [{"pr": 1, "tier": "M", "gain": 0.002, "first_seen": "t1"}, {"pr": 2, "tier": "L", "gain": 0.003, "first_seen": "t3"},
         {"pr": 3, "tier": "L", "gain": 0.004, "first_seen": "t4"}, {"pr": 4, "tier": "L", "gain": 0.004, "first_seen": "t2"},
         {"pr": 5, "tier": "none", "gain": 0.0, "first_seen": "t0"}]
    assert pr_bot.pick_merge_first(c)["pr"] == 4
    assert pr_bot.pick_merge_first([c[-1]]) is None


def test_every_head_that_cannot_earn_is_closed_and_paid_or_pending_ones_stay_open():
    prs = [{"number": n, "head": {"sha": f"{n}" * 40}} for n in (1, 2, 3, 4, 5, 6, 7, 8)]
    state = {f"{n}-{str(n) * 12}": e for n, e in (
        (1, {"status": "gate", "tier": "REJECT"}), (2, {"status": "duplicate", "tier": "none"}),
        (3, {"status": "dominated", "tier": "none"}), (4, {"status": "frontier", "tier": "none"}),   # below XS
        (5, {"status": "frontier", "tier": "S"}), (6, {"status": "queued"}), (7, {"status": "provisional"}))}
    state["8-" + "f" * 12] = {"status": "gate"}   # an older head of #8: its new head is still unmeasured
    assert pr_bot.to_close(prs, state) == [1, 2, 3, 4]


def test_comment_shows_the_nearest_frontier_results_and_bolds_only_clear_wins():
    box = {"rp_kl": [0.0, 0.3], "decode_tps": [60.0, 120.0], "prefill_tps": [2000.0, 20000.0], "peak_gpu_gib": [14.0, 32.0]}

    def row(name, kl, pre, mem, frontier=True):
        return {"name": name, "rp_kl": kl, "decode_tps": 96.0, "prefill_tps": pre, "peak_gpu_gib": mem, "tasks_passed": 570,
                "tasks_n": 784, "holdout": "PASS", "valid": True, "frontier": frontier, "frontier_gain": 0.0003,
                "gate_failures": [], "dominated_by": []}
    frontier = {"evaluator_epoch": "e", "incumbent": "V0", "box": box,
                "internal": [row("mine", 0.120, 12000, 20.5), row("V0", 0.136, 14760, 22.0), row("near", 0.125, 11000, 21.0),
                             row("far", 0.200, 3000, 30.0), row("gone", 0.121, 12001, 20.5, frontier=False)]}
    body = pr_bot.render_comment("mine", "cid", frontier, None, "frontier", [], pr_of={"near": 5})
    assert "`near` (#5)" in body and "`far`" in body and "`gone`" not in body   # off-frontier rows are never shown
    assert "**0.1200**" in body and "**20.50**" in body and "**12,000**" not in body   # V0 is faster at prefill
    assert "Epoch `e2`" in pr_bot.render_comment("x", "cid", None, None, "gate", ["failed"], epoch="e2")


def test_merge_first_is_not_promised_while_merging_is_off():
    from types import SimpleNamespace

    class GH:
        def __init__(self):
            self.added, self.removed = [], []

        def add_label(self, n, name):
            self.added.append(n)

        def remove_label(self, n, name):
            self.removed.append(n)

    prs = [{"number": 7, "head": {"sha": "a" * 40}, "labels": [{"name": "bt:merge-first"}]},
           {"number": 8, "head": {"sha": "b" * 40}, "labels": []}]
    state = {"7-" + "a" * 12: {"status": "frontier", "tier": "S", "gain": 0.001, "first_seen": "t1"},
             "8-" + "b" * 12: {"status": "frontier", "tier": "XS", "gain": 0.0002, "first_seen": "t2"}}
    for merging, added, removed in ((True, [], []), (False, [], [7])):
        ev = pr_bot.Evaluator.__new__(pr_bot.Evaluator)
        ev.gh, ev.state, ev.args = GH(), state, SimpleNamespace(auto_merge=merging)
        assert ev.mark_merge_first(prs) == 7
        assert (ev.gh.added, ev.gh.removed) == (added, removed)


def test_the_closing_comment_is_plain():
    assert "earn" not in pr_bot.CLOSE_COMMENT.lower() and ":" not in pr_bot.CLOSE_COMMENT


def test_error_summary_shows_only_the_failing_steps_error():
    build = ("\n$ python -m bittrellis.cli manifest m.yaml\n✓ m.yaml: gdn  id=d2cb\n"
             "\n$ python -m bittrellis.cli build m.yaml --out ckpt\nTraceback (most recent call last):\n"
             '  File "build.py", line 168, in build\n    writer.add(t.name)\n'
             "OSError: [Errno 28] No space left on device\n[build] gdn (d2cb): 2103 tensors -> ckpt\n[build]   250/2103 tensors, 16s\n")
    assert pr_bot.error_summary(build) == "OSError: [Errno 28] No space left on device"
    invalid = "\n$ [sandbox:bt-sandbox] python -m bittrellis.cli manifest m.yaml\n✗ m.yaml: unknown quantizer 'x'\n"
    assert pr_bot.error_summary(invalid) == "✗ m.yaml: unknown quantizer 'x'"
    assert pr_bot.error_summary("\n$ python -m x\n") == "(no output)"


def test_a_pr_left_for_a_maintainer_is_told_the_real_reason():
    two = ["manifests/a.yaml", "manifests/b.yaml"]
    assert pr_bot.classify(two)[0] == "other"
    assert "adds 2 manifests (`manifests/a.yaml`, `manifests/b.yaml`)" in pr_bot.not_evaluated_reason("other", two, two)
    assert "protected paths" in pr_bot.not_evaluated_reason("evaluator", ["configs/hpc01.yaml"], [])
    assert "adds no manifest" in pr_bot.not_evaluated_reason("other", ["README.md"], [])
    files = ["manifests/a.yaml", "README.md", "scripts/x.sh"]
    assert "(`README.md`, `scripts/x.sh`)" in pr_bot.not_evaluated_reason("other", files, ["manifests/a.yaml"])


def test_each_pr_belongs_to_exactly_one_track():
    hpc01 = ["manifests/my-recipe.yaml"]
    hpc02 = ["manifests/hpc02/my-recipe.yaml"]
    enc02 = ["bittrellis/hpc02_encoders/my_kq.py", "tests/unit/test_my_kq.py", "manifests/hpc02/uses-it.yaml"]
    assert [pr_bot.track_of(f) for f in (hpc01, hpc02, enc02)] == ["HPC-01", "HPC-02", "HPC-02"]
    assert pr_bot.classify(hpc01, "HPC-01") == ("manifest", hpc01)
    assert pr_bot.classify(hpc02, "HPC-02") == ("manifest", hpc02)
    assert pr_bot.classify(enc02, "HPC-02") == ("code", ["manifests/hpc02/uses-it.yaml"])
    # HPC-01's evaluator never takes an HPC-02 recipe for one of its own manifests
    assert pr_bot.classify(hpc02, "HPC-01")[1] == []
    # the GGUF pipeline is evaluator code: a PR changing it is left for a maintainer
    assert pr_bot.classify(["bittrellis/hpc02.py", "manifests/hpc02/x.yaml"], "HPC-02")[0] == "evaluator"
