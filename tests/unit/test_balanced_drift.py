"""hpc01-e4: section-balanced drift, the holdout transfer rule built on it, and history carried across epochs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bittrellis import holdout
from bittrellis.eval.logits import balance_scale, balanced, mean_drift

STREAMS = {"short-general": 0.05, "short-math": 0.20, "short-code": 0.08, "short-tools": 0.30, "short-multilingual": 0.08,
           "long-8k": 0.04, "long-16k": 0.04, "long-32k": 0.20}


def positions(cut: dict[str, float] | None = None, scale: float = 1.0, n: int = 512, seed: int = 0) -> dict:
    """Synthetic per-position drift: each stream at its level × `scale`, reduced by `cut[stream]` (0.5 = halved)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, level in STREAMS.items():
        base = rng.gamma(4.0, level * scale / 4.0, size=n if s.startswith("short") else n // 4)
        out[f"{s}.kl"] = (base * (1 - (cut or {}).get(s, 0.0))).astype(np.float32)
    return out


def test_the_reference_keeps_its_drift_and_every_section_counts_the_same():
    ref = positions()
    scale = balance_scale(ref)
    assert mean_drift(balanced(ref, scale)) == pytest.approx(mean_drift(ref))
    gain = lambda cut: 1 - mean_drift(balanced(positions(cut), scale)) / mean_drift(ref)  # noqa: E731
    # halving the section where the reference drifts most counts exactly as much as halving the calmest one
    assert gain({"short-math": 0.5}) == pytest.approx(gain({"short-general": 0.5}), abs=1e-9) == pytest.approx(0.5 / 6, abs=1e-9)
    # the long streams are one section between them
    assert gain({"long-8k": 0.5, "long-16k": 0.5, "long-32k": 0.5}) == pytest.approx(0.5 / 6, abs=1e-9)


TRACK = {"frontier": {"drift": "section-balanced", "epsilon_floor": {"rp_kl": 0.002}},
         "evaluation": {"holdout": {"min_gain_ratio": 0.5}}}


def verdict(pub_cut, hold_cut, hold_scale=0.35, seed=1):
    return holdout.transfer(TRACK, positions(), positions(pub_cut), positions(scale=hold_scale, seed=seed),
                            positions(hold_cut, scale=hold_scale, seed=seed))


def test_an_easier_holdout_no_longer_fails_an_improvement_that_carries_over():
    everywhere = {s: 0.2 for s in STREAMS}
    t = verdict(everywhere, everywhere)                    # same cut on both sides, holdout drifts 3x less
    assert not t["fails"] and t["holdout_gain"] == pytest.approx(t["public_gain"], rel=0.05)


def test_a_gain_confined_to_one_public_section_that_does_not_carry_over_still_fails():
    assert verdict({"short-math": 0.5}, {})["fails"]


def test_every_credited_gain_is_checked_not_only_significant_ones():
    small = {s: 0.05 for s in STREAMS}                     # far from significant on noisy data, but above the floor
    t = verdict(small, {})
    assert t["public_gain"] > 0.002 / 0.12 and t["fails"]
    assert not verdict({}, {})["fails"]                    # no gain claimed, nothing to carry over


spec = importlib.util.spec_from_file_location("pr_bot", Path(__file__).resolve().parents[2] / "evaluator/pr_bot.py")
pr_bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_bot)


def test_a_new_epoch_keeps_first_seen_and_merged_results_but_takes_seeds_from_the_repository(tmp_path):
    ledger = tmp_path / "ledger"
    old = ledger / "hpc01-e3"
    (old / "observations").mkdir(parents=True)
    (old / "observations" / "pr-000007-aaaaaaaaaaaa.json").write_text(json.dumps({"pr": 7, "first_seen": "t0"}))
    for name in ("merged-recipe", "V1-all-q4k"):
        (old / "accepted" / name).mkdir(parents=True)
        (old / "accepted" / name / "candidate.json").write_text("{}")
    seeds = tmp_path / "seeds" / "V1-all-q4k"
    seeds.mkdir(parents=True)
    (seeds / "candidate.json").write_text("{}")
    ev = pr_bot.Evaluator.__new__(pr_bot.Evaluator)
    ev.epoch, ev.accepted = "hpc01-e4", tmp_path / "eval" / "accepted"
    ev.accepted.mkdir(parents=True)
    ev.obs = pr_bot.G.Observations(tmp_path / "eval")
    ev.args = SimpleNamespace(ledger_remote=None, seeds=str(tmp_path / "seeds"))
    ev.restore(ledger)
    assert (ev.obs.dir / "pr-000007-aaaaaaaaaaaa.json").exists()          # submission priority survives the rule change
    assert (ev.accepted / "merged-recipe").exists()                       # merged PRs stay on the table
    assert not (ev.accepted / "V1-all-q4k").exists()                      # seeds are judged afresh from the repository
