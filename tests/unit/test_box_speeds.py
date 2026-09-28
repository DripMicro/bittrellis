"""hpc01-e5: ranked results use speeds re-measured on the evaluator's machine."""

import json
import shutil
import sys
from pathlib import Path

from bittrellis.frontier.report import load_row

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "results/feasibility/artifacts"
sys.path.insert(0, str(ROOT / "evaluator"))
import speeds  # noqa: E402

FLOORS = {"rp_kl": 0.002, "decode_tps": 0.01, "prefill_tps": 0.03, "peak_gpu_gib": 0.10}


def test_a_ranked_result_takes_its_speed_from_this_machine(tmp_path):
    v0 = SEEDS / "V0-baseline-rebuild"
    cid = json.loads((v0 / "candidate.json").read_text())["id"]
    (tmp_path / cid).mkdir()
    here = {**json.loads((v0 / "performance.json").read_text()), "prefill_tps": 16650.0}
    (tmp_path / cid / "performance.json").write_text(json.dumps(here))
    stored = json.loads((v0 / "performance.json").read_text())["prefill_tps"]
    assert load_row(v0).prefill_tps == stored                     # stored: the machine that measured the seeds
    assert load_row(v0, tmp_path).prefill_tps == 16650.0          # this machine
    assert load_row(v0, tmp_path / "none").prefill_tps == stored


def test_external_references_keep_their_own_numbers(tmp_path):
    ext = next(d for d in SEEDS.iterdir() if json.loads((d / "candidate.json").read_text()).get("kind") == "external"
               and (d / "performance.json").exists() and (d / "quality.json").exists())
    cid = json.loads((ext / "candidate.json").read_text())["id"]
    (tmp_path / cid).mkdir()
    shutil.copy(ext / "performance.json", tmp_path / cid / "performance.json")
    (tmp_path / cid / "performance.json").write_text(json.dumps({"decode_tps": 1, "prefill_tps": 1, "peak_gpu_gib": 1}))
    assert load_row(ext, tmp_path).decode_tps == load_row(ext).decode_tps


def test_drift_is_judged_by_the_noise_floors():
    base = {"decode_tps": 95.0, "prefill_tps": 16650.0, "peak_gpu_gib": 22.0}
    assert speeds.drifted(base, {**base, "decode_tps": 95.5, "prefill_tps": 16300.0}, FLOORS) == []   # within 1% / 3%
    assert speeds.drifted(base, {**base, "prefill_tps": 14760.0}, FLOORS) == ["prefill_tps 16650 -> 14760"]
    assert speeds.drifted(base, {**base, "peak_gpu_gib": 22.3}, FLOORS) == ["peak_gpu_gib 22 -> 22.3"]


def test_only_ranked_references_with_a_recipe_are_measured():
    names = {p.name for p in speeds.references([SEEDS])}
    assert "V0-baseline-rebuild" in names and len(names) == 9     # the two external references are context only
