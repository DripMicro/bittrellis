"""Evaluate one checkpoint end to end and write its artifact directory.

    artifacts/<candidate>/
      candidate.json      what was evaluated: manifest, expanded assignments, executed formats, lineage
      audit.json          HPC-01 rule checks (candidates only)
      quality.json        RP-KL / top-1 / NLL vs BF16 on the public corpus, per category, needles
      correctness.json    runtime-correctness checks (see RUNTIME_CORRECTNESS)
      holdout.json        private holdout PASS/FAIL only (validators)
      tasks.json          SparkInfer quality-suite guard
      performance.json    two independent runs: decode + prefill per context, peak GPU, host RAM
      environment.json    GPU, driver, CUDA, SparkInfer commit, effective SPARKINFER_* environment
      kl_positions.npz    per-position RP-KL for paired comparisons
      scores/*.npz        raw per-stream candidate log-probs on the reference partition
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

from .. import __version__
from ..runtime import RefScore, SparkInfer, gpu_memory_used_mib, wait_gpu_idle
from ..track import REPO_ROOT, Track
from . import logits
from .reference import load_reference
from .tasks import run_tasks

RUNTIME_CORRECTNESS = (
    "checkpoint loads in the pinned runtime (every scoring and benchmark process started and exited 0)",
    "no NaN/Inf log-probabilities on any scored position",
    "argmax token ids are inside the vocabulary",
    "executed format of every unit matches the manifest (audit)",
    "no SPARKINFER_* variable other than the track's pinned env reached the runtime",
)


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def effective_env(track: Track, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = track.clean_env(extra)
    keep = ("SPARKINFER_", "CUDA_", "NVIDIA_", "LD_LIBRARY_PATH", "OMP_")
    return {k: v for k, v in sorted(env.items()) if k.startswith(keep)}


def environment(si: SparkInfer, track: Track) -> dict:
    nvcc = _sh(["nvcc", "--version"])
    return {
        "gpu": _sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "cuda": nvcc.splitlines()[-1] if nvcc else "",
        "cpu": _sh(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip(),
        "kernel": platform.release(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "sparkinfer_commit": si.commit(),
        "bittrellis_version": __version__,
        "bittrellis_commit": _sh(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "evaluator_epoch": track["evaluation"]["epoch"],
        "effective_env": {
            "scoring": effective_env(track, track["evaluation"]["score"].get("env")),
            "benchmark": effective_env(track),
        },
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def evaluate_quality(si: SparkInfer, track: Track, model_dir: Path, corpus: dict, ref_dir: Path, out: Path,
                     streams: set[str] | None = None, log=print) -> tuple[dict, dict]:
    cfg = track["evaluation"]["score"]
    refs = load_reference(ref_dir, corpus, expected_k=cfg["reference_topk"])
    (out / "scores").mkdir(parents=True, exist_ok=True)
    vocab = track["model"]["vocab"]
    selected = [s for s in corpus["streams"] if not streams or s["id"] in streams]
    todo = [s for s in selected if not (out / "scores" / f"{s['id']}.npz").exists()]
    info: dict = {}
    if todo:
        t0 = time.time()
        scored, info = si.refscore_many(model_dir, [(s["id"], s["ids"], refs[s["id"]].top_ids, s["score_from"]) for s in todo],
                                        out / "scores" / "_work", env=cfg.get("env"))
        for sid, rs in scored.items():
            rs.save(out / "scores" / f"{sid}.npz")
        log(f"[quality] {len(todo)} streams in one process: {time.time() - t0:.0f}s")
    results, bad_argmax = [], 0
    for s in selected:
        rs = RefScore.load(out / "scores" / f"{s['id']}.npz")
        bad_argmax += int(((rs.argmax < 0) | (rs.argmax >= vocab)).sum())
        results.append(logits.compare(refs[s["id"]], rs, s))
    logits.save_positions(results, out / "kl_positions.npz")
    q = logits.summarize(results, cfg["reference_topk"])
    q["corpus"] = {"version": corpus["version"], "split": corpus["split"], "sha256": corpus["sha256"]}
    correctness = {
        "definition": list(RUNTIME_CORRECTNESS),
        "loads_and_exits_cleanly": True,
        "nonfinite_logprobs": q["nonfinite_logprobs"],
        "argmax_out_of_vocab": bad_argmax,
        "scoring_env": effective_env(track, cfg.get("env")),
        "peak_host_mib_scoring": info.get("peak_host_mib"),
    }
    correctness["ok"] = correctness["nonfinite_logprobs"] == 0 and bad_argmax == 0
    return q, correctness


def evaluate_performance(si: SparkInfer, track: Track, model_dir: Path, corpus: dict, out: Path, log=print) -> dict:
    """The track's two performance repetitions, each a fresh process with one model load."""
    cfg = track["evaluation"]["performance"]
    longest = max(corpus["streams"], key=lambda s: len(s["ids"]))
    prompt = out / "bench_prompt_ids.txt"
    prompt.write_text(" ".join(str(i) for i in longest["ids"]))
    runs = []
    for rep in range(int(cfg["runs"])):
        wait_gpu_idle()
        idle = gpu_memory_used_mib() or 0
        r = si.bench_run(model_dir, cfg["contexts"], cfg["decode_tokens"], prompt)
        runs.append({"sweep": {str(k): v for k, v in sorted(r["sweep"].items())},
                     "peak_gpu_gib": round((r["peak_gpu_mib"] - idle) / 1024.0, 3),
                     "resident_after_load_gib": r["vram_gib_reported"], "peak_host_gib": round(r["peak_host_mib"] / 1024.0, 3),
                     "idle_gpu_mib": idle, "wall_seconds": r["wall_seconds"]})
        log(f"[perf] run {rep + 1}/{cfg['runs']}: {r['wall_seconds']}s")
    prompt.unlink(missing_ok=True)
    ctx = str(cfg["primary_context"])

    def stat(key: str) -> tuple[float, float]:
        vals = [run["sweep"][ctx][key] for run in runs]
        mean = sum(vals) / len(vals)
        return mean, (max(vals) - min(vals)) / mean if mean else 0.0

    decode, decode_spread = stat("decode_tps")
    prefill, prefill_spread = stat("prefill_pp")
    return {
        "protocol": {"runs": cfg["runs"], "contexts": cfg["contexts"], "decode_tokens": cfg["decode_tokens"],
                     "primary_context": cfg["primary_context"], "memory_workload": cfg["memory_workload"]},
        "decode_tps": decode, "decode_spread": decode_spread,
        "prefill_tps": prefill, "prefill_spread": prefill_spread,
        "peak_gpu_gib": max(run["peak_gpu_gib"] for run in runs),
        "resident_after_load_gib": max(run["resident_after_load_gib"] or 0 for run in runs),
        "peak_host_gib": max(run["peak_host_gib"] for run in runs),
        "runs": runs,
    }


def evaluate(track: Track, si: SparkInfer, model_dir: Path, out: Path, corpus: dict, ref_dir: Path,
             identity: dict, stages: tuple[str, ...] = ("quality", "performance", "tasks"), log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    si.check_pinned()
    _write(out / "candidate.json", identity)
    _write(out / "environment.json", environment(si, track))
    result: dict = {"candidate": identity}
    timings = json.loads((out / "timings.json").read_text()) if (out / "timings.json").exists() else {}
    if "quality" in stages:
        t0 = time.time()
        q, correctness = evaluate_quality(si, track, model_dir, corpus, ref_dir, out, log=log)
        timings["quality_seconds"] = round(time.time() - t0, 1)
        if identity.get("audit_ok") is not None:
            correctness["execution_map_matches_manifest"] = bool(identity["audit_ok"])
            correctness["ok"] = correctness["ok"] and bool(identity["audit_ok"])
        result["quality"], result["correctness"] = q, correctness
        _write(out / "quality.json", q)
        _write(out / "correctness.json", correctness)
    if "tasks" in stages:
        tcfg = track["evaluation"]["tasks"]
        t0 = time.time()
        dspark = track["runtime"].get("dspark")
        result["tasks"] = run_tasks(si, model_dir, out / "tasks", tcfg["tier"], tcfg["server_ctx"], log=log,
                                    draft_model=REPO_ROOT / dspark["dir"] if dspark else None)
        timings["tasks_seconds"] = round(time.time() - t0, 1)
        _write(out / "tasks.json", result["tasks"])
    if "performance" in stages:
        t0 = time.time()
        result["performance"] = evaluate_performance(si, track, model_dir, corpus, out, log)
        timings["performance_seconds"] = round(time.time() - t0, 1)
        _write(out / "performance.json", result["performance"])
    _write(out / "timings.json", timings)
    return result


def checkpoint_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
