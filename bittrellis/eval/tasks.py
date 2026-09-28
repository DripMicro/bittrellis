"""Task-level guard: SparkInfer's own quality suite, run against the candidate on the server.

BitTrellis reuses the pinned SparkInfer checkout's `bench/quality` data, prompt builders and
scorers (IFEval, GSM8K, MMLU-Pro, HumanEval, function calling) rather than shipping its own
benchmark. Two things differ from `run_quality.py`, both because it was not written for Qwen3.8:

* Requests go to `/v1/chat/completions` with thinking disabled. Its `/v1/completions` route
  returns the end-of-turn token in the text (`"F<|im_end|>"`), and the MMLU-Pro scorer's
  last-letter fallback then reads the "D" of "END".
* Token caps are raised so Qwen3.8's step-by-step answers are not cut before the final line.

Task scores are a *guard*, not the ranking signal. Per-item outcomes are kept so the guard can compare
a candidate with V0 question by question (see frontier/pareto.py `task_guard`).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from ..runtime import SparkInfer

MAX_TOKENS = {"gsm8k": 1024, "humaneval": 768, "mmlu_pro": 64, "ifeval": 768, "bfcl": 256}


def _load_suite(si: SparkInfer):
    qdir = si.root / "bench/quality"
    sys.path.insert(0, str(qdir))
    try:
        spec = importlib.util.spec_from_file_location("si_run_quality", qdir / "run_quality.py")
        rq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rq)
    finally:
        sys.path.remove(str(qdir))
    return rq


def _chat(port: int, prompt: str, max_tokens: int) -> str:
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"] or ""


def run_tasks(si: SparkInfer, model_dir: Path, out_dir: Path, tier: str, ctx: int, port: int = 18080,
              log=print, draft_model: Path | None = None) -> dict:
    """Run the suite one request at a time against a deterministic server on `model_dir`.

    One at a time on purpose: concurrent requests are batched, and batching changed 70 of 784 answers
    between two identical runs even in deterministic mode (Qwen3.8 is not SparkInfer's qualified
    batch-invariant model). `draft_model` turns on DSpark, which answers identically, only faster.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rq = _load_suite(si)
    items = rq.load(set(), 0, "" if tier == "full" else tier)  # "full": every item SparkInfer ships (784)
    # Deterministic mode makes answers bit-reproducible (SparkInfer server/README.md, "Determinism"). Without
    # it, two runs of one model differ on dozens of the 784 answers, which the paired task guard compares.
    env = {"SPARKINFER_DETERMINISTIC": "1"}
    if draft_model and Path(draft_model).exists():
        env["SPARKINFER_DRAFT_MODEL"] = str(draft_model)
    elif draft_model:
        log(f"[tasks] DSpark drafter {draft_model} missing: running without it (same answers, slower)")
    proc = si.start_server(model_dir, port, ctx, out_dir / "server.log", env=env)
    t0 = time.time()
    jsonl = out_dir / "tasks.jsonl"

    def one(it: dict) -> dict:
        b = it["benchmark"]
        try:
            text = _chat(port, rq.build_prompt(it), MAX_TOKENS.get(b, 512))
            r = rq.scorers.SCORERS[b](it, text)
        except Exception as e:  # noqa: BLE001 - one malformed item must not stop the guard
            text, r = "", {"score": 0.0, "pass": False, "detail": "ERROR: " + repr(e)}
        return {"id": it["id"], "benchmark": b, "pass": r["pass"], "score": r["score"], "detail": r["detail"],
                "output": text[-2000:]}

    try:
        with open(jsonl, "w") as fh:
            for it in items:
                fh.write(json.dumps(one(it)) + "\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    log(f"[tasks] {len(items)} items in {time.time() - t0:.0f}s{' with DSpark' if 'SPARKINFER_DRAFT_MODEL' in env else ''}")
    return summarize_tasks(jsonl)


def summarize_tasks(jsonl: Path) -> dict:
    per: dict[str, list[int]] = {}
    items: dict[str, dict[str, int]] = {}
    for line in Path(jsonl).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        per.setdefault(r["benchmark"], []).append(1 if r["pass"] else 0)
        items.setdefault(r["benchmark"], {})[str(r["id"])] = 1 if r["pass"] else 0
    suites = {b: {"passed": sum(v), "n": len(v), "rate": sum(v) / len(v)} for b, v in sorted(per.items())}
    total = sum(s["passed"] for s in suites.values())
    n = sum(s["n"] for s in suites.values())
    return {"suites": suites, "passed": total, "n": n, "rate": total / n if n else None,
            "items": items, "max_tokens": MAX_TOKENS}
