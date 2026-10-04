"""Per-box speeds (hpc01-e5): every ranked reference is re-measured on the machine that measures the PR.

The same checkpoint scores identically on two boxes for RP-KL, tasks and the holdout, but its decode and
prefill move with the GPU driver, CUDA and CPU (V0's prefill: 14,760 tok/s on the box that measured the seeds,
16,650-16,715 on two later ones). So a PR is ranked against references' speeds taken here, not stored ones.
Only the performance stage runs; each reference is rebuilt from its own recipe and deleted again.

    <root>/speeds/machine.json            the machine and the time these speeds were measured
    <root>/speeds/<candidate id>/performance.json
    <root>/speeds/v0-checkpoint/          kept: V0 is re-measured next to every PR to catch drift
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

OBJECTIVES = ("decode_tps", "prefill_tps", "peak_gpu_gib")


def machine(sparkinfer: str) -> dict:
    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    return {"gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "cuda": (sh(["nvcc", "--version"]).splitlines() or [""])[-1],
            "cpu": sh(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
            "sparkinfer_commit": sh(["git", "-C", sparkinfer, "rev-parse", "HEAD"])}


def references(paths: list[Path]) -> list[Path]:
    """Ranked (internal) reference artifacts that carry the recipe they were built from."""
    arts = []
    for p in paths:
        for d in ([p] if (p / "candidate.json").exists() else sorted(p.iterdir()) if p.is_dir() else []):
            cand = json.loads((d / "candidate.json").read_text()) if (d / "candidate.json").exists() else {}
            if cand.get("kind") in ("candidate", "internal") and cand.get("manifest"):
                arts.append(d)
    return arts


def speed_manifest(manifest: dict) -> tuple[dict, bool]:
    """The recipe as built for a speed run: same formats, with every regenerable encoder replaced by its format's
    default bytes. Speed and memory do not depend on the byte values within a format (on this evaluator, #28's and
    #32's recipes measured within 0.4% of their real bytes, inside every noise floor), while an encoder such as
    nvfp4_blockfit takes ~17 min to rebuild a checkpoint that the defaults write in under a minute."""
    from bittrellis import quantizers as Q

    m, changed = json.loads(json.dumps(manifest)), False

    def swap(fmt, q):
        return q and q in Q.REGISTRY and Q.REGISTRY[q].lineage == "regenerable" and q != Q.DEFAULT_FOR_FORMAT.get(fmt)

    for r in m.get("rules", []):
        if swap(r.get("format"), r.get("quantizer")):
            r.pop("quantizer")
            r.pop("params", None)
            changed = True
    for fmt, q in list((m.get("quantizers") or {}).items()):
        if swap(fmt, q):
            m["quantizers"][fmt] = Q.DEFAULT_FOR_FORMAT[fmt]
            changed = True
    if changed:
        m.pop("expanded", None)
        m["name"] = m["name"] + "-speed"
    return m, changed


def kernels(manifest: dict) -> dict:
    """Per unit: the stored format and the kernels it runs on -- what a speed run depends on."""
    return {u: (e.get("source_format"), json.dumps(e.get("execution"), sort_keys=True))
            for u, e in (manifest.get("expanded") or {}).items()}


def drifted(stored: dict, now: dict, floors: dict) -> list[str]:
    """Objectives on which a re-measurement moved beyond the noise floor (relative for speed, GiB for memory)."""
    out = []
    for k in OBJECTIVES:
        a, b = stored[k], now[k]
        if (abs(b - a) if k == "peak_gpu_gib" else abs(b - a) / a) > floors[k]:
            out.append(f"{k} {a:g} -> {b:g}")
    return out


class BoxSpeeds:
    def __init__(self, root: Path, run, py: list[str], env_args: list[str], sparkinfer: str, floors: dict, incumbent: str):
        self.dir = Path(root) / "speeds"
        self.run, self.py, self.env_args, self.sparkinfer = run, py, env_args, sparkinfer
        self.floors, self.incumbent = floors, incumbent
        self.log = self.dir / "speeds.log"

    @property
    def stamp(self) -> str | None:
        """When this machine's reference speeds were measured; results ranked with other speeds are re-ranked."""
        f = self.dir / "machine.json"
        return json.loads(f.read_text()).get("measured_utc") if f.exists() else None

    def has(self, cid: str) -> bool:
        return (self.dir / cid / "performance.json").exists()

    def missing(self, refs: list[Path]) -> list[Path]:
        """References without a speed from this machine. A different machine invalidates every stored speed."""
        f = self.dir / "machine.json"
        if f.exists() and json.loads(f.read_text())["machine"] != machine(self.sparkinfer):
            self.reset()
        return [a for a in refs if not self.has(json.loads((a / "candidate.json").read_text())["id"])]

    def reset(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def adopt(self, art: Path) -> bool:
        """Take a reference's own speed instead of rebuilding it, when it was measured on this machine after the
        last full re-measurement -- a PR merged here was benchmarked by this evaluator an hour earlier, and
        rebuilding a 19 GB checkpoint for a one-minute benchmark repeats that work."""
        f, env_f, perf = self.dir / "machine.json", art / "environment.json", art / "performance.json"
        if not (f.exists() and env_f.exists() and perf.exists()):
            return False
        doc, env = json.loads(f.read_text()), json.loads(env_f.read_text())
        since = doc.get("calibrated_utc") or doc.get("measured_utc")
        if {k: env.get(k) for k in doc["machine"]} != doc["machine"] or not since or env.get("time_utc", "") < since:
            return False
        cid = json.loads((art / "candidate.json").read_text())["id"]
        (self.dir / cid).mkdir(exist_ok=True)
        shutil.copyfile(perf, self.dir / cid / "performance.json")
        return True

    def measure(self, refs: list[Path]) -> list[str]:
        """Rebuild and benchmark each reference; returns the ones that failed (they keep their stored speed)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / "machine.json"
        old = json.loads(f.read_text()) if f.exists() else {}
        failed = []
        for art in refs:
            cand = json.loads((art / "candidate.json").read_text())
            perf = self._benchmark(cand)
            if perf is None:
                failed.append(cand["name"])
                continue
            (self.dir / cand["id"]).mkdir(exist_ok=True)
            (self.dir / cand["id"] / "performance.json").write_text(json.dumps(perf, indent=2) + "\n")
        if len(failed) < len(refs):
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # calibrated_utc: when every reference was last measured together (only a full re-measurement moves it)
            f.write_text(json.dumps({"machine": machine(self.sparkinfer), "measured_utc": now,
                                     "calibrated_utc": old.get("calibrated_utc") or old.get("measured_utc") or now}, indent=2) + "\n")
        return failed

    def check_drift(self, incumbent_art: Path) -> list[str]:
        """Re-measure V0 now; the objectives that moved beyond noise since the references were measured.

        A move must show at two PRs in a row -- V0 is measured next to every PR anyway, so this costs no extra
        run -- because one reading a little past the floor (V0's peak memory 22.016 -> 22.135 GiB once) is not
        worth re-measuring every reference for."""
        cand = json.loads((incumbent_art / "candidate.json").read_text())
        stored = self.dir / cand["id"] / "performance.json"
        now = self._benchmark(cand)
        if now is None or not stored.exists():
            return []
        moved = drifted(json.loads(stored.read_text()), now, self.floors)
        pending = self.dir / "drift-pending.json"
        if moved and not pending.exists():
            pending.write_text(json.dumps({"moved": moved, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            return []
        pending.unlink(missing_ok=True)
        return moved

    def _benchmark(self, cand: dict) -> dict | None:
        work = self.dir / "work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        is_v0 = cand["name"] == self.incumbent  # V0's checkpoint is kept: it is re-measured next to every PR
        ckpt = self.dir / "v0-checkpoint" if is_v0 else work / "checkpoint"
        fast_bytes = False   # V0 is built from its own recipe (default bytes already) and kept
        try:
            if not (ckpt / "bittrellis_build.json").exists():
                shutil.rmtree(ckpt, ignore_errors=True)
                manifest = work / "manifest.yaml"
                built, fast_bytes = speed_manifest(cand["manifest"])
                manifest.write_text(json.dumps(built))  # YAML is a superset of JSON
                if self.run(self.py + ["build", str(manifest), "--out", str(ckpt)] + self.env_args, self.dir, self.log) != 0:
                    return None
            out = work / "artifact"
            if self.run(self.py + ["benchmark", str(ckpt), "--out", str(out), "--sparkinfer", self.sparkinfer] + self.env_args,
                        self.dir, self.log) != 0:
                return None
            got = json.loads((out / "candidate.json").read_text())
            # exactly the reference that is ranked -- or, with default bytes, the same formats on the same kernels
            same = (kernels(got.get("manifest", {})) == kernels(cand["manifest"])) if fast_bytes else got["id"] == cand["id"]
            if not same:
                with open(self.log, "a") as fh:
                    fh.write(f"[speeds] {cand['name']}: rebuilt checkpoint is not the ranked reference ({got['id']})\n")
                return None
            return json.loads((out / "performance.json").read_text())
        finally:
            if not is_v0:
                shutil.rmtree(ckpt, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
