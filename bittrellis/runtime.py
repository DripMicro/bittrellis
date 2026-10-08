"""Thin, strict wrappers around the pinned SparkInfer binaries and their text output."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .track import REPO_ROOT, Track

_SWEEP = re.compile(r"^SWEEP_JSON\s+(\{.*\})\s*$", re.M)
_VRAM = re.compile(r"^VRAM used\s*:\s*([\d.]+)\s*GB", re.M)
_SCORE_LINE = re.compile(r"^S i=(\d+) tgt=(\d+) am=(\d+) lp=(\S+) top=(.*)$")


class RuntimeError_(RuntimeError):
    pass


@dataclass
class ScoreDump:
    """Teacher-forced output of qwen3_gguf_score for one token stream."""

    pos: np.ndarray        # int32 [N] position i (predicting token i+1)
    target: np.ndarray     # int32 [N]
    argmax: np.ndarray     # int32 [N]
    lp_target: np.ndarray  # float32 [N]
    top_ids: np.ndarray    # int32 [N, K]
    top_lp: np.ndarray     # float32 [N, K]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, pos=self.pos, target=self.target, argmax=self.argmax,
                            lp_target=self.lp_target, top_ids=self.top_ids, top_lp=self.top_lp)

    @classmethod
    def load(cls, path: Path) -> ScoreDump:
        z = np.load(path)
        return cls(z["pos"], z["target"], z["argmax"], z["lp_target"], z["top_ids"], z["top_lp"])


@dataclass
class RefScore:
    """Candidate log-probabilities on the fixed BF16 reference partition for one stream."""

    pos: np.ndarray        # int32 [M] position i (predicting token i+1)
    argmax: np.ndarray     # int32 [M]
    lp_target: np.ndarray  # float32 [M]
    lp_ref: np.ndarray     # float32 [M, K] log-prob of each reference top-K token id

    def save(self, path: Path) -> None:
        np.savez_compressed(path, pos=self.pos, argmax=self.argmax, lp_target=self.lp_target, lp_ref=self.lp_ref)

    @classmethod
    def load(cls, path: Path) -> RefScore:
        z = np.load(path)
        return cls(z["pos"], z["argmax"], z["lp_target"], z["lp_ref"])


def write_refscore_inputs(tokens: list[int], ref_ids: np.ndarray, tok_path: Path, ref_path: Path) -> None:
    np.asarray(tokens, "<i4").tofile(tok_path)
    k = np.asarray([ref_ids.shape[1]], "<i4")
    with open(ref_path, "wb") as fh:
        fh.write(k.tobytes())
        fh.write(np.ascontiguousarray(ref_ids, "<i4").tobytes())


def read_refscore_output(path: Path, prefix_len: int) -> RefScore:
    data = Path(path).read_bytes()
    if data[:4] != b"BTRS":
        raise RuntimeError_(f"{path}: not a refscore output")
    m, k = (int(x) for x in np.frombuffer(data[4:12], "<i4"))
    am = np.frombuffer(data, "<i4", m, 12)
    lpt = np.frombuffer(data, "<f4", m, 12 + 4 * m)
    lpr = np.frombuffer(data, "<f4", m * k, 12 + 8 * m).reshape(m, k)
    return RefScore(np.arange(prefix_len, prefix_len + m, dtype=np.int32), am.copy(), lpt.copy(), lpr.copy())


def run_monitored(cmd: list[str], env: dict[str, str], timeout: int) -> tuple[subprocess.CompletedProcess, int, int]:
    """Run a command while polling peak GPU memory (device-wide, MiB) and the process's peak host RSS (MiB)."""
    with PeakMemory() as peak:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        peak.watch_pid(proc.pid)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            raise RuntimeError_(f"timed out after {timeout}s: {' '.join(cmd[:2])}") from None
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err), peak.peak_mib, peak.peak_host_mib


def parse_score(text: str) -> ScoreDump:
    rows = []
    for line in text.splitlines():
        m = _SCORE_LINE.match(line)
        if not m:
            continue
        top = [kv.split(":") for kv in m.group(5).split(",")]
        rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)),
                     [int(a) for a, _ in top], [float(b) for _, b in top]))
    if not rows:
        raise RuntimeError_("no score lines in output")
    k = min(len(r[4]) for r in rows)
    return ScoreDump(
        pos=np.array([r[0] for r in rows], np.int32),
        target=np.array([r[1] for r in rows], np.int32),
        argmax=np.array([r[2] for r in rows], np.int32),
        lp_target=np.array([r[3] for r in rows], np.float32),
        top_ids=np.array([r[4][:k] for r in rows], np.int32),
        top_lp=np.array([r[5][:k] for r in rows], np.float32),
    )


def parse_sweep(text: str) -> dict:
    m = _SWEEP.findall(text)
    if not m:
        raise RuntimeError_("benchmark printed no SWEEP_JSON line")
    sweep = json.loads(m[-1])
    vram = [float(v) for v in _VRAM.findall(text)]
    return {"sweep": {int(k): v for k, v in sweep.items()}, "vram_gib_reported": max(vram) if vram else None}


class SparkInfer:
    def __init__(self, root: str | Path, track: Track):
        self.root = Path(root)
        self.track = track
        self.bench_bin = self.root / "build/runtime/qwen3_gguf_bench"
        self.refscore_bin = self.root / "build/bittrellis_refscore"
        self.server_bin = self.root / "build/server/sparkinfer_server"

    def commit(self) -> str:
        return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()

    def check_pinned(self) -> None:
        want = self.track["runtime"]["commit"]
        got = self.commit()
        if got != want:
            raise RuntimeError_(f"SparkInfer at {got}, track pins {want}")
        dirty = subprocess.run(["git", "-C", str(self.root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            raise RuntimeError_("SparkInfer checkout has local modifications")
        for b in (self.bench_bin, self.refscore_bin):
            if not b.exists():
                raise RuntimeError_(f"missing binary {b}; run scripts/setup_sparkinfer.sh")

    def refscore_many(self, model_dir: Path, streams: list[tuple[str, list[int], np.ndarray, int]], workdir: Path,
                      env: dict[str, str] | None = None, timeout: int = 7200) -> tuple[dict[str, RefScore], dict]:
        """Score several streams on the fixed reference partition in one process (one model load).

        `streams` holds (stream id, tokens, reference top-K ids [M, K], prefix length).
        """
        workdir.mkdir(parents=True, exist_ok=True)
        lines, outs = [], {}
        for sid, tokens, ref_ids, prefix_len in streams:
            tok, ref, out = workdir / f"{sid}.tokens.bin", workdir / f"{sid}.refids.bin", workdir / f"{sid}.out.bin"
            write_refscore_inputs(tokens, ref_ids, tok, ref)
            lines.append(f"{tok} {ref} {out} {prefix_len}")
            outs[sid] = (out, prefix_len, tok, ref)
        jobs = workdir / "jobs.txt"
        jobs.write_text("\n".join(lines) + "\n")
        r, peak_gpu, peak_host = run_monitored([str(self.refscore_bin), str(model_dir), "--jobs", str(jobs)],
                                               self.track.clean_env(env), timeout)
        if r.returncode != 0 or "\nOK " not in "\n" + r.stdout:
            raise RuntimeError_(f"refscore failed ({r.returncode}): {r.stdout[-800:]} {r.stderr[-1500:]}")
        result = {}
        for sid, (out, prefix_len, tok, ref) in outs.items():
            result[sid] = read_refscore_output(out, prefix_len)
            for f in (tok, ref, out):
                f.unlink(missing_ok=True)
        jobs.unlink(missing_ok=True)
        nonfinite = int(r.stdout.split("nonfinite_logits=")[1].split()[0]) if "nonfinite_logits=" in r.stdout else 0
        return result, {"nonfinite_logits": nonfinite, "peak_gpu_mib": peak_gpu, "peak_host_mib": peak_host}

    def bench_run(self, model_dir: Path, contexts: list[int], n_decode: int, prompt_file: Path,
                  env: dict[str, str] | None = None, timeout: int = 3600) -> dict:
        """One performance repetition: a fresh process, one model load, one pass over `contexts`."""
        extra = dict(env or {})
        extra.update({
            "SPARKINFER_BENCH_SWEEP_CTXS": ",".join(str(c) for c in contexts),
            "SPARKINFER_BENCH_SWEEP_REPS": "1",
            "SPARKINFER_BENCH_PROMPT_FILE": str(prompt_file),
        })
        cmd = [str(self.bench_bin), str(model_dir), str(n_decode), "sweep"]
        t0 = time.time()
        r, peak_gpu, peak_host = run_monitored(cmd, self.track.clean_env(extra), timeout)
        if r.returncode != 0:
            raise RuntimeError_(f"bench failed ({r.returncode}): {r.stdout[-800:]} {r.stderr[-1500:]}")
        out = parse_sweep(r.stdout)
        out.update({"peak_gpu_mib": peak_gpu, "peak_host_mib": peak_host, "wall_seconds": round(time.time() - t0, 1),
                    "log_tail": r.stdout[-2000:]})
        return out

    def start_server(self, model_dir: Path, port: int, ctx: int, log: Path,
                     env: dict[str, str] | None = None) -> subprocess.Popen:
        # A checkpoint directory carries its tokenizer; a single GGUF file (HPC-02) uses the track's pinned one.
        tok = Path(model_dir) / "tokenizer.json" if Path(model_dir).is_dir() else REPO_ROOT / self.track["model"]["paths"]["tokenizer"]
        cmd = [str(self.server_bin), "-m", str(model_dir), "--tokenizer", str(tok),
               "--ctx", str(ctx), "--host", "127.0.0.1", "--port", str(port)]
        extra = {"SPARKINFER_SAMPLING_DEFAULTS": "greedy", **(env or {})}
        fh = open(log, "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=self.track.clean_env(extra))
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError_(f"server exited with {proc.returncode}; see {log}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return proc
            except OSError:
                pass
            time.sleep(1)
        proc.kill()
        raise RuntimeError_("server did not become healthy in 900s")


def gpu_memory_used_mib() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _rss_mib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


class PeakMemory:
    """Poll device memory (and optionally one process's host RSS) in a background thread.

    nvidia-smi is sampled every `interval` seconds, so allocations that live shorter than that can be
    missed; the reported peak is a lower bound on the true device peak.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_mib = 0
        self.peak_host_mib = 0
        self._pid: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def watch_pid(self, pid: int) -> None:
        self._pid = pid

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak_mib = max(self.peak_mib, used)
            if self._pid is not None:
                self.peak_host_mib = max(self.peak_host_mib, _rss_mib(self._pid))
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def wait_gpu_idle(max_mib: int = 1024, timeout: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = gpu_memory_used_mib()
        if used is None or used < max_mib:
            return
        time.sleep(2)
    raise RuntimeError_(f"GPU still has {gpu_memory_used_mib()} MiB in use; refusing to benchmark on a busy card")
