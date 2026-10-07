"""Build a deployable SparkInfer checkpoint from a manifest.

Sources are the hash-pinned checkpoints in configs/sources.lock.json:

    base              canonical BF16 weights: Q4_K units store these bytes, regenerable quantizers read them
    gittensor_nvfp4   the shipped checkpoint: every non-searchable tensor, and `baseline` NVFP4 bytes
    unsloth_nvfp4     attested calibrated NVFP4 bytes (`unsloth`), MLP layers 0-55

Each unit's tensors come from its quantizer (bittrellis/quantizers). Everything else is copied
byte-for-byte from the shipped checkpoint. No GPU is needed and the build is deterministic, so the
same manifest always yields the same shard hashes.
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import __version__, build_jobs
from . import quantizers as Q
from .calibration import open_calibration
from .lineage import require_verified
from .manifest import Assignment, Manifest, candidate_hash, summarize
from .model.qwen38 import Qwen38Arch, Unit
from .precision import FP8, NVFP4
from .quantizers.builtin import CT_SUFFIXES, MODELOPT_SUFFIXES
from .safetensors_io import SafeTensorsDir, ShardWriter, file_sha256
from .track import Track

LINEAR_SUFFIXES = tuple(sorted(set(MODELOPT_SUFFIXES + CT_SUFFIXES)))


@dataclass
class PlannedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    make: Callable[[], object]
    memo: _Memo | None = None       # set for independent regenerable encodings, which may run in a worker


class _Memo:
    def __init__(self, fn, job: tuple[str, str] | None = None):
        self.fn, self.value, self.job = fn, None, job   # job: (unit id, Linear prefix)

    def __call__(self):
        if self.value is None:
            self.value = self.fn()
        return self.value


def plan_tensors(units: list[Unit], assignments: dict[str, Assignment], ctx: Q.QuantContext,
                 frozen: SafeTensorsDir) -> list[PlannedTensor]:
    plan: list[PlannedTensor] = []
    searchable: set[str] = set()
    for q in {Q.get(a.quantizer) for a in assignments.values()}:
        q.begin(ctx)
    # Pipeline order (layer ascending, unit order) matters for sequential quantizers.
    for u in units:
        a = assignments[u.id]
        qz = Q.get(a.quantizer)
        why = qz.available(ctx, u, a.format)
        if why:
            raise ValueError(f"{u.id}: {a.format}@{a.quantizer}: {why}")
        for lin in u.linears:
            searchable.add(lin.prefix)
            if qz.replay_mode == "sequential":
                produced = encode_unit(qz, ctx, u, lin, a)  # must run now, in pipeline order
                for suf, dtype, shape, data in produced:
                    plan.append(PlannedTensor(lin.prefix + suf, dtype, tuple(shape), lambda d=data: d))
                continue
            job = (u.id, lin.prefix) if qz.lineage == "regenerable" else None
            memo = _Memo(lambda qz=qz, u=u, lin=lin, a=a: encode_unit(qz, ctx, u, lin, a), job)
            for suf, dtype, shape in _declared(qz, ctx, u, lin, a.format):
                plan.append(PlannedTensor(lin.prefix + suf, dtype, shape,
                                          lambda m=memo, suf=suf: next(d for s, _, _, d in m() if s == suf), memo))
    for name, ref in frozen.tensors.items():
        prefix, _, suffix = name.rpartition(".")
        if prefix in searchable and "." + suffix in LINEAR_SUFFIXES:
            continue
        plan.append(PlannedTensor(name, ref.dtype, ref.shape, lambda n=name: frozen.raw(n)))
    plan.sort(key=lambda t: t.name)
    names = [t.name for t in plan]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate tensor in build plan")
    return plan


def encode_unit(qz: Q.Quantizer, ctx: Q.QuantContext, u: Unit, lin, a: Assignment):
    ctx.params = dict(a.params)
    return qz.encode(ctx, u, lin, a.format)


# Independent regenerable encodings run in forked workers: each depends only on its own Linear's BF16 weight,
# so the bytes are the same as a serial build, only sooner. The writer still consumes them in plan order.
_WORKER: dict = {}


def _encode_job(job: tuple[str, str]):
    ctx, units, assignments = _WORKER["ctx"], _WORKER["units"], _WORKER["assignments"]
    u = units[job[0]]
    lin = next(x for x in u.linears if x.prefix == job[1])
    a = assignments[u.id]
    return [(suf, dt, tuple(sh), data) for suf, dt, sh, data in encode_unit(Q.get(a.quantizer), ctx, u, lin, a)]


def _write_plan(plan: list[PlannedTensor], writer, ctx, units, assignments, jobs: int, log, t0: float) -> None:
    order: list[_Memo] = []
    for t in plan:
        if t.memo is not None and t.memo.job is not None and t.memo not in order:
            order.append(t.memo)
    pool = None
    if jobs > 1 and order:
        _WORKER.update(ctx=ctx, units={u.id: u for u in units}, assignments=assignments)
        pool = multiprocessing.get_context("fork").Pool(jobs)
    try:
        pending: deque = deque()
        queued = iter(order)

        def top_up():
            while pool is not None and len(pending) < 2 * jobs:
                m = next(queued, None)
                if m is None:
                    return
                pending.append((m, pool.apply_async(_encode_job, (m.job,))))

        top_up()
        for i, t in enumerate(plan):
            if t.memo is not None and t.memo.value is None and pending and any(m is t.memo for m, _ in pending):
                while True:   # results arrive in submission order, which is plan order
                    m, r = pending.popleft()
                    m.value = r.get()
                    top_up()
                    if m is t.memo:
                        break
            writer.add(t.name, t.dtype, t.shape, t.make())
            if (i + 1) % 250 == 0:
                log(f"[build]   {i + 1}/{len(plan)} tensors, {time.time() - t0:.0f}s")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _WORKER.clear()


def _declared(qz: Q.Quantizer, ctx: Q.QuantContext, u: Unit, lin, fmt: str) -> list[tuple[str, str, tuple]]:
    """Tensor names/dtypes/shapes a quantizer will produce, without computing the data."""
    if qz.lineage == "attested":
        return [(s, dt, tuple(sh)) for s, dt, sh, _ in qz.encode(ctx, u, lin, fmt)]  # splices are views: cheap
    if qz.lineage == "runtime":
        return [(".weight", "BF16", (lin.rows, lin.cols))]
    if fmt == FP8:
        return [(".weight", "F8_E4M3", (lin.rows, lin.cols)), (".weight_scale", "BF16", (lin.rows, 1))]
    if fmt == NVFP4:
        return [(".weight", "U8", (lin.rows, lin.cols // 2)), (".weight_scale", "F8_E4M3", (lin.rows, lin.cols // 16)),
                (".weight_scale_2", "F32", ())]
    raise ValueError(f"{qz.name}: cannot declare outputs for {fmt}")


def quant_config(units: list[Unit], assignments: dict[str, Assignment], frozen_cfg: dict) -> dict:
    layers: dict[str, dict] = {}
    q4k: list[str] = []
    for u in units:
        a = assignments[u.id]
        for lin in u.linears:
            if a.format == NVFP4:
                layers[lin.prefix] = {"quant_algo": "NVFP4", "group_size": 16, "quantizer": a.quantizer_ref}
            elif a.format == FP8:
                layers[lin.prefix] = {"quant_algo": "FP8_PER_CHANNEL", "quantizer": a.quantizer_ref}
            else:
                q4k.append(lin.prefix)
    old = frozen_cfg.get("quantization_config", {})
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "bittrellis", "version": __version__},
        "quantized_layers": layers,
        "ignore": sorted(set(old.get("ignore", [])) | set(q4k)),
        "runtime_fit": {"Q4_K": q4k},
    }


def open_context(assignments: dict[str, Assignment], source_dirs: dict[str, Path], verify: bool,
                 log: Callable[[str], None]) -> tuple[Q.QuantContext, list[SafeTensorsDir]]:
    needed = {"base", "gittensor_nvfp4"} | {Q.get(a.quantizer).source_id for a in assignments.values()
                                            if Q.get(a.quantizer).lineage == "attested"}
    missing = sorted(s for s in needed if s not in source_dirs)
    if missing:
        raise ValueError(f"missing source checkouts: {missing}")
    if verify:
        for s in sorted(needed):
            require_verified(s, source_dirs[s], log=log)
    opened = {s: SafeTensorsDir(source_dirs[s]) for s in needed}
    # The calibration statistics are opened whenever they are provided: whether an encoder reads them is
    # its own business, and the sandbox gets the same view as the trusted audit.
    calib = open_calibration(source_dirs["calibration"], verify, log) if "calibration" in source_dirs else None
    ctx = Q.QuantContext(base=opened["base"], sources=opened, calibration=calib)
    return ctx, list(opened.values()) + ([calib] if calib else [])


def build(manifest: Manifest, track: Track, source_dirs: dict[str, Path], out_dir: Path,
          log: Callable[[str], None] = print, shard_bytes: int = 5 * 1024**3, verify: bool = True) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    if manifest.track != track.id:
        raise ValueError(f"manifest is for {manifest.track}, track is {track.id}")
    frozen_dir = Path(source_dirs["gittensor_nvfp4"])
    frozen_cfg = json.loads((frozen_dir / "config.json").read_text())
    units = Qwen38Arch.from_config(frozen_cfg).units()
    assignments = manifest.expand_assignments(units)
    cid = candidate_hash(track.id, assignments)
    t0 = time.time()
    ctx, handles = open_context(assignments, source_dirs, verify, log)
    try:
        plan = plan_tensors(units, assignments, ctx, ctx.sources["gittensor_nvfp4"])
        log(f"[build] {manifest.name} ({cid}): {len(plan)} tensors -> {out_dir}")
        writer = ShardWriter(out_dir, shard_bytes)
        _write_plan(plan, writer, ctx, units, assignments, build_jobs(), log, t0)
        writer.close(metadata={"producer": "bittrellis", "candidate_id": cid})
    finally:
        for h in handles:
            h.close()
    cfg = dict(frozen_cfg)
    cfg["quantization_config"] = quant_config(units, assignments, frozen_cfg)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for f in track["model"]["aux_files"]:
        if (frozen_dir / f).exists():
            shutil.copy2(frozen_dir / f, out_dir / f)
    (out_dir / "precision_manifest.yaml").write_text(yaml.safe_dump(manifest.to_dict(units), sort_keys=False))
    files = sorted(p.name for p in out_dir.glob("*.safetensors"))
    used = sorted({Q.get(a.quantizer).ref for a in assignments.values()})
    record = {
        "candidate_id": cid,
        "name": manifest.name,
        "track": track.id,
        "bittrellis_version": __version__,
        "quantizers": {r: {"lineage": Q.get(r.split("@")[0]).lineage, "replay_mode": Q.get(r.split("@")[0]).replay_mode,
                           "source": Q.get(r.split("@")[0]).source_id} for r in used},
        "sources_verified": verify,
        "summary": summarize(assignments, units),
        "checkpoint_bytes": sum((out_dir / f).stat().st_size for f in files),
        "files": {f: file_sha256(out_dir / f) for f in files},
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "bittrellis_build.json").write_text(json.dumps(record, indent=2) + "\n")
    log(f"[build] done in {record['build_seconds']}s, {record['checkpoint_bytes'] / 1e9:.2f} GB")
    return record
