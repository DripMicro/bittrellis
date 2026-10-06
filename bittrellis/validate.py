"""Audit a checkpoint before anything is benchmarked.

`describe()` answers "what will SparkInfer execute?" for any Qwen3.8 checkpoint. `audit()` enforces
the HPC-01 rules for candidates, in this order:

1. sources      base and shipped checkpoints (plus any attested source used) match the hash lock
2. config       config.json is the shipped config except `quantization_config`
3. tensor set   exactly the tensors the manifest's quantizers produce plus the frozen tensors
4. frozen       every non-searchable tensor is byte-identical to the shipped checkpoint
5. execution    each unit resolves, under the pinned loader rules, to the manifest's format
6. lineage      per quantizer class:
                  runtime      stored BF16 is byte-identical to the base model
                  attested     unit bytes are byte-identical to the verified source
                  regenerable  sampled units are rebuilt (replaying the pipeline in order for
                               sequential quantizers) and must match byte-for-byte
7. anomaly      reconstruction error vs round-to-nearest is reported; > 5x is rejected as
                substituted bytes (calibrated encoders measure 1.2-1.6x, so this is diagnostic)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import quantizers as Q
from .build import _declared, encode_unit
from .calibration import open_calibration
from .lineage import LineageError, require_verified
from .manifest import Assignment, Manifest
from .model.qwen38 import Qwen38Arch, Unit
from .precision import FP8, NVFP4, resolve_checkpoint
from .quant.formats import (
    bf16_to_f32,
    dequantize_fp8_per_channel,
    dequantize_nvfp4,
    quantize_fp8_per_channel,
    quantize_nvfp4,
)
from .safetensors_io import SafeTensorsDir

SAMPLE_ROWS = 24
REGEN_SAMPLES_PER_QUANTIZER = 6
ANOMALY_WARN_RATIO = 2.0
ANOMALY_REJECT_RATIO = 5.0


@dataclass
class AuditResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, dict] = field(default_factory=dict)
    anomaly: dict[str, float] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def to_dict(self) -> dict:
        worst = sorted(self.anomaly.items(), key=lambda kv: -kv[1])[:10]
        return {"ok": self.ok, "errors": self.errors[:200], "n_errors": len(self.errors),
                "warnings": self.warnings[:50], "runtime": self.runtime, "lineage": self.lineage,
                "anomaly_worst_ratio": dict(worst)}


def describe(ckpt_dir: str | Path) -> dict[str, str]:
    """Selected format per unit as the loader resolves it, e.g. {'L3.attn.q': 'Q4_K(fp8)'}."""
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
    units = Qwen38Arch.from_config(cfg).units()
    with SafeTensorsDir(ckpt_dir) as ck:
        return {uid: r.label for uid, r in resolve_checkpoint(ck, units).items()}


def _rows(n_rows: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_rows - 1, SAMPLE_ROWS).astype(np.int64))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a))


def anomaly_ratio(ck: SafeTensorsDir, base: SafeTensorsDir, lin, fmt: str) -> float:
    """Sampled-row reconstruction error of the stored tensor divided by round-to-nearest's."""
    rows = _rows(lin.rows)
    w = bf16_to_f32(base.array(lin.prefix + ".weight", "<u2")[rows].reshape(-1)).reshape(len(rows), lin.cols)
    if fmt == NVFP4:
        if ck.get(lin.prefix + ".weight_packed") is not None:
            packed = ck.array(lin.prefix + ".weight_packed", "u1").reshape(lin.rows, -1)[rows]
            ws2 = 1.0 / float(ck.array(lin.prefix + ".weight_global_scale", "<f4").reshape(-1)[0])
        else:
            packed = ck.array(lin.prefix + ".weight", "u1").reshape(lin.rows, -1)[rows]
            ws2 = float(ck.array(lin.prefix + ".weight_scale_2", "<f4").reshape(-1)[0])
        scales = ck.array(lin.prefix + ".weight_scale", "u1").reshape(lin.rows, -1)[rows]
        got = _rel(dequantize_nvfp4(packed, scales, ws2), w)
        ref = _rel(dequantize_nvfp4(*quantize_nvfp4(w, global_amax=ws2 * 6.0 * 448.0)), w)
    else:
        codes = ck.array(lin.prefix + ".weight", "u1").reshape(lin.rows, lin.cols)[rows]
        scale = ck.array(lin.prefix + ".weight_scale", "<u2").reshape(-1)[rows]
        got = _rel(dequantize_fp8_per_channel(codes, scale), w)
        ref = _rel(dequantize_fp8_per_channel(*quantize_fp8_per_channel(w)), w)
    return (got + 1e-9) / (ref + 1e-9)


def _bytes_of(data) -> bytes:
    return np.ascontiguousarray(data).tobytes() if isinstance(data, np.ndarray) else bytes(data)


def _sample_units(units: list[Unit], seed: str, k: int) -> list[Unit]:
    if len(units) <= k:
        return list(units)
    ranked = sorted(units, key=lambda u: hashlib.sha256(f"{seed}:{u.id}".encode()).hexdigest())
    picked = {units[0].id, units[-1].id} | {u.id for u in ranked[: k - 2]}
    return [u for u in units if u.id in picked]


def replay_cache_dir() -> Path:
    return Path(os.environ.get("BITTRELLIS_REPLAY_CACHE", Path.home() / ".cache/bittrellis/replay"))


def _replay_key(qz: Q.Quantizer, units: list[Unit], assignments: dict[str, Assignment], unit: Unit) -> str:
    """Identity of a regenerated unit: the quantizer version, the base weights, and -- for sequential
    quantizers -- every assignment of that quantizer up to and including the unit (its pipeline prefix) --
    plus the pinned calibration files, which a calibrated encoder reads."""
    from .lineage import load_lock

    lock = load_lock()["sources"]
    base_rev = lock["base"]["revision"]
    calib = lock.get("calibration", {}).get("files")
    if qz.replay_mode == "sequential":
        prefix = [f"{u.id}={assignments[u.id].key()}" for u in units[: units.index(unit) + 1] if assignments[u.id].quantizer == qz.name]
    else:
        prefix = [f"{unit.id}={assignments[unit.id].key()}"]
    key = [qz.ref, base_rev, prefix] + ([calib] if calib else [])
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def regenerate(qz: Q.Quantizer, units: list[Unit], assignments: dict[str, Assignment], ctx: Q.QuantContext,
               targets: set[str]) -> dict[str, bytes]:
    """The tensors `qz` produces for the `targets` units, replaying its pipeline in order when sequential."""
    mine = [u for u in units if assignments[u.id].quantizer == qz.name]
    remaining = set(targets)
    out: dict[str, bytes] = {}
    qz.begin(ctx)
    for u in (mine if qz.replay_mode == "sequential" else [u for u in mine if u.id in targets]):
        if not remaining:
            break
        remaining.discard(u.id)
        for lin in u.linears:
            produced = encode_unit(qz, ctx, u, lin, assignments[u.id])
            if u.id in targets:
                out.update({lin.prefix + suf: _bytes_of(data) for suf, _dtype, _shape, data in produced})
    return out


def regenerable_samples(units: list[Unit], assignments: dict[str, Assignment], qname: str, seed: str) -> list[Unit]:
    """Units of quantizer `qname` the audit rebuilds. `seed` is the candidate id, plus the evaluator's
    secret when one is set, so a submission cannot know in advance which units are checked."""
    return _sample_units([u for u in units if assignments[u.id].quantizer == qname], seed, REGEN_SAMPLES_PER_QUANTIZER)


def _check_regenerable(qz: Q.Quantizer, units: list[Unit], assignments: dict[str, Assignment],
                       ck: SafeTensorsDir, ctx: Q.QuantContext, seed: str, res: AuditResult,
                       regenerated_dir: Path | None = None) -> dict:
    mine = [u for u in units if assignments[u.id].quantizer == qz.name]
    sampled = regenerable_samples(units, assignments, qz.name, seed)
    targets = {u.id for u in sampled}
    info = {"lineage": "regenerable", "replay_mode": qz.replay_mode, "units": len(mine), "sampled": sorted(targets)}
    if regenerated_dir is not None:
        # Contributed code regenerated the samples in isolation, without access to this checkpoint;
        # only the comparison happens here.
        compared = 0
        for u in sampled:
            for lin in u.linears:
                files = sorted(Path(regenerated_dir).glob(f"{lin.prefix}.*.bin"))
                if not files:
                    res.fail(f"{lin.prefix}: no regeneration by {qz.ref} was provided")
                for f in files:
                    name = f.name[: -len(".bin")]
                    compared += 1
                    if ck.get(name) is None or bytes(ck.raw(name)) != f.read_bytes():
                        res.fail(f"{name}: does not match a regeneration by {qz.ref} -- not produced by the declared quantizer")
        return {**info, "replay_mode": "isolated", "tensors_compared": compared}
    cache = replay_cache_dir()
    # Replays are shared: a unit whose replay key was regenerated before (by any candidate) is checked
    # against the cached tensor hashes instead of being rebuilt.
    cached_hits = 0
    for u in sampled:
        entry = cache / f"{_replay_key(qz, units, assignments, u)}.json"
        if entry.exists():
            want = json.loads(entry.read_text())
            for name, digest in want.items():
                if ck.get(name) is None or ck.sha256(name) != digest:
                    res.fail(f"{name}: does not match a regeneration by {qz.ref} -- not produced by the declared quantizer")
            targets.discard(u.id)
            cached_hits += 1
    replayed = 0
    produced = regenerate(qz, units, assignments, ctx, targets) if targets else {}
    for name, blob in produced.items():
        replayed += 1
        if ck.get(name) is None or bytes(ck.raw(name)) != blob:
            res.fail(f"{name}: does not match a regeneration by {qz.ref} -- not produced by the declared quantizer")
    for u in sampled:
        digests = {n: hashlib.sha256(b).hexdigest() for n, b in produced.items() if n.rsplit(".", 1)[0] in {lin.prefix for lin in u.linears}}
        if digests:
            try:
                cache.mkdir(parents=True, exist_ok=True)
                (cache / f"{_replay_key(qz, units, assignments, u)}.json").write_text(json.dumps(digests))
            except OSError:
                pass
    return {**info, "replayed_tensors": replayed, "replay_cache_hits": cached_hits}


def audit(ckpt_dir: str | Path, manifest: Manifest, source_dirs: dict[str, Path],
          check_bytes: bool = True, verify_sources: bool = True, log=None, sample_secret: str = "",
          regenerated_dir: Path | None = None) -> AuditResult:
    """`sample_secret` makes the regenerated samples unpredictable; `regenerated_dir` holds tensors that
    foreign (contributed) quantizers regenerated in isolation, compared here instead of being executed."""
    res = AuditResult()
    ckpt_dir = Path(ckpt_dir)
    frozen_dir = Path(source_dirs["gittensor_nvfp4"])
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    frozen_cfg = json.loads((frozen_dir / "config.json").read_text())
    if "quantization_config" not in cfg:
        res.fail("config.json has no quantization_config (the loader rejects the directory)")
    strip = lambda c: {k: v for k, v in c.items() if k != "quantization_config"}  # noqa: E731
    if strip(cfg) != strip(frozen_cfg):
        res.fail("config.json differs from the shipped config outside quantization_config")
    units = Qwen38Arch.from_config(frozen_cfg).units()
    try:
        assignments = manifest.expand_assignments(units)
    except ValueError as e:
        res.fail(f"manifest: {e}")
        return res

    used_sources = {"base", "gittensor_nvfp4"} | {Q.get(a.quantizer).source_id for a in assignments.values()
                                                  if Q.get(a.quantizer).lineage == "attested"}
    if verify_sources:
        for s in sorted(used_sources):
            if s not in source_dirs:
                res.fail(f"source {s} not provided")
                return res
            try:
                require_verified(s, source_dirs[s], log=log)
            except LineageError as e:
                res.fail(str(e))
                return res

    handles = {s: SafeTensorsDir(source_dirs[s]) for s in used_sources}
    calib = None
    try:
        if "calibration" in source_dirs:
            try:
                calib = open_calibration(source_dirs["calibration"], verify_sources, log)
            except LineageError as e:
                res.fail(str(e))
                return res
        base, frozen = handles["base"], handles["gittensor_nvfp4"]
        ctx = Q.QuantContext(base=base, sources=handles, calibration=calib)
        with SafeTensorsDir(ckpt_dir) as ck:
            resolved = resolve_checkpoint(ck, units)
            res.runtime = {uid: r.label for uid, r in resolved.items()}
            expected: set[str] = set()
            searchable: set[str] = set()
            for u in units:
                a = assignments[u.id]
                got = resolved[u.id]
                for lin in u.linears:
                    searchable.add(lin.prefix)
                if got.precision is None:
                    res.fail(f"{u.id}: does not load ({got.note})")
                    continue
                if got.label != a.format:
                    res.fail(f"{u.id}: manifest selects {a.format}, the loader would execute {got.label}")
                    continue
                qz = Q.get(a.quantizer)
                for lin in u.linears:
                    if qz.lineage == "regenerable":
                        names = [lin.prefix + s for s, _, _ in _declared(qz, ctx, u, lin, a.format)]
                    else:
                        names = [lin.prefix + s for s, _, _, _ in qz.encode(ctx, u, lin, a.format)]
                    expected.update(names)
                    if check_bytes and qz.lineage in ("runtime", "attested"):
                        src = base if qz.lineage == "runtime" else handles[qz.source_id]
                        for n in names:
                            if ck.get(n) is None or src.get(n) is None or ck.sha256(n) != src.sha256(n):
                                res.fail(f"{n}: bytes differ from {'the BF16 base' if qz.lineage == 'runtime' else qz.source_id}")
                    if a.format in (NVFP4, FP8) and all(ck.get(n) is not None for n in names):
                        ratio = anomaly_ratio(ck, base, lin, a.format)
                        res.anomaly[lin.prefix] = round(ratio, 3)
                        if ratio > ANOMALY_REJECT_RATIO:
                            res.fail(f"{lin.prefix}: reconstruction error {ratio:.1f}x round-to-nearest -- substituted bytes")
                        elif ratio > ANOMALY_WARN_RATIO:
                            res.warnings.append(f"{lin.prefix}: reconstruction error {ratio:.2f}x round-to-nearest")
            for name, ref in frozen.tensors.items():
                prefix, _, _ = name.rpartition(".")
                if prefix in searchable:
                    continue
                expected.add(name)
                t = ck.get(name)
                if t is None:
                    res.fail(f"{name}: missing (present in the shipped checkpoint)")
                elif (t.dtype, t.shape) != (ref.dtype, ref.shape):
                    res.fail(f"{name}: {t.dtype}{list(t.shape)} != shipped {ref.dtype}{list(ref.shape)}")
                elif check_bytes and ck.sha256(name) != frozen.sha256(name):
                    res.fail(f"{name}: bytes differ from the shipped checkpoint")
            extra = sorted(set(ck.tensors) - expected)
            if extra:
                res.fail(f"{len(extra)} unexpected tensors, e.g. {extra[:5]}")

            if res.ok:
                seed = manifest.candidate_id(units) + (f":{sample_secret}" if sample_secret else "")
                for qname in sorted({a.quantizer for a in assignments.values()}):
                    qz = Q.get(qname)
                    if qz.lineage == "regenerable":
                        isolated = regenerated_dir if isinstance(qz, Q.Foreign) else None
                        res.lineage[qz.ref] = _check_regenerable(qz, units, assignments, ck, ctx, seed, res, isolated)
                    else:
                        res.lineage[qz.ref] = {"lineage": qz.lineage, "source": qz.source_id or "base",
                                               "units": sum(1 for a in assignments.values() if a.quantizer == qname),
                                               "bytes_checked": check_bytes}
    finally:
        for h in [*handles.values(), *([calib] if calib else [])]:
            h.close()
    return res
