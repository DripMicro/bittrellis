"""Behavioural fingerprints: what a quantizer produces, and what a checkpoint's bytes look like.

The copycat guard judges quantizers by their output, not their source text. Renaming a class,
reordering functions or rewriting loops changes the code but not the bytes an encoder emits, and
the audit already requires every regenerable quantizer to be deterministic.

Two fingerprints:

* **probe**   each regenerable quantizer encodes every unit it supports on a tiny synthetic model
              whose weights come from an evaluator-chosen seed. Cheap, CPU only, before any build.
* **sketch**  sampled bytes of a built checkpoint's tensors at offsets derived from an evaluator
              secret. It fingerprints the bytes that were actually measured, so a quantizer that
              behaves differently on probes than on real weights is still compared on real ones.

Similarity is the fraction of identical bytes over tensors both sides produced.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from . import quantizers as Q
from .model.qwen38 import Qwen38Arch
from .safetensors_io import SafeTensorsDir

SKETCH_BYTES = 1 << 16


def probe(names: list[str] | None, seed: int) -> dict[str, bytes]:
    """{"quantizer@vN|FORMAT|unit|suffix": bytes} for every regenerable quantizer (or `names`)."""
    from .synthetic import TINY_TEXT, make_tiny, make_tiny_calibration

    selected = [q for q in Q.REGISTRY.values() if q.lineage == "regenerable" and (names is None or q.name in names)]
    out: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        base, baseline, ct = make_tiny(Path(tmp), seed=seed)
        units = Qwen38Arch.from_config({"text_config": TINY_TEXT}).units()
        handles = {"base": SafeTensorsDir(base), "gittensor_nvfp4": SafeTensorsDir(baseline), "unsloth_nvfp4": SafeTensorsDir(ct)}
        calib = SafeTensorsDir(make_tiny_calibration(Path(tmp), seed=seed))
        try:
            for qz in selected:
                for fmt in qz.formats:
                    ctx = Q.QuantContext(base=handles["base"], sources=handles, calibration=calib)
                    qz.begin(ctx)
                    for u in units:
                        if not qz.supports(u, fmt) or qz.available(ctx, u, fmt) is not None:
                            continue
                        for lin in u.linears:
                            ctx.params = {}
                            for suffix, _dtype, _shape, data in qz.encode(ctx, u, lin, fmt):
                                out[f"{qz.ref}|{fmt}|{lin.prefix}|{suffix}"] = _bytes(data)
        finally:
            for h in [*handles.values(), calib]:
                h.close()
    return out


def _bytes(data) -> bytes:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).tobytes()
    return bytes(data)


def save_probe(fp: dict[str, bytes], path: Path) -> None:
    np.savez_compressed(path, **{k: np.frombuffer(v, dtype=np.uint8) for k, v in fp.items()})


def load_probe(path: Path) -> dict[str, bytes]:
    with np.load(path) as z:
        return {k: z[k].tobytes() for k in z.files}


def by_quantizer(fp: dict[str, bytes]) -> dict[str, dict[str, bytes]]:
    """{"name@vN": {"FORMAT|tensor|suffix": bytes}}."""
    out: dict[str, dict[str, bytes]] = {}
    for k, v in fp.items():
        ref, rest = k.split("|", 1)
        out.setdefault(ref, {})[rest] = v
    return out


def similarity(a: dict[str, bytes], b: dict[str, bytes]) -> tuple[float, int]:
    """(fraction of identical bytes, bytes compared) over keys present in both with equal length."""
    same = total = 0
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if len(x) != len(y) or not x:
            continue
        xa, ya = np.frombuffer(x, dtype=np.uint8), np.frombuffer(y, dtype=np.uint8)
        same += int((xa == ya).sum())
        total += len(x)
    return (same / total if total else 0.0), total


CODES = (".weight", ".weight_packed")   # NVFP4 packed codes (ModelOpt / compressed-tensors) or FP8 values
SCALES = (".weight_scale",)


def normalized(name: str) -> str | None:
    """`prefix|codes` or `prefix|scales` for tensors comparable across layouts, else None."""
    for suf in CODES:
        if name.endswith(suf):
            return name[: -len(suf)] + "|codes"
    for suf in SCALES:
        if name.endswith(suf):
            return name[: -len(suf)] + "|scales"
    return None


def sketch_bytes(name: str, data, secret: str, nbytes: int = SKETCH_BYTES) -> bytes:
    """`nbytes` bytes of `data` at offsets derived from (secret, name), plus the length."""
    buf = np.frombuffer(data, dtype=np.uint8)
    seed = int.from_bytes(hashlib.sha256(f"{secret}:{name}".encode()).digest()[:8], "little")
    if len(buf) > nbytes:
        idx = np.sort(np.random.default_rng(seed).integers(0, len(buf), size=nbytes))
        out = buf[idx].tobytes()
    else:
        out = buf.tobytes()
    return out + len(buf).to_bytes(8, "little")


def sketch(checkpoint: Path, prefixes: list[str], secret: str, nbytes: int = SKETCH_BYTES) -> dict[str, bytes]:
    """Sketches of the codes and scales a checkpoint stores for `prefixes`, keyed by normalized name."""
    out: dict[str, bytes] = {}
    with SafeTensorsDir(checkpoint) as ck:
        for prefix in prefixes:
            for suf in CODES + SCALES:
                if ck.get(prefix + suf) is not None:
                    key = normalized(prefix + suf)
                    out[key] = sketch_bytes(key, ck.raw(prefix + suf), secret, nbytes)
    return out


def reference_sketches(source_dirs: dict[str, Path], units: list, fmt_by_unit: dict[str, str], secret: str,
                       nbytes: int = SKETCH_BYTES) -> dict[str, dict[str, bytes]]:
    """{quantizer ref: sketches} of what every quantizer on this code base stores for `units` at their formats.

    Regenerable quantizers are run on the real BF16 weights; attested ones are read from their pinned
    sources. Compared with a submission's sketches, this catches an encoder that reproduces an existing
    one on real weights, including one that splices attested bytes.
    """
    handles = {k: SafeTensorsDir(v) for k, v in source_dirs.items() if k != "calibration" or Path(v).is_dir()}
    out: dict[str, dict[str, bytes]] = {}
    try:
        for qz in Q.REGISTRY.values():
            if qz.lineage == "runtime":
                continue
            ctx = Q.QuantContext(base=handles["base"], sources=handles, calibration=handles.get("calibration"))
            began = False
            for u in units:
                fmt = fmt_by_unit[u.id]
                if not qz.supports(u, fmt) or qz.available(ctx, u, fmt) is not None:
                    continue
                if qz.lineage == "regenerable" and qz.replay_mode != "independent":
                    continue  # a sequential encoder's output depends on the whole pipeline before the unit
                if not began:
                    qz.begin(ctx)
                    began = True
                for lin in u.linears:
                    ctx.params = {}
                    for suffix, _dtype, _shape, data in qz.encode(ctx, u, lin, fmt):
                        key = normalized(lin.prefix + suffix)
                        if key:
                            out.setdefault(qz.ref, {})[key] = sketch_bytes(key, _bytes(data), secret, nbytes)
    finally:
        for h in handles.values():
            h.close()
    return out


def save_sketches(sk: dict[str, bytes], path: Path) -> None:
    save_probe(sk, path)


def load_sketches(path: Path) -> dict[str, bytes]:
    return load_probe(path)
