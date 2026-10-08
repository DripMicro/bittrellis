"""HPC-02: Qwen3.6-35B-A3B as GGUF on one RTX 5090. Units, encoders, manifests, build and audit.

A manifest uses the HPC-01 syntax (default, ordered rules, modules); it expands to one (format, encoder)
assignment per searchable GGUF tensor. Searchable tensors are the template's BF16 weight matrices; every other
tensor (norms, router, recurrent constants) is copied from the template byte for byte.

Units (one GGUF tensor each; experts are stored per layer, so a format applies to all 256 experts at once):

    L{i}.exps.gate / .up / .down     routed experts          ffn_{gate,up,down}_exps
    L{i}.shexp.gate / .up / .down    shared expert           ffn_{gate,up,down}_shexp
    L{i}.gdn.qkv / .z / .out         recurrent (GDN) layers  attn_qkv / attn_gate / ssm_out
    L{i}.attn.q / .k / .v / .o       full-attention layers   attn_q / attn_k / attn_v / attn_output
    embed, lm_head                                            token_embd / output

Encoders turn a tensor's float32 rows into GGUF block bytes. `kq_rtn` (regenerable) is the built-in baseline;
`unsloth_ud` (attested) copies the bytes of the pinned unsloth UD-Q4_K_M GGUF where it stores that format.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import kquant

TRACK = "HPC-02"
FORMATS = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")
KINDS = {"ffn_gate_exps": "exps.gate", "ffn_up_exps": "exps.up", "ffn_down_exps": "exps.down",
         "ffn_gate_shexp": "shexp.gate", "ffn_up_shexp": "shexp.up", "ffn_down_shexp": "shexp.down",
         "attn_qkv": "gdn.qkv", "attn_gate": "gdn.z", "ssm_out": "gdn.out",
         "attn_q": "attn.q", "attn_k": "attn.k", "attn_v": "attn.v", "attn_output": "attn.o"}
TOP = {"token_embd.weight": "embed", "output.weight": "lm_head"}
# Recurrent per-head projections the template stores BF16 and llama.cpp releases store F32 (exact): not searchable.
F32_ALWAYS = ("ssm_alpha", "ssm_beta")


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    id: str            # L12.exps.down
    tensor: str        # blk.12.ffn_down_exps.weight
    layer: int | None
    kind: str          # exps.down
    cols: int          # row length (ne0)
    rows: int          # every row of every expert
    numel: int


def units(template_tensors) -> list[Unit]:
    out = []
    for t in template_tensors:
        if t.tensor_type.name != "BF16" or len(t.shape) < 2:
            continue
        cols, rows = int(t.shape[0]), int(np.prod([int(s) for s in t.shape[1:]]))
        if t.name in TOP:
            out.append(Unit(TOP[t.name], t.name, None, TOP[t.name], cols, rows, cols * rows))
            continue
        parts = t.name.split(".")
        if parts[0] != "blk" or parts[2] in F32_ALWAYS:
            continue
        kind = KINDS.get(parts[2])
        if kind is None:
            raise ValueError(f"template tensor {t.name}: no HPC-02 unit kind")
        i = int(parts[1])
        out.append(Unit(f"L{i}.{kind}", t.name, i, kind, cols, rows, cols * rows))
    order = {k: n for n, k in enumerate(KINDS.values())}

    def place(u: Unit):                   # embed, then layer by layer in a fixed kind order, then lm_head
        if u.layer is None:
            return (-1 if u.id == "embed" else 10**6, 0)
        return (u.layer, order[u.kind])
    return sorted(out, key=place)


def fixed_f32(template_tensors) -> list[str]:
    """Template BF16 tensors written as F32 (lossless) in every candidate."""
    return [t.name for t in template_tensors if t.tensor_type.name == "BF16" and t.name.split(".")[2:3]
            and t.name.split(".")[2] in F32_ALWAYS]


# ------------------------------------------------------------------ encoders


@dataclass
class EncodeContext:
    rows: np.ndarray                      # float32 [rows, cols], the template's BF16 values
    unit: Unit
    params: dict
    calibration: object | None = None     # SafeTensorsDir of the pinned HPC-02 statistics (moe_calibration)


@dataclass(frozen=True)
class Encoder:
    name: str
    version: int
    lineage: str                          # regenerable | attested
    fn: Callable[[EncodeContext, str], bytes] | None = None
    formats: tuple = FORMATS

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def _kq_rtn(ctx: EncodeContext, fmt: str) -> bytes:
    return kquant.RTN[fmt](ctx.rows)


ENCODERS: dict[str, Encoder] = {
    "kq_rtn": Encoder("kq_rtn", 1, "regenerable", _kq_rtn),
    "unsloth_ud": Encoder("unsloth_ud", 1, "attested"),
}


def register(enc: Encoder) -> Encoder:
    if enc.name in ENCODERS and ENCODERS[enc.name] is not enc:
        raise ValueError(f"encoder {enc.name!r} already registered")
    ENCODERS[enc.name] = enc
    return enc


# ------------------------------------------------------------------ manifests


@dataclass(frozen=True)
class Assignment:
    format: str
    encoder: str
    params: tuple = ()

    def key(self) -> str:
        s = f"{self.format}@{ENCODERS[self.encoder].ref}"
        if self.params:
            s += "+" + hashlib.sha256(json.dumps(dict(self.params), sort_keys=True).encode()).hexdigest()[:12]
        return s


def _layers(spec, n: int) -> set[int]:
    if spec is None:
        return set(range(n))
    out: set[int] = set()
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def load_manifest(path: Path) -> dict:
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict) or d.get("track") != TRACK:
        raise ManifestError(f"{path}: not an {TRACK} manifest")
    return d


def expand(d: dict, us: list[Unit], ud_formats: dict[str, str] | None = None) -> dict[str, Assignment]:
    """One assignment per unit. `ud_formats` ({tensor: format} of the pinned UD GGUF) limits `unsloth_ud`."""
    n_layers = 1 + max(u.layer for u in us if u.layer is not None)
    default_enc = (d.get("encoders") or {})

    def check(u: Unit, fmt, enc, params, where) -> Assignment:
        if fmt not in FORMATS:
            raise ManifestError(f"{where}: unknown format {fmt!r} (one of {', '.join(FORMATS)})")
        enc = enc or default_enc.get(fmt) or "kq_rtn"
        if enc not in ENCODERS:
            raise ManifestError(f"{where}: unknown encoder {enc!r} (known: {sorted(ENCODERS)})")
        if fmt not in ENCODERS[enc].formats:
            raise ManifestError(f"{where}: {enc} cannot produce {fmt}")
        if u.cols % kquant.BLOCK[fmt][1]:
            raise ManifestError(f"{where}: {u.id} rows of {u.cols} do not fit {fmt} blocks")
        if enc == "unsloth_ud" and (ud_formats or {}).get(u.tensor) != fmt:
            raise ManifestError(f"{where}: unsloth_ud stores {u.id} as {(ud_formats or {}).get(u.tensor)}, not {fmt}")
        return Assignment(fmt, enc, tuple(sorted((params or {}).items())))

    if "default" not in d:
        raise ManifestError("manifest needs a default format")
    # Collect each unit's final (format, encoder, params) first, then validate only that: a default or an
    # early rule that a later rule overrides never has to be valid on its own.
    raw = {u.id: (d["default"], None, None, "default") for u in us}
    by_id = {u.id: u for u in us}
    for i, rule in enumerate(d.get("rules") or []):
        extra = set(rule) - {"match", "layers", "format", "encoder", "params", "note"}
        if extra or "match" not in rule or "format" not in rule:
            raise ManifestError(f"rule {i}: needs match and format; unknown keys {sorted(extra)}")
        layers = _layers(rule.get("layers"), n_layers)
        hits = [u for u in us if fnmatch.fnmatchcase(u.id, str(rule["match"]))
                and (u.layer is None or u.layer in layers or rule.get("layers") is None)]
        if not hits:
            raise ManifestError(f"rule {i} ({rule['match']!r}, layers={rule.get('layers')}) matches no unit")
        for u in hits:
            raw[u.id] = (rule["format"], rule.get("encoder"), rule.get("params"), f"rule {i}")
    for uid, spec in (d.get("modules") or {}).items():
        if uid not in by_id:
            raise ManifestError(f"modules: unknown unit {uid!r}")
        spec = spec if isinstance(spec, dict) else {"format": spec}
        raw[uid] = (spec.get("format"), spec.get("encoder"), spec.get("params"), f"modules.{uid}")
    out = {uid: check(by_id[uid], *raw[uid]) for uid in raw}
    return out


def candidate_id(assignments: dict[str, Assignment]) -> str:
    payload = json.dumps([TRACK, sorted((u, a.key()) for u, a in assignments.items())])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ sources


def template_paths(template_dir: Path) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(Path(template_dir) / "**" / "*.gguf"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no GGUF files in {template_dir}")
    return paths


def read_template(template_dir: Path):
    from gguf import GGUFReader

    return [t for p in template_paths(template_dir) for t in GGUFReader(p).tensors]


def ud_formats(ud_gguf: Path) -> dict[str, str]:
    from gguf import GGUFReader

    return {t.name: t.tensor_type.name for t in GGUFReader(ud_gguf).tensors}


# ------------------------------------------------------------------ build


def _assign(us: list[Unit], a: dict[str, Assignment], template, ud: Path | None, calibration) -> dict:
    """{tensor: (format, fn(fmt, rows) -> bytes)} for gguf_build: every unit plus the fixed F32 tensors."""
    from gguf import GGUFReader

    ud_tensors = {t.name: t for t in GGUFReader(ud).tensors} if ud and any(x.encoder == "unsloth_ud" for x in a.values()) else {}
    out = {}
    for u in us:
        x = a[u.id]
        enc = ENCODERS[x.encoder]
        if enc.lineage == "attested":
            out[u.tensor] = (x.format, lambda fmt, rows, t=ud_tensors[u.tensor]: np.asarray(t.data).tobytes())
        else:
            out[u.tensor] = (x.format, lambda fmt, rows, u=u, x=x, enc=enc: enc.fn(
                EncodeContext(rows, u, dict(x.params), calibration), fmt))
    for name in fixed_f32(template):
        out[name] = ("F32", lambda fmt, rows: kquant.to_f32(rows))
    return out


def build(manifest: Path, template_dir: Path, out: Path, ud: Path | None = None, calibration_dir: Path | None = None,
          jobs: int | None = None, log=print) -> dict:
    from .gguf_build import build_gguf
    from .safetensors_io import SafeTensorsDir

    d = load_manifest(manifest)
    template = read_template(template_dir)
    us = units(template)
    a = expand(d, us, ud_formats(ud) if ud else None)
    cid = candidate_id(a)
    calib = SafeTensorsDir(calibration_dir) if calibration_dir else None
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = build_gguf(template_paths(template_dir), out, _assign(us, a, template, ud, calib), jobs=jobs, log=log)
    summary = {}
    for x in a.values():
        summary[x.key()] = summary.get(x.key(), 0) + 1
    record = {"candidate_id": cid, "name": d.get("name"), "track": TRACK, "summary": summary, **rec}
    Path(str(out) + ".build.json").write_text(json.dumps(record, indent=2) + "\n")
    log(f"[hpc02] built {d.get('name')} ({cid}): {rec['bytes'] / 1e9:.2f} GB")
    return record


# ------------------------------------------------------------------ audit

SAMPLES_PER_ENCODER = 6


def audit(gguf_path: Path, manifest: Path, template_dir: Path, ud: Path | None = None, calibration_dir: Path | None = None,
          secret: str = "", log=print) -> dict:
    """The HPC-01 rules for a GGUF candidate: same metadata and tensors as the template, frozen tensors byte for
    byte, each searchable tensor in its manifest format, attested bytes identical to their source, sampled
    regenerable tensors rebuilt byte for byte (samples from the candidate id and the evaluator's secret)."""
    from gguf import GGUFReader

    from .gguf_build import SKIP_KEYS, _bf16_rows
    from .safetensors_io import SafeTensorsDir

    errors: list[str] = []
    d = load_manifest(manifest)
    tpaths = template_paths(template_dir)
    treaders = [GGUFReader(p) for p in tpaths]
    template = [t for r in treaders for t in r.tensors]
    us = units(template)
    a = expand(d, us, ud_formats(ud) if ud else None)
    cid = candidate_id(a)
    ck = GGUFReader(gguf_path)
    tf = {f.name: f for f in treaders[0].fields.values() if not f.name.startswith(SKIP_KEYS)}
    cf = {f.name: f for f in ck.fields.values() if not f.name.startswith(SKIP_KEYS)}
    if set(tf) != set(cf):
        errors.append(f"metadata keys differ from the template: {sorted(set(tf) ^ set(cf))[:5]}")
    for k in sorted(set(tf) & set(cf)):
        if tf[k].contents() != cf[k].contents():
            errors.append(f"metadata {k} differs from the template")
    T = {t.name: t for t in template}
    C = {t.name: t for t in ck.tensors}
    if set(T) != set(C):
        errors.append(f"tensor set differs from the template: {sorted(set(T) ^ set(C))[:5]}")
    unit_of = {u.tensor: u for u in us}
    f32 = set(fixed_f32(template))
    for name, t in T.items():
        c = C.get(name)
        if c is None:
            continue
        want_shape = [int(s) for s in t.shape]
        if len(want_shape) == 2 and want_shape[1] == 1:
            want_shape = want_shape[:1]
        if [int(s) for s in c.shape] != want_shape:
            errors.append(f"{name}: shape {list(map(int, c.shape))} != template {want_shape}")
        if name in unit_of:
            fmt = a[unit_of[name].id].format
            if c.tensor_type.name != fmt:
                errors.append(f"{name}: stored {c.tensor_type.name}, manifest selects {fmt}")
        elif name in f32:
            if c.tensor_type.name != "F32" or np.asarray(c.data).tobytes() != kquant.to_f32(_bf16_rows(t)):
                errors.append(f"{name}: not the template's values as F32")
        elif np.asarray(c.data).tobytes() != np.asarray(t.data).tobytes():
            errors.append(f"{name}: bytes differ from the template")
    lineage: dict[str, dict] = {}
    udt = {t.name: t for t in GGUFReader(ud).tensors} if ud else {}
    calib = SafeTensorsDir(calibration_dir) if calibration_dir else None
    for enc_name in sorted({x.encoder for x in a.values()}):
        enc = ENCODERS[enc_name]
        mine = [u for u in us if a[u.id].encoder == enc_name]
        if enc.lineage == "attested":
            bad = [u.id for u in mine if u.tensor in C and np.asarray(C[u.tensor].data).tobytes() != np.asarray(udt[u.tensor].data).tobytes()]
            errors += [f"{uid}: bytes differ from unsloth_ud" for uid in bad]
            lineage[enc.ref] = {"lineage": "attested", "units": len(mine), "checked": len(mine)}
            continue
        ranked = sorted(mine, key=lambda u: hashlib.sha256(f"{cid}:{secret}:{u.id}".encode()).hexdigest())
        sample = {mine[0].id, mine[-1].id} | {u.id for u in ranked[:SAMPLES_PER_ENCODER - 2]}
        for u in [u for u in mine if u.id in sample]:
            x = a[u.id]
            want = enc.fn(EncodeContext(_bf16_rows(T[u.tensor]), u, dict(x.params), calib), x.format)
            if u.tensor in C and np.asarray(C[u.tensor].data).tobytes() != want:
                errors.append(f"{u.id}: does not match a regeneration by {enc.ref}")
        lineage[enc.ref] = {"lineage": "regenerable", "units": len(mine), "sampled": sorted(sample)}
    res = {"ok": not errors, "errors": errors[:200], "n_errors": len(errors), "candidate_id": cid, "lineage": lineage}
    log(f"[hpc02] audit {'PASS' if res['ok'] else 'FAIL'} ({cid}): {len(errors)} errors")
    return res
