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


def register_foreign(refs: list[str]) -> list[Encoder]:
    """Stand-ins for contributed encoders this (trusted) process must not execute: name@vN, formats unknown."""
    added = []
    for ref in refs:
        name, _, v = ref.partition("@v")
        if name not in ENCODERS:
            added.append(register(Encoder(name, int(v), "regenerable", None)))
    return added


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
    return candidate_id_keys({u: a.key() for u, a in assignments.items()})


def candidate_id_keys(keys: dict[str, str]) -> str:
    payload = json.dumps([TRACK, sorted(keys.items())])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def predicted_bytes(us: list[Unit], a: dict[str, Assignment]) -> int:
    """Stored bytes of the searchable tensors (the rest is the same for every candidate)."""
    return sum(u.rows * kquant.row_bytes(a[u.id].format, u.cols) for u in us)


def load_contributed() -> list[Encoder]:
    """Import every module in bittrellis/hpc02_encoders/; each registers its encoders on import."""
    import importlib
    import pkgutil

    from . import hpc02_encoders

    before = set(ENCODERS)
    for m in pkgutil.iter_modules(hpc02_encoders.__path__):
        importlib.import_module(f"{hpc02_encoders.__name__}.{m.name}")
    return [ENCODERS[n] for n in sorted(set(ENCODERS) - before)]


def probe(seed: int, names: list[str] | None = None) -> dict[str, bytes]:
    """{"enc@vN|FORMAT|probe": bytes}: every regenerable encoder on seeded synthetic rows, per format. Encoders
    are compared by these bytes (duplicate and determinism screens), never by their source text."""
    rng = np.random.default_rng(seed)
    rows = (rng.standard_normal((64, 1024)) * rng.uniform(0.005, 0.05, (64, 1)) * rng.uniform(0.3, 3, 1024)).astype(np.float32)
    rows[rng.integers(0, 64, 8), rng.integers(0, 1024, 8)] *= 20.0          # a few outliers, as real weights have
    unit = Unit("probe", "probe.weight", 0, "exps.gate", 1024, 64, 64 * 1024)
    out = {}
    for enc in ENCODERS.values():
        if enc.lineage != "regenerable" or (names and enc.name not in names):
            continue
        for fmt in enc.formats:
            out[f"{enc.ref}|{fmt}|probe"] = enc.fn(EncodeContext(rows.copy(), unit, {}, None), fmt)
    return out


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


# The unit list and UD's formats, committed so a recipe can be checked without the 69 GB template; the evaluator
# checks it against the pinned template in verify-sources.
UNITS_FILE = Path(__file__).resolve().parents[1] / "configs" / "hpc02_units.json"


def units_doc(us: list[Unit], udf: dict[str, str]) -> dict:
    return {"units": [[u.id, u.tensor, u.layer, u.kind, u.cols, u.rows] for u in us],
            "ud_formats": {u.tensor: udf.get(u.tensor) for u in us}}


def read_units_file(path: Path = UNITS_FILE) -> tuple[list[Unit], dict[str, str]]:
    d = json.loads(Path(path).read_text())
    us = [Unit(i, t, layer, k, c, r, c * r) for i, t, layer, k, c, r in d["units"]]
    return us, d["ud_formats"]


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
    from .safetensors_io import file_sha256

    record = {"candidate_id": cid, "name": d.get("name"), "track": TRACK, "summary": summary, **rec,
              "sha256": file_sha256(out)}
    Path(str(out) + ".build.json").write_text(json.dumps(record, indent=2) + "\n")
    Path(str(out) + ".manifest.yaml").write_text(Path(manifest).read_text())
    log(f"[hpc02] built {d.get('name')} ({cid}): {rec['bytes'] / 1e9:.2f} GB")
    return record


def samples(us: list[Unit], a: dict[str, Assignment], enc_name: str, seed: str) -> list[Unit]:
    """Units of an encoder the audit rebuilds: first, last and secretly chosen ones (seed = id + secret)."""
    mine = [u for u in us if a[u.id].encoder == enc_name]
    if not mine:
        return []
    ranked = sorted(mine, key=lambda u: hashlib.sha256(f"{seed}:{u.id}".encode()).hexdigest())
    pick = {mine[0].id, mine[-1].id} | {u.id for u in ranked[:SAMPLES_PER_ENCODER - 2]}
    return [u for u in mine if u.id in pick]


def regenerate(manifest: Path, template_dir: Path, targets: list[str], out_dir: Path, ud: Path | None = None,
               calibration_dir: Path | None = None) -> int:
    """Contributed-code side of an isolated audit: encode `targets` from the template only, never the candidate."""
    from .gguf_build import _bf16_rows
    from .safetensors_io import SafeTensorsDir

    template = read_template(template_dir)
    us = units(template)
    a = expand(load_manifest(manifest), us, ud_formats(ud) if ud else None)
    T = {t.name: t for t in template}
    calib = SafeTensorsDir(calibration_dir) if calibration_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {u.id: u for u in us}
    for uid in targets:
        u, x = by_id[uid], a[uid]
        (out_dir / f"{uid}.bin").write_bytes(ENCODERS[x.encoder].fn(EncodeContext(_bf16_rows(T[u.tensor]), u, dict(x.params), calib), x.format))
    return len(targets)


# ------------------------------------------------------------------ sources

SOURCES = {"template": "qwen36_gguf", "calibration": "hpc02_calibration"}


def verify_sources(template_dir: Path, ud: Path | None, calibration_dir: Path | None, log=print) -> list[str]:
    """Every pinned file this candidate's bytes come from, hashed against configs/sources.lock.json (cached by
    size and mtime next to the files, as HPC-01's sources are)."""
    from .lineage import load_lock, verify_source

    lock = load_lock()["sources"]
    errors: list[str] = []
    gguf_dir = Path(template_dir).parent              # .../Qwen3.6-35B-A3B-GGUF holds BF16/ and the UD file
    checks = [(SOURCES["template"], gguf_dir, [n for n in lock[SOURCES["template"]]["files"]
                                               if n.startswith("BF16/") or (ud and n == Path(ud).name)])]
    if calibration_dir:
        checks.append((SOURCES["calibration"], Path(calibration_dir), None))
    for sid, d, files in checks:
        res = verify_source(sid, d, files, log=log)
        errors += [f"source {sid}: {e}" for e in res.errors]
    return errors


# ------------------------------------------------------------------ audit

SAMPLES_PER_ENCODER = 6


def audit(gguf_path: Path, manifest: Path, template_dir: Path, ud: Path | None = None, calibration_dir: Path | None = None,
          secret: str = "", regenerated_dir: Path | None = None, verify: bool = True, log=print) -> dict:
    """The HPC-01 rules for a GGUF candidate: same metadata and tensors as the template, frozen tensors byte for
    byte, each searchable tensor in its manifest format, attested bytes identical to their source, sampled
    regenerable tensors rebuilt byte for byte (samples from the candidate id and the evaluator's secret)."""
    from gguf import GGUFReader

    from .gguf_build import SKIP_KEYS, _bf16_rows
    from .safetensors_io import SafeTensorsDir

    errors: list[str] = verify_sources(template_dir, ud, calibration_dir, log=log) if verify else []
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
        sample = samples(us, a, enc_name, f"{cid}:{secret}")
        isolated = enc.fn is None          # contributed code: its sandboxed regeneration is compared, never run here
        for u in sample:
            if isolated:
                f = Path(regenerated_dir or "/nonexistent") / f"{u.id}.bin"
                want = f.read_bytes() if f.exists() else None
                if want is None:
                    errors.append(f"{u.id}: no regeneration by {enc.ref} was provided")
                    continue
            else:
                x = a[u.id]
                want = enc.fn(EncodeContext(_bf16_rows(T[u.tensor]), u, dict(x.params), calib), x.format)
            if u.tensor in C and np.asarray(C[u.tensor].data).tobytes() != want:
                errors.append(f"{u.id}: does not match a regeneration by {enc.ref}")
        lineage[enc.ref] = {"lineage": "regenerable", "units": len(mine), "sampled": sorted(u.id for u in sample),
                            **({"replay_mode": "isolated"} if isolated else {})}
    res = {"ok": not errors, "errors": errors[:200], "n_errors": len(errors), "candidate_id": cid, "lineage": lineage}
    log(f"[hpc02] audit {'PASS' if res['ok'] else 'FAIL'} ({cid}): {len(errors)} errors")
    return res
