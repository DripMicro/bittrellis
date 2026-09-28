"""bittrellis — command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from . import __version__
from . import quantizers as Q
from .manifest import Manifest, ManifestError, distance, summarize
from .model.qwen38 import Qwen38Arch
from .precision import SPACE, execution, stored_bytes, stored_for, weight_bytes
from .track import REPO_ROOT, load_track

ENV = os.environ.get
DEFAULTS = {
    "base": ENV("BITTRELLIS_BASE", str(REPO_ROOT / "models/Qwen3.8-27B")),
    "shipped": ENV("BITTRELLIS_SHIPPED", str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090")),
    "unsloth": ENV("BITTRELLIS_UNSLOTH", str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-unsloth")),
    "sparkinfer": ENV("BITTRELLIS_SPARKINFER", str(REPO_ROOT / "third_party/sparkinfer")),
    "llamacpp": ENV("BITTRELLIS_LLAMACPP", str(REPO_ROOT / "third_party/llama.cpp")),
    "corpus": str(REPO_ROOT / "data/corpus/hpc01-public-v2.json"),
    "reference": ENV("BITTRELLIS_REFERENCE", str(REPO_ROOT / "data/reference/hpc01-public-v2-k256")),
}
SOURCE_ARGS = {"base": "base", "gittensor_nvfp4": "shipped", "unsloth_nvfp4": "unsloth"}
GIB = 1024**3


def _source_dirs(args) -> dict[str, Path]:
    return {sid: Path(getattr(args, arg)) for sid, arg in SOURCE_ARGS.items() if getattr(args, arg, None)}


def _units(args=None):
    cfg = Path(getattr(args, "shipped", DEFAULTS["shipped"])) / "config.json"
    return (Qwen38Arch.from_config(cfg) if cfg.exists() else Qwen38Arch()).units()


def _json(obj) -> None:
    print(json.dumps(obj, indent=2))


# ------------------------------------------------------------------ inspection


def cmd_track(args) -> int:
    print(yaml.safe_dump(load_track(args.track).data, sort_keys=False), end="")
    return 0


def cmd_quantizers(args) -> int:
    for q in Q.REGISTRY.values():
        print(f"  {q.ref:14s} {','.join(q.formats):12s} {q.lineage:12s} {q.replay_mode:12s} {q.source_id or '':16s} {q.description}")
    return 0


def cmd_inventory(args) -> int:
    rows = []
    for u in _units(args):
        legal = []
        for fmt in SPACE[u.kind]:
            for q in Q.REGISTRY.values():
                if q.supports(u, fmt):
                    legal.append({"format": fmt, "quantizer": q.ref, "lineage": q.lineage,
                                  "source": q.source_id, "execution": execution(u.kind, fmt)})
        rows.append({
            "unit": u.id, "kind": u.kind, "role": u.role, "layer": u.layer, "params": u.numel,
            "tensors": [{"name": lin.prefix, "rows": lin.rows, "cols": lin.cols} for lin in u.linears],
            "legal_assignments": legal,
            "stored_bytes": {f: sum(stored_bytes(lin, stored_for(f)) for lin in u.linears) for f in SPACE[u.kind]},
            "decode_weight_bytes": {f: weight_bytes(u, f) for f in SPACE[u.kind]},
            "loader_constraints": {
                "gdn": "NVFP4 native; FP8 native only with one BF16 scale per row; anything else is fit to Q4_K",
                "attn": "NVFP4 native per tensor; anything else is fit to Q4_K",
                "mlp": "gate/up/down share one decision keyed on gate_proj; NVFP4 gate requires NVFP4 up/down",
                "lm_head": "batch-1 decode always executes a Q4_K fit of the stored head",
            }[u.kind],
        })
    doc = {"track": args.track, "model": "Qwen3.8-27B", "units": rows,
           "totals": {"units": len(rows), "params": sum(r["params"] for r in rows),
                      "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in SPACE}}}
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.out}: {doc['totals']}")
    else:
        _json(doc["totals"])
    return 0


def cmd_manifest(args) -> int:
    track = load_track(args.track)
    units = _units(args)
    by_id = {u.id: u for u in units}
    seen: dict[str, tuple[str, dict]] = {}
    if args.against:
        for p in sorted(Path(args.against).glob("*.yaml")):
            try:
                m = Manifest.load(p)
                seen[str(p)] = (m.candidate_id(units), m.expand_assignments(units))
            except (ManifestError, yaml.YAMLError):
                continue
    status = 0
    for path in args.manifests:
        try:
            m = Manifest.load(path)
            if m.track != track.id:
                raise ManifestError(f"manifest track {m.track} != {track.id}")
            asg = m.expand_assignments(units)
        except (ManifestError, OSError, yaml.YAMLError) as e:
            print(f"✗ {path}: {e}")
            status = 1
            continue
        cid = m.candidate_id(units)
        est = sum(weight_bytes(by_id[k], a.format) for k, a in asg.items()) / GIB
        print(f"✓ {path}: {m.name}  id={cid}")
        for kind, counts in summarize(asg, units).items():
            print(f"    {kind:8s} " + "  ".join(f"{p}×{n}" for p, n in sorted(counts.items())))
        print(f"    decode weights ≈ {est:.2f} GiB (estimate; peak GPU memory is measured)")
        for other, (ocid, oasg) in seen.items():
            if Path(other).resolve() == Path(path).resolve():
                continue
            d = distance(asg, oasg)
            if d == 0:
                print(f"    ✗ duplicate of {other} (same candidate id {ocid})")
                status = 1
            elif d <= args.min_distance:
                print(f"    ! near-duplicate of {other}: only {d} unit(s) differ")
        if args.expand:
            _json(m.to_dict(units)["expanded"])
        if args.ids_out:
            Path(args.ids_out).write_text(json.dumps({"name": m.name, "id": cid, "keys": {k: a.key() for k, a in asg.items()}},
                                                     sort_keys=True) + "\n")
    return status


def cmd_describe(args) -> int:
    from collections import Counter

    from .validate import describe

    labels = describe(args.checkpoint)
    counts = Counter((uid.split(".")[1] if uid.startswith("L") else uid, lab) for uid, lab in labels.items())
    for (kind, lab), n in sorted(counts.items()):
        print(f"  {kind:8s} {lab:12s} ×{n}")
    if args.out:
        Path(args.out).write_text(json.dumps(labels, indent=1) + "\n")
    return 0


def cmd_verify_sources(args) -> int:
    from .lineage import verify_source

    dirs = _source_dirs(args)
    ok = True
    for sid in args.source or sorted(dirs):
        res = verify_source(sid, dirs[sid], log=print)
        ok &= res.ok
        print(f"{'✓' if res.ok else '✗'} {sid}: {res.checked} hashed, {res.cached} cached" + (f"; {res.errors}" if res.errors else ""))
    return 0 if ok else 1


# ------------------------------------------------------------------ build + audit


def cmd_build(args) -> int:
    from .build import build

    track = load_track(args.track)
    m = Manifest.load(args.manifest)
    out = Path(args.out) if args.out else REPO_ROOT / "models/candidates" / f"{m.name}-{m.candidate_id(_units(args))}"
    build(m, track, _source_dirs(args), out, verify=not args.no_verify)
    print(out)
    return 0


def _secret(args) -> str:
    path = getattr(args, "sample_secret_file", None)
    return Path(path).read_text().strip() if path else ""


def _register_foreign(ckpt: Path) -> list[str]:
    """Stub every quantizer the checkpoint's config names that this code base does not have (never executed)."""
    from . import quantizers as Q

    layers = json.loads((Path(ckpt) / "config.json").read_text()).get("quantization_config", {}).get("quantized_layers", {})
    refs: dict[str, set[str]] = {}
    for layer in layers.values():
        ref = layer.get("quantizer")
        if ref and ref.partition("@v")[0] not in Q.REGISTRY:
            refs.setdefault(ref, set()).add("FP8" if layer["quant_algo"].startswith("FP8") else "NVFP4")
    return [q.ref for q in Q.register_foreign({r: tuple(sorted(f)) for r, f in refs.items()})]


def cmd_regenerate(args) -> int:
    """Contributed-code side of an isolated audit: regenerate units from the sources only, never the checkpoint."""
    from . import quantizers as Q
    from .build import open_context
    from .validate import regenerate

    m = Manifest.load(args.manifest)
    units = _units(args)
    assignments = m.expand_assignments(units)
    targets = {u for u in args.units.split(",") if u}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx, handles = open_context(assignments, _source_dirs(args), verify=False, log=print)
    try:
        for qname in sorted({assignments[u].quantizer for u in targets}):
            qz = Q.get(qname)
            if qz.lineage != "regenerable":
                continue
            mine = {u for u in targets if assignments[u].quantizer == qname}
            for name, blob in regenerate(qz, units, assignments, ctx, mine).items():
                (out / f"{name}.bin").write_bytes(blob)
    finally:
        for h in handles:
            h.close()
    print(f"regenerated {len(list(out.glob('*.bin')))} tensors for {len(targets)} units")
    return 0


def cmd_audit(args) -> int:
    from .validate import audit

    if args.regenerated:
        print("foreign quantizers (compared, not executed):", ", ".join(_register_foreign(Path(args.checkpoint))) or "none")
    m = Manifest.load(args.manifest or Path(args.checkpoint) / "precision_manifest.yaml")
    res = audit(args.checkpoint, m, _source_dirs(args), check_bytes=not args.fast, verify_sources=not args.fast, log=print,
                sample_secret=_secret(args), regenerated_dir=Path(args.regenerated) if args.regenerated else None)
    if args.out:
        build_rec = Path(args.checkpoint) / "bittrellis_build.json"
        files = json.loads(build_rec.read_text())["files"] if build_rec.exists() else None
        Path(args.out).write_text(json.dumps({**res.to_dict(), "fast": args.fast, "checkpoint_files": files}, indent=2) + "\n")
    for e in res.errors[:30]:
        print(f"  ✗ {e}")
    for w in res.warnings[:10]:
        print(f"  ! {w}")
    print("AUDIT PASS" if res.ok else f"AUDIT FAIL ({len(res.errors)} errors)")
    return 0 if res.ok else 1


# ------------------------------------------------------------------ evaluation


def cmd_corpus(args) -> int:
    from .eval.corpus import build_corpus, load_corpus, save_corpus

    if args.action == "build":
        c = build_corpus(Path(args.cache), split=args.split)
        save_corpus(c, Path(args.out))
        print(f"{args.out}: {len(c['streams'])} streams, sha256 {c['sha256']}")
    else:
        c = load_corpus(Path(args.out))
        print(f"{args.out}: OK sha256 {c['sha256']}")
    return 0


def cmd_reference(args) -> int:
    from .eval.corpus import load_corpus
    from .eval.reference import build_reference
    from .lineage import require_verified

    track = load_track(args.track)
    require_verified("base", args.base, log=print)
    corpus = load_corpus(Path(args.corpus))
    build_reference(Path(args.base), corpus, Path(args.out), topk=track["evaluation"]["score"]["reference_topk"],
                    gpu_gib=args.gpu_gib)
    return 0


def _identity(args, track, ckpt: Path) -> dict | None:
    from .eval.candidate import checkpoint_bytes
    from .validate import audit, describe

    if args.external:
        ref = track["external_references"][args.external]
        ident = {"id": args.external, "name": ref["name"], "kind": "external", "source": ref.get("source")}
    else:
        _register_foreign(ckpt)
        m = Manifest.load(ckpt / "precision_manifest.yaml")
        units = Qwen38Arch.from_config(ckpt / "config.json").units()
        ident = {"id": m.candidate_id(units), "name": m.name, "kind": "internal", "manifest": m.to_dict(units),
                 "build": json.loads((ckpt / "bittrellis_build.json").read_text())}
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        files = ident["build"]["files"]
        prior = json.loads(Path(args.audit_json).read_text()) if getattr(args, "audit_json", None) else None
        if prior is not None and prior.get("fast") is False and prior.get("checkpoint_files") == files:
            audit_doc = prior  # audited earlier on exactly these shard bytes (pipelined maintainer runs)
        else:
            res = audit(ckpt, m, _source_dirs(args), check_bytes=True, verify_sources=True, log=print, sample_secret=_secret(args))
            audit_doc = {**res.to_dict(), "candidate_id": ident["id"], "checkpoint_files": files}
        (out / "audit.json").write_text(json.dumps(audit_doc, indent=2) + "\n")
        ident["audit_ok"] = audit_doc["ok"]
        if not audit_doc["ok"]:
            print("AUDIT FAIL — not evaluating:", *audit_doc["errors"][:10], sep="\n  ")
            (out / "candidate.json").write_text(json.dumps(ident, indent=2) + "\n")
            return None
    ident["executed_formats"] = describe(ckpt)
    ident["checkpoint_bytes"] = checkpoint_bytes(ckpt)
    return ident


def _evaluate(args, stages: tuple[str, ...]) -> int:
    from .eval.candidate import evaluate
    from .eval.corpus import load_corpus
    from .runtime import SparkInfer

    track = load_track(args.track)
    ckpt = Path(args.checkpoint)
    identity = _identity(args, track, ckpt)
    if identity is None:
        return 1
    si = SparkInfer(args.sparkinfer, track)
    evaluate(track, si, ckpt, Path(args.out), load_corpus(Path(args.corpus)), Path(args.reference), identity, stages)
    print(f"wrote {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    return _evaluate(args, tuple(s for s in args.stages.split(",") if s))


def cmd_evaluate_public(args) -> int:
    return _evaluate(args, ("quality",))


def cmd_benchmark(args) -> int:
    return _evaluate(args, ("performance",))


def cmd_evaluate_llamacpp(args) -> int:
    from .eval.corpus import load_corpus
    from .eval.llamacpp import evaluate_llamacpp
    from .lineage import require_verified

    track = load_track(args.track)
    gguf = Path(args.gguf)
    require_verified("unsloth_gguf", gguf.parent, log=print)
    evaluate_llamacpp(track, "R2", Path(args.llamacpp), gguf, load_corpus(Path(args.corpus)), Path(args.reference), Path(args.out))
    print(f"wrote {args.out}")
    return 0


def cmd_holdout(args) -> int:
    from . import holdout
    from .runtime import SparkInfer

    track = load_track(args.track)
    if args.action == "inventory":
        inv = holdout.inventory(Path(args.private), Path(args.shipped) / "tokenizer.json")
        epoch = inv.pop("epoch")
        print(f"epoch {epoch['epoch'] if epoch else 'MISSING epoch.json'}")
        print(f"  {'category':14s} {'files':>5s} {'tokens':>8s} {'needs':>8s}  status")
        for cat, d in inv.items():
            if d["repeated_spans"] > holdout.MAX_REPEATED_SPANS:
                print(f"  {cat:14s} {d['files']:5d} {d['tokens']:8,d} {d['needs']:8,d}  "
                      f"{d['repeated_spans']:.0%} of its {holdout.SHINGLE_WORDS}-word spans repeat: add new text, not copies")
                continue
            state = f"needs {d['missing']:,} more tokens (~{d['missing'] * 3 // 4:,} words)" if d["missing"] else (
                f"ok (~{d['short_of_recommended'] * 3 // 4:,} more words would give the 8K/16K/32K streams distinct text)"
                if d.get("short_of_recommended") else "ok")
            print(f"  {cat:14s} {d['files']:5d} {d['tokens']:8,d} {d['needs']:8,d}  {state}")
        missing = sum(d["missing"] for d in inv.values())
        print("ready to build" if not missing and epoch else "not ready: add the text above, and epoch.json with a secret seed")
        return 0 if not missing and epoch else 1
    if args.action == "build":
        try:
            c = holdout.build_private_corpus(Path(args.private), Path(args.shipped) / "tokenizer.json")
        except holdout.NotEnoughText as e:
            print(f"{e}\n  run: bittrellis holdout inventory --private {args.private}")
            return 1
        print(f"private holdout {c['version']}: {len(c['streams'])} streams, sha256 {c['sha256'][:16]}")
        return 0
    if args.action == "recheck":  # re-judge from cached holdout positions under the current rule; no GPU
        verdict = holdout.recheck(track, Path(args.artifact), Path(args.private), Path(args.incumbent_artifact))
        print(f"HOLDOUT {verdict}")
        return 0 if verdict == "PASS" else 1
    verdict = holdout.check(SparkInfer(args.sparkinfer, track), track, Path(args.checkpoint), Path(args.artifact),
                            Path(args.private), Path(args.shipped), Path(args.incumbent_artifact))
    print(f"HOLDOUT {verdict}")
    return 0 if verdict == "PASS" else 1


def cmd_frontier(args) -> int:
    from .frontier.report import load_rows, render_table, seed_rows, write_frontier

    track = load_track(args.track)
    paths = [Path(p) for p in args.artifacts] + (seed_rows(track) if args.with_seeds else [])
    rows, _ = load_rows(paths, track, args.speeds)
    doc = write_frontier(rows, track, Path(args.out) if args.out else None)
    print(render_table(rows))
    print(f"{len(doc['frontier'])} on the internal frontier ({doc['frontier_gain_version']}, epoch {doc['evaluator_epoch']})")
    return 0


def cmd_compare(args) -> int:
    from .frontier.report import compare

    _json(compare(Path(args.a), Path(args.b), load_track(args.track), args.speeds))
    return 0


def cmd_report(args) -> int:
    from .frontier.report import write_report

    print(f"wrote {write_report([Path(p) for p in args.artifacts], load_track(args.track), Path(args.out))}")
    return 0


def cmd_fingerprint(args) -> int:
    from . import fingerprint as F

    names = [n for n in args.quantizers.split(",") if n] if args.quantizers else None
    first = F.probe(names, args.seed)
    F.save_probe(first, Path(args.out))
    refs = sorted({k.split("|", 1)[0] for k in first})
    print(f"probe seed {args.seed}: {len(first)} tensors from {', '.join(refs) or 'no regenerable quantizer'}")
    if args.repeat_out:
        F.save_probe(F.probe(names, args.seed), Path(args.repeat_out))
    return 0


def cmd_search(args) -> int:
    from .search import neighbors

    base = Manifest.load(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = neighbors(base, _units(args), limit=args.limit)
    for m in made:
        (out / f"{m.name}.yaml").write_text(yaml.safe_dump(m.to_dict(), sort_keys=False))
    print(f"wrote {len(made)} neighbor manifests to {out}")
    return 0


def cmd_doctor(args) -> int:
    from .lineage import verify_source
    from .runtime import RuntimeError_, SparkInfer

    track = load_track(args.track)
    ok = True

    def check(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}{(' — ' + detail) if detail else ''}")

    print(f"bittrellis {__version__} · track {track.id} · epoch {track['evaluation']['epoch']}")
    check("nvidia-smi", shutil.which("nvidia-smi") is not None)
    for sid, d in _source_dirs(args).items():
        if not (d / "config.json").exists():
            check(f"source {sid}", False, f"{d} missing")
            continue
        res = verify_source(sid, d, files=["config.json"])
        check(f"source {sid}", res.ok, f"{d} (weights are hash-verified on first build/audit)")
    try:
        SparkInfer(args.sparkinfer, track).check_pinned()
        check("SparkInfer pinned + built", True, track["runtime"]["commit"][:12])
    except (RuntimeError_, OSError) as e:
        check("SparkInfer pinned + built", False, str(e)[:120])
    check("public corpus", Path(args.corpus).exists(), args.corpus)
    check("BF16 reference", (Path(args.reference) / "reference.json").exists(), args.reference)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bittrellis", description="Search the best quantization topology for an LLM on real hardware.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--track", default="HPC-01")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def paths(p, *keys):
        for k in keys:
            p.add_argument(f"--{k}", default=DEFAULTS[k])

    def sp(name: str, fn, help_: str, *keys):
        p = sub.add_parser(name, help=help_)
        paths(p, *keys)
        p.set_defaults(fn=fn)
        return p

    sp("doctor", cmd_doctor, "check the environment against the track pins", "base", "shipped", "unsloth", "sparkinfer", "corpus", "reference")
    sp("track", cmd_track, "print the pinned track definition")
    sp("quantizers", cmd_quantizers, "list registered quantizers, their formats and lineage")
    p = sp("inventory", cmd_inventory, "every searchable unit: tensors, legal assignments, executed formats", "shipped")
    p.add_argument("--out")
    p = sp("manifest", cmd_manifest, "validate manifests; show assignments, id and duplicates", "shipped")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--expand", action="store_true")
    p.add_argument("--against", help="directory of existing manifests to check for duplicates")
    p.add_argument("--min-distance", type=int, default=2, help="warn when this few units differ")
    p.add_argument("--ids-out", help="write {name, id, keys} of the (last) manifest as JSON")
    p = sp("verify-sources", cmd_verify_sources, "hash-verify local source checkouts against the lock", "base", "shipped", "unsloth")
    p.add_argument("--source", action="append")
    p = sp("describe", cmd_describe, "what SparkInfer executes for any checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--out")
    p = sp("build", cmd_build, "build a deployable checkpoint from a manifest (CPU only)", "base", "shipped", "unsloth")
    p.add_argument("manifest")
    p.add_argument("--out")
    p.add_argument("--no-verify", action="store_true", help="skip source hash verification (development only)")
    p = sp("audit", cmd_audit, "HPC-01 rule and lineage checks for a built checkpoint", "base", "shipped", "unsloth")
    p.add_argument("checkpoint")
    p.add_argument("--manifest")
    p.add_argument("--fast", action="store_true", help="skip byte and source hashing (development only)")
    p.add_argument("--out")
    p.add_argument("--sample-secret-file", help="evaluator secret that makes the regenerated samples unpredictable")
    p.add_argument("--regenerated", help="tensors contributed quantizers regenerated in isolation (compared, not executed)")
    p = sp("regenerate", cmd_regenerate, "regenerate units from the sources only (the contributed-code side of an isolated audit)",
           "base", "shipped", "unsloth")
    p.add_argument("manifest")
    p.add_argument("--units", required=True, help="comma-separated unit ids")
    p.add_argument("--out", required=True)
    p = sp("corpus", cmd_corpus, "build or verify the public corpus")
    p.add_argument("action", choices=["build", "verify"])
    p.add_argument("--out", default=DEFAULTS["corpus"])
    p.add_argument("--cache", default=str(REPO_ROOT / "data/cache"))
    p.add_argument("--split", default="public", choices=["public", "public-validation"])
    p = sp("reference", cmd_reference, "BF16 reference distributions on the fixed partition (once per corpus)", "base", "corpus")
    p.add_argument("--out", default=DEFAULTS["reference"])
    p.add_argument("--gpu-gib", type=int, default=14)
    for name, fn, help_ in (("evaluate", cmd_evaluate, "audit + public fidelity + tasks + performance"),
                            ("evaluate-public", cmd_evaluate_public, "audit + public fidelity (RP-KL, long-context guard)"),
                            ("benchmark", cmd_benchmark, "audit + decode, 4K prefill and peak memory (2 runs)")):
        p = sp(name, fn, help_, "base", "shipped", "unsloth", "sparkinfer", "corpus", "reference")
        p.add_argument("checkpoint")
        p.add_argument("--out", required=True)
        p.add_argument("--external", help="evaluate an external reference (R1) instead of a candidate")
        p.add_argument("--audit-json", help="reuse a full audit of the same checkpoint files (from `bittrellis audit --out`)")
        p.add_argument("--sample-secret-file", help="evaluator secret that makes the audit's regenerated samples unpredictable")
        if name == "evaluate":
            p.add_argument("--stages", default="quality,tasks,performance")
    p = sp("evaluate-llamacpp", cmd_evaluate_llamacpp, "external reference R2 through pinned llama.cpp", "llamacpp", "corpus", "reference")
    p.add_argument("--gguf", default=str(REPO_ROOT / "models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"))
    p.add_argument("--out", required=True)
    p = sp("holdout", cmd_holdout, "validators: inventory or build the private holdout, check a candidate (PASS/FAIL), "
           "or recheck one already scored under the current rule", "shipped", "sparkinfer")
    p.add_argument("action", choices=["inventory", "build", "check", "recheck"])
    p.add_argument("--private", required=True)
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--artifact")
    p.add_argument("--incumbent-artifact")
    p = sp("frontier", cmd_frontier, "gate and rank the internal frontier; FG-2")
    p.add_argument("artifacts", nargs="*")
    p.add_argument("--with-seeds", action="store_true", help="include the track's seed artifacts")
    p.add_argument("--out")
    p.add_argument("--speeds", help="per-machine speeds dir (<id>/performance.json) used for ranked results")
    p = sp("compare", cmd_compare, "paired comparison of two artifacts")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--speeds", help="per-machine speeds dir (<id>/performance.json) used for ranked results")
    p = sp("report", cmd_report, "frontier.json, comparison.csv, plots")
    p.add_argument("artifacts", nargs="+")
    p.add_argument("--out", default=str(REPO_ROOT / "results/feasibility"))
    p = sp("fingerprint", cmd_fingerprint, "run regenerable quantizers on a seeded synthetic model; save their bytes")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--quantizers", help="comma-separated names (default: every regenerable quantizer)")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat-out", help="run again and save here, for a determinism check")
    p = sp("search", cmd_search, "baseline search helpers", "shipped")
    p.add_argument("action", choices=["neighbors"])
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
