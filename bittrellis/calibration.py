"""The pinned calibration source: public calibration text, and BF16 activation statistics computed from it.

Calibrated encoders (GPTQ-style, Hessian-weighted rounding) need to know which input directions matter.
They read them from `ctx.calibration` through `quantizers.base.input_hessian`: for each Linear that reads
a module's input directly, the mean of x xᵀ over every calibration token, where x is that input as the
BF16 model computes it.

Three steps, all pinned:

1. `build_text`   327,680 tokens of public text (32 sequences of 2,048 tokens per category: general, math,
                  code, tools, multilingual) from pinned dataset revisions. Every document that shares a
                  run of 50 normalized characters with the public drift corpus or with the task questions
                  is dropped, so the calibration text is disjoint from everything that is scored.
                  Committed as data/calibration/<version>.json and checked by hash.
2. `capture`      runs the BF16 model once (transformers, CPU offload) and stores the statistics as
                  safetensors in about 4 minutes on one RTX 5090. The sums run in float32 on the GPU: a
                  rerun on the same machine gave identical bytes, other hardware may differ in the last
                  bits, so the published files, pinned by sha256 in configs/sources.lock.json (source
                  `calibration`), are the reference.
3. `fetch`        downloads the published files and verifies them against the lock.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

from .eval.corpus import CHAT_END, CHAT_USER, DOC_SEP, TOKENIZER, _order, _parquet_rows, fetch, wikitext_articles

VERSION = "hpc01-calib-v1"
SEQ_TOKENS = 2048
SEQS_PER_CATEGORY = 32
MAX_DOC_TOKENS = 1024          # longer documents are cut, so no single document dominates a category
OVERLAP_CHARS = 50             # a shared run of this many normalized characters drops a document
SEED = "hpc01-calib"
FW2_LANGS = ["cmn_Hani", "jpn_Jpan", "kor_Hang", "deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl", "arb_Arab",
             "hin_Deva", "vie_Latn"]   # the drift corpus's ten WMT24++ target languages
SOURCES = {
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
                 ["wikitext-103-raw-v1/train-00000-of-00002.parquet", "wikitext-103-raw-v1/train-00001-of-00002.parquet"]),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", ["main/train-00000-of-00001.parquet"]),
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
             [f"full/{s}-00000-of-00001.parquet" for s in ("train", "test", "validation", "prompt")]),
    "glaive": ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
               ["glaive-function-calling-v2.json"]),
    "fineweb2": ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9",
                 [f"data/{lang}/test/000_00000.parquet" for lang in FW2_LANGS]),
}
# Which module's input each searchable Linear reads directly. Down, o and out projections read internal
# activations and have no statistics here.
INPUT_OF = {"gate_proj": "mlp", "up_proj": "mlp", "q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn",
            "in_proj_qkv": "linear_attn", "in_proj_z": "linear_attn"}
KINDS = {"mlp": ("mlp",), "attn": ("self_attn", "linear_attn")}
RELEASE_URL = "https://github.com/coderbench/bittrellis/releases/download/{tag}/{file}"


def stat_name(lin_prefix: str) -> str | None:
    """Calibration tensor holding the input statistics of a Linear, e.g. `...layers.3.mlp.input_xtx`."""
    parent, _, leaf = lin_prefix.rpartition(".")
    module = INPUT_OF.get(leaf)
    if module is None or not parent.endswith("." + module):
        return None
    return parent + ".input_xtx"


# ------------------------------------------------------------------ text


def _norm(text: str) -> str:
    return "".join(re.findall(r"\w", unicodedata.normalize("NFKC", text).lower()))


def _h(s: str) -> bytes:
    return hashlib.blake2b(s.encode(), digest_size=8).digest()


def windows(text: str) -> set[bytes]:
    t = _norm(text)
    return {_h(t[i:i + OVERLAP_CHARS]) for i in range(len(t) - OVERLAP_CHARS + 1)}


def overlaps(text: str, excluded: set[bytes]) -> bool:
    t = _norm(text)
    return any(_h(t[i:i + OVERLAP_CHARS]) in excluded for i in range(len(t) - OVERLAP_CHARS + 1))


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def excluded_windows(tok, corpus: dict, task_files: list[Path]) -> set[bytes]:
    """Every 50-character window of the drift corpus's text and of every string in the task files."""
    out: set[bytes] = set()
    for s in corpus["streams"]:
        out |= windows(tok.decode(s["ids"]))
    for f in task_files:
        for line in f.read_text().splitlines():
            if line.strip():
                for s in _strings(json.loads(line)):
                    out |= windows(s)
    return out


def chat(q: str, a: str) -> str:
    return CHAT_USER.format(q=q.strip()) + a.strip() + CHAT_END


GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):")


def glaive_chat(r: dict) -> str:
    """A Glaive conversation in Qwen's chat layout: system (with the function list), then user, assistant
    and tool turns."""
    system = r["system"].strip()
    system = system[len("SYSTEM:"):].strip() if system.startswith("SYSTEM:") else system
    out = [f"<|im_start|>system\n{system}<|im_end|>\n"] if system else []
    parts = GLAIVE_TURN.split(r["chat"])
    for who, text in zip(parts[1::2], parts[2::2], strict=True):
        text = text.replace("<|endoftext|>", "").strip()
        role = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}[who]
        if text:
            out.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    return "".join(out)


def pack(tok, docs, excluded: set[bytes], n_seqs: int = SEQS_PER_CATEGORY, seq_tokens: int = SEQ_TOKENS) -> tuple[list[list[int]], dict]:
    """Fill `n_seqs` sequences of exactly `seq_tokens`, each starting with a fresh <|endoftext|>-opened document.

    Documents are cut at MAX_DOC_TOKENS; a document that does not fit the rest of a sequence is cut there
    (its remainder is not reused). Documents overlapping `excluded` are skipped and counted."""
    seqs: list[list[int]] = []
    cur: list[int] = []
    used = dropped = 0
    for d in docs:
        if overlaps(d, excluded):
            dropped += 1
            continue
        ids = tok.encode(DOC_SEP + d).ids[:MAX_DOC_TOKENS]
        cur.extend(ids[: seq_tokens - len(cur)])
        used += 1
        if len(cur) == seq_tokens:
            seqs.append(cur)
            cur = []
            if len(seqs) == n_seqs:
                return seqs, {"documents_used": used, "documents_dropped_overlap": dropped}
    raise ValueError(f"not enough source text: {len(seqs)} of {n_seqs} sequences")


def _round_robin(groups: list[list]) -> list:
    out = []
    for i in range(max(len(g) for g in groups)):
        out.extend(g[i] for g in groups if i < len(g))
    return out


def text_hash(body: dict) -> str:
    payload = {"version": body["version"], "sequences": body["sequences"]}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def build_text(cache: Path, corpus: dict, task_files: list[Path], log=print) -> dict:
    from tokenizers import Tokenizer

    tok_bytes = fetch(*TOKENIZER, cache=cache, dataset=False)
    tok = Tokenizer.from_str(tok_bytes.decode())
    excluded = excluded_windows(tok, corpus, task_files)
    log(f"[calibration] {len(excluded):,} excluded windows from the drift corpus and {len(task_files)} task files")

    def get(src: str) -> list[bytes]:
        repo, rev, files = SOURCES[src]
        return [fetch(repo, rev, f, cache) for f in files]

    wiki = wikitext_articles([r for data in get("wikitext") for r in _parquet_rows(data)])
    gsm = [r for data in get("gsm8k") for r in _parquet_rows(data)]
    mbpp = [r for data in get("mbpp") for r in _parquet_rows(data)]
    glaive = json.loads(get("glaive")[0])
    fw2 = [_order(_parquet_rows(data), lambda r: r["id"], SEED) for data in get("fineweb2")]
    docs = {
        "general": _order(wiki, lambda a: a[:200], SEED),
        "math": [chat(r["question"], r["answer"].replace("####", "The answer is"))
                 for r in _order(gsm, lambda r: r["question"], SEED)],
        "code": [chat(r["text"], "```python\n" + r["code"].strip() + "\n```") for r in _order(mbpp, lambda r: str(r["task_id"]), SEED)],
        "tools": [glaive_chat(r) for r in _order(glaive, lambda r: r["system"] + r["chat"], SEED)],
        "multilingual": [r["text"] for r in _round_robin(fw2)],
    }
    sequences, stats = [], {}
    for cat, ds in docs.items():
        seqs, stats[cat] = pack(tok, ds, excluded)
        sequences += [{"category": cat, "ids": s} for s in seqs]
        log(f"[calibration] {cat}: {stats[cat]}")
    body = {
        "version": VERSION,
        "tokenizer": {"repo": TOKENIZER[0], "revision": TOKENIZER[1], "sha256": hashlib.sha256(tok_bytes).hexdigest()},
        "sources": {k: {"repo": v[0], "revision": v[1], "files": v[2]} for k, v in SOURCES.items()},
        "disjoint_from": {"drift_corpus_sha256": corpus["sha256"],
                          "task_files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in task_files},
                          "rule": f"documents sharing {OVERLAP_CHARS} normalized characters (NFKC, lowercase, letters and digits) are dropped"},
        "seq_tokens": SEQ_TOKENS, "max_doc_tokens": MAX_DOC_TOKENS,
        "categories": stats,
        "tokens": SEQ_TOKENS * len(sequences),
        "sequences": sequences,
    }
    body["sha256"] = text_hash(body)
    return body


def save_text(body: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in body.items() if k != "sequences"}
    # One sequence per line keeps the file diffable and small.
    lines = ",\n".join(json.dumps(s, separators=(",", ":")) for s in body["sequences"])
    text = json.dumps(head, indent=1)[:-2] + ',\n "sequences": [\n' + lines + "\n ]\n}\n"
    path.write_text(text)


def load_text(path: Path) -> dict:
    body = json.loads(Path(path).read_text())
    if text_hash(body) != body["sha256"]:
        raise ValueError(f"{path}: calibration text hash mismatch")
    return body


# ------------------------------------------------------------------ statistics (GPU, maintainers)


def _modules(model, kinds: list[str]) -> dict[str, tuple[int, str, object]]:
    """{checkpoint prefix: (layer, module kind, module)} for the language model's layers."""
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.(mlp|self_attn|linear_attn)$")
    wanted = {m for k in kinds for m in KINDS[k]}
    out = {}
    for name, mod in model.named_modules():
        if "visual" in name or "mtp" in name:
            continue
        m = pat.search(name)
        if m and m.group(2) in wanted:
            i, kind = int(m.group(1)), m.group(2)
            out[f"model.language_model.layers.{i}.{kind}"] = (i, kind, mod)
    return out


def capture(model_dir: Path, text: dict, out_dir: Path, kinds: list[str], gpu_gib: int = 10, cpu_gib: int = 54,
            batch: int = 8, shard_bytes: int = 1536 * 1024**2, log=print) -> dict:
    """Mean x xᵀ of each selected module's BF16 input over every calibration token, one safetensors
    directory per kind (`<out>/<kind>/`), plus calibration.json describing it."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from .safetensors_io import ShardWriter

    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="auto",
                                                 max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"})
    model.eval()
    mods = _modules(model, kinds)
    hidden = model.config.get_text_config().hidden_size
    layers = model.config.get_text_config().num_hidden_layers
    per_kind = {k: sum(1 for p in mods if p.rsplit(".", 1)[1] in KINDS[k]) for k in kinds}
    if per_kind.get("mlp", layers) != layers or per_kind.get("attn", layers) != layers:
        raise RuntimeError(f"expected {layers} modules per kind, found {per_kind}")
    log(f"[calibration] loaded in {time.time() - t0:.0f}s; {len(mods)} modules, hidden {hidden}")
    dev = torch.device("cuda:0")
    acc = {p: torch.zeros(hidden, hidden, dtype=torch.float32, device=dev) for p in mods}
    hooks = []
    for p, (_, _, mod) in mods.items():
        def pre(module, args, kwargs, p=p):
            x = args[0] if args else kwargs["hidden_states"]
            x = x.to(dev).reshape(-1, hidden).float()
            acc[p].addmm_(x.T, x)
        hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    body = model.model
    seqs = [s["ids"] for s in text["sequences"]]
    try:
        with torch.inference_mode():
            for a in range(0, len(seqs), batch):
                t1 = time.time()
                body(input_ids=torch.tensor(seqs[a:a + batch], dtype=torch.long), use_cache=False)
                log(f"[calibration] {min(a + batch, len(seqs))}/{len(seqs)} sequences, {time.time() - t1:.0f}s")
    finally:
        for h in hooks:
            h.remove()
    n = sum(len(s) for s in seqs)
    record = {}
    for kind in kinds:
        d = Path(out_dir) / kind
        w = ShardWriter(d, shard_bytes)
        for p in sorted(p for p in mods if p.rsplit(".", 1)[1] in KINDS[kind]):
            m = (acc[p].double() / n).float().cpu().numpy()
            m = ((m + m.T) / 2).astype("<f4")   # exactly symmetric
            w.add(p + ".input_xtx", "F32", m.shape, np.ascontiguousarray(m))
        w.close(metadata={"producer": "bittrellis", "calibration": VERSION})
        meta = {"version": VERSION, "kind": kind, "text_sha256": text["sha256"], "tokens": n,
                "sequences": len(seqs), "seq_tokens": len(seqs[0]),
                "statistic": "mean over calibration tokens of x x^T, x = the module's BF16 input; float32 GPU sums",
                "tensors": "model.language_model.layers.<i>.<module>.input_xtx, float32 [hidden, hidden]",
                "model": str(model_dir), "seconds": round(time.time() - t0)}
        (d / "calibration.json").write_text(json.dumps(meta, indent=2) + "\n")
        record[kind] = meta
        log(f"[calibration] wrote {d}")
    return record


# ------------------------------------------------------------------ reading


def open_calibration(path: Path, verify: bool, log=None):
    """The pinned statistics as a SafeTensorsDir; with `verify`, every file must match the lock first."""
    from .lineage import LineageError, verify_source
    from .safetensors_io import SafeTensorsDir

    if verify:
        res = verify_source("calibration", path, log=log)
        if not res.ok:
            raise LineageError(f"calibration at {path} does not match the lock: " + "; ".join(res.errors[:5]))
    return SafeTensorsDir(path)


# ------------------------------------------------------------------ download


def fetch_release(dest: Path, log=print) -> None:
    """Download the pinned calibration files (configs/sources.lock.json, source `calibration`) and verify them."""
    from .lineage import LineageError, load_lock, verify_source

    entry = load_lock()["sources"]["calibration"]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, pin in sorted(entry["files"].items()):
        p = dest / name
        if p.exists() and p.stat().st_size == pin["size"]:
            continue
        log(f"[calibration] downloading {name} ({pin['size'] / 1e9:.2f} GB)")
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(RELEASE_URL.format(tag=entry["revision"], file=name), timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(64 << 20):
                fh.write(chunk)
        tmp.replace(p)
    res = verify_source("calibration", dest, log=log)
    if not res.ok:
        raise LineageError(f"calibration at {dest} does not match the lock: {res.errors[:5]}")
    log(f"[calibration] {dest}: verified")
