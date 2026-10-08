"""BF16 reference distributions, produced once per corpus with Hugging Face transformers.

The BF16 model (~52 GiB) does not fit a 32 GB card and SparkInfer cannot execute BF16 Linears,
so the reference runs through transformers with CPU offload. It is slow but runs exactly once;
the result is hashed and every candidate is compared against the same files.

Requires the `reference` extra (torch, transformers, accelerate).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..runtime import ScoreDump
from ..safetensors_io import SafeTensorsDir


def reference_path(ref_dir: Path, stream_id: str) -> Path:
    return Path(ref_dir) / f"{stream_id}.npz"


def load_bf16(model_dir: Path, gpu_gib: int, cpu_gib: int, offload: Path | None = None):
    """(model, text body) in BF16 across GPU, CPU and (with `offload`) disk.

    Qwen3.6-35B-A3B (model_type qwen3_5_moe) ships as a multimodal checkpoint whose weights are named
    model.language_model.*; loaded as a causal LM, transformers attaches none of them and runs random
    weights without an error. It is loaded as the multimodal model and its language model is the body.
    The HPC-01 model keeps its original loader."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    kw = {"dtype": torch.bfloat16, "device_map": "auto", "max_memory": {0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"}}
    if offload:
        kw["offload_folder"] = str(offload)
    if cfg.get("model_type") == "qwen3_5_moe":
        model = AutoModelForImageTextToText.from_pretrained(model_dir, **kw)
        body = model.model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
        body = model.model
    model.eval()
    return model, body


def build_reference(model_dir: Path, corpus: dict, out_dir: Path, topk: int = 256, gpu_gib: int = 14,
                    cpu_gib: int = 58, chunk: int = 1024, offload: Path | None = None, log=print) -> dict:
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, body = load_bf16(model_dir, gpu_gib, cpu_gib, offload)
    log(f"[reference] loaded in {time.time() - t0:.0f}s: {type(model).__name__}")
    # The head is applied by hand from the pinned safetensors bytes: accelerate may offload it to
    # "meta", and a 2.4 GiB BF16 matrix on the GPU is cheaper than paging it in per chunk.
    with SafeTensorsDir(model_dir) as st:
        t = st.get("lm_head.weight")
        head_w = torch.frombuffer(bytearray(st.raw("lm_head.weight")), dtype=torch.bfloat16).reshape(t.shape).to("cuda:0")
    head_dev = head_w.device
    record = {"corpus_sha256": corpus["sha256"], "model_dir": str(model_dir), "topk": topk, "streams": {}}
    for s in corpus["streams"]:
        dest = reference_path(out_dir, s["id"])
        if dest.exists():
            log(f"[reference] {s['id']}: exists, skipping")
        else:
            t1 = time.time()
            ids = torch.tensor([s["ids"]], dtype=torch.long)
            with torch.inference_mode():
                hidden = body(input_ids=ids, use_cache=False).last_hidden_state[0]
                log(f"[reference] {s['id']}: body forward {time.time() - t1:.0f}s")
                start = s["score_from"]
                n = len(s["ids"]) - 1 - start
                top_ids = np.empty((n, topk), np.int32)
                top_lp = np.empty((n, topk), np.float32)
                lp_t = np.empty(n, np.float32)
                tgt = torch.tensor(s["ids"][start + 1 :], dtype=torch.long)
                for a in range(0, n, chunk):
                    b = min(n, a + chunk)
                    h = hidden[start + a : start + b].to(head_dev)
                    logp = torch.log_softmax((h.to(torch.bfloat16) @ head_w.T).float(), dim=-1)
                    v, i = torch.topk(logp, topk, dim=-1)
                    top_ids[a:b] = i.cpu().numpy()
                    top_lp[a:b] = v.cpu().numpy()
                    lp_t[a:b] = logp.gather(1, tgt[a:b].to(head_dev)[:, None])[:, 0].cpu().numpy()
                del hidden
            pos = np.arange(start, start + n, dtype=np.int32)
            dump = ScoreDump(pos, tgt.numpy().astype(np.int32), top_ids[:, 0].copy(), lp_t, top_ids, top_lp)
            dump.save(dest)
            log(f"[reference] {s['id']}: {n} positions in {time.time() - t1:.0f}s")
        record["streams"][s["id"]] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (out_dir / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_reference(ref_dir: Path, corpus: dict, expected_k: int | None = None) -> dict[str, ScoreDump]:
    meta = json.loads((Path(ref_dir) / "reference.json").read_text())
    if meta["corpus_sha256"] != corpus["sha256"]:
        raise ValueError("reference was built for a different corpus")
    if expected_k is not None and meta.get("topk") != expected_k:
        raise ValueError(f"reference partition K={meta.get('topk')} but the track pins K={expected_k}")
    out = {}
    for s in corpus["streams"]:
        p = reference_path(ref_dir, s["id"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != meta["streams"][s["id"]]:
            raise ValueError(f"{p}: hash mismatch")
        out[s["id"]] = ScoreDump.load(p)
    return out
