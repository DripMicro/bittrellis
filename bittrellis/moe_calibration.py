"""Calibration statistics for the HPC-02 track (Qwen3.6-35B-A3B), from the BF16 model on the pinned calibration text.

Per layer i (names follow the GGUF tensors they serve):

  blk.{i}.attn_input.xtx      [2048, 2048]  mean x xᵀ of the attention / recurrent input (attn_qkv, attn_q/k/v,
                                            attn_gate read it)
  blk.{i}.ffn_input.xtx       [2048, 2048]  mean x xᵀ of the MoE block input (router, shared expert gate/up,
                                            routed experts' gate/up)
  blk.{i}.ffn_down_shexp.xtx  [512, 512]    mean h hᵀ of the shared expert's down input
  blk.{i}.exps_input.sumsq    [256, 2048]   per expert: sum of x² over the tokens routed to it (gate/up input)
  blk.{i}.exps_down.sumsq     [256, 512]    per expert: sum of h² over its routed tokens (down input)
  blk.{i}.exps.count          [256]         tokens routed to each expert

Means are over all calibration tokens; per-expert sums divide by `count` (llama.cpp's imatrix keeps the same
per-expert diagonal). float32 GPU sums, float64 division.
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import numpy as np


def capture(model_dir: Path, text: dict, out_dir: Path, gpu_gib: int = 20, cpu_gib: int = 34, batch: int = 4,
            offload: Path | None = None, log=print) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText

    from .safetensors_io import ShardWriter

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="auto",
                                                        max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"},
                                                        offload_folder=str(offload) if offload else None)
    model.eval()
    body = model.model.language_model
    cfg = body.config
    L, E, H, Ie = cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
    Is = cfg.shared_expert_intermediate_size
    z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)  # noqa: E731
    acc = {i: {"attn_input.xtx": z(H, H), "ffn_input.xtx": z(H, H), "ffn_down_shexp.xtx": z(Is, Is),
               "exps_input.sumsq": z(E, H), "exps_down.sumsq": z(E, Ie), "exps.count": z(E)} for i in range(L)}
    hooks = []

    def add(i, key, x):
        x = x.to(dev).reshape(-1, x.shape[-1]).float()
        acc[i][key].addmm_(x.T, x)

    for i, layer in enumerate(body.layers):
        mixer = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
        hooks.append(mixer.register_forward_pre_hook(
            lambda mod, a, kw, i=i: add(i, "attn_input.xtx", a[0] if a else kw["hidden_states"]), with_kwargs=True))
        hooks.append(layer.mlp.register_forward_pre_hook(lambda mod, a, i=i: add(i, "ffn_input.xtx", a[0])))
        hooks.append(layer.mlp.shared_expert.down_proj.register_forward_pre_hook(
            lambda mod, a, i=i: add(i, "ffn_down_shexp.xtx", a[0])))
        layer.mlp.experts._bt_layer = i

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        """transformers' eager expert loop, recording each expert's inputs on the way."""
        i = self._bt_layer
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            xf = x.to(dev).float()
            acc[i]["exps_input.sumsq"][e] += (xf * xf).sum(0)
            acc[i]["exps.count"][e] += len(tok)
            gate, up = F.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            hf = h.to(dev).float()
            acc[i]["exps_down.sumsq"][e] += (hf * hf).sum(0)
            y = F.linear(h, self.down_proj[e]) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out

    # accelerate keeps each offloaded module's original forward as `_old_forward` and calls that, so the
    # recording loop is installed on every expert block itself, not on the class.
    patched = []
    for layer in body.layers:
        ex = layer.mlp.experts
        attr = "_old_forward" if hasattr(ex, "_old_forward") else "forward"
        patched.append((ex, attr, getattr(ex, attr)))
        setattr(ex, attr, types.MethodType(experts_forward, ex))
    seqs = [s["ids"] for s in text["sequences"]]
    try:
        with torch.inference_mode():
            for a in range(0, len(seqs), batch):
                t1 = time.time()
                body(input_ids=torch.tensor(seqs[a:a + batch], dtype=torch.long), use_cache=False)
                log(f"[moe-calibration] {min(a + batch, len(seqs))}/{len(seqs)} sequences, {time.time() - t1:.0f}s")
    finally:
        for ex, attr, fn in patched:
            setattr(ex, attr, fn)
        for h in hooks:
            h.remove()
    n = sum(len(s) for s in seqs)
    w = ShardWriter(out_dir, 1536 * 1024**2)
    for i in range(L):
        for key, v in acc[i].items():
            a = v.double()
            if key.endswith(".xtx"):
                a = a / n
                a = (a + a.T) / 2
            w.add(f"blk.{i}.{key}", "F32", tuple(a.shape), np.ascontiguousarray(a.float().cpu().numpy().astype("<f4")))
    w.close(metadata={"producer": "bittrellis", "calibration": "hpc02-calib-v1"})
    meta = {"version": "hpc02-calib-v1", "text_sha256": text["sha256"], "tokens": n, "layers": L, "experts": E,
            "statistics": __doc__.split("Per layer i")[1].strip(), "seconds": round(time.time() - t0)}
    (Path(out_dir) / "calibration.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
