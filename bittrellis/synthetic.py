"""A tiny synthetic Qwen3.8-shaped model: BF16 base, ModelOpt-NVFP4 baseline and a compressed-tensors source.

Used by the tests and by `bittrellis fingerprint`, which runs quantizers on it with an evaluator-chosen seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model.qwen38 import Qwen38Arch
from .quant.formats import f32_to_bf16, quantize_nvfp4
from .safetensors_io import ShardWriter

TINY_TEXT = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4, "full_attention_interval": 4,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "attn_output_gate": True,
    "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 8,
    "linear_value_head_dim": 8, "vocab_size": 64,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
}


def _bf16(rng, shape, scale=0.05):
    return f32_to_bf16((rng.standard_normal(shape) * scale).astype(np.float32)).reshape(shape)


def make_tiny(root: Path, seed: int = 0) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(seed)
    arch = Qwen38Arch.from_config({"text_config": TINY_TEXT})
    base_dir, bl_dir = root / "base", root / "baseline"
    base, bl = ShardWriter(base_dir, 1 << 20), ShardWriter(bl_dir, 1 << 20)
    tensors: dict[str, np.ndarray] = {}
    lm = "model.language_model."
    tensors[lm + "embed_tokens.weight"] = _bf16(rng, (64, 32))
    tensors[lm + "norm.weight"] = _bf16(rng, (32,))
    for i in range(4):
        b = f"{lm}layers.{i}."
        tensors[b + "input_layernorm.weight"] = _bf16(rng, (32,))
        tensors[b + "post_attention_layernorm.weight"] = _bf16(rng, (32,))
        if arch.is_linear(i):
            for n, s in (("A_log", (2,)), ("dt_bias", (2,)), ("norm.weight", (8,)), ("conv1d.weight", (48, 1, 4)),
                         ("in_proj_a.weight", (2, 32)), ("in_proj_b.weight", (2, 32))):
                tensors[b + "linear_attn." + n] = _bf16(rng, s)
        else:
            tensors[b + "self_attn.q_norm.weight"] = _bf16(rng, (16,))
            tensors[b + "self_attn.k_norm.weight"] = _bf16(rng, (16,))
    for u in arch.units():
        for lin in u.linears:
            tensors[lin.prefix + ".weight"] = _bf16(rng, (lin.rows, lin.cols))
    tensors["model.visual.blocks.0.attn.proj.weight"] = _bf16(rng, (8, 8))
    searchable = {lin.prefix for u in arch.units() for lin in u.linears}
    from .quant.formats import bf16_to_f32

    for name in sorted(tensors):
        arr = tensors[name]
        base.add(name, "BF16", arr.shape, arr)
        prefix = name[: -len(".weight")]
        if name.endswith(".weight") and prefix in searchable:
            p, s, ws2 = quantize_nvfp4(bf16_to_f32(arr.reshape(-1)).reshape(arr.shape))
            bl.add(prefix + ".weight", "U8", p.shape, p)
            bl.add(prefix + ".weight_scale", "F8_E4M3", s.shape, s)
            bl.add(prefix + ".weight_scale_2", "F32", (), np.asarray(ws2, "<f4"))
            bl.add(prefix + ".input_scale", "F32", (), np.asarray(1.0, "<f4"))
        else:
            bl.add(name, "BF16", arr.shape, arr)
    base.close()
    bl.close()
    # A compressed-tensors NVFP4 source (unsloth layout) for MLP layers 0-2 only.
    ct_dir = root / "ct"
    ct = ShardWriter(ct_dir, 1 << 20)
    for i in range(3):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.language_model.layers.{i}.mlp.{proj}"
            arr = tensors[prefix + ".weight"]
            p, s, ws2 = quantize_nvfp4(bf16_to_f32(arr.reshape(-1)).reshape(arr.shape))
            ct.add(prefix + ".weight_packed", "U8", p.shape, p)
            ct.add(prefix + ".weight_scale", "F8_E4M3", s.shape, s)
            ct.add(prefix + ".weight_global_scale", "F32", (), np.asarray(1.0 / ws2, "<f4"))
            ct.add(prefix + ".input_global_scale", "F32", (), np.asarray(1.0, "<f4"))
    ct.close()
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": TINY_TEXT,
           "vision_config": {"depth": 1}}
    (base_dir / "config.json").write_text(json.dumps(cfg))
    (bl_dir / "config.json").write_text(json.dumps({**cfg, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}))
    (bl_dir / "tokenizer.json").write_text("{}")
    return base_dir, bl_dir, ct_dir


def make_tiny_calibration(root: Path, seed: int = 0) -> Path:
    """Seeded stand-in for the pinned calibration statistics of the tiny model: one positive definite
    mean x xᵀ per layer's MLP input (bittrellis/calibration.py)."""
    rng = np.random.default_rng([seed, 7])
    w = ShardWriter(root / "calibration", 1 << 20)
    h = TINY_TEXT["hidden_size"]
    for i in range(TINY_TEXT["num_hidden_layers"]):
        x = rng.standard_normal((4 * h, h)) * rng.uniform(0.2, 3.0, h)   # uneven channel scales, like real activations
        w.add(f"model.language_model.layers.{i}.mlp.input_xtx", "F32", (h, h), (x.T @ x / len(x)).astype("<f4"))
    w.close()
    return root / "calibration"
