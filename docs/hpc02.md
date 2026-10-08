# HPC-02 — Qwen3.6-35B-A3B on one RTX 5090 (GGUF)

> The second track: a mixture-of-experts model, stored as GGUF, run by a pinned SparkInfer. Same rules and
> scoring as HPC-01 ([specification](specification.md)); what differs is below.

**The model.** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): 40 layers (30 recurrent,
10 full attention), 256 routed experts per layer of which 8 run per token, one shared expert. 35B weights,
about 3B used per token. In BF16 it is 67 GB; one 32 GB card needs it compressed.

**The runtime.** SparkInfer at the commit pinned in [`configs/hpc02.yaml`](../configs/hpc02.yaml), with three
of its speed shortcuts off so every recipe runs what it stores: the load-time refit of attention and expert-down
tensors to Q4_K, and the 4,096-token sliding window it uses from 16K context (with it, no recipe recalls the
32K needles the BF16 model recalls).

**The incumbent (V0)** is the popular free GGUF, unsloth's UD-Q4_K_M, rebuilt from its own bytes.

## Recipes

One file per PR: `manifests/hpc02/<name>.yaml`, the HPC-01 syntax with GGUF formats.

```yaml
schema: bittrellis/manifest@2
track: HPC-02
name: exps-q5k-down-q6k-attn-q8
description: what you changed and why it should carry over to unseen text
default: Q8_0                     # Q4_K | Q5_K | Q6_K | Q8_0
encoders: {Q4_K: kq_rtn}          # default encoder per format (optional; kq_rtn otherwise)
rules:                            # applied in order; later rules win
  - match: "L*.exps.gate"
    format: Q5_K
  - match: "L*.exps.down"
    layers: "30-39"
    format: Q6_K
    encoder: my_kq                # a merged contributed encoder
modules:                          # exact units, applied last
  lm_head: Q6_K
```

**Units** (one GGUF tensor each; GGUF stores a layer's 256 experts in one tensor, so a format applies to all of
them):

| unit | tensor | layers |
|---|---|---|
| `L{i}.exps.gate` / `.up` / `.down` | routed experts | all 40 |
| `L{i}.shexp.gate` / `.up` / `.down` | shared expert | all 40 |
| `L{i}.gdn.qkv` / `.z` / `.out` | recurrent projections | 30 recurrent layers |
| `L{i}.attn.q` / `.k` / `.v` / `.o` | attention | 10 full-attention layers |
| `embed`, `lm_head` | token embeddings, output head | — |

Every other tensor (norms, router, recurrent constants) is the template's, byte for byte.
`bittrellis --track HPC-02 manifest manifests/hpc02/<name>.yaml` validates a recipe and prints its id and size;
it needs no model download (the unit list is in [`configs/hpc02_units.json`](../configs/hpc02_units.json)).

**Encoders.** `kq_rtn` (built in) rounds each block to its own range. `unsloth_ud` copies unsloth's bytes for a
unit where UD-Q4_K_M stores exactly that format. Anything better is a contributed encoder.

## Contributed encoders

One module in `bittrellis/hpc02_encoders/`, plus a test and one recipe using it, in one PR. Contributed code
runs only in the evaluator's sandbox (no GPU, no network) after a maintainer adds `eval-approved`.

```python
from bittrellis.hpc02 import Encoder, register
from bittrellis import kquant


def _encode(ctx, fmt):
    # ctx.rows        float32 [rows, cols]: the tensor's BF16 values (all experts' rows, expert-major)
    # ctx.unit        the unit (id, kind, layer, cols, rows)
    # ctx.params      the recipe's `params` for this unit
    # ctx.calibration the pinned statistics below (a safetensors directory), or None
    # return the tensor's GGUF block bytes for `fmt`, in llama.cpp's layout
    return kquant.RTN[fmt](ctx.rows)


register(Encoder("my_kq", 1, "regenerable", _encode))
```

- **Deterministic:** the audit rebuilds secretly chosen tensors byte for byte; a test that encodes twice and
  compares bytes is required. NumPy on the CPU in float64 is the safe way.
- **Own tensor only:** read `ctx.rows` and `ctx.calibration`; nothing else.
- **A new name and a version;** bump the version on any change to the bytes. Encoders are compared by their
  output bytes on a probe: one that reproduces an existing encoder is rejected.
- `bittrellis/kquant.py` holds the four block layouts (and gguf-py's decoder as their reference).

## Calibration statistics

The BF16 model's activations over the pinned calibration text
([`data/calibration`](../data/calibration/README.md), disjoint from the scored text), per layer `i`:

| tensor | shape | |
|---|---|---|
| `blk.{i}.attn_input.xtx` | 2048 × 2048 | mean x xᵀ of the attention / recurrent input |
| `blk.{i}.ffn_input.xtx` | 2048 × 2048 | mean x xᵀ of the MoE block input (router, shared expert, routed gate/up) |
| `blk.{i}.ffn_down_shexp.xtx` | 512 × 512 | mean h hᵀ of the shared expert's down input |
| `blk.{i}.exps_input.sumsq` | 256 × 2048 | per expert: sum of x² over the tokens routed to it |
| `blk.{i}.exps_down.sumsq` | 256 × 512 | per expert: sum of h² of its down input |
| `blk.{i}.exps.count` | 256 | tokens routed to each expert |

Expert `e`'s rows in `ctx.rows` are `e * rows_per_expert` to `(e + 1) * rows_per_expert`; its importance per
input channel is `exps_input.sumsq[e] / count[e]` (llama.cpp's imatrix keeps the same per-expert diagonal).

## Scoring

As HPC-01: section-balanced RP-KL against the BF16 model, decode and prefill at 4K and peak memory measured on
the evaluator's machine, the 784-question task guard against V0, the private holdout transfer test, then FG-2
over this track's own frontier, with the same tiers. The frontier box is HPC-01's scaled by V0's own decode,
prefill and peak memory, so the same relative improvement earns about the same gain on both tracks
([`configs/hpc02.yaml`](../configs/hpc02.yaml)).

| seed | recipe | RP-KL | decode tok/s | prefill 4K tok/s | peak GiB | tasks |
|---|---|---:|---:|---:|---:|---:|
| V0 | unsloth UD-Q4_K_M (its bytes) | 0.0450 | 397 | 39,225 | 25.5 | 570/784 |
| V1 | V0's map, `kq_rtn` | 0.0460 | 397 | 39,333 | 25.5 | 563/784 |
| V2 | everything Q4_K | 0.1207 | 466 | 40,265 | 23.3 | 573/784 |
| V3 | experts Q5_K, downs Q6_K, rest Q8_0 | 0.0340 | 382 | 36,996 | 29.2 | 564/784 |

RP-KL here is section-balanced against V0, as ranked.
