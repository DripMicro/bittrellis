# Quantizer contract and lineage

> Add an encoder: one searchable Linear in, the tensors a checkpoint stores out.

Why: an encoder can move fidelity as much as the format map. Calibrated NVFP4 bytes cut public
RP-KL 11.5% at equal decode and memory — but carried only 9% of that to the private holdout and so
earned nothing ([report](../results/feasibility/feasibility_report.md)). A gain has to survive unseen text.

## The contract

Subclass [`base.py`](../bittrellis/quantizers/base.py):

```python
class MyQuantizer(Quantizer):
    name = "gptq_nvfp4"          # used in manifests: quantizer: gptq_nvfp4
    version = 1                  # bump on ANY change to produced bytes: it is part of every candidate id
    formats = ("NVFP4",)         # execution formats it can produce
    lineage = "regenerable"      # runtime | regenerable | attested
    replay_mode = "sequential"   # independent | sequential   (regenerable only)

    def begin(self, ctx):        # once per build or audit replay, before the first encode
        ...

    def encode(self, ctx, unit, lin, fmt):
        w = f32_weight(ctx, lin)                  # canonical BF16 weight, float32
        params = ctx.params                       # per-unit params from the manifest
        # ... your algorithm ...
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
```

Register it in [`__init__.py`](../bittrellis/quantizers/__init__.py), then reference it from a manifest
rule: `quantizer: gptq_nvfp4`, optionally with `params: {damp: 0.01, blocksize: 128}`.

### What `encode` may do

| Allowed | Not allowed (audit or reviewers reject) |
|---|---|
| read the unit's BF16 weight (`ctx.base`) | read or emit other tensors' bytes |
| read the pinned public calibration manifest (`ctx.calibration`) — **not provided yet**: it is always `None`, so calibrated encoders cannot work in this epoch | fetch network data at build time |
| search scale, clipping, rounding, act-order, error feedback inside the tensor | move scales into norms or neighbouring Linears (SmoothQuant/AWQ-style migration) |
| keep cross-unit state (`ctx.state`) if `replay_mode = "sequential"` | depend on wall-clock time, unseeded randomness or GPU nondeterminism |
| emit NVFP4 (ModelOpt layout) or FP8 (E4M3 + one BF16 scale per row) | emit a format the loader silently converts ([precision_space.md](precision_space.md)) |

**Determinism is the whole game:** bytes are rebuilt and compared exactly. On GPU, pin
deterministic kernels or round on CPU in float64.

## Lineage: how bytes are proven (`bittrellis quantizers` lists built-ins)

- **`runtime`** (nobody adds one): stored BF16 byte-identical to base; SparkInfer fits Q4_K at load.
  Built-in: `runtime@v1` (Lloyd fit).
- **`regenerable`** (anyone, by PR; runs contributed code, so evaluated after maintainer approval):
  audit rebuilds sampled units byte for byte. Built-in `rtn@v1`, independent:
  round-to-nearest from BF16 (NVFP4 max-calibrated, FP8 per row).
- **`attested`** (maintainers only): bytes match a source whose files are all sha256-pinned in
  `configs/sources.lock.json`. Built-in NVFP4: `baseline@v1` (`gittensor_nvfp4`, shipped ModelOpt
  round-to-nearest bytes), `unsloth@v1` (`unsloth_nvfp4`, llm-compressor calibrated, MLP 0–55 only).

**Regenerable replay:**

- **Independent:** six units per quantizer, always incl. first and last, rebuilt byte for byte. The evaluator picks them from the candidate id plus its secret, so they cannot be predicted.
- **Sequential:** in order (layer ascending, unit order), first unit to last sampled: deep
  tensors can depend on earlier ones.
- **Replay cache:** keyed by quantizer version + base revision + the unit's *pipeline prefix* (its
  assignments to that quantizer through this unit). Shared prefix, shared replay.

**Anomaly check**, per NVFP4/FP8 tensor: sampled-row reconstruction error ÷ round-to-nearest error (same rows). Calibrated encoders
measure 1.2–1.6× (unsloth's layer-0 `down_proj`: 1.61×); > 2× warns, > 5× rejects as substituted
bytes. Diagnostic only: lineage gives legitimacy.

## Checklist for a quantizer PR

- [ ] one module in `bittrellis/quantizers/`, registered in `__init__.py`
- [ ] a **new name** (the sandboxed evaluator knows it by name and version only)
- [ ] `version` set; bumped if produced bytes change
- [ ] does not reproduce an existing encoder's bytes (output bytes are compared, not code)
- [ ] deterministic: a `pytest` test encodes the tiny synthetic model twice and compares bytes
- [ ] one manifest in `manifests/` using it, hypothesis in `description`
- [ ] no changes to evaluator paths (see [CONTRIBUTING.md](../CONTRIBUTING.md))
