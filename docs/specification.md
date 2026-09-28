# BitTrellis specification v2.1 (authoritative)

> What a candidate may change, how it is measured, and what it must pass to rank. Terms: [README](../README.md#key-terms).

*Every layer doesn't deserve the same bits. And the same bits don't deserve the same quantizer.*

Supersedes the original blueprint and v2: v2's rules plus six fixes ([§14](#14-changes-from-v2)).

## 1. Track HPC-01

- **Stack:** Qwen3.8-27B (architecture, tokenizer, BF16 weights frozen) × SparkInfer `b1ed168` (v0.5.8; kernels unmodified; every `SPARKINFER_*` knob pinned) × 1× RTX 5090 32 GB.
- **Search:** execution format · quantizer/encoder · module mapping.
- **Objectives:** RP-KL ↓ · decode tok/s ↑ · 4K prefill tok/s ↑ · peak GPU memory ↓.
- **Out of scope:** pruning, distillation, fine-tuning/QAT, adapters, architecture/tokenizer/kernel changes, other models/GPUs/runtimes.

Pins: [`configs/hpc01.yaml`](../configs/hpc01.yaml), [`configs/sources.lock.json`](../configs/sources.lock.json). Changing evaluation, frontier or gates needs a new **evaluator epoch** (versioned rule set; now `hpc01-e5`: speeds measured on the evaluator's machine, [frontier.md](frontier.md#speeds-from-the-evaluators-machine-hpc01-e5)).

## 2. Runtime truth

A manifest names what SparkInfer **executes** ([precision_space.md](precision_space.md)). GDN = Gated DeltaNet (recurrent layers); Q4_K = 4-bit block format the runtime fits at load.

| Unit kind | Units | Legal formats | Notes |
|---|---:|---|---|
| `L{i}.gdn.qkv/z/out` | 144 | NVFP4 · FP8 · Q4_K | FP8 needs one BF16 scale per output row |
| `L{i}.attn.q/k/v/o` | 64 | NVFP4 · Q4_K | attention only in layers 3, 7, …, 63 |
| `L{i}.mlp` | 64 | NVFP4 · Q4_K | gate/up/down move together |
| `lm_head` | 1 | NVFP4 · Q4_K | see rule 3 |

1. `bittrellis inventory` derives the 273 units from config and loader contract; tests assert the count.
2. **Q4_K is a runtime fit:** quantizer `runtime`, stored tensor = frozen BF16; no encoder choice or source-weight preparation.
3. **lm_head is dual.** Batch-1 decode executes a Q4_K fit of the stored bytes; a stored NVFP4 head also runs wide packed decode as NVFP4 if ≥ 3 GiB VRAM is free at load. Both paths are recorded:
   ```yaml
   lm_head: {source_format: NVFP4, quantizer: baseline@v1,
             execution: {decode_b1: Q4_K(nvfp4), packed_wide: "NVFP4 if free VRAM ... else Q4_K(nvfp4)"}}
   ```
4. A runtime Q4_K fit may come from the unit's legal stored representation (BF16; head: BF16 or NVFP4), making the shipped V0 legal (v2 §6 contradicted this).

## 3. Quantizers and lineage

Each unit gets `format @ quantizer@version [+params]` ([contract](quantizer_contract.md), [`bittrellis/quantizers/base.py`](../bittrellis/quantizers/base.py)). Lineage proves bytes legitimate:

| Lineage | Proof | Built-in |
|---|---|---|
| runtime | stored BF16 byte-identical to the base | `runtime` (Q4_K) |
| regenerable | sampled units rebuilt and byte-compared; `sequential` quantizers replay in pipeline order from the first unit | `rtn` (NVFP4, FP8) |
| attested | identical to a maintainer-approved source, every file hash-locked | `baseline` (shipped NVFP4), `unsloth` (calibrated NVFP4, MLP 0–55) |

1. Only a maintainer change to `sources.lock.json` adds an attested source.
2. A quantizer writes only its own Linear's tensors. Scale migration into norms or neighbours (SmoothQuant/AWQ-style) is illegal; in-tensor scale, clipping, rounding and act-order search is legal.
3. Reconstruction error vs round-to-nearest: warn > 2×, reject > 5× as substituted bytes (calibrated encoders reach 1.6×).
4. Regenerated units are cached across candidates. Sequential key: quantizer version + base revision + pipeline prefix (the quantizer's assignments up to and including the unit), because GPTQ-style tensors depend on everything quantized before. Independent quantizers key on the unit's own assignment.

## 4. Fidelity: Reference-Partition KL

1. **Reference:** BF16 runs once (transformers, CPU offload), storing top-256 ids and log-probs per scored position; hash-pinned per corpus and epoch.
2. **Fixed partition:** BF16 top-256 + one tail bucket, chosen before any candidate exists.
3. **Scoring:** [`tools/sparkinfer_refscore.cpp`](../tools/sparkinfer_refscore.cpp) links the *unmodified* pinned runtime library, drives SparkInfer's load → `cache_prefix` → `forward_token` path, and returns exact log-probs of the 256 ids, which shipped tools cannot (`qwen3_gguf_score` prints its own top-k; `/v1/score` caps at 20). Allowed: library, kernels and loader are untouched.
4. **Metric:** `RP-KL = Σ p_i log(p_i/q_i) + p_tail log(p_tail/q_tail)`, the KL between both projections: never above full-vocabulary KL, never called "KL". Outside-reference mass is a diagnostic.
5. **Uncertainty:** scoring is deterministic (`SPARKINFER_DETERMINISTIC=1`), so it is corpus-only: paired block bootstrap over contiguous 128-position blocks.

## 5. Corpora

- `public` ([`data/corpus/hpc01-public-v2.json`](../data/corpus/)): everyone; development set, official RP-KL.
- `public-validation`: everyone (rebuildable); local overfitting check, **not** a holdout.
- private holdout: validators only, rotated per epoch; PASS/FAIL gate ([holdout.md](holdout.md)).

Public: 77,824 tokens, 21,624 scored positions; 5 × 4K streams (general, math, code, tools,
multilingual) + 8K/16K/32K streams with three needles and a 384-position scored tail each.
Documents begin with `<|endoftext|>`. New categories (e.g. instruction following) need a new epoch.

## 6. Performance and memory

- **Runs:** exactly **two** independent processes, each one model load and a 128/4K/16K sweep with **512 decode tokens**; no repeats inside a run (`SPARKINFER_BENCH_SWEEP_REPS=1`).
- **Speed:** decode, 4K prefill = mean of both runs; spreads recorded.
- **Memory:** peak device memory over the run (weights, KV at `max_seq` 16,384, prefill arena, CUDA context) minus idle; 250 ms sampling makes it a lower bound.
- **Reported only:** resident memory after load, peak host RSS, checkpoint bytes.
- **16K, not the 262K default:** at 262K context allocation dominates and shifts throughput; a long-context memory track would be a new epoch.

## 7. Frontier

- **Ranked:** internal rows (legal manifests, pinned runtime). Incumbent **V0** = shipped checkpoint rebuilt byte for byte.
- **Seeded** with every maintainer-measured feasibility candidate (V1–V13); resubmissions earn nothing.
- **Never ranked** (no dominance or FG-2): R1 unsloth (FP8 attention runs as Q4_K refit, so not a legal manifest), R2 llama.cpp UD-Q4_K_M, R3 NVIDIA (compatibility boundary).

**ε-dominance** (only differences beyond noise count). A is *materially better* on:

| RP-KL | decode / prefill | peak memory |
|---|---|---|
| lower by > 0.002 and paired 95% CI excludes 0 | higher by > max(floor, either row's two-run spread); floors 1% / 3% | lower by > 0.1 GiB |

A dominates B if materially better somewhere and worse nowhere. **FG-2** = the normalized 4-D dominated hypervolume a valid internal row adds, **noise-aware**: the row is first handicapped by the ε floors on every objective (RP-KL +0.002, decode and prefill −max(floor, spread), memory +0.1 GiB), and it earns nothing unless it is materially better than every other valid row on some objective; versioned. The paid tier is a fixed bucket of FG-2 (`rewards.tiers_fg2`: `eval:XL` ≥ 0.50%, `L` ≥ 0.25%, `M` ≥ 0.12%, `S` ≥ 0.035%, `XS` ≥ 0.005%), never a manual judgment; `eval:none` for no gain, `eval:REJECT` for a failed gate, audit or screen ([rewards.md](rewards.md)).

## 8. Gates

Frontier entry requires all of:

1. **Audit:** sources, config, tensor set, frozen bytes, execution map, lineage, anomaly.
2. **Runtime correctness:** every scoring/benchmark process loads and exits 0; no NaN/Inf log-probs; argmax ids in vocabulary; executed map = manifest; no unpinned `SPARKINFER_*` reaches the runtime.
3. **Fidelity:** RP-KL ≤ 0.30, top-1 ≥ 0.80.
4. **Long context:** every needle BF16 retrieves at 8K/16K/32K is retrieved (success 1.0 per length). Long-context RP-KL is reported, not gated.
5. **Tasks:** all 784 SparkInfer quality questions, compared with V0 question by question: fail if losses significantly exceed gains (exact one-sided McNemar, p < 0.05 overall, p < 0.01 per suite) or any suite keeps fewer than half of V0's passes.
6. **Holdout:** PASS (§9).

## 9. Holdout

Private text, same stream structure, validator-held seed and epoch; only `holdout.json` = `{epoch, result}` is published.

**PASS** needs all of: valid audit · holdout RP-KL ≤ 0.30 · every retrievable holdout needle kept ·
if the public RP-KL gain over V0 is significant, a holdout gain over V0 of ≥ 50% of it.

- **Probing defense:** PASS/FAIL only, per-epoch rotation, no item output, rate limits.
- **Accepted risk:** attested sources' calibration data is unknown, so non-overlap cannot be proven; the holdout is the protection.

## 10. Evaluation flow

```text
observe PR heads ─▶ SCREEN (no GPU): queue share · validate · duplicate · near-copy · memory · new-encoder bytes
        │
   build (sources hash-verified) ─▶ audit ─▶ new encoder's stored bytes ──fail──▶ reject
        │
   public RP-KL + correctness + needles ──gate fail──▶ stop
        │
   2 performance runs (decode, 4K prefill, peak GPU, host RAM) ──dominated──▶ stop (tasks, holdout skipped)
        │
   task guard ─▶ private holdout (PASS/FAIL) ─▶ gates ─▶ ε-frontier vs seeds + accepted + earlier open PRs ─▶ FG-2
```

[`evaluator/pr_bot.py`](../evaluator/pr_bot.py) runs it. Manifest-only PRs use trusted `main` code; contributed code (quantizers, search) waits for a maintainer's `eval-approved` label; PRs touching evaluator paths are never auto-evaluated.

**Copies** are judged by expanded recipe and encoder output bytes, not source text. The first-observed head is the original; a later PR ranks against other authors' earlier open PRs on the frontier, earning only what it adds ([guards.md](guards.md)). **Contributed quantizer code** runs only in an unprivileged sandbox account (expand, probe, build, regenerate); trusted `main` code audits and measures its output. **Payment:** each pass the bot marks one `bt:merge-first` result and merges it at the head SHA it measured, re-checking every condition first; the tier on a merged PR is final, and Gittensor pays that tier. **No private holdout PASS, no paid tier:** such a result is `bt:provisional` ([rewards.md](rewards.md)).

## 11. Artifacts

```text
artifacts/<name>/
  candidate.json     manifest, expanded assignments (source format, quantizer@version, execution), build record
  audit.json         audit result, lineage per quantizer, anomaly ratios
  quality.json       RP-KL, CI, per category, outside mass, needles by length
  correctness.json   runtime-correctness checks and the effective scoring environment
  tasks.json         task guard
  performance.json   both runs, spreads, peak GPU, resident, host RAM, protocol
  holdout.json       {epoch, result}
  environment.json   GPU, driver, CUDA, kernel, SparkInfer commit, evaluator epoch, effective env
  kl_positions.npz   per-position RP-KL for paired comparisons
```

Cache key: candidate id (includes every unit's quantizer name and version) + evaluator epoch, so **a quantizer code change must bump its version**.

## 12. Realistic results

The shipped map already uses the fastest kernels, Q4_K and FP8 cost prefill, and no map beats V0's
decode beyond noise. Credible headlines: fidelity at a stated small cost ("RP-KL −11.5% at unchanged
decode and memory, −4.4% prefill"), memory savings with prefill cost stated, or long-context fidelity
at the same execution format. +x% decode/prefill claims need a measured, beyond-ε result.

## 13. What does not count

Metadata relabeling or stored-format changes with the same execution · Q4_K "quantizers" · edits to
frozen source weights · env-variable tricks · public-corpus overfitting · architecture changes ·
fine-tuning disguised as quantization · disk-size savings with no runtime effect.

## 14. Changes from v2

1. Unmodified-library scorer joins the evaluator (§4): pinned binaries cannot emit log-probs for chosen ids.
2. Q4_K fits from the legal stored representation (§2), resolving the V0 legality contradiction.
3. ε has floors (§7): a two-sample spread can be near zero by chance.
4. Holdout PASS rule defined (§9): public RP-KL ranks; the holdout guards transfer.
5. Replay cached by pipeline prefix; cache key includes quantizer versions (§3, §11): configuration-only sharing is wrong for sequential encoders.
6. Benchmarks split in reporting, not processes (§6): separate commands multiply model loads.

Epoch decisions: 16K sweep for peak memory (§6); feasibility candidates seed the frontier (§7).
