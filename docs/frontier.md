# Frontier and Frontier Gain (FG-2)

> Which results move the frontier, and what do they earn? See [Key terms](../README.md#key-terms).

**Internal rows** (legal manifests on the pinned SparkInfer) enter dominance and FG-2: incumbent V0
(shipped Gittensor checkpoint, rebuilt byte for byte from a manifest), every maintainer-measured
seed (`results/feasibility/artifacts`), and accepted miner results. The frontier never starts empty;
resubmitting a seed earns nothing. **External rows** (R1 unsloth checkpoint, R2 llama.cpp UD-Q4_K_M,
R3 NVIDIA compatibility boundary) are context only.

## Section-balanced drift (hpc01-e4)

The fidelity objective is RP-KL made **section-balanced**: V0's RP-KL × the mean, over sections, of the
candidate's drift relative to V0's in that section. Sections are the five short categories (general,
maths, code, tools, multilingual) and the three long-context streams together. V0 keeps its own RP-KL,
and halving drift counts the same in every section.

In hpc01-e3 RP-KL was the plain mean over positions, so the section where V0 drifts most decided most of
every score: for about half of all results most of the public gain came from the maths stream alone. The
plain mean stays in `frontier.json` as `rp_kl_raw`. Paired comparisons (dominance, the holdout transfer
test) use the same weights, so every test measures the same quantity.

## ε-dominance

A difference counts only beyond noise margin ε. *Materially better* means:

| Objective (box) | Measure | Margin |
|---|---|---|
| Fidelity (0–0.30) | RP-KL vs BF16, lower | > 0.002 **and** paired 95% interval excludes 0 |
| Decode (60–120) | tok/s @ 4K, batch 1, mean of 2 runs, higher | > max(1%, either row's two-run spread) |
| Prefill (2,000–20,000) | tok/s @ 4K, mean of 2 runs, higher | > max(3%, either row's two-run spread) |
| Memory (14–32) | peak GPU GiB over a run, lower | > 0.1 GiB |

A dominates B if materially better somewhere and materially worse nowhere. The frontier: valid
internal rows no other valid internal row dominates. Prefill counts because several maps save
memory at equal fidelity and decode by giving up 20–50% of prefill: free on three axes, a trade-off on four.

## FG-2

The share of normalized 4-D space a result adds **beyond noise**. Map valid internal rows into the unit
box (1 = best edge per axis); HV is the dominated hypervolume, the union of boxes `[0, point]`:

```text
FG-2(candidate) = HV(frontier ∪ {handicap(candidate)}) − HV(frontier)
handicap: RP-KL + 0.002 · decode × (1 − max(1%, spread)) · prefill × (1 − max(3%, spread)) · memory + 0.1 GiB
```

- **Only gains beyond noise count.** The handicap removes the part of every improvement that is within
  the ε floors, so a copy of V0 that differs by 0.01 tok/s earns exactly 0.
- **Distinct or nothing.** A result earns only if it is materially better than *every* other valid
  internal row on at least one objective (the same test as dominance, RP-KL significance included).
- Dominated, invalid and external rows earn 0. Any axis counts: a slower but much more faithful map
  earns like a faster one.

Paid tiers (`eval:XL` … `eval:XS`) are fixed buckets of FG-2, never manual judgments
([rewards.md](rewards.md)); formula, box and floors are versioned with the evaluator epoch. Rank:
`bittrellis frontier --with-seeds artifacts/mine`.
