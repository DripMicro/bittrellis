# Guards: screening and copies

> What runs before GPU time, who counts as first, and what a copy earns.

Code: [`evaluator/guards.py`](../evaluator/guards.py), [`bittrellis/fingerprint.py`](../bittrellis/fingerprint.py),
[`evaluator/pr_bot.py`](../evaluator/pr_bot.py). Thresholds: `evaluation.screen` in
[`configs/hpc01.yaml`](../configs/hpc01.yaml). Terms: [README key terms](../README.md#key-terms).

## The whole flow

```text
 observe every open PR head (first-seen record)  →  queue in first-seen order
                                                          │
 SCREEN (CPU, seconds) ───────────────────────────────────┤
   1 queue share       an author's 4th waiting PR waits its turn        bt:queued
   2 manifest          valid, legal, right track                        bt:invalid-manifest
   3 duplicate         same expanded recipe as a seed, an accepted      bt:duplicate
                       result or an earlier PR
   4 near-copy         ≤ 2% of weights differ from an earlier PR by     bt:derivative (measured)
                       another author                                   bt:copy-review (3rd in an epoch)
   5 memory            predicted peak > 31.5 GiB                         bt:memory
   6 new quantizers    deterministic, and not an existing encoder       bt:nondeterministic
                       (probe bytes, run in the sandbox)                bt:same-encoder
                                                          │
 BUILD (sandbox for code PRs) + AUDIT (trusted, CPU) ────┤
   7 stored bytes      a new encoder's real bytes vs every existing     bt:same-encoder
                       encoder's bytes for the same tensors
                                                          │
 MEASURE (GPU) ───────────────────────────────────────────┤
   quality ~4.5 min → quality gates fail? stop                          bt:gate-fail
   2 speed runs ~1 min → dominated? stop, skip tasks + holdout          bt:dominated
                       → below the lowest tier (XS)? stop, same skip    bt:frontier + eval:none
   tasks ~4 min → private holdout → rank                                bt:frontier / bt:dominated / bt:gate-fail
```

Stage times: evaluator dry run on the pinned RTX 5090 (PR #1). A dominated result uses about 6 GPU
minutes instead of about 15.

**Skipping is safe:** tasks and the holdout can fail a result, never lift one, so a result that is dominated or
already below XS ends the same way without them. If a re-rank later
makes a skipped result non-dominated or lifts it to a tier (e.g. an earlier PR it was ranked against closes), the bot
rebuilds it and runs the skipped stages before labelling it `bt:frontier`.

## Who was first

```text
<root>/observations/pr-000042-3f9c1a2b7d10.json   {pr, author, head, first_seen, candidate_id, keys, ...}
```

- Written once, outside every checkout; `first_seen`, `author` and `head` never change.
- **A force-push is a new head** with its own time: an early empty PR later filled with copied content is not the original.
- All open heads are observed **before** any evaluation, so an original is measured before its near-copy.

## Recipes: judged by the expanded recipe

Manifests are compared after expansion to all 273 units (`FORMAT@quantizer@version[+params]` per
unit): rewritten rules, reordered rules or other layer ranges that expand identically are the same recipe.

| Compared with | Rule |
|---|---|
| seed, accepted result, open or merged PR observed earlier | same candidate id = duplicate |
| earlier open or merged PR by **another** author | share of searchable **weights** assigned differently ≤ `near_copy_max_share` = near-copy, ranked with the original |
| your own earlier PR | never a copy: iterating is normal |
| a closed, unmerged PR | never a reference: it earns nothing, so its idea is free to build on |

Distance is weighted by parameter count: one attention projection moves far less than 2%, a block of MLP layers far more.

## Credit: only what you add

A PR's FG-2 is the space it adds over seeds, accepted results **and earlier measured, still-open PRs by other authors**.

- A pure copy earns 0, even past the screen; a real improvement earns exactly the improvement.
- When an earlier PR merges or closes, the bot **re-ranks** every later PR measured against it.
- Nothing is blocked automatically: the space is small and `bittrellis search neighbors` ships, so honest miners often find the same recipe.

## Quantizers: judged by the bytes they produce

Rewritten code does not change output, and regenerable quantizers are deterministic, so bytes identify an encoder.

1. **Probe (before build).** Each new quantizer encodes, twice, a tiny synthetic model seeded from the evaluator's secret; both runs must be byte-identical. ≥ `quantizer_same_bytes` (99%) identical bytes with any quantizer on `main` or an earlier PR's new quantizer = same encoder.
2. **Stored bytes (after build, before GPU).** Probes have recognisable tiny shapes, so bytes sampled at secret offsets from up to four units the new quantizer encoded are compared with every existing encoder's bytes for the same tensors, and with earlier PRs' stored bytes. `rtn` runs on the real weights; `baseline` and `unsloth` are read from their pinned sources, across both NVFP4 layouts. This catches a "new" encoder that splices attested bytes.

A version bump with identical bytes (`rtn@v2` = `rtn@v1`) is also the same encoder.

## Isolation: contributed code produces, trusted code judges

Contributed code runs **only** as the unprivileged account `bt-sandbox`
([`evaluator/sandbox.py`](../evaluator/sandbox.py)), in four CPU steps:

| Step | Runs as | Can read | Writes |
|---|---|---|---|
| expand the recipe | sandbox | its checkout, the models | `untrusted/ids.json` |
| probe the new encoder | sandbox | same | `untrusted/probe*.npz` |
| build the checkpoint | sandbox | same | `untrusted/checkpoint`, then **sealed** (moved to a root-only directory) |
| regenerate the audit samples | sandbox | same, but **not** the sealed checkpoint | `untrusted/regen`, then sealed |
| audit, quality, speed, tasks, holdout, ranking | evaluator, `main` code | everything | the artifact |

- **Regeneration without the answer.** Regenerated units are chosen from the candidate id and the evaluator's secret, so a submission cannot predict them or copy bytes from its sealed checkpoint. The trusted audit compares them byte for byte and never imports the contributed quantizer; it knows it only by name and version.
- **Allow-listed environment.** `runuser -u bt-sandbox -- env -i` with only `HOME`, `PATH`, `CUDA_VISIBLE_DEVICES=""` (no GPU), `PYTHONDONTWRITEBYTECODE=1` and `BITTRELLIS_REPLAY_CACHE`, plus `LANG`/`LC_ALL`/`TZ` when set. No token, holdout path or secrets.
- **Clean between steps.** The account's processes are killed; its writes to its home, `/tmp`, `/var/tmp` and `/dev/shm` are deleted.
- **No network.** Outbound traffic must be blocked (the setup script adds an `iptables` owner rule), or a build could upload its checkpoint and regeneration download it back.
- **Refuses an unsafe host.** The account must not read `secret.txt`, `state.json` or the token file, read or write the private holdout, accepted results or the first-seen record, or reach the network, and must read the models. Otherwise: `bt:eval-error`, nothing runs.
- **New code needs a new name.** Changed bytes under an existing quantizer's name fail the audit (the trusted side runs the old code).

Set up once, as root:

```bash
BT_EVAL_ROOT=/workspace/bt-eval BT_PRIVATE=/secure/holdout-epoch evaluator/setup_sandbox.sh
```

## Labels

Green: credit. Grey: nothing new. Blue/yellow: waiting. Red: failure (darker is worse). Purple: evaluator's fault. Meanings follow badge order.

| Group | Labels | Meaning |
|---|---|---|
| Credited | ![bt:frontier](https://img.shields.io/badge/bt%3Afrontier-0e8a16?style=flat-square) | moves the internal frontier |
| Partial credit | ![bt:derivative](https://img.shields.io/badge/bt%3Aderivative-fff3b0?style=flat-square) | near-copy of another author's PR; credited only for what it adds |
| Nothing new | ![bt:dominated](https://img.shields.io/badge/bt%3Adominated-bfc5cc?style=flat-square) ![bt:duplicate](https://img.shields.io/badge/bt%3Aduplicate-e1e4e8?style=flat-square) | another result is at least as good on every objective · known recipe, not measured |
| Waiting | ![bt:queued](https://img.shields.io/badge/bt%3Aqueued-c5def5?style=flat-square) ![bt:needs-approval](https://img.shields.io/badge/bt%3Aneeds--approval-fbca04?style=flat-square) ![bt:copy-review](https://img.shields.io/badge/bt%3Acopy--review-e99a1c?style=flat-square) ![bt:touches-evaluator](https://img.shields.io/badge/bt%3Atouches--evaluator-1d76db?style=flat-square) ![bt:provisional](https://img.shields.io/badge/bt%3Aprovisional-bfd4f2?style=flat-square) | author's earlier PRs are ahead · contributed code, needs `eval-approved` · repeated near-copies, needs `copy-cleared` · evaluator paths, maintainer review, never evaluated · measured without a private holdout PASS, no paid tier |
| Fix your submission | ![bt:invalid-manifest](https://img.shields.io/badge/bt%3Ainvalid--manifest-f4a6a6?style=flat-square) ![bt:build-fail](https://img.shields.io/badge/bt%3Abuild--fail-f4a6a6?style=flat-square) ![bt:memory](https://img.shields.io/badge/bt%3Amemory-e8590c?style=flat-square) | manifest invalid · checkpoint or probe did not build · would exceed GPU memory, not measured |
| Failed | ![bt:gate-fail](https://img.shields.io/badge/bt%3Agate--fail-d73a4a?style=flat-square) ![bt:nondeterministic](https://img.shields.io/badge/bt%3Anondeterministic-d73a4a?style=flat-square) | failed a quality, task, runtime or holdout gate · identical runs gave different bytes |
| Integrity | ![bt:audit-fail](https://img.shields.io/badge/bt%3Aaudit--fail-b60205?style=flat-square) ![bt:same-encoder](https://img.shields.io/badge/bt%3Asame--encoder-b60205?style=flat-square) | not a legal encoding of the pinned weights · reproduces an existing encoder's bytes |
| Evaluator's fault | ![bt:eval-error](https://img.shields.io/badge/bt%3Aeval--error-8250df?style=flat-square) | not the submission's fault; retried up to three times per head |
| Merge order | ![bt:merge-first](https://img.shields.io/badge/bt%3Amerge--first-2da44e?style=flat-square) | highest-scoring open result; merge this one first |
| Maintainer actions | ![eval-approved](https://img.shields.io/badge/eval--approved-0e8a16?style=flat-square) ![copy-cleared](https://img.shields.io/badge/copy--cleared-0e8a16?style=flat-square) | evaluate contributed code in the sandbox · measure a PR held in `bt:copy-review` |
| Paid tier | `eval:XL` … `eval:XS`, `eval:none`, `eval:REJECT` | the label Gittensor pays, see [rewards.md](rewards.md) |

State: `<root>/state.json`. Every pass also publishes the first-seen records and each PR's outcome to the public score record ([rewards.md](rewards.md#the-public-record)).

## Known limits

- **The sandbox is an account, not a VM.** It shares the kernel; `eval-approved` is a real review.
- **Accidental near-copies.** Independent recipes can collide; the later one is still credited for what it adds, and one near-copy never triggers review.
- **Sketches sample, not prove.** 65,536 bytes per tensor at secret offsets tells encoders apart (unrelated encoders share far fewer identical bytes); it is not a proof of identity.
