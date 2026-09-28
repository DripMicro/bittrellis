# Private holdout (validators)

> Is a fidelity gain real, or fitted to the public corpus miners can calibrate against?

- **Private text:** never published. A public-dataset split is not a holdout (anyone can rebuild it); `public-validation` is for local checks only.
- **Same structure:** five 4K category streams plus 8K, 16K, 32K needle streams, same code (`assemble_streams`).
- **PASS/FAIL only:** the artifact gets `{"epoch": ..., "result": "PASS"}`; scores, positions, failing items stay private.
- **Rotation:** a new epoch (new text, seed) replaces the old on a declared schedule; retired text may be published for audits and disputes.

Provisioning a box and restoring this directory: [evaluator_runbook.md](evaluator_runbook.md).

## Set up and check

```text
/secure/holdout-2026w38/            never inside the repository, never synced to public storage
  epoch.json                        {"epoch": "hpc01-h2026w38", "seed": "<long random secret>"}
  docs/general/*.txt                one document per file, scored as written
  docs/math/*.txt                   (chat-formatted Q&A where that matters)
  docs/code/*.txt
  docs/tools/*.txt
  docs/multilingual/*.txt
  docs/long/*.txt                   long filler for the 8K/16K/32K needle streams (≥ 60K tokens total)
```

Add text a bit at a time; the inventory says what is still missing:

```bash
bittrellis holdout inventory --private /secure/holdout-2026w38
#   category       files   tokens    needs  status
#   general            3    7,412    4,096  ok
#   math               0        0    4,096  needs 4,096 more tokens (~3,072 words)
```

Text must never have been published, and must not be model-generated: both make the check meaningless.

```bash
bittrellis holdout build --private /secure/holdout-2026w38
bittrellis reference --corpus /secure/holdout-2026w38/corpus.json --out /secure/holdout-2026w38/reference
bittrellis holdout check models/candidates/<name> --private /secure/holdout-2026w38 \
    --artifact artifacts/<name> --incumbent-artifact results/feasibility/artifacts/V0-baseline-rebuild
```

Build fails loudly if a short category has under 4,096 tokens. The PR bot runs the check when started
with `--private`; V0 is scored once per epoch, cached.

## PASS rule (epoch hpc01-e4)

1. audit passed;
2. holdout RP-KL ≤ `gates.rp_kl_max` (0.30);
3. keeps every holdout needle BF16 retrieves;
4. **transfer:** if the public drift gain over V0 is above the RP-KL floor (the gain FG-2 would credit), the
   holdout gain over V0 is ≥ `evaluation.holdout.min_gain_ratio` (0.5) of it. Both gains are *relative* and
   *section-balanced* ([frontier.md](frontier.md#section-balanced-drift-hpc01-e4)).

Why it changed from hpc01-e3, which compared absolute nats and only for significant public gains:

- the holdout text drifts about 2.8× less than the public text, so even a gain that carried over fully
  showed about 36% in absolute terms, under the 50% bar;
- the public maths stream (GSM8K worked answers), where V0 drifts most, carried most of many results'
  gains, and those gains did not appear on the holdout even for untuned seeds;
- a gain too uncertain to be significant skipped the comparison entirely, so weaker evidence faced the
  easier test.

Re-judged under hpc01-e4 from the cached holdout positions, V3 and #15 pass and #17 fails; every other
result checked keeps its verdict. `bittrellis holdout recheck` re-judges a result already scored on this
holdout without a GPU.

A paid tier requires an explicit PASS: an evaluator without a private holdout labels results `bt:provisional`. Ranking uses reproducible *public* RP-KL; the holdout only rejects gains that do not carry over, so
miners can optimize public fidelity only until it stops generalizing.

## Known limits

- **Unknown calibration data:** attested sources (e.g. unsloth's) used data nobody here can inspect; holdout overlap cannot be proven absent. The holdout mitigates; risk is accepted per source.
- **Probing:** even PASS/FAIL leaks a little per submission; evaluators should rate-limit per miner and rotate epochs.
