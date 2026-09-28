# Rewards: from a measured result to TAO

> [Gittensor](https://github.com/entrius/gittensor) (Bittensor subnet 74) pays a merged PR by one `eval:*` label. This page shows how your result earns that label and what it is worth.

## How Gittensor pays

```text
earned = fixed_base_score × label multiplier × time decay × review factor × open-PR spam factor
```

Source: [`scoring.py`](https://github.com/entrius/gittensor/blob/main/gittensor/validator/oss_contributions/scoring.py),
[`label_resolution.py`](https://github.com/entrius/gittensor/blob/main/gittensor/validator/oss_contributions/label_resolution.py).
Your emission share: your earned score against everyone else's.

| Factor | For a miner |
|---|---|
| **Merged only** | an open PR earns nothing; it reserves collateral (20% of its potential score) |
| **Label multiplier** | the tier below; no tier label means ×0 |
| **Time decay** | full score for 4 hours after merge, half of it by about 3.3 days, never below 5% |
| **Review factor** | each maintainer "changes requested" review lowers it |
| **Credibility gate** | merged ÷ (merged + closed); the minimum here is 0.0, so closing a passed-by recipe costs you nothing |
| **Spam factor** | too many open PRs in the repository sets your score there to 0 |

## Tiers

FG-2 ([key terms](../README.md#key-terms)) buckets into SparkInfer's tiers. Thresholds: `rewards.tiers_fg2` in [`configs/hpc01.yaml`](../configs/hpc01.yaml), calibrated to the seeds.

| Label | Noise-aware FG-2 | Multiplier | Seed at this level |
|---|---|---:|---|
| ![eval:XL](https://img.shields.io/badge/eval%3AXL-0e8a16?style=flat-square) | ≥ 0.50% | ×4.0 | V0, the shipped map (2.25%) |
| ![eval:L](https://img.shields.io/badge/eval%3AL-2da44e?style=flat-square) | ≥ 0.25% | ×2.5 | — |
| ![eval:M](https://img.shields.io/badge/eval%3AM-4ac26b?style=flat-square) | ≥ 0.12% | ×1.5 | — |
| ![eval:S](https://img.shields.io/badge/eval%3AS-8ddb8c?style=flat-square) | ≥ 0.035% | ×1.0 | V4 (0.10%), V1 all Q4_K (0.09%) |
| ![eval:XS](https://img.shields.io/badge/eval%3AXS-c6efce?style=flat-square) | ≥ 0.005% | ×0.5 | V6 (0.02%) |
| ![eval:none](https://img.shields.io/badge/eval%3Anone-bfc5cc?style=flat-square) | below 0.005%, dominated, or a duplicate | ×0 | V7, V9; a copy of V0 within noise |
| ![eval:REJECT](https://img.shields.io/badge/eval%3AREJECT-b60205?style=flat-square) | failed a gate, the audit, the holdout or a screen | ×0 | V13 and V3 (holdout), V5 (task guard) |

FG-2 only counts gains beyond measurement noise ([frontier.md](frontier.md#fg-2)).

- A PR still waiting (queue, approval, maintainer) has **no** tier label.
- **No private holdout PASS, no paid tier.** A result measured without one is `bt:provisional`.
- FG-2 counts merged results and earlier open PRs by other authors: a near-copy earns only what it adds ([guards.md](guards.md)).
- Only maintainers and the evaluator can label PRs here, so a miner cannot label their own.

## Merging

Merging is payment, so the bot merges only what it has just measured and re-checked.

1. Each pass the bot marks **one** open result `bt:merge-first`: highest tier, then largest FG-2, then first observed.
2. It merges that one itself, at the exact head SHA it measured, and it joins the frontier. The merge is
   refused — and simply retried next pass — if the head moved, the PR is a draft, GitHub does not call the
   branch clean (a conflict or a failing check), or contributed code is missing `eval-approved`. Run the
   evaluator with `BT_AUTO_MERGE=0` to leave merging to a maintainer instead.
3. Next pass, other open results are re-ranked against it; one whose gain the merge covered drops to `eval:none`.
4. **A merged PR's tier is final**; the bot never relabels it.

The bot closes every PR that cannot earn, with a comment: rejected (`eval:REJECT`), duplicate, dominated, and results on the frontier below the lowest tier (`eval:none`). An open PR counts toward your open-PR limit and reserves collateral, and closing costs no credibility here (`min_credibility` is 0). A revised recipe is welcome as a new PR; if a result you were ranked against later closes, reopen yours to have it re-scored.

## The public record

After every pass the evaluator writes and pushes a score record to [its own repository](https://github.com/coderbench/bittrellis-ledger)
([`evaluator/ledger.py`](../evaluator/ledger.py), [`publish_ledger.py`](../evaluator/publish_ledger.py)):
one write-once record per evaluated PR head (status, tier, FG-2, measured row, screen result), the
first-seen records, the current frontier, and the artifacts of merged results. The GPU box is rented;
the record outlives it. Re-derive any score yourself:

```bash
git clone https://github.com/coderbench/bittrellis-ledger && cd bittrellis-ledger
bittrellis frontier hpc01-e4/accepted <your artifact>
```

The private holdout never appears there: records carry PASS or FAIL only.

**The record is what a replacement box restores from.** A rented box is replaced, not repaired, and
everything that decides what a contributor is owed lived only on its disk. On startup the evaluator
fetches the published history into its ledger directory and takes back the first-seen records
(submission priority and copy credit) and the accepted artifacts (what later PRs are ranked against),
so the same submission keeps its place and its score across a box change. Both stores are write-once,
so restoring only copies what is missing. A push that fails is retried on the next pass even when
that pass wrote nothing — one network failure must not strand the history.
