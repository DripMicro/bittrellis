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
| **Label multiplier** | the tier below (`label_multipliers` in the registry entry); no tier label means ×0 |
| **Time decay** | half the score is gone about 10 days after merge |
| **Review factor** | each maintainer "changes requested" review lowers it |
| **Credibility gate** | merged ÷ (merged + closed), against the registry entry's `min_credibility`. SparkInfer sets 0.2; we propose 0.0 here, so closing a passed-by recipe costs you nothing |
| **Spam factor** | too many open PRs in the repository sets your score there to 0 |

## Tiers

FG-2 ([key terms](../README.md#key-terms)) buckets into SparkInfer's tiers. Thresholds: `rewards.tiers_fg2` in [`configs/hpc01.yaml`](../configs/hpc01.yaml), calibrated to the seeds.

| Label | Noise-aware FG-2 | Proposed multiplier | Seed at this level |
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

## Merging

Merging is payment, so the bot merges only what it has just measured and re-checked.

1. Each pass the bot marks **one** open result `bt:merge-first`: highest tier, then largest FG-2, then first observed.
2. It merges that one itself, at the exact head SHA it measured, and it joins the frontier. The merge is
   refused — and simply retried next pass — if the head moved, the PR is a draft, GitHub does not call the
   branch clean (a conflict or a failing check), or contributed code is missing `eval-approved`. Run the
   evaluator with `BT_AUTO_MERGE=0` to leave merging to a maintainer instead.
3. Next pass, other open results are re-ranked against it; one whose gain the merge covered drops to `eval:none`.
4. **A merged PR's tier is final**; the bot never relabels it.

The bot closes rejected (`eval:REJECT`) and duplicate PRs itself, with a comment: they cannot earn, and an open PR counts toward your open-PR limit. Closing costs no credibility here (`min_credibility` is 0). Dominated PRs stay open, because a re-rank can revive them; close them yourself once you stop pursuing them, since they reserve collateral and count toward the spam limit.

## The public record

After every pass the evaluator writes and pushes a score record to [its own repository](https://github.com/coderbench/bittrellis-ledger)
([`evaluator/ledger.py`](../evaluator/ledger.py), [`publish_ledger.py`](../evaluator/publish_ledger.py)):
one write-once record per evaluated PR head (status, tier, FG-2, measured row, screen result), the
first-seen records, the current frontier, and the artifacts of merged results. The GPU box is rented;
the record outlives it. Re-derive any score yourself:

```bash
git clone https://github.com/coderbench/bittrellis-ledger && cd bittrellis-ledger
bittrellis frontier hpc01-e3/accepted <your artifact>
```

The private holdout never appears there: records carry PASS or FAIL only.

**The record is what a replacement box restores from.** A rented box is replaced, not repaired, and
everything that decides what a contributor is owed lived only on its disk. On startup the evaluator
fetches the published history into its ledger directory and takes back the first-seen records
(submission priority and copy credit) and the accepted artifacts (what later PRs are ranked against),
so the same submission keeps its place and its score across a box change. Both stores are write-once,
so restoring only copies what is missing. A push that fails is retried on the next pass even when
that pass wrote nothing — one network failure must not strand the history.

## Listing status

**BitTrellis is not in `master_repositories.json` yet, so every tier below is worth ×0 today.** As of
2026-09-23 the registry lists `gittensor-ai-lab/sparkinfer` and `gittensor-model-hub/spark-hermes`.
Until an entry for `coderbench/bittrellis` is merged there, the evaluator still measures, labels and
publishes score records — a merged PR simply earns no emissions. The proposal below is what we are
asking for; the Gittensor team sets the final numbers.

## Proposed registry entry

For Gittensor's `gittensor/validator/weights/master_repositories.json`:

```json
"coderbench/bittrellis": {
  "default_label_multiplier": 0.0,
  "eligibility": {"min_credibility": 0.0, "min_issue_credibility": 0.0, "min_valid_merged_prs": 0, "min_valid_solved_issues": 0},
  "emission_share": "<set by Gittensor>",
  "fixed_base_score": 1.0,
  "issue_discovery_share": 0.0,
  "label_multipliers": {"eval:XL": 4.0, "eval:L": 2.5, "eval:M": 1.5, "eval:S": 1.0, "eval:XS": 0.5, "eval:none": 0.0, "eval:REJECT": 0.0},
  "maintainer_cut": 0.5,
  "scoring": {
    "standard_issue_multiplier": 1.0,
    "maintainer_issue_multiplier": 1.0,
    "time_decay": {"grace_period_hours": 4, "sigmoid_midpoint_days": 3.33, "sigmoid_steepness": 1.2, "min_multiplier": 0.05}
  },
  "trusted_label_pipeline": true
}
```

Where this departs from SparkInfer's entry, and why:

| Field | SparkInfer | Here | Why |
|---|---|---|---|
| `min_credibility` | 0.2 | **0.0** | A search track asks authors to close recipes the frontier has passed by. Charging them credibility for that would punish exactly the housekeeping the queue depends on — and the spam factor already caps open PRs. The newest accepted entry (spark-hermes) sets 0.0 for the same reason. |
| `maintainer_cut` | 0.4 | **0.5** | Matches the newest accepted entry. |
| `time_decay` | default (~10-day half-life) | **explicit, faster** | Copied from spark-hermes. A candidate's value decays as the frontier moves, not on a calendar, and a short window keeps the board honest. |
| `label_multipliers` | tiers | **tiers** | Kept. BitTrellis evaluates continuously rather than in rounds, so there is no single per-round winner to crown — several results can each add frontier space at once. |

`trusted_label_pipeline: true` accepts the evaluator account's labels whatever its GitHub association.
Only maintainers and the evaluator can label here, so miners cannot label their own PRs.
