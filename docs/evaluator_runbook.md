# Evaluator runbook

> Rent a GPU, get the evaluator running, and shut it down again — copy-paste commands.

Everything below assumes Ubuntu 24.04, one RTX 5090, root over SSH, and that you keep a backup of the
private holdout ([holdout.md](holdout.md)). Times are from a real run on 2026-09-18.

## 1. Provision the box (~1.5–2 h, mostly downloads)

```bash
ssh -i ~/.ssh/<key> -p <port> root@<host>

curl -fsSL https://raw.githubusercontent.com/coderbench/bittrellis/main/scripts/provision_box.sh | bash
```

It installs CUDA 12.8, CMake, Rust, the repo and a virtualenv, downloads the three pinned models
(~90 GiB), builds SparkInfer and the scorer, computes the public BF16 reference, and ends with
`bittrellis doctor`. Re-running it skips whatever is already done.

To keep the session alive, run it inside `tmux`:

```bash
tmux new -d -s prov 'curl -fsSL https://raw.githubusercontent.com/coderbench/bittrellis/main/scripts/provision_box.sh | bash > /workspace/provision.log 2>&1'
tail -f /workspace/provision.log
```

## 2. Restore the private holdout (~3 min)

From the machine holding your backup:

```bash
scp -i ~/.ssh/<key> -P <port> -r ./holdout-backup root@<host>:/secure/holdout-2026w38
```

Then on the box:

```bash
export PATH=/workspace/bittrellis/venv/bin:$PATH
cd /workspace/bittrellis/repo
chmod -R go-rwx /secure

bittrellis holdout inventory --private /secure/holdout-2026w38    # every category must say "ok"
[ -f /secure/holdout-2026w38/corpus.json ] || bittrellis holdout build --private /secure/holdout-2026w38
[ -d /secure/holdout-2026w38/reference ]   || bittrellis reference --corpus /secure/holdout-2026w38/corpus.json \
                                                 --out /secure/holdout-2026w38/reference
```

Keep `epoch.json` from the backup: its seed fixes how your text is arranged, so results stay
comparable with earlier epochs. A new epoch means new text and a new seed.

## 3. Isolate contributed code (~1 min, once)

```bash
BT_EVAL_ROOT=/workspace/bt-eval BT_PRIVATE=/secure/holdout-2026w38 evaluator/setup_sandbox.sh
runuser -u bt-sandbox -- test -r /workspace/bt-eval/secret.txt && echo UNSAFE || echo ok
```

This creates the `bt-sandbox` account and blocks its outbound traffic. Without it the bot refuses
quantizer PRs ([guards.md](guards.md#isolation-contributed-code-produces-trusted-code-judges)).

## 4. Run the evaluator

```bash
umask 077
mkdir -p /workspace/bt-private
printf '%s' '<github-token>' > /workspace/bt-private/.gh_token
chmod 600 /workspace/bt-private/.gh_token
# fine-grained token on bittrellis + bittrellis-ledger:
#   Contents: write (merges, ledger pushes) · Pull requests: write · Issues: write (labels, comments)
# Or, from your own machine, without typing it again:
#   ssh -p <port> -i ~/.ssh/<key> root@<host> \
#       'umask 077; mkdir -p /workspace/bt-private; cat > /workspace/bt-private/.gh_token' < ~/.bittrellis-token

cat > /workspace/run-evaluator.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/workspace/bittrellis/venv/bin:$PATH
export GITHUB_TOKEN="$(cat /workspace/bt-private/.gh_token)"
export BT_TOKEN_FILE=/workspace/bt-private/.gh_token
export BT_EVAL_ROOT=/workspace/bt-eval
export BT_PRIVATE=/secure/holdout-2026w38
export BT_LEDGER_TOKEN="$GITHUB_TOKEN"          # pushes the public score records
# BT_LEDGER_REMOTE defaults to the ledger repository. Leave it set: on a replacement box the
# evaluator fetches the published record from it at startup and restores the first-seen records and
# accepted artifacts, so submission priority and scores survive returning the box.
cd /workspace/bittrellis/repo
exec evaluator/run_bot.sh --interval 600
EOF
chmod +x /workspace/run-evaluator.sh

tmux new -d -s bot '/workspace/run-evaluator.sh > /workspace/evaluator.log 2>&1'
tail -f /workspace/evaluator.log
```

One pass only, to try it: add `--once` to `run_bot.sh`.

Under systemd instead of tmux:

```ini
# /etc/systemd/system/bittrellis-evaluator.service
[Unit]
Description=BitTrellis PR evaluator
After=network-online.target
[Service]
ExecStart=/workspace/run-evaluator.sh
Restart=always
RestartSec=60
[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now bittrellis-evaluator
journalctl -u bittrellis-evaluator -f
```

## 5. Check it is working

```bash
bittrellis doctor                                  # pins, models, runtime, corpus, reference
bittrellis frontier --with-seeds                   # the current ranking
cat /workspace/bt-eval/state.json                  # per-PR status
ls /workspace/bt-eval/ledger/hpc01-e4/results/     # published score records
```

## 6. Before returning the box

Only the private holdout is irreplaceable; models and builds are re-downloadable, and every result is
committed or published.

```bash
scp -i ~/.ssh/<key> -P <port> -r root@<host>:/secure/holdout-2026w38 ./holdout-backup
scp -i ~/.ssh/<key> -P <port> -r root@<host>:/workspace/bt-eval/ledger ./ledger-backup   # if a push ever failed
```

The evaluator stops when the box does: open PRs simply wait until a new box picks them up, because
the first-seen record and the score history live in the
[score records repository](https://github.com/coderbench/bittrellis-ledger), not on the disk.

## Timings from a real provision

| Step | Time |
|---|---|
| packages, CUDA, Rust, venv | ~7 min |
| model downloads (~90 GiB) | ~45 min |
| SparkInfer build | ~25 min |
| public BF16 reference | ~2 min |
| holdout corpus + reference | ~3 min |
| one PR: screen → build → audit → quality → speed | ~10 min |
| adding tasks + holdout for a frontier candidate | ~25 min |
