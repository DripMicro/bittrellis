#!/usr/bin/env bash
# Provision a fresh Ubuntu 24.04 + RTX 5090 host for BitTrellis, from nothing to `bittrellis doctor`.
#
#   curl -fsSL https://raw.githubusercontent.com/coderbench/bittrellis/main/scripts/provision_box.sh | bash
#   # or, from a checkout:  scripts/provision_box.sh
#
# Installs CUDA 12.8 (HPC-01) and 13.0 (HPC-02), CMake, Rust, the repo and a virtualenv, downloads the pinned
# models of both tracks (~260 GB), builds each track's SparkInfer and scorer, and computes each public BF16
# reference. Roughly 2-3 hours, almost all of it download time. Safe to re-run: done steps are skipped.
#
#   BT_ROOT     where everything lives            (default /workspace/bittrellis)
#   BT_BRANCH   branch to check out               (default main)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
BT_ROOT="${BT_ROOT:-/workspace/bittrellis}"
BT_BRANCH="${BT_BRANCH:-main}"
step() { echo; echo "=== $(date -u +%H:%M:%S) $* ==="; }

[ "$(id -u)" = 0 ] || { echo "run as root (it installs packages)" >&2; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader || { echo "no NVIDIA GPU visible" >&2; exit 1; }

step "system packages"
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev build-essential cmake ninja-build curl ca-certificates git git-lfs pkg-config tmux

step "CUDA toolkit 12.8 (the track pins it; provider images ship whatever they like)"
# Test for 12.8 itself, not for "an nvcc": an image carrying CUDA 13 at /usr/local/cuda would
# otherwise skip this step and SparkInfer would be built against a toolkit the epoch never used.
CUDA_HOME=/usr/local/cuda-12.8
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  curl -fsSLo /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq && apt-get install -y -qq cuda-toolkit-12-8
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "CUDA 12.8 is pinned but could not be installed" >&2; exit 1; }
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version | tail -1

step "CUDA toolkit 13.0 (HPC-02's SparkInfer pin needs it)"
if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  [ -f /tmp/cuda-keyring.deb ] || { curl -fsSLo /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq; }
  apt-get install -y -qq cuda-toolkit-13-0
fi
[ -x /usr/local/cuda-13.0/bin/nvcc ] || { echo "CUDA 13.0 is pinned for HPC-02 but could not be installed" >&2; exit 1; }
/usr/local/cuda-13.0/bin/nvcc --version | tail -1

step "rust (kept off encrypted mounts: cargo's archiver fails on some of them)"
export RUSTUP_HOME="$BT_ROOT/rust/rustup" CARGO_HOME="$BT_ROOT/rust/cargo"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME"
[ -x "$CARGO_HOME/bin/rustc" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q --default-toolchain stable --no-modify-path
export PATH="$CARGO_HOME/bin:$PATH"

step "repository and virtualenv"
mkdir -p "$BT_ROOT"
[ -d "$BT_ROOT/repo/.git" ] || git clone -q https://github.com/coderbench/bittrellis.git "$BT_ROOT/repo"
git -C "$BT_ROOT/repo" fetch -q origin && git -C "$BT_ROOT/repo" checkout -q "$BT_BRANCH" && git -C "$BT_ROOT/repo" pull -q
[ -x "$BT_ROOT/venv/bin/python" ] || python3 -m venv "$BT_ROOT/venv"
export PATH="$BT_ROOT/venv/bin:$PATH"
# Provider images pin torch to the CUDA build they ship (/etc/pip.conf -> constraints.txt). That pin
# is for the pod's system interpreter; this venv is separate and follows the track's pin instead.
: > "$BT_ROOT/no-constraints.txt"
export PIP_CONSTRAINT="$BT_ROOT/no-constraints.txt"
pip install -q -U pip wheel
cd "$BT_ROOT/repo"
pip install -q -e ".[dev,eval,corpus]"
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers>=5.0" accelerate
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'sees the GPU')"

step "pinned models (~97 GiB: base 52, shipped 17, unsloth 22, calibration 6.7 GB)"
scripts/setup_models.sh base shipped unsloth dspark calibration

step "HPC-02 models (~160 GB: Qwen3.6-35B-A3B BF16 67, GGUF template + UD-Q4_K_M 91, statistics 1.4)"
scripts/setup_models.sh qwen36 qwen36_gguf calibration02

step "SparkInfer at the pinned commit, plus the scorer"
scripts/setup_sparkinfer.sh

step "HPC-02's SparkInfer at its own pinned commit (CUDA 13.0), plus the scorer"
scripts/setup_sparkinfer.sh HPC-02

step "public BF16 reference (once per corpus)"
[ -f data/reference/hpc01-public-v2-k256/reference.json ] || bittrellis reference --out data/reference/hpc01-public-v2-k256
[ -f data/reference/hpc02-public-v2-k256/reference.json ] || bittrellis --track HPC-02 reference

step "doctor"
bittrellis doctor
echo
echo "PROVISION DONE — next: restore the private holdout, then evaluator/setup_sandbox.sh (docs/evaluator_runbook.md)"
