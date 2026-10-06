#!/usr/bin/env bash
# Download pinned sources into $MODELS_DIR, then hash-verify them against configs/sources.lock.json.
#   base      Qwen/Qwen3.8-27B BF16 (52 GiB)          builds, audits, BF16 reference
#   shipped   gittensor NVFP4 (17 GiB)                 builds (frozen tensors, `baseline`), V0
#   unsloth   unsloth NVFP4 (22 GiB)                   `unsloth` quantizer, external R1
#   gguf      unsloth UD-Q4_K_M GGUF (15 GiB)          external R2 (validators only)
#   dspark    DSpark drafter (1.3 GiB)                 faster task suite (answers unchanged)
#   calibration  pinned calibration statistics (6.7 GB, GitHub release)   calibrated encoders
# Usage: scripts/setup_models.sh [base] [shipped] [unsloth] [gguf] [dspark] [calibration]
#        (default: base shipped unsloth dspark calibration)
set -euo pipefail
source "$(dirname "$0")/_pins.sh"
command -v hf >/dev/null || pip install -q "huggingface_hub[cli]>=0.34"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(base shipped unsloth dspark calibration)
for w in "${want[@]}"; do
  case "$w" in
    base)    hf download "$PIN_BASE_REPO" --revision "$PIN_BASE_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B" --max-workers 16 ;;
    shipped) hf download "$PIN_GITTENSOR_NVFP4_REPO" --revision "$PIN_GITTENSOR_NVFP4_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4-RTX5090" --exclude "assets/*" ;;
    unsloth) hf download "$PIN_UNSLOTH_NVFP4_REPO" --revision "$PIN_UNSLOTH_NVFP4_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4-unsloth" ;;
    dspark)  hf download "$PIN_DSPARK_REPO" --revision "$PIN_DSPARK_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-DSpark-NVFP4" ;;
    gguf)    hf download "$PIN_UNSLOTH_GGUF_REPO" "$PIN_R2_FILE" --revision "$PIN_UNSLOTH_GGUF_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-GGUF" ;;
    calibration) bittrellis calibration fetch --calibration "$MODELS_DIR/hpc01-calib-v1" ;;
    *) echo "unknown source '$w'" >&2; exit 2 ;;
  esac
done
bittrellis verify-sources
