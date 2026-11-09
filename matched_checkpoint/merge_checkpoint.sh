#!/usr/bin/env bash
# Merge the FSDP shards of a matched-checkpoint SFT run into a Hugging Face model directory.
#
# Usage (from the repository root, in the verl environment):
#   bash matched_checkpoint/merge_checkpoint.sh c0          # merges global_step_150
#   bash matched_checkpoint/merge_checkpoint.sh c1
#   bash matched_checkpoint/merge_checkpoint.sh c1 120      # another saved step
#
# Environment overrides:
#   CKPT_ROOT    checkpoint root   (default: outputs/matched_checkpoint/checkpoints)
#   MODEL_ROOT   merged models     (default: outputs/matched_checkpoint/models)
set -euo pipefail

COND="${1:?usage: merge_checkpoint.sh c0|c1 [STEP]}"
STEP="${2:-150}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${SAFETY_COST_OUTPUT_DIR:-$REPO_ROOT/outputs}/matched_checkpoint"
CKPT_ROOT="${CKPT_ROOT:-$OUT_ROOT/checkpoints}"
MODEL_ROOT="${MODEL_ROOT:-$OUT_ROOT/models}"

EXP="bsft-qwen3-8b-${COND}"
SRC="$CKPT_ROOT/$EXP/global_step_$STEP"
DST="$MODEL_ROOT/$EXP"

ls "$SRC"/model_world_size_*.pt >/dev/null 2>&1 || { echo "no complete checkpoint in $SRC"; exit 1; }
rm -rf "$DST"
mkdir -p "$MODEL_ROOT"

echo "=== merging $SRC -> $DST ==="
python -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST"

if [ -f "$DST/model.safetensors" ] || [ -f "$DST/model.safetensors.index.json" ]; then
  echo "=== merged: $DST ==="
else
  echo "merge failed: no model weights in $DST"; exit 1
fi
