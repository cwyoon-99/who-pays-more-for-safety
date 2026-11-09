#!/usr/bin/env bash
# Full-parameter SFT of Qwen3-8B-Base on one matched-checkpoint mixture with verl (FSDP).
#
#   c0 = instruction data + non-safety filler  (genuinely unaligned)
#   c1 = instruction data + English safety      (safety-aligned)
#
# Both runs are identical except for the training file: 1 epoch, lr 2e-5 (cosine, 3% warmup),
# bf16, effective batch size 512, max length 2048 (right truncation), micro batch 4 per GPU.
# The chat template / EOS token come from the Qwen3-8B (instruct) tokenizer.
#
# Usage (from the repository root, in an environment with verl installed):
#   bash matched_checkpoint/train_sft.sh c0
#   bash matched_checkpoint/train_sft.sh c1
#
# Environment overrides:
#   DATA_DIR   training mixtures   (default: data/matched_checkpoint)
#   CKPT_ROOT  checkpoint root     (default: outputs/matched_checkpoint/checkpoints)
#   NGPU       GPUs on this node   (default: 8; the effective batch size stays 512 for any
#              NGPU via gradient accumulation)
#   HF_TOKEN   read from the environment if needed
#
# Requires verl (`verl.trainer.fsdp_sft_trainer`, FSDP2 strategy).
set -euo pipefail

COND="${1:?usage: train_sft.sh c0|c1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${SAFETY_COST_DATA_DIR:-$REPO_ROOT/data}/matched_checkpoint}"
CKPT_ROOT="${CKPT_ROOT:-${SAFETY_COST_OUTPUT_DIR:-$REPO_ROOT/outputs}/matched_checkpoint/checkpoints}"
NGPU="${NGPU:-8}"

BASE="Qwen/Qwen3-8B-Base"
TOKENIZER="Qwen/Qwen3-8B"
WRAP="Qwen3DecoderLayer"
MICRO=4
MAXLEN=2048
BSIZE=512
LR=2e-5

case "$COND" in
  c0) DATA="$DATA_DIR/c0.parquet" ;;
  c1) DATA="$DATA_DIR/c1_english.parquet" ;;
  *) echo "unknown condition: $COND (expected c0 or c1)"; exit 1 ;;
esac
[ -f "$DATA" ] || { echo "missing training data: $DATA"; exit 1; }

EXP="bsft-qwen3-8b-${COND}"
CKPT="$CKPT_ROOT/$EXP"
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use GPUs 0..NGPU-1 unless the caller already restricted the visible devices.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU - 1)))}"

echo "=== $EXP | base=$BASE tokenizer=$TOKENIZER gpus=$NGPU data=$DATA ==="

# data.val_files is required by the trainer; validation is disabled (test_freq=-1).
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA" \
  data.val_files="$DATA" \
  data.max_length=$MAXLEN \
  data.truncation=right \
  data.micro_batch_size_per_gpu=$MICRO \
  data.train_batch_size=$BSIZE \
  data.prompt_key=prompt \
  data.response_key=response \
  use_remove_padding=False \
  +data.tokenizer="$TOKENIZER" \
  trainer.logger='["console"]' \
  trainer.project_name=base-sft \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$CKPT" \
  trainer.nnodes=1 \
  trainer.save_freq=30 \
  trainer.max_ckpt_to_keep=2 \
  trainer.test_freq=-1 \
  +trainer.log_freq=10 \
  trainer.total_epochs=1 \
  optim.lr=$LR \
  optim.warmup_steps_ratio=0.03 \
  optim.lr_scheduler=cosine \
  model.fsdp_config.model_dtype=bf16 \
  +model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[$WRAP] \
  model.partial_pretrain="$BASE" \
  hydra.run.dir="$CKPT/hydra"

# Checkpoints are saved every 30 steps (see merge_checkpoint.sh).
echo "=== training done: $CKPT ==="
