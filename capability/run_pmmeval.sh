#!/usr/bin/env bash
# Evaluate the aligned/unaligned model pairs on P-MMEval with OpenCompass + vLLM, then
# build the capability tables. Run from the repository root inside the OpenCompass
# environment (see capability/README.md). Each model uses 4 GPUs (tensor parallel).
#
# Usage:
#   bash capability/run_pmmeval.sh                         # all four pairs
#   bash capability/run_pmmeval.sh qwen3_32b gemma3_27b    # selected pairs
#
# Environment:
#   HF_TOKEN   Hugging Face token (Llama-3.3 and Gemma-3 are gated); read from the environment.
#   WORK_DIR   OpenCompass work directory (default: outputs/capability/opencompass).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${SAFETY_COST_OUTPUT_DIR:-$REPO_ROOT/outputs}/capability/opencompass}"
export VLLM_NO_USAGE_STATS=1

if [ "$#" -gt 0 ]; then
  PAIRS=("$@")
else
  PAIRS=(llama3_3_70b qwen2_5_72b gemma3_27b qwen3_32b)
fi

for pair in "${PAIRS[@]}"; do
  config="$REPO_ROOT/capability/configs/eval_${pair}.py"
  [ -f "$config" ] || { echo "unknown pair: $pair (no $config)"; exit 1; }
  echo "=== P-MMEval: $pair ==="
  opencompass "$config" -w "$WORK_DIR/$pair"
done

python "$REPO_ROOT/capability/summarize.py" "$WORK_DIR"
