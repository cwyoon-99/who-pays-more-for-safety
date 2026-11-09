#!/usr/bin/env bash
# End-to-end pipeline for the main results (Sec. 3-4).
# Requires: OPENAI_API_KEY (translation + judges), HF_TOKEN (gated models), and GPUs for vLLM.
# Each stage skips files that already exist, so the script can be re-run after interruptions.
set -euo pipefail
cd "$(dirname "$0")/.."

PAIRS=(llama-3.3-70b qwen2.5-72b gemma-3-27b qwen3-32b)

# 1. Evaluation data
python scripts/00_translate_benchmarks.py
python scripts/01_prepare_multijail.py

# 2. Generation (each model is loaded once for all benchmarks/languages)
for pair in "${PAIRS[@]}"; do
  for which in aligned unaligned; do
    python scripts/10_generate.py --pair "$pair" --which "$which" \
      --benchmarks xstest or-bench-80k or-bench-hard-1k multijail
  done
done

# 3. Judging (OpenAI Batch API; add --yes to skip the cost confirmation)
python scripts/20_judge_pairwise.py   # Safety Cost
python scripts/21_judge_unsafe.py     # Safety Gain
python scripts/22_judge_refusal.py    # refusal rates (Fig. 3)
python scripts/23_meta_judge.py       # implicit-degradation taxonomy (Sec. 4.2)

# 4. Tables and figures
python analysis/safety_cost_table.py
python analysis/plot_safety_utility.py
python analysis/plot_butterfly.py
python analysis/plot_label_distribution.py
python analysis/length_analysis.py
