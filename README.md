# Who Pays More for Safety?

Code for **[Who Pays More for Safety? Measuring the Disparate Cost of Safety Alignment across Languages](https://arxiv.org/abs/2608.22490)** (EMNLP 2026).

We measure **Safety Cost**, the utility loss caused by safety alignment, as the win rate of a
refusal-ablated (unaligned) model over its safety-aligned counterpart on over-refusal prompts,
and compare it across eight languages.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY, HF_TOKEN
```

## Main pipeline

```bash
python scripts/00_translate_benchmarks.py   # XSTest, OR-Bench, OR-Bench-Hard -> 7 languages
python scripts/01_prepare_multijail.py      # MultiJail prompts

# once per model (pairs: llama-3.3-70b, qwen2.5-72b, gemma-3-27b, qwen3-32b)
python scripts/10_generate.py --pair gemma-3-27b --which aligned \
    --benchmarks xstest or-bench-80k or-bench-hard-1k multijail

python scripts/20_judge_pairwise.py   # Safety Cost (GPT-5-mini)
python scripts/21_judge_unsafe.py     # Safety Gain (GPT-4.1)
python scripts/22_judge_refusal.py    # refusal rates
python scripts/23_meta_judge.py       # reasons behind the judgments
```

`scripts/run_all.sh` runs the whole pipeline, including the analysis scripts.
Models, languages, and judge settings are defined in `safety_cost/config.py`; Qwen3 runs in
thinking mode. Judges use the OpenAI Batch API: `--yes` skips the cost confirmation and
`--resume <batch dir>` continues an interrupted job. Prompts are written to `data/` and all
results to `outputs/`.

## Tables and figures

```bash
python analysis/safety_cost_table.py        # Safety Cost per benchmark and language
python analysis/plot_safety_utility.py      # Safety Gain vs. Safety Cost
python analysis/plot_butterfly.py           # refusal rate vs. Safety Cost
python analysis/plot_label_distribution.py  # meta-judge labels
python analysis/length_analysis.py          # judge length sensitivity
python analysis/plot_confound_map.py --capability_csv outputs/capability/capability_drop.csv
```

## Repository

| Directory | Contents |
|---|---|
| `safety_cost/` | shared config, prompts, judge parsing, OpenAI Batch API helpers |
| `scripts/` | main pipeline |
| `analysis/` | tables and figures |
| `capability/` | benign capability on P-MMEval (OpenCompass) |
| `matched_checkpoint/` | matched checkpoints trained from Qwen3-8B-Base |
| `reliability/` | translation quality and judge robustness |
| `ordinary_benign/` | ordinary-benign control |

Each appendix directory has its own README.

## Citation

```bibtex
@inproceedings{yoon2026whopays,
  title     = {Who Pays More for Safety? Measuring the Disparate Cost of Safety Alignment across Languages},
  author    = {Yoon, Chanwoong and Park, Jungsoo and Ritter, Alan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

MIT
