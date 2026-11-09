# Benign capability (P-MMEval)

Scores each aligned/unaligned pair on six P-MMEval subsets (MHellaSwag, MIFEval, MLogiQA,
MMMLU, XNLI, MGSM) with OpenCompass and vLLM. The output feeds the capability tables and the
x-axis of the confound map.

## Install

```bash
git clone https://github.com/open-compass/opencompass.git && cd opencompass
git checkout 0d99e7c50baafa136f6a55486944bcfdb23276e1
git apply <repo>/capability/opencompass.patch   # empty-prediction fix (MLogiQA), max length for Gemma-3
pip install -e . && pip install vllm==0.10.1
```

## Run

From the repository root, with `HF_TOKEN` set:

```bash
bash capability/run_pmmeval.sh                 # all pairs, or e.g. `qwen3_32b gemma3_27b`
python capability/summarize.py outputs/capability/opencompass
```

`summarize.py` writes `capability_drop.csv`, `alignment_gap_difference.csv`, and
`multilingual_performance.csv` to `outputs/capability/`.

Both models of a pair share the same sampling parameters (`configs/eval_<pair>.py`). Qwen3
runs in thinking mode and the reasoning trace is removed before scoring.
