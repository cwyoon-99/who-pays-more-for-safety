# Ordinary-benign control

Compares aligned and unaligned models on 300 ordinary Just-Eval instructions, which are not
expected to trigger safety behavior, and contrasts the resulting cross-lingual disparity with
the over-refusal results.

```bash
python ordinary_benign/build_prompts.py   # 300 Just-Eval instructions
python ordinary_benign/translate.py       # 5 languages
python ordinary_benign/generate.py --pair gemma-3-27b --which aligned --tp 4   # per model
python ordinary_benign/judge.py           # GPT-5-mini pairwise judge
python ordinary_benign/report.py          # disparity table (needs the main pairwise judgments)
```
