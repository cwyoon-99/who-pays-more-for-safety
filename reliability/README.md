# Reliability checks

Run from the repository root after the main pipeline. Results go to `outputs/reliability/`.

| Check | Scripts |
|---|---|
| Translation quality (CometKiwi-XL, xCOMET-XL) | `translation_qe.py` |
| Native-speaker validation | `native_validation/prepare_annotation.py`, `native_validation/analyze.py` |
| Back-translated re-judging | `back_translation/back_translate.py`, `judge_backtranslated.py`, `compare.py` |
| Second judge (GPT-5.2) | `cross_judge/run_cross_judge.py`, `compare_judges.py` |

Requirements: `pip install unbabel-comet requests` (QE needs a GPU and access to
`Unbabel/wmt23-cometkiwi-da-xl`); `GOOGLE_TRANSLATE_API_KEY` for back-translation.

```bash
python reliability/translation_qe.py
python reliability/back_translation/back_translate.py
python reliability/back_translation/judge_backtranslated.py
python reliability/back_translation/compare.py
python reliability/cross_judge/run_cross_judge.py
python reliability/cross_judge/compare_judges.py
```

`native_validation/` creates annotation sheets and summarizes the completed ones (see `--help`).
Annotations are not released.
