"""Turn OpenCompass P-MMEval results into the capability tables.

Inputs (any mix; later inputs override earlier ones for the same model/subset/language):
  * OpenCompass work directories, searched recursively for ``results/<model_abbr>/*.json``
  * ``results/<model_abbr>/`` directories
  * OpenCompass summary CSVs with per-language rows (written when the summarizer from
    ``configs/pmmeval_datasets.py`` is used)

Outputs (printed and written to ``--output_dir``):
  * capability_drop.csv           Table capability_drop: per language, the mean over the six
                                  P-MMEval subsets (Flores-200 and HumanEval-XL excluded) for the
                                  aligned (Aln) and unaligned (Unl) model, and Delta = Aln - Unl.
  * alignment_gap_difference.csv  Confound map x-axis: Delta_L - Delta_English.
  * multilingual_performance.csv  Table multilingual_performance: per subset, the mean over the
                                  10 P-MMEval languages, plus the ``avg`` column.

Rounding follows the paper: per-language values are rounded half-up to two decimals (the values
used for the confound map) and then to one decimal for the table; per-subset means are formatted
with two decimals.

Usage:
    python capability/summarize.py outputs/capability/opencompass
    python capability/summarize.py run1/ run2/results/qwen3-32b-vllm/ summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import LANGUAGES, MODEL_PAIRS, OUTPUT_DIR

# Subsets that enter the tables and the metric reported for each.
SUBSETS = ("mhellaswag", "mifeval", "mlogiqa", "mmmlu", "xnli", "mgsm")
METRICS = {"mifeval": "strict_acc"}  # everything else: accuracy
# ``avg`` averages these five subsets (mgsm is reported separately).
DEFAULT_AVG_SUBSETS = ("mhellaswag", "mifeval", "mlogiqa", "mmmlu", "xnli")

# P-MMEval language codes and display names.
LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ar": "Arabic", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "th": "Thai", "vi": "Vietnamese",
}
MMMLU_CODES = {
    "EN-US": "en", "ZH-CN": "zh", "AR-XY": "ar", "ES-LA": "es", "FR-FR": "fr",
    "JA-JP": "ja", "KO-KR": "ko", "PT-BR": "pt", "TH-TL": "th", "VI-VT": "vi",
}
# Evaluation languages covered by P-MMEval (Ukrainian is not in P-MMEval).
NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}
TABLE_LANGS = tuple(NAME_TO_CODE[lang] for lang in LANGUAGES if lang in NAME_TO_CODE)

# OpenCompass model abbreviations (see configs/eval_*.py): pair -> (aligned, unaligned).
OC_ABBRS = {
    "llama-3.3-70b": ("llama3_3-70b-instruct-vllm", "llama3_3-70b-instruct-abliterated-vllm"),
    "qwen2.5-72b": ("qwen2_5-72b-instruct-vllm", "qwen2_5-72b-instruct-abliterated-vllm"),
    "gemma-3-27b": ("gemma-3-27b-it-vllm", "gemma-3-27b-it-abliterated-vllm"),
    "qwen3-32b": ("qwen3-32b-vllm", "qwen3-32b-abliterated-vllm"),
}


def half_up(x: Decimal, places: str) -> Decimal:
    return x.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def parse_dataset_abbr(abbr: str) -> tuple[str, str] | None:
    """``mmmlu-EN-US`` -> (``mmmlu``, ``en``); ``None`` for subsets that are not used."""
    subset, _, lang = abbr.partition("-")
    if subset not in SUBSETS or not lang:
        return None
    lang = MMMLU_CODES.get(lang, lang.lower())
    return (subset, lang) if lang in LANG_NAMES else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results_dir(model_dir: Path, scores: dict) -> int:
    n = 0
    for path in sorted(model_dir.glob("*.json")):
        key = parse_dataset_abbr(path.stem)
        if key is None:
            continue
        with open(path, encoding="utf-8") as f:
            value = json.load(f).get(METRICS.get(key[0], "accuracy"))
        if value is not None:
            scores[model_dir.name][key] = value
            n += 1
    return n


def load_summary_csv(path: Path, scores: dict) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    model_abbrs = header[4:]  # dataset, version, metric, mode, <models...>
    for row in body:
        key = parse_dataset_abbr(row[0])
        if key is None or row[2] != METRICS.get(key[0], "accuracy"):
            continue
        for abbr, value in zip(model_abbrs, row[4:]):
            if value not in ("-", ""):
                scores[abbr][key] = float(value)
                n += 1
    return n


def load_inputs(paths: list[Path]) -> dict:
    """scores[model_abbr][(subset, lang)] = score."""
    scores: dict = defaultdict(dict)
    for path in paths:
        if path.is_file() and path.suffix == ".csv":
            n = load_summary_csv(path, scores)
        elif path.is_dir() and any(path.glob("*.json")):
            n = load_results_dir(path, scores)
        elif path.is_dir():
            n = 0
            # <work_dir>/<timestamp>/results/<model_abbr>/; timestamps sort chronologically.
            for model_dir in sorted(p for p in path.glob("**/results/*") if p.is_dir()):
                n += load_results_dir(model_dir, scores)
        else:
            raise FileNotFoundError(path)
        print(f"loaded {n} scores from {path}")
    return scores


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def per_language(aligned: dict, unaligned: dict, lang: str):
    """Mean over the subsets scored for both models (exact decimal arithmetic)."""
    subsets = [s for s in SUBSETS if (s, lang) in aligned and (s, lang) in unaligned]
    if not subsets:
        return None
    aln = sum(Decimal(str(aligned[(s, lang)])) for s in subsets) / len(subsets)
    unl = sum(Decimal(str(unaligned[(s, lang)])) for s in subsets) / len(subsets)
    return {"aln": half_up(aln, "0.01"), "unl": half_up(unl, "0.01"),
            "delta": half_up(aln - unl, "0.01"), "n_subsets": len(subsets)}


def per_subset(model_scores: dict, subset: str) -> float | None:
    """Mean over languages (sorted by code), as a float."""
    values = [model_scores[(subset, lang)] for lang in sorted(LANG_NAMES) if (subset, lang) in model_scores]
    return sum(values) / len(values) if values else None


def model_average(model_scores: dict, subsets) -> float | None:
    values = [model_scores[(s, lang)] for lang in sorted(LANG_NAMES) for s in subsets if (s, lang) in model_scores]
    return sum(values) / len(values) if values else None


def write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pairs", nargs="+", default=list(OC_ABBRS), choices=list(OC_ABBRS))
    parser.add_argument("--pair_abbrs", nargs="+", default=[], metavar="PAIR=ALIGNED,UNALIGNED",
                        help="Override the OpenCompass model abbreviations of a pair.")
    parser.add_argument("--avg_subsets", nargs="+", default=list(DEFAULT_AVG_SUBSETS), choices=SUBSETS)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR / "capability")
    args = parser.parse_args()

    abbrs = dict(OC_ABBRS)
    for item in args.pair_abbrs:
        pair, _, models = item.partition("=")
        abbrs[pair] = tuple(models.split(","))

    scores = load_inputs(args.inputs)
    pairs = [p for p in args.pairs if abbrs[p][0] in scores and abbrs[p][1] in scores]
    for p in args.pairs:
        if p not in pairs:
            print(f"warning: no results for {p} ({abbrs[p][0]} / {abbrs[p][1]}); skipped")

    # ---- capability_drop + alignment gap difference -------------------------------
    drop_rows, gap_rows = [], []
    table = {}
    for pair in pairs:
        aligned, unaligned = scores[abbrs[pair][0]], scores[abbrs[pair][1]]
        for lang in LANG_NAMES:
            stats = per_language(aligned, unaligned, lang)
            if stats is None:
                continue
            if stats["n_subsets"] < len(SUBSETS):
                print(f"warning: {pair} {LANG_NAMES[lang]}: only {stats['n_subsets']}/{len(SUBSETS)} subsets")
            table[(pair, lang)] = stats
            drop_rows.append([pair, LANG_NAMES[lang], stats["n_subsets"], stats["aln"], stats["unl"], stats["delta"]])
        if (pair, "en") in table:
            en_delta = table[(pair, "en")]["delta"]
            gaps = []
            for lang in TABLE_LANGS:
                if lang != "en" and (pair, lang) in table:
                    gap = table[(pair, lang)]["delta"] - en_delta
                    gaps.append(abs(gap))
                    gap_rows.append([pair, LANG_NAMES[lang], table[(pair, lang)]["delta"], en_delta, gap])
            if gaps:
                print(f"{MODEL_PAIRS[pair].family}: mean |Alignment Gap Difference| = "
                      f"{half_up(sum(gaps) / len(gaps), '0.01')}")

    write_csv(args.output_dir / "capability_drop.csv",
              ["pair", "language", "n_subsets", "aligned", "unaligned", "delta"], drop_rows)
    write_csv(args.output_dir / "alignment_gap_difference.csv",
              ["pair", "language", "delta_language", "delta_english", "delta_language_minus_english"], gap_rows)

    print("\nTable capability_drop (Aln / Unl / Delta, P-MMEval w/o Flores-200 and HumanEval-XL)")
    print(f"{'Language':<12}" + "".join(f"{MODEL_PAIRS[p].family:>24}" for p in pairs))
    for lang in TABLE_LANGS:
        cells = []
        for pair in pairs:
            s = table.get((pair, lang))
            cells.append("{:>24}".format("-" if s is None else
                         " / ".join(f"{half_up(s[k], '0.1')}" for k in ("aln", "unl", "delta"))))
        print(f"{LANG_NAMES[lang]:<12}" + "".join(cells))

    # ---- multilingual_performance -------------------------------------------------
    perf_rows = []
    print(f"\nTable multilingual_performance (mean over languages; avg = mean of {', '.join(args.avg_subsets)})")
    print(f"{'Model':<42}" + "".join(f"{s:>11}" for s in SUBSETS) + f"{'avg':>11}")
    for pair in pairs:
        for which, abbr in zip(("aligned", "unaligned"), abbrs[pair]):
            means = [per_subset(scores[abbr], s) for s in SUBSETS]
            avg = model_average(scores[abbr], args.avg_subsets)
            cells = ["-" if m is None else f"{m:.2f}" for m in means] + ["-" if avg is None else f"{avg:.2f}"]
            perf_rows.append([pair, which, abbr] + cells)
            print(f"{MODEL_PAIRS[pair].family + ' ' + which:<42}" + "".join(f"{c:>11}" for c in cells))
    write_csv(args.output_dir / "multilingual_performance.csv",
              ["pair", "model", "abbr", *SUBSETS, "avg"], perf_rows)


if __name__ == "__main__":
    main()
