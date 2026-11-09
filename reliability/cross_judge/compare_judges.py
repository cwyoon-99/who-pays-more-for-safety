"""Compare GPT-5-mini (main judge) with an additional judge on the same instances.

Reads the files written by ``run_cross_judge.py`` (each re-judged instance has both
``avg_score`` from GPT-5-mini and ``avg_score_<judge>``) and writes per-cell, per-(language,
benchmark), per-language, per-model, and overall rows to
``outputs/reliability/cross_judge/comparison_<judge>.csv``.

Columns: mean_gpt5mini / mean_new = aligned win rate (mean ``avg_score``) under each judge,
mean_diff = new - gpt5mini, pearson_r = instance-level correlation, sign_flip_pct = % of
instances whose winner (avg_score above/below 0.5) changes. The script also prints the
win rate and the gap to English per language.

Usage:
    python reliability/cross_judge/compare_judges.py [--judge gpt-5.2]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.common import RELIABILITY_PAIRS, cell_path, judge_suffix, pearson, reliability_dir, winner
from reliability.cross_judge.run_cross_judge import CROSS_JUDGE, CROSS_JUDGE_LANGUAGES
from safety_cost.config import BENCHMARKS, MODEL_PAIRS
from safety_cost.utils import load_json

FIELDS = ["lang", "benchmark", "model", "n", "mean_gpt5mini", "mean_new", "mean_diff", "pearson_r",
          "sign_flip_pct"]


def paired(entries: list[dict], field: str) -> list[dict]:
    return [e for e in entries if e.get("avg_score") is not None and e.get(field) is not None]


def cell_metrics(entries: list[dict], field: str) -> dict:
    pairs = [(e["avg_score"], e[field]) for e in paired(entries, field)]
    if not pairs:
        return {"n": 0}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    n = len(pairs)
    return {
        "n": n,
        "mean_gpt5mini": round(sum(a) / n, 4),
        "mean_new": round(sum(b) / n, 4),
        "mean_diff": round(sum(b) / n - sum(a) / n, 4),
        "pearson_r": pearson(a, b),
        "sign_flip_pct": round(100 * sum(1 for x, y in pairs if winner(x) != winner(y)) / n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", default=CROSS_JUDGE)
    parser.add_argument("--pairs", nargs="+", default=list(RELIABILITY_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--languages", nargs="+", default=list(CROSS_JUDGE_LANGUAGES))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    suffix = judge_suffix(args.judge)
    field = f"avg_score_{suffix}"
    root = reliability_dir("cross_judge", args.output_dir)
    pairs = sorted((MODEL_PAIRS[k] for k in args.pairs), key=lambda p: p.unaligned_tag)

    rows = []
    by_lang_bench, by_lang, by_model = defaultdict(list), defaultdict(list), defaultdict(list)
    for benchmark in sorted(BENCHMARKS):
        for language in sorted(args.languages):
            for pair in pairs:
                path = cell_path(root / suffix, benchmark, language, pair)
                if not path.exists():
                    continue
                entries = load_json(path)
                rows.append({"lang": language, "benchmark": benchmark, "model": pair.unaligned_tag,
                             **cell_metrics(entries, field)})
                both = paired(entries, field)
                by_lang_bench[(language, benchmark)].extend(both)
                by_lang[language].extend(both)
                by_model[pair.unaligned_tag].extend(both)
    if not rows:
        sys.exit(f"No cross-judge files under {root / suffix}")

    for (language, benchmark), entries in sorted(by_lang_bench.items()):
        rows.append({"lang": language, "benchmark": benchmark, "model": "all", **cell_metrics(entries, field)})
    for language, entries in sorted(by_lang.items()):
        rows.append({"lang": language, "benchmark": "all", "model": "all", **cell_metrics(entries, field)})
    for model, entries in sorted(by_model.items()):
        rows.append({"lang": "all", "benchmark": "all", "model": model, **cell_metrics(entries, field)})
    overall = cell_metrics(sum(by_lang.values(), []), field)
    rows.append({"lang": "all", "benchmark": "all", "model": "all", **overall})

    out_csv = root / f"comparison_{suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {out_csv} ({len(rows)} rows)\n")

    # Aligned win rate per language and its gap to English (Eng - L) under both judges.
    lang_rows = {r["lang"]: r for r in rows if r["benchmark"] == "all" and r["model"] == "all" and r["lang"] != "all"}
    print(f"{'Language':<10} {'n':>5} {'5-mini':>8} {args.judge:>8} {'gap 5-mini':>11} {'gap ' + args.judge:>11}")
    english = lang_rows.get("english")
    for language in args.languages:
        r = lang_rows.get(language)
        if r is None or not r["n"]:
            continue
        gaps = ("", "")
        if english and language != "english":
            gaps = (f"{english['mean_gpt5mini'] - r['mean_gpt5mini']:+.3f}",
                    f"{english['mean_new'] - r['mean_new']:+.3f}")
        print(f"{language:<10} {r['n']:>5} {r['mean_gpt5mini']:>8.3f} {r['mean_new']:>8.3f} "
              f"{gaps[0]:>11} {gaps[1]:>11}")
    print(f"{'overall':<10} {overall['n']:>5} {overall['mean_gpt5mini']:>8.3f} {overall['mean_new']:>8.3f}")


if __name__ == "__main__":
    main()
