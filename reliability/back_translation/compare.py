"""Compare target-language judging with judging of the back-translated English responses.

Every judged instance has ``avg_score`` (original, target-language judging) and
``avg_score_bt`` (back-translated). Writes per-cell, per-(language, benchmark), per-language,
per-model, and overall rows to ``outputs/reliability/back_translation/comparison.csv``.

Columns: mean_orig / mean_bt = mean ``avg_score`` (aligned model's win rate),
mean_diff = mean_bt - mean_orig, pearson_r = instance-level correlation,
sign_flip_pct = % of instances whose winner (above/below 0.5) changes,
tie_db_flip_pct = the same with a +-0.1 dead band around 0.5.

Usage:
    python reliability/back_translation/compare.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.back_translation.back_translate import BACK_TRANSLATION_LANGUAGES
from reliability.common import RELIABILITY_PAIRS, cell_path, pearson, reliability_dir, winner
from safety_cost.config import BENCHMARKS, MODEL_PAIRS
from safety_cost.utils import load_json

FIELDS = ["lang", "benchmark", "model", "n", "mean_orig", "mean_bt", "mean_diff", "pearson_r",
          "sign_flip_pct", "tie_db_flip_pct"]


def paired(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("avg_score") is not None and e.get("avg_score_bt") is not None]


def cell_metrics(entries: list[dict]) -> dict:
    pairs = [(e["avg_score"], e["avg_score_bt"]) for e in paired(entries)]
    if not pairs:
        return {"n": 0}
    orig = [p[0] for p in pairs]
    bt = [p[1] for p in pairs]
    n = len(pairs)
    mean_orig = round(sum(orig) / n, 4)
    mean_bt = round(sum(bt) / n, 4)
    flip = sum(1 for o, b in pairs if winner(o) != winner(b))
    flip_db = sum(1 for o, b in pairs if winner(o, 0.1) != winner(b, 0.1)
                  and winner(o, 0.1) is not None and winner(b, 0.1) is not None)
    return {
        "n": n,
        "mean_orig": mean_orig,
        "mean_bt": mean_bt,
        "mean_diff": round(mean_bt - mean_orig, 4),
        "pearson_r": pearson(orig, bt),
        "sign_flip_pct": round(100 * flip / n, 1),
        "tie_db_flip_pct": round(100 * flip_db / n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(RELIABILITY_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--languages", nargs="+", default=list(BACK_TRANSLATION_LANGUAGES))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    root = reliability_dir("back_translation", args.output_dir)
    pairs = sorted((MODEL_PAIRS[k] for k in args.pairs), key=lambda p: p.unaligned_tag)

    rows = []
    by_lang_bench, by_lang, by_model = defaultdict(list), defaultdict(list), defaultdict(list)
    for benchmark in sorted(BENCHMARKS):
        for language in sorted(args.languages):
            for pair in pairs:
                path = cell_path(root / "judged", benchmark, language, pair)
                if not path.exists():
                    continue
                entries = load_json(path)
                rows.append({"lang": language, "benchmark": benchmark, "model": pair.unaligned_tag,
                             **cell_metrics(entries)})
                both = paired(entries)
                by_lang_bench[(language, benchmark)].extend(both)
                by_lang[language].extend(both)
                by_model[pair.unaligned_tag].extend(both)
    if not rows:
        sys.exit(f"No judged back-translation files under {root / 'judged'}")

    for (language, benchmark), entries in sorted(by_lang_bench.items()):
        rows.append({"lang": language, "benchmark": benchmark, "model": "all", **cell_metrics(entries)})
    for language, entries in sorted(by_lang.items()):
        rows.append({"lang": language, "benchmark": "all", "model": "all", **cell_metrics(entries)})
    for model, entries in sorted(by_model.items()):
        rows.append({"lang": "all", "benchmark": "all", "model": model, **cell_metrics(entries)})
    rows.append({"lang": "all", "benchmark": "all", "model": "all", **cell_metrics(sum(by_lang.values(), []))})

    out_csv = root / "comparison.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {out_csv} ({len(rows)} rows)\n")

    print(f"{'Group':<36} {'n':>5} {'orig':>7} {'BT-EN':>7} {'delta':>8} {'r':>6}")
    for r in rows:
        if r["benchmark"] != "all":
            continue
        group = r["model"] if r["lang"] == "all" and r["model"] != "all" else r["lang"]
        group = "overall" if r["lang"] == "all" and r["model"] == "all" else group
        print(f"{group:<36} {r['n']:>5} {r['mean_orig']:>7.3f} {r['mean_bt']:>7.3f} "
              f"{r['mean_diff']:>+8.3f} {r['pearson_r']:>6.2f}")


if __name__ == "__main__":
    main()
