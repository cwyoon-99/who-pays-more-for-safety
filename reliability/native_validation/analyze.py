"""Summarize the native-speaker validation ratings (Appendix B, Table "native_validation").

Reads the completed annotation sheets (``naturalness`` and ``fidelity`` filled with integers
1-5; anything else counts as missing) and reports mean +- sample std per language and per
(language, benchmark), the score distribution, a paired cross-language comparison on the
shared English sources, and the annotators' notes. The CometKiwi-XL / xCOMET-XL language
means from ``translation_qe.py`` are added for reference when available.

Usage:
    python reliability/native_validation/analyze.py \
        --annotations chinese=path/to/human_eval_chinese_done.csv korean=path/to/human_eval_korean_done.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.common import pearson, reliability_dir

BENCHMARKS = ("xstest", "or-bench-80k", "or-bench-hard-1k")
FIELDS = ["lang", "benchmark", "n", "naturalness_mean", "naturalness_std", "fidelity_mean", "fidelity_std",
          "cometkiwi_xl", "xcomet_xl"]


def parse_score(value) -> int | None:
    try:
        v = int(str(value).strip())
    except ValueError:
        return None
    return v if 1 <= v <= 5 else None


def stat(xs: list[int]):
    if not xs:
        return None, None
    return round(statistics.mean(xs), 3), (round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0)


def fmt(m, s) -> str:
    return f"{m:.2f} +- {s:.2f}" if m is not None else "n/a"


def load_qe_reference(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["lang"]: (round(float(r["cometkiwi_mean"]), 3), round(float(r["xcomet_mean"]), 3))
                for r in csv.DictReader(f) if r["benchmark"] == "all" and r["cometkiwi_mean"] and r["xcomet_mean"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", nargs="+", required=True, metavar="LANGUAGE=CSV",
                        help="completed sheets, e.g. chinese=human_eval_chinese_done.csv")
    parser.add_argument("--qe_summary", type=Path, default=None,
                        help="summary.csv of translation_qe.py (default: outputs/reliability/translation_qe/)")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    rows_by_lang: dict[str, list[dict]] = {}
    for spec in args.annotations:
        language, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["naturalness_int"] = parse_score(r.get("naturalness"))
            r["fidelity_int"] = parse_score(r.get("fidelity"))
        rows_by_lang[language] = rows
    languages = list(rows_by_lang)
    qe = load_qe_reference(args.qe_summary or reliability_dir("translation_qe", args.output_dir) / "summary.csv")

    def scores(rows, metric):
        return [r[metric] for r in rows if r[metric] is not None]

    summary = []
    print(f"{'Lang':<10} {'n':<5} {'Naturalness':<16} {'Fidelity':<16} {'CK-XL':<8} {'xC-XL':<8}")
    for language in languages:
        nat, fid = scores(rows_by_lang[language], "naturalness_int"), scores(rows_by_lang[language], "fidelity_int")
        (nm, ns), (fm, fs) = stat(nat), stat(fid)
        ck, xc = qe.get(language, ("", ""))
        summary.append({"lang": language, "benchmark": "all", "n": len(nat), "naturalness_mean": nm,
                        "naturalness_std": ns, "fidelity_mean": fm, "fidelity_std": fs,
                        "cometkiwi_xl": ck, "xcomet_xl": xc})
        print(f"{language:<10} {len(nat):<5} {fmt(nm, ns):<16} {fmt(fm, fs):<16} {ck!s:<8} {xc!s:<8}")

    print(f"\n{'Lang':<10} {'Benchmark':<20} {'n':<5} {'Naturalness':<16} {'Fidelity':<16}")
    for language in languages:
        for benchmark in BENCHMARKS:
            sub = [r for r in rows_by_lang[language] if r["benchmark"] == benchmark]
            nat, fid = scores(sub, "naturalness_int"), scores(sub, "fidelity_int")
            (nm, ns), (fm, fs) = stat(nat), stat(fid)
            summary.append({"lang": language, "benchmark": benchmark, "n": len(nat), "naturalness_mean": nm,
                            "naturalness_std": ns, "fidelity_mean": fm, "fidelity_std": fs})
            print(f"{language:<10} {benchmark:<20} {len(nat):<5} {fmt(nm, ns):<16} {fmt(fm, fs):<16}")

    print(f"\nScore distribution\n{'Lang':<10} {'Metric':<14} {'1':>4} {'2':>4} {'3':>4} {'4':>4} {'5':>4}")
    for language in languages:
        for metric in ("naturalness_int", "fidelity_int"):
            counts = Counter(scores(rows_by_lang[language], metric))
            print(f"{language:<10} {metric[:-4]:<14} " + " ".join(f"{counts.get(i, 0):>4}" for i in range(1, 6)))

    if len(languages) == 2:
        first, second = languages
        by_source = defaultdict(dict)
        for language in languages:
            for r in rows_by_lang[language]:
                by_source[r["english_source"]][language] = r
        paired = [(d[first], d[second]) for d in by_source.values()
                  if first in d and second in d
                  and d[first]["naturalness_int"] is not None and d[second]["naturalness_int"] is not None]
        print(f"\nPaired comparison on the same English source: {len(paired)} prompts")
        for metric in ("naturalness_int", "fidelity_int"):
            diffs = [b[metric] - a[metric] for a, b in paired]
            m, s = stat(diffs)
            r = pearson([a[metric] for a, _ in paired], [b[metric] for _, b in paired])
            print(f"  {metric[:-4]:<12} mean ({second} - {first}) = {m:+.2f} +- {s:.2f}; per-prompt Pearson r = {r}")

    print("\nAnnotator notes (non-empty)")
    notes = [(language, r) for language in languages for r in rows_by_lang[language] if (r.get("notes") or "").strip()]
    for language, r in notes:
        print(f"  [{r['id']}] (n={r['naturalness_int']}, f={r['fidelity_int']}) {r['notes'].strip()[:200]}")
    if not notes:
        print("  (none)")

    out_csv = reliability_dir("native_validation", args.output_dir) / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in summary:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
