"""Judge length sensitivity (Appendix B, Tables length_ratio and length_matched).

1. Median response-length ratio r = len(unaligned) / len(aligned) over the judged pairs
   (tiktoken ``o200k_base`` tokens), per language and model.
2. Safety Cost disparity relative to English (Cost_L - Cost_English) on all judged pairs
   vs. only length-matched pairs (r in [0.67, 1.5]). Pairs are pooled over the three
   over-refusal benchmarks.

Usage:
    python analysis/length_analysis.py [--output_dir outputs] [--band 0.67 1.5] [--pairs ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from common import figures_dir

from safety_cost.config import BENCHMARKS, DEFAULT_PAIRS, MODEL_PAIRS, MULTIJAIL_LANGUAGES, pairwise_path
from safety_cost.utils import load_json

LANGUAGES = list(MULTIJAIL_LANGUAGES)  # the six languages of the main figure


def load_pairs(output_dir, pair_keys) -> pd.DataFrame:
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    records = []
    for key in pair_keys:
        pair = MODEL_PAIRS[key]
        for benchmark in BENCHMARKS:
            for language in LANGUAGES:
                path = pairwise_path(benchmark, language, pair, output_dir)
                if not path.exists():
                    continue
                for item in load_json(path):
                    if item.get("avg_score") is None:
                        continue
                    records.append({
                        "family": pair.family, "benchmark": benchmark, "language": language,
                        "aligned_tokens": len(enc.encode(item["aligned_response"])),
                        "unaligned_tokens": len(enc.encode(item["unaligned_response"])),
                        "cost": 100.0 * (1.0 - item["avg_score"]),
                    })
    df = pd.DataFrame(records)
    if not df.empty:
        df["ratio"] = df["unaligned_tokens"] / df["aligned_tokens"].clip(lower=1)
    return df


def disparity(df: pd.DataFrame) -> pd.DataFrame:
    cost = df.groupby(["family", "language"])["cost"].mean().unstack("family")
    return cost.sub(cost.loc["english"], axis=1).drop(index="english")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--band", nargs=2, type=float, default=[0.67, 1.5])
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), choices=list(MODEL_PAIRS))
    args = parser.parse_args()

    df = load_pairs(args.output_dir, args.pairs)
    if df.empty:
        sys.exit("No pairwise judgments found; run scripts/20_judge_pairwise.py first.")
    families = [MODEL_PAIRS[k].family for k in args.pairs if MODEL_PAIRS[k].family in set(df["family"])]
    order = [l for l in LANGUAGES if l in set(df["language"])]

    ratio = df.groupby(["language", "family"])["ratio"].median().unstack("family").loc[order, families]
    print(f"Median length ratio (unaligned / aligned) over {len(df):,} judged pairs:")
    print(ratio.round(2).to_string(), "\n")

    lo, hi = args.band
    matched = df[(df["ratio"] >= lo) & (df["ratio"] <= hi)]
    full, filt = disparity(df), disparity(matched)
    rows = [l for l in order if l != "english"]
    table = pd.DataFrame({f: [f"{full.loc[l, f]:+.1f} -> {filt.loc[l, f]:+.1f}" for l in rows] for f in families},
                         index=rows)
    table.loc["mean"] = [f"{full[f].mean():+.1f} -> {filt[f].mean():+.1f}" for f in families]
    print(f"Cost_L - Cost_English: all pairs -> length-matched r in [{lo}, {hi}] "
          f"({100 * len(matched) / len(df):.0f}% of pairs retained):")
    print(table.to_string())

    out = figures_dir(args.output_dir)
    ratio.to_csv(out / "length_ratio.csv")
    table.to_csv(out / "length_matched.csv")
    print(f"\nSaved {out / 'length_ratio.csv'} and {out / 'length_matched.csv'}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
