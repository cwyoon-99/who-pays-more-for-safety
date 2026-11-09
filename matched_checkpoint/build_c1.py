"""Build the c1 (safety-aligned) mixture: the 60K instruction data + 20K English safety.

The 20K English safety examples are sampled (seed 42) from the pool built by
build_english_safety.py. The mixture is shuffled with seed 42.

Output (in --data_dir):
    c1_english.parquet            columns prompt, response (80K rows)
    c1_english_labeled.parquet    + source (instruction / safety)

Usage:
    python matched_checkpoint/build_c1.py
    python matched_checkpoint/build_c1.py --report_tokens   # response-token budget per source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import MC_DATA_DIR

N_SAFETY = 20000
SEED = 42


def report_response_tokens(c1: pd.DataFrame, tokenizer_name: str) -> None:
    """Response tokens per source (the SFT loss is computed on response tokens only)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    c1 = c1.copy()
    c1["rtoks"] = c1["response"].map(lambda s: len(tok(s, add_special_tokens=False).input_ids))
    print(c1.groupby("source")["rtoks"].agg(["count", "median", "mean", "sum"]).round(1))
    total = c1["rtoks"].sum()
    for src in ("instruction", "safety"):
        s = c1[c1.source == src]["rtoks"].sum()
        print(f"  {src:12s}: {s:,} response tokens ({100 * s / total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=MC_DATA_DIR)
    parser.add_argument("--instruction_base", type=Path, default=None,
                        help="Default: <data_dir>/instruction_base.parquet")
    parser.add_argument("--english_safety", type=Path, default=None,
                        help="Default: <data_dir>/english_safety_30K.parquet")
    parser.add_argument("--report_tokens", action="store_true")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    inst = pd.read_parquet(args.instruction_base or args.data_dir / "instruction_base.parquet").copy()
    inst["source"] = "instruction"
    saf = pd.read_parquet(args.english_safety or args.data_dir / "english_safety_30K.parquet")
    saf = saf.sample(n=N_SAFETY, random_state=SEED).copy()
    saf["source"] = "safety"
    print(f"instruction: {len(inst)}  English safety (sampled): {len(saf)}")

    c1 = pd.concat([inst[["prompt", "response", "source"]], saf[["prompt", "response", "source"]]],
                   ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = args.data_dir / "c1_english.parquet"
    c1[["prompt", "response"]].to_parquet(out, index=False)
    c1.to_parquet(args.data_dir / "c1_english_labeled.parquet", index=False)
    print(f"c1: {len(c1)} rows -> {out}")

    if args.report_tokens:
        report_response_tokens(c1, args.tokenizer)


if __name__ == "__main__":
    main()
