"""Build the shared 60K multilingual instruction data ("Data 1" of Table matched_data).

Per target language (English, Chinese, Vietnamese, Arabic, Korean, Thai) up to 8,000 examples
are drawn from the human-written CohereForAI/aya_dataset; any deficit is backfilled from
CohereForAI/aya_collection_language_split. 12,000 additional examples (20%) come from the
non-target languages of aya_dataset. Every response must have at least 50 characters.

Output (in --output_dir):
    instruction_base.parquet            columns prompt, response
    instruction_base_labeled.parquet    + language, source (for reporting)

Usage:
    python matched_checkpoint/build_instruction_base.py
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets  # noqa: F401  (imported before seeding to keep the sampling order)
import pandas as pd

from common import (ALL_TARGET_HUMAN_NAMES, MC_DATA_DIR, TARGET_COLLECTION_CFG, TARGET_HUMAN,
                    collect_from_collection, load_aya_human)

N_TARGET = 8000         # per target language
N_SIDE_TOTAL = 12000    # non-target languages (~20%)
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=MC_DATA_DIR)
    args = parser.parse_args()

    random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    human = load_aya_human()

    rows, summary = [], []

    # Target languages: human data first, backfill from aya_collection.
    for canon, names in TARGET_HUMAN.items():
        sub = human[human.language.isin(names)]
        take = min(N_TARGET, len(sub))
        s = sub.sample(n=take, random_state=SEED) if take > 0 else sub
        for _, r in s.iterrows():
            inp = str(r["inputs"]).strip()
            tgt = str(r["targets"]).strip()
            if inp and tgt:
                rows.append({"prompt": inp, "response": tgt, "language": canon, "source": "aya_human"})
        got_human = sum(1 for x in rows if x["language"] == canon and x["source"] == "aya_human")
        deficit = N_TARGET - got_human
        got_coll = 0
        if deficit > 0:
            print(f"[{canon}] human={got_human}, backfilling {deficit} from "
                  f"aya_collection[{TARGET_COLLECTION_CFG[canon]}]...")
            coll = collect_from_collection(TARGET_COLLECTION_CFG[canon], deficit, seed=SEED)
            for c in coll:
                c["language"] = canon
            rows.extend(coll)
            got_coll = len(coll)
        summary.append((canon, got_human, got_coll, got_human + got_coll))
        print(f"  -> {canon}: human={got_human} + collection={got_coll} = {got_human + got_coll}")

    # Non-target languages from aya human (diversity).
    nontarget = human[~human.language.isin(ALL_TARGET_HUMAN_NAMES)]
    side = nontarget.sample(n=min(N_SIDE_TOTAL, len(nontarget)), random_state=SEED)
    n_side = 0
    for _, r in side.iterrows():
        inp = str(r["inputs"]).strip()
        tgt = str(r["targets"]).strip()
        if inp and tgt:
            rows.append({"prompt": inp, "response": tgt, "language": f"[nt]{r['language']}",
                         "source": "aya_human_nontarget"})
            n_side += 1
    print(f"[non-target] {n_side} from {nontarget.language.nunique()} languages")

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    out = args.output_dir / "instruction_base.parquet"
    df[["prompt", "response"]].to_parquet(out, index=False)
    df.to_parquet(args.output_dir / "instruction_base_labeled.parquet", index=False)

    print(f"\ninstruction_base: {len(df)} rows -> {out}")
    print(f"{'language':12s}{'human':>8}{'collection':>12}{'total':>8}")
    for canon, h, c, t in summary:
        print(f"{canon:12s}{h:>8}{c:>12}{t:>8}")
    print(f"{'(non-target)':12s}{'':>8}{'':>12}{n_side:>8}")
    for k, v in Counter(df["source"]).most_common():
        print(f"  {k:24s}: {v} ({100 * v / len(df):.1f}%)")


if __name__ == "__main__":
    main()
