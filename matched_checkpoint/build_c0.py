"""Build the c0 (unaligned) mixture: the 60K instruction data + 20K non-safety filler.

The filler is the non-safety counterpart of c1's 20K English safety slice: it comes from the
same Aya sources with the same distribution (2,667 per target language, aya_dataset first and
aya_collection backfill, plus 4,000 non-target aya_dataset examples) and is disjoint from the
instruction data (no shared prompts). Sampling uses seed 43 so that it does not repeat the
draws of build_instruction_base.py.

Output (in --data_dir):
    c0.parquet            columns prompt, response
    c0_labeled.parquet    + source, language

Usage:
    python matched_checkpoint/build_c0.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets  # noqa: F401  (imported before seeding to keep the sampling order)
import pandas as pd

from common import (ALL_TARGET_HUMAN_NAMES, MC_DATA_DIR, TARGET_COLLECTION_CFG, TARGET_HUMAN,
                    collect_from_collection, load_aya_human)

N_TARGET = 2667        # per target language (2667 * 6 ~= 16000)
N_SIDE_TOTAL = 4000    # non-target languages (total filler ~= 20000)
SEED = 43


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=MC_DATA_DIR)
    parser.add_argument("--instruction_base", type=Path, default=None,
                        help="Default: <data_dir>/instruction_base.parquet")
    args = parser.parse_args()

    random.seed(SEED)
    base = pd.read_parquet(args.instruction_base or args.data_dir / "instruction_base.parquet")
    exclude = set(base["prompt"].tolist())
    print(f"instruction_base: {len(base)}, excluded prompts: {len(exclude)}")

    human = load_aya_human()
    human["inp_stripped"] = human["inputs"].str.strip()  # instruction_base prompts are stripped

    rows, summary = [], []
    for canon, names in TARGET_HUMAN.items():
        sub = human[human.language.isin(names) & ~human["inp_stripped"].isin(exclude)]
        take = min(N_TARGET, len(sub))
        s = sub.sample(n=take, random_state=SEED) if take > 0 else sub
        for _, r in s.iterrows():
            inp = str(r["inputs"]).strip()
            tgt = str(r["targets"]).strip()
            if inp and tgt:
                rows.append({"prompt": inp, "response": tgt, "language": canon, "source": "aya_human"})
                exclude.add(inp)
        got_human = sum(1 for x in rows if x["language"] == canon and x["source"] == "aya_human")
        deficit = N_TARGET - got_human
        got_coll = 0
        if deficit > 0:
            coll = collect_from_collection(TARGET_COLLECTION_CFG[canon], deficit, seed=SEED, exclude=exclude)
            for c in coll:
                c["language"] = canon
                exclude.add(c["prompt"])
            rows.extend(coll)
            got_coll = len(coll)
        summary.append((canon, got_human, got_coll))
        print(f"  filler {canon}: human={got_human} collection={got_coll}")

    nontarget = human[~human.language.isin(ALL_TARGET_HUMAN_NAMES) & ~human["inp_stripped"].isin(exclude)]
    side = nontarget.sample(n=min(N_SIDE_TOTAL, len(nontarget)), random_state=SEED)
    n_side = 0
    for _, r in side.iterrows():
        inp = str(r["inputs"]).strip()
        tgt = str(r["targets"]).strip()
        if inp and tgt:
            rows.append({"prompt": inp, "response": tgt, "language": f"[nt]{r['language']}",
                         "source": "aya_human_nontarget"})
            n_side += 1

    filler = pd.DataFrame(rows)
    print(f"filler: {len(filler)} (targets {sum(h + c for _, h, c in summary)}, non-target {n_side})")

    base2 = base.copy()
    base2["source"] = "instruction"
    base2["language"] = "[base]"
    c0 = pd.concat([base2[["prompt", "response", "source", "language"]],
                    filler[["prompt", "response", "source", "language"]]], ignore_index=True)
    c0 = c0.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = args.data_dir / "c0.parquet"
    c0[["prompt", "response"]].to_parquet(out, index=False)
    c0.to_parquet(args.data_dir / "c0_labeled.parquet", index=False)
    print(f"c0: {len(c0)} rows -> {out}")
    print(f"filler/instruction prompt overlap: {len(set(filler['prompt']) & set(base['prompt']))} (should be 0)")


if __name__ == "__main__":
    main()
