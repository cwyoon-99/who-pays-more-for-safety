"""Build the English safety pool (OLMo 3 safety recipe: WildJailbreak + CoCoNot).

30K examples are sampled from the union of allenai/wildjailbreak (train) and
allenai/coconot ("original", train) so that every WildJailbreak data type
(vanilla/adversarial x harmful/benign) and CoCoNot keep their share of the full union.
build_c1.py then samples the 20K "English safety" slice of c1 from this pool.

WildJailbreak prompts are the vanilla prompt followed (for adversarial examples) by the
adversarial prompt, joined by a space; responses are the WildJailbreak completions and the
CoCoNot responses.

Output: <output_dir>/english_safety_30K.parquet (columns prompt, response; 29,998 rows)

Both datasets are gated on Hugging Face: accept their terms and export HF_TOKEN.

Usage:
    python matched_checkpoint/build_english_safety.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from datasets import load_dataset

from common import MC_DATA_DIR

SIZE = 30000
SEED = 42


def wildjailbreak_to_sft(dataset) -> pd.DataFrame:
    rows = []
    for item in dataset:
        parts = [p for p in (item.get("vanilla"), item.get("adversarial")) if p]
        prompt = " ".join(parts) if parts else str(item.get("input", ""))
        rows.append({"prompt": prompt, "response": item.get("completion", "")})
    return pd.DataFrame(rows)


def coconot_to_sft(dataset) -> pd.DataFrame:
    return pd.DataFrame([{"prompt": item["prompt"], "response": item["response"]} for item in dataset])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=MC_DATA_DIR)
    args = parser.parse_args()

    wildjailbreak = load_dataset("allenai/wildjailbreak", "train", split="train",
                                 delimiter="\t", keep_default_na=False)
    coconot = load_dataset("allenai/coconot", "original", split="train")
    print(f"WildJailbreak: {len(wildjailbreak):,}  CoCoNot: {len(coconot):,}")

    # Row indices per WildJailbreak data type, in order of first appearance.
    data_type_groups: dict[str, list[int]] = {}
    for idx, data_type in enumerate(wildjailbreak["data_type"]):
        data_type_groups.setdefault(data_type, []).append(idx)

    total_all = len(wildjailbreak) + len(coconot)
    counts = {dt: min(int(SIZE * (len(indices) / total_all)), len(indices))
              for dt, indices in data_type_groups.items()}
    n_coconot = min(int(SIZE * (len(coconot) / total_all)), len(coconot))
    for dt, n in counts.items():
        print(f"  {dt}: {n:,}")
    print(f"  coconot: {n_coconot:,}")

    np.random.seed(SEED)
    wj_indices = []
    for dt, n in counts.items():
        if n > 0:
            wj_indices.extend(np.random.choice(data_type_groups[dt], size=n, replace=False).tolist())
    wj_sampled = wildjailbreak.select(wj_indices)
    cn_indices = np.random.choice(len(coconot), size=n_coconot, replace=False)
    cn_sampled = coconot.select(cn_indices.tolist())

    df = pd.concat([wildjailbreak_to_sft(wj_sampled), coconot_to_sft(cn_sampled)], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "english_safety_30K.parquet"
    df.to_parquet(out, index=False)
    print(f"English safety pool: {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
