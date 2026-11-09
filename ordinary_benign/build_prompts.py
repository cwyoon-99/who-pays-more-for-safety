"""Sample the 300 ordinary benign instructions of the control experiment (Appendix C).

Just-Eval (re-align/just-eval-instruct, 1,000 instructions) is restricted to its helpfulness
part (``category == "regular"``, 800 instructions; the 200 safety instructions are removed),
de-duplicated on the instruction text (first occurrence kept, 781 left), and 300
instructions are sampled with ``random.seed(42)``.

Output:
    data/ordinary_benign/ordinary_benign_300.json   (the sampled Just-Eval records)
    data/ordinary_benign/english.json               ({id, prompt, prompt_original}; id = 0..299)

Usage:
    python ordinary_benign/build_prompts.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import DATA_DIR, SEED, prompts_path
from safety_cost.utils import save_json

JUST_EVAL = "re-align/just-eval-instruct"
BENCHMARK = "ordinary_benign"
N_PROMPTS = 300


def select_prompts(rows: list[dict], n: int) -> list[dict]:
    regular = [r for r in rows if r["category"] == "regular"]
    seen, unique = set(), []
    for r in regular:
        if r["instruction"] not in seen:
            seen.add(r["instruction"])
            unique.append(r)
    print(f"Just-Eval: {len(rows)} total, {len(regular)} regular, {len(unique)} after de-duplication")
    random.seed(SEED)
    return random.sample(unique, n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=N_PROMPTS)
    parser.add_argument("--data_dir", type=Path, default=None)
    args = parser.parse_args()

    import datasets

    sampled = select_prompts(list(datasets.load_dataset(JUST_EVAL, split="test")), args.n)

    out_dir = Path(args.data_dir or DATA_DIR) / BENCHMARK
    save_json(sampled, out_dir / f"{BENCHMARK}_{args.n}.json")
    save_json([{"id": i, "prompt": r["instruction"], "prompt_original": r["instruction"]}
               for i, r in enumerate(sampled)], prompts_path(BENCHMARK, "english", args.data_dir))
    print(f"wrote {len(sampled)} prompts to {out_dir}")


if __name__ == "__main__":
    main()
