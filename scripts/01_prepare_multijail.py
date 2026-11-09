"""Export the MultiJail prompts (Deng et al., 2024) used to measure Safety Gain.

Output: data/multijail/<language>.json with fields id, lang, prompt, source, tags
(315 prompts per language).

Usage:
    python scripts/01_prepare_multijail.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import MULTIJAIL_LANGUAGES, prompts_path
from safety_cost.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    args = parser.parse_args()

    import datasets

    rows = list(datasets.load_dataset("DAMO-NLP-SG/MultiJail", split="train"))
    for language, code in MULTIJAIL_LANGUAGES.items():
        items = [{"id": str(row["id"]), "lang": code, "prompt": row[code],
                  "source": row["source"], "tags": row["tags"]} for row in rows]
        path = prompts_path("multijail", language, args.data_dir)
        save_json(items, path)
        print(f"{language}: {len(items)} prompts -> {path}")


if __name__ == "__main__":
    main()
