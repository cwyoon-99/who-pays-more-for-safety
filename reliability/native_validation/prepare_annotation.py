"""Build the native-speaker validation sheets for the translated prompts (Appendix B).

For each language, 50 prompts are drawn with a stratified sample of 17 XSTest, 17 OR-Bench,
and 16 OR-Bench-Hard prompts. Indices are drawn with a per-benchmark seed
(``random.Random("42|<benchmark>")``), so every language is rated on the same English sources.

Each CSV row holds the English source and its translation, plus empty ``naturalness``,
``fidelity`` (both 1-5) and ``notes`` columns for the annotator (see instructions_<language>.md).
For Chinese, an optional Korean rendering of each translated prompt (prepared by a human
translator and passed as a JSON file ``{row_id: text}``) is added as the auxiliary column
``chinese_to_korean_reference``.

Input:  data/<benchmark>/<language>.json
Output: outputs/reliability/native_validation/human_eval_<language>.csv

Usage:
    python reliability/native_validation/prepare_annotation.py
    python reliability/native_validation/prepare_annotation.py --chinese_reference zh_to_ko.json
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.common import reliability_dir
from safety_cost.config import SEED, prompts_path
from safety_cost.utils import load_json

LANGUAGES = ("chinese", "korean")
SAMPLE_PER_BENCHMARK = {"xstest": 17, "or-bench-80k": 17, "or-bench-hard-1k": 16}
REFERENCE_COLUMN = "chinese_to_korean_reference"


def stratified_indices(language: str, data_dir: Path | None) -> dict[str, list[int]]:
    """Same indices for every language: the seed depends only on the benchmark."""
    out = {}
    for benchmark, n in SAMPLE_PER_BENCHMARK.items():
        population = len(load_json(prompts_path(benchmark, language, data_dir)))
        rng = random.Random(f"{SEED}|{benchmark}")
        out[benchmark] = sorted(rng.sample(range(population), min(n, population)))
    return out


def build_rows(language: str, indices: dict[str, list[int]], data_dir: Path | None,
               reference: dict[str, str] | None) -> list[dict]:
    rows = []
    for benchmark, idxs in indices.items():
        data = load_json(prompts_path(benchmark, language, data_dir))
        for idx in idxs:
            item = data[idx]
            row = {
                "id": f"{language}_{benchmark}_{idx:04d}",
                "benchmark": benchmark,
                "category": item.get("category") or item.get("focus") or item.get("type") or "",
                "english_source": item.get("prompt_original", ""),
                "translated_prompt": item.get("prompt", ""),
            }
            if reference is not None:
                row[REFERENCE_COLUMN] = reference.get(row["id"], "")
            row.update({"naturalness": "", "fidelity": "", "notes": ""})
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--chinese_reference", type=Path, default=None,
                        help="JSON {row_id: Korean rendering} added to the Chinese sheet")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    indices = stratified_indices(args.languages[0], args.data_dir)
    for benchmark, idxs in indices.items():
        print(f"{benchmark}: {len(idxs)} indices, e.g. {idxs[:5]}")

    out_dir = reliability_dir("native_validation", args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for language in args.languages:
        reference = None
        if language == "chinese" and args.chinese_reference:
            reference = load_json(args.chinese_reference)
        rows = build_rows(language, indices, args.data_dir, reference)
        out_csv = out_dir / f"human_eval_{language}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
