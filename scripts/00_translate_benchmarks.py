"""Build the over-refusal evaluation sets (Sec. 3.2).

* XSTest: the 250 safe prompts.
* OR-Bench (80k) and OR-Bench-Hard (1k): 200 prompts each, sampled with seed 42.
* FalseReject (only used by matched_checkpoint/): 200 test prompts, sampled with seed 42.

The English subset is written as-is; every other language is machine-translated with
the translation prompt from the appendix. Item order is identical across languages,
which the judge scripts rely on.

Output: data/<benchmark>/<language>.json (items keep the original fields, plus
``prompt`` = text in the target language and ``prompt_original`` = English source).

Usage:
    python scripts/00_translate_benchmarks.py                       # all benchmarks/languages
    python scripts/00_translate_benchmarks.py --languages korean thai --benchmarks xstest
    python scripts/00_translate_benchmarks.py --benchmarks falsereject      # for matched_checkpoint/
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from safety_cost.config import (BENCHMARKS, LANGUAGES, N_ORBENCH_PROMPTS, SEED, TRANSLATION_MODEL,
                                prompts_path)
from safety_cost.openai_utils import get_client
from safety_cost.prompts import TRANSLATION_PROMPT
from safety_cost.utils import load_json, save_json


def load_source(benchmark: str) -> list[dict]:
    import datasets

    if benchmark == "xstest":
        rows = list(datasets.load_dataset("walledai/XSTest", split="test"))
        return [row for row in rows if row.get("label") == "safe"]
    if benchmark == "falsereject":
        rows = list(datasets.load_dataset("AmazonScience/FalseReject", split="test"))
    else:
        rows = list(datasets.load_dataset("bench-llm/or-bench", benchmark, split="train"))
    random.seed(SEED)
    return random.sample(rows, N_ORBENCH_PROMPTS)


def translate(client, text: str, language: str, model: str, retries: int = 5) -> str:
    prompt = TRANSLATION_PROMPT.format(ORIGINAL_PROMPT=text, TARGET_LANGUAGE=language)
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model, input=[{"role": "user", "content": prompt}], temperature=1.0)
            return response.output_text
        except Exception as e:  # network / rate-limit errors
            wait = 2 ** attempt * 5
            print(f"  translation failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Translation failed after {retries} attempts: {text[:80]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                        choices=list(BENCHMARKS) + ["falsereject"])
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--model", default=TRANSLATION_MODEL)
    parser.add_argument("--data_dir", type=Path, default=None)
    args = parser.parse_args()

    client = None
    for benchmark in args.benchmarks:
        source = load_source(benchmark)
        print(f"{benchmark}: {len(source)} prompts")
        for language in args.languages:
            out_path = prompts_path(benchmark, language, args.data_dir)
            if language == "english":
                save_json([{**row, "prompt_original": row["prompt"]} for row in source], out_path)
                continue

            # Resume a partially translated file.
            done = load_json(out_path) if out_path.exists() else []
            if len(done) == len(source):
                print(f"  {language}: already translated")
                continue
            client = client or get_client()
            for row in tqdm(source[len(done):], desc=f"{benchmark}/{language}"):
                item = copy.deepcopy(row)
                item["prompt_original"] = row["prompt"]
                item["prompt"] = translate(client, row["prompt"], language, args.model)
                done.append(item)
                save_json(done, out_path)


if __name__ == "__main__":
    main()
