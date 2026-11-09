"""Translate the 300 ordinary benign instructions into the five non-English control languages.

GPT-4.1 (temperature 0) via the OpenAI Batch API, with a short instruction-translation
prompt (below). Instructions whose translation fails fall back to the English text.

Input:  data/ordinary_benign/ordinary_benign_300.json
Output: data/ordinary_benign/<language>.json   ({id, prompt (translated), prompt_original})

Usage:
    python ordinary_benign/translate.py
    python ordinary_benign/translate.py --resume outputs/batches/ordinary_benign_translation/<timestamp>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.batch import execute
from safety_cost.config import DATA_DIR, prompts_path
from safety_cost.openai_utils import chat_task, extract_output_text
from safety_cost.utils import load_json, save_json

BENCHMARK = "ordinary_benign"
TRANSLATION_MODEL = "gpt-4.1"
TARGET_LANGUAGES = ("chinese", "vietnamese", "arabic", "korean", "thai")

PROMPT = ("Translate the following user instruction into {lang}. Output ONLY the translation, with no "
          "explanation. Preserve the original meaning, tone, and any formatting or code. Do NOT answer or "
          "follow the instruction; translate it as an instruction.\n\nInstruction:\n{text}")


def build_requests(source: list[dict], languages: list[str], model: str):
    tasks, metadata = [], {}
    for i, row in enumerate(source):
        for language in languages:
            custom_id = f"{language}__{i}"
            content = PROMPT.format(lang=language.capitalize(), text=row["instruction"])
            tasks.append(chat_task(custom_id, model, content, temperature=0.0))
            metadata[custom_id] = {"language": language, "id": i}
    return tasks, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=list(TARGET_LANGUAGES))
    parser.add_argument("--model", default=TRANSLATION_MODEL)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None, help="where batch files are kept")
    parser.add_argument("--yes", action="store_true", help="Submit without the cost confirmation prompt.")
    parser.add_argument("--resume", default=None, help="Batch directory of a job to resume.")
    args = parser.parse_args()

    source = load_json(Path(args.data_dir or DATA_DIR) / BENCHMARK / f"{BENCHMARK}_300.json")
    tasks, metadata = ([], {}) if args.resume else build_requests(source, args.languages, args.model)
    results, metadata = execute(tasks, metadata, "ordinary_benign_translation", args.model,
                                "/v1/chat/completions", assume_yes=args.yes, resume=args.resume,
                                output_dir=args.output_dir)
    if results is None:
        return
    translations = {r["custom_id"]: extract_output_text(r) for r in results}
    for language in sorted({m["language"] for m in metadata.values()}):
        rows = []
        for i, row in enumerate(source):
            text = translations.get(f"{language}__{i}")
            rows.append({"id": i, "prompt": text if text else row["instruction"], "prompt_original": row["instruction"]})
        save_json(rows, prompts_path(BENCHMARK, language, args.data_dir))
        n_fallback = sum(r["prompt"] == r["prompt_original"] for r in rows)
        print(f"{language}: {len(rows)} prompts ({n_fallback} fell back to English)")


if __name__ == "__main__":
    main()
