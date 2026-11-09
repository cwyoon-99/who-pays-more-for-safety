"""Back-translate judged responses to English (Appendix B, back-translated re-evaluation).

For every (pair, benchmark, language) pairwise file of the main evaluation, the aligned and
unaligned responses are translated to English with the Google Cloud Translation API (v2,
Basic). The output is a copy of the pairwise file with ``aligned_response_en`` and
``unaligned_response_en`` added. Re-running resumes: entries already translated are kept.

Requires GOOGLE_TRANSLATE_API_KEY (environment variable or .env).

Input:  outputs/judgments/pairwise/<benchmark>/<language>/<aligned>__<unaligned>.json
Output: outputs/reliability/back_translation/translated/<benchmark>/<language>/<aligned>__<unaligned>.json

Usage:
    python reliability/back_translation/back_translate.py
    python reliability/back_translation/back_translate.py --languages korean --limit 5   # quick test
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.common import (GOOGLE_LANGUAGE_CODES, RELIABILITY_PAIRS, cell_path, google_api_key,
                                google_translate, reliability_dir)
from safety_cost.config import BENCHMARKS, MODEL_PAIRS, pairwise_path
from safety_cost.utils import load_json, save_json

BACK_TRANSLATION_LANGUAGES = ("chinese", "korean", "thai")
FIELDS = ("aligned_response", "unaligned_response")


def process_cell(src: Path, out_file: Path, source_code: str, key: str, max_workers: int,
                 limit: int | None) -> dict:
    data = load_json(src)
    if limit is not None:
        data = data[:limit]
    prior = {e["index"]: e for e in load_json(out_file)} if out_file.exists() else {}

    todo = []
    for entry in data:
        previous = prior.get(entry["index"], {})
        for field in FIELDS:
            if previous.get(f"{field}_en") is None:
                todo.append((entry["index"], f"{field}_en", entry.get(field, "")))

    translated, failed = {}, 0
    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(google_translate, text, source_code, "en", key): (idx, en_field)
                       for idx, en_field, text in todo}
            for n_done, future in enumerate(as_completed(futures), 1):
                idx, en_field = futures[future]
                try:
                    translated[(idx, en_field)] = future.result()
                except Exception as e:
                    print(f"  FAIL {src.name} index={idx} {en_field}: {type(e).__name__}", file=sys.stderr)
                    translated[(idx, en_field)] = ""
                    failed += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    print(f"  {n_done}/{len(todo)}", flush=True)

    out = []
    for entry in data:
        idx = entry["index"]
        merged = dict(prior.get(idx) or entry)
        for field in FIELDS:
            en_field = f"{field}_en"
            if (idx, en_field) in translated:
                merged[en_field] = translated[(idx, en_field)]
            else:
                merged.setdefault(en_field, prior.get(idx, {}).get(en_field, ""))
        out.append(merged)
    save_json(out, out_file)
    return {"translated": len(todo) - failed, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(RELIABILITY_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--languages", nargs="+", default=list(BACK_TRANSLATION_LANGUAGES),
                        choices=[lang for lang in GOOGLE_LANGUAGE_CODES if lang != "english"])
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N entries per file (testing)")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    key = google_api_key()
    out_root = reliability_dir("back_translation", args.output_dir) / "translated"
    totals = {"translated": 0, "failed": 0}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for language in args.languages:
            for benchmark in args.benchmarks:
                src = pairwise_path(benchmark, language, pair, args.output_dir)
                if not src.exists():
                    print(f"WARN: missing {src}", file=sys.stderr)
                    continue
                print(f"[{pair_key} / {benchmark} / {language}]", flush=True)
                stats = process_cell(src, cell_path(out_root, benchmark, language, pair),
                                     GOOGLE_LANGUAGE_CODES[language], key, args.max_workers, args.limit)
                for k in totals:
                    totals[k] += stats[k]
    print(f"done: {totals}")


if __name__ == "__main__":
    main()
