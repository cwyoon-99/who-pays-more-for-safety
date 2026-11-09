"""Re-run pairwise judging on the back-translated (English) responses.

Identical to the main pairwise evaluation (GPT-5-mini, same prompt, two position-swapped
judgments, Arena-Hard scoring) except for the inputs: the English source prompt
(``prompt_original``) and the back-translated responses (``*_response_en``).

Input:  outputs/reliability/back_translation/translated/<benchmark>/<language>/<pair>.json
Output: outputs/reliability/back_translation/judged/<benchmark>/<language>/<pair>.json
        (the input plus judgment_1_bt, judgment_2_bt, raw_output_*_bt, scores_bt, avg_score_bt)

Usage:
    python reliability/back_translation/judge_backtranslated.py
    python reliability/back_translation/judge_backtranslated.py --resume outputs/batches/back_translation/<timestamp>
    python reliability/back_translation/compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.back_translation.back_translate import BACK_TRANSLATION_LANGUAGES
from reliability.common import (RELIABILITY_PAIRS, cell_path, collect_swaps, reliability_dir, score_swaps,
                                swap_tasks)
from safety_cost.batch import execute
from safety_cost.config import BENCHMARKS, MODEL_PAIRS, PAIRWISE_JUDGE, PAIRWISE_REASONING_EFFORT
from safety_cost.utils import load_json, save_json


def build_requests(args, translated_root: Path):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for language in args.languages:
            for benchmark in args.benchmarks:
                src = cell_path(translated_root, benchmark, language, pair)
                if not src.exists():
                    print(f"WARN: missing {src}", file=sys.stderr)
                    continue
                data = load_json(src)
                if args.limit is not None:
                    data = data[:args.limit]
                for entry in data:
                    en_aligned = entry.get("aligned_response_en", "")
                    en_unaligned = entry.get("unaligned_response_en", "")
                    if not en_aligned or not en_unaligned:
                        continue
                    prefix = f"{pair_key}:{benchmark}:{language}:{entry['index']}"
                    for swap, custom_id, task in swap_tasks(prefix, entry["prompt_original"], en_aligned,
                                                            en_unaligned, args.judge, args.reasoning_effort):
                        tasks.append(task)
                        metadata[custom_id] = {"benchmark": benchmark, "language": language, "pair": pair_key,
                                               "index": entry["index"], "swap": swap}
    return tasks, metadata


def write_outputs(results, metadata, translated_root: Path, out_root: Path):
    summary = {"cells_written": 0, "instances_judged": 0, "incomplete_swaps": 0}
    for (benchmark, language, pair_key), by_index in collect_swaps(results, metadata).items():
        pair = MODEL_PAIRS[pair_key]
        src = cell_path(translated_root, benchmark, language, pair)
        rows = []
        for entry in load_json(src):
            row = dict(entry)
            swaps = by_index.get(entry["index"], {})
            if "swap1" in swaps and "swap2" in swaps:
                for field, value in score_swaps(swaps["swap1"], swaps["swap2"]).items():
                    row[f"{field}_bt"] = value
                summary["instances_judged"] += 1
            else:
                summary["incomplete_swaps"] += 1
            rows.append(row)
        save_json(rows, cell_path(out_root, benchmark, language, pair))
        summary["cells_written"] += 1
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", default=PAIRWISE_JUDGE)
    parser.add_argument("--reasoning_effort", default=PAIRWISE_REASONING_EFFORT)
    parser.add_argument("--pairs", nargs="+", default=list(RELIABILITY_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--languages", nargs="+", default=list(BACK_TRANSLATION_LANGUAGES))
    parser.add_argument("--limit", type=int, default=None, help="only the first N entries per file (testing)")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--yes", action="store_true", help="Submit without the cost confirmation prompt.")
    parser.add_argument("--resume", default=None, help="Batch directory of a job to resume.")
    args = parser.parse_args()

    root = reliability_dir("back_translation", args.output_dir)
    tasks, metadata = ([], {}) if args.resume else build_requests(args, root / "translated")
    results, metadata = execute(tasks, metadata, "back_translation", args.judge, "/v1/responses",
                                assume_yes=args.yes, resume=args.resume, output_dir=args.output_dir)
    if results is not None:
        write_outputs(results, metadata, root / "translated", root / "judged")


if __name__ == "__main__":
    main()
