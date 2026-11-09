"""Judge robustness (Appendix B, Table "cross_judge"): re-judge a subset with GPT-5.2.

The inputs are exactly those of the main pairwise evaluation (target-language prompt and
target-language responses), restricted to the first ``--limit`` (50) instances of every
(pair, benchmark, language) pairwise file. Each instance is judged twice with swapped
positions, using the same prompt, parsing, and Arena-Hard scoring as the main judge.

Input:  outputs/judgments/pairwise/<benchmark>/<language>/<aligned>__<unaligned>.json
Output: outputs/reliability/cross_judge/<judge>/<benchmark>/<language>/<aligned>__<unaligned>.json
        (a copy of the input with judgment_1_<judge>, judgment_2_<judge>, raw_output_*_<judge>,
        scores_<judge>, avg_score_<judge> added to the re-judged instances; <judge> = gpt52)

Usage:
    python reliability/cross_judge/run_cross_judge.py
    python reliability/cross_judge/run_cross_judge.py --resume outputs/batches/cross_judge_gpt52/<timestamp>
    python reliability/cross_judge/compare_judges.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.common import (RELIABILITY_PAIRS, cell_path, collect_swaps, judge_suffix,
                                reliability_dir, score_swaps, swap_tasks)
from safety_cost.batch import execute
from safety_cost.config import BENCHMARKS, MODEL_PAIRS, PAIRWISE_REASONING_EFFORT, pairwise_path
from safety_cost.utils import load_json, save_json

CROSS_JUDGE = "gpt-5.2"
CROSS_JUDGE_LANGUAGES = ("english", "chinese", "korean", "thai")
N_SUBSET = 50


def build_requests(args):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for language in args.languages:
            for benchmark in args.benchmarks:
                src = pairwise_path(benchmark, language, pair, args.output_dir)
                if not src.exists():
                    print(f"WARN: missing {src}", file=sys.stderr)
                    continue
                for entry in load_json(src)[:args.limit]:  # deterministic subset of the 100
                    if not entry["aligned_response"] or not entry["unaligned_response"]:
                        continue
                    prefix = f"{pair_key}:{benchmark}:{language}:{entry['index']}"
                    for swap, custom_id, task in swap_tasks(prefix, entry["prompt"], entry["aligned_response"],
                                                            entry["unaligned_response"], args.judge,
                                                            args.reasoning_effort):
                        tasks.append(task)
                        metadata[custom_id] = {"benchmark": benchmark, "language": language, "pair": pair_key,
                                               "index": entry["index"], "swap": swap}
    return tasks, metadata


def write_outputs(results, metadata, args):
    suffix = judge_suffix(args.judge)
    out_root = reliability_dir("cross_judge", args.output_dir) / suffix
    summary = {"cells_written": 0, "instances_judged": 0, "incomplete": 0}
    for (benchmark, language, pair_key), by_index in collect_swaps(results, metadata).items():
        pair = MODEL_PAIRS[pair_key]
        rows = []
        for entry in load_json(pairwise_path(benchmark, language, pair, args.output_dir)):
            row = dict(entry)
            swaps = by_index.get(entry["index"], {})
            if "swap1" in swaps and "swap2" in swaps:
                for field, value in score_swaps(swaps["swap1"], swaps["swap2"]).items():
                    row[f"{field}_{suffix}"] = value
                summary["instances_judged"] += 1
            elif entry["index"] in by_index:
                summary["incomplete"] += 1
            rows.append(row)
        save_json(rows, cell_path(out_root, benchmark, language, pair))
        summary["cells_written"] += 1
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", default=CROSS_JUDGE)
    parser.add_argument("--reasoning_effort", default=PAIRWISE_REASONING_EFFORT)
    parser.add_argument("--limit", type=int, default=N_SUBSET, help="first N instances per file")
    parser.add_argument("--pairs", nargs="+", default=list(RELIABILITY_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--languages", nargs="+", default=list(CROSS_JUDGE_LANGUAGES))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--yes", action="store_true", help="Submit without the cost confirmation prompt.")
    parser.add_argument("--resume", default=None, help="Batch directory of a job to resume.")
    args = parser.parse_args()

    tasks, metadata = ([], {}) if args.resume else build_requests(args)
    results, metadata = execute(tasks, metadata, f"cross_judge_{judge_suffix(args.judge)}", args.judge,
                                "/v1/responses", assume_yes=args.yes, resume=args.resume,
                                output_dir=args.output_dir)
    if results is not None:
        write_outputs(results, metadata, args)


if __name__ == "__main__":
    main()
