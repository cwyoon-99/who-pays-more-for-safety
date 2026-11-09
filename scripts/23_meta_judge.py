"""Meta-judging of pairwise rationales (Sec. 4.2, Appendix C.1).

GPT-4.1 reads each pairwise judge rationale and assigns 1-3 labels from the taxonomy
(INFO_USABILITY, COVERAGE_SCOPE, CLARITY_STRUCTURE, CORRECTNESS_RELIABILITY,
CALIBRATION_NUANCE, SAFETY_ETHICS, TONE_HELPFULNESS, OUTPUT_FAILURE) explaining why one
response was preferred. Both rationales of every judged pair are labelled.

Output: outputs/judgments/meta_judge/<benchmark>/<language>/<aligned>__<unaligned>.json
(the pairwise items plus ``meta_judge_{1,2}`` and ``meta_judge_labels_{1,2}``).

Usage:
    python scripts/23_meta_judge.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.batch import add_batch_args, execute
from safety_cost.config import (BENCHMARKS, DEFAULT_PAIRS, LANGUAGES, META_JUDGE, MODEL_PAIRS,
                                meta_judge_path, pairwise_path)
from safety_cost.judging import parse_meta_labels
from safety_cost.openai_utils import extract_output_text, responses_task
from safety_cost.prompts import META_JUDGE_PROMPT
from safety_cost.utils import load_json, save_json


def build_requests(args):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for benchmark in args.benchmarks:
            for language in args.languages:
                src = pairwise_path(benchmark, language, pair, args.output_dir)
                out = meta_judge_path(benchmark, language, pair, args.output_dir)
                if not src.exists() or (out.exists() and not args.overwrite):
                    continue
                for entry in load_json(src):
                    for n in (1, 2):
                        rationale = entry.get(f"raw_output_{n}")
                        if not rationale:
                            continue
                        custom_id = f"meta:{pair_key}:{benchmark}:{language}:{entry['index']}:{n}"
                        tasks.append(responses_task(custom_id, args.judge,
                                                    META_JUDGE_PROMPT.format(rationale=rationale)))
                        metadata[custom_id] = {"source": str(src), "save_path": str(out),
                                               "index": entry["index"], "n": n}
    return tasks, metadata


def process_results(results, metadata):
    labels = defaultdict(dict)  # (source, save_path) -> (index, n) -> (raw, labels)
    for result in results:
        meta = metadata.get(result.get("custom_id"))
        if meta is None:
            continue
        raw = extract_output_text(result) or "Error"
        labels[(meta["source"], meta["save_path"])][(meta["index"], meta["n"])] = (raw, parse_meta_labels(raw))
    for (source, save_path), by_key in labels.items():
        entries = load_json(source)
        for entry in entries:
            for n in (1, 2):
                if (entry["index"], n) in by_key:
                    raw, parsed = by_key[(entry["index"], n)]
                    entry[f"meta_judge_{n}"] = raw
                    entry[f"meta_judge_labels_{n}"] = parsed
        save_json(entries, save_path)
        print(f"saved {len(entries)} meta-judged entries -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--judge", default=META_JUDGE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_batch_args(parser)
    args = parser.parse_args()
    if args.summarize_only:
        print("Use analysis/plot_label_distribution.py to aggregate meta-judge labels.")
        return

    tasks, metadata = ([], {}) if args.resume else build_requests(args)
    results, metadata = execute(tasks, metadata, "meta_judge", args.judge, "/v1/responses",
                                assume_yes=args.yes, resume=args.resume, output_dir=args.output_dir)
    if results is not None:
        process_results(results, metadata)


if __name__ == "__main__":
    main()
