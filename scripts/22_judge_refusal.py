"""Refusal classification of each response (Sec. 4.1, Fig. 3 left).

The same 100 prompts used for pairwise judging are classified as direct answer (1),
direct refusal (2) or indirect refusal (3); classes 2 and 3 count as refusals.

Output:
    outputs/judgments/refusal/<benchmark>/<language>/<model_tag>.json
    outputs/judgments/refusal/summary.csv   (refusal rates and aligned - unaligned difference)

Usage:
    python scripts/22_judge_refusal.py
    python scripts/22_judge_refusal.py --summarize_only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.batch import add_batch_args, execute
from safety_cost.config import (BENCHMARKS, DEFAULT_PAIRS, LANGUAGES, MODEL_PAIRS, N_JUDGE_SAMPLES,
                                REFUSAL_JUDGE, REFUSAL_REASONING_EFFORT, generation_path, judgment_dir,
                                refusal_path)
from safety_cost.judging import is_refusal, parse_refusal_label
from safety_cost.openai_utils import extract_output_text, responses_task
from safety_cost.prompts import REFUSAL_PROMPT
from safety_cost.utils import fix_degenerate, load_json, sample_indices, save_json


def build_requests(args):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for which in ("aligned", "unaligned"):
            tag = pair.tag(which)
            for benchmark in args.benchmarks:
                for language in args.languages:
                    out_path = refusal_path(benchmark, language, tag, args.output_dir)
                    gen_path = generation_path(benchmark, language, tag, args.output_dir)
                    if out_path.exists() and not args.overwrite:
                        continue
                    if not gen_path.exists():
                        print(f"missing generations: {gen_path}; skipped")
                        continue
                    items = [r for r in load_json(gen_path) if r.get("label") != "unsafe"]
                    for idx, orig_idx in enumerate(sample_indices(len(items), args.n_samples)):
                        item = items[orig_idx]
                        response = fix_degenerate(item.get("generated_response", ""))
                        if "</think>" in response:
                            response = response.split("</think>", 1)[1].strip()
                        if not item.get("prompt") or not response:
                            continue
                        custom_id = f"refusal:{tag}:{benchmark}:{language}:{idx}"
                        content = REFUSAL_PROMPT.format(prompt=item["prompt"], response=response)
                        tasks.append(responses_task(custom_id, args.judge, content, args.reasoning_effort))
                        metadata[custom_id] = {
                            "save_path": str(out_path), "index": idx, "original_index": orig_idx,
                            "prompt": item["prompt"], "prompt_original": item.get("prompt_original", ""),
                            "response": response,
                        }
    return tasks, metadata


def process_results(results, metadata, judge):
    grouped = defaultdict(list)
    for result in results:
        meta = metadata.get(result.get("custom_id"))
        if meta is None:
            continue
        raw = extract_output_text(result)
        label = parse_refusal_label(raw)
        grouped[meta["save_path"]].append({
            **{k: v for k, v in meta.items() if k != "save_path"},
            f"judge_{judge}": raw, f"judge_{judge}_label": label, f"refusal_{judge}": is_refusal(label),
        })
    for save_path, items in grouped.items():
        items.sort(key=lambda x: x["index"])
        save_json(items, save_path)
        print(f"saved {len(items)} refusal labels -> {save_path}")


def refusal_rate(items, judge):
    values = [it[f"refusal_{judge}"] for it in items if it.get(f"refusal_{judge}") in (0, 1)]
    return 100.0 * sum(values) / len(values) if values else float("nan")


def summarize(args):
    rows = []
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for benchmark in args.benchmarks:
            for language in args.languages:
                rates = {}
                for which in ("aligned", "unaligned"):
                    path = refusal_path(benchmark, language, pair.tag(which), args.output_dir)
                    if path.exists():
                        rates[which] = refusal_rate(load_json(path), args.judge)
                if len(rates) == 2:
                    rows.append({"family": pair.family, "benchmark": benchmark, "language": language,
                                 "refusal_aligned": round(rates["aligned"], 2),
                                 "refusal_unaligned": round(rates["unaligned"], 2),
                                 "difference": round(rates["aligned"] - rates["unaligned"], 2)})
    if not rows:
        print("No judgment files found.")
        return
    out = judgment_dir("refusal", args.output_dir) / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRefusal rates (%) -> {out}")
    for r in rows:
        print(f"  {r['family']:<14} {r['benchmark']:<17} {r['language']:<11} aligned={r['refusal_aligned']:6.2f} "
              f"unaligned={r['refusal_unaligned']:6.2f} diff={r['difference']:+6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--n_samples", type=int, default=N_JUDGE_SAMPLES, help="-1 = all prompts")
    parser.add_argument("--judge", default=REFUSAL_JUDGE)
    parser.add_argument("--reasoning_effort", default=REFUSAL_REASONING_EFFORT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_batch_args(parser)
    args = parser.parse_args()

    if not args.summarize_only:
        tasks, metadata = ([], {}) if args.resume else build_requests(args)
        results, metadata = execute(tasks, metadata, "refusal", args.judge, "/v1/responses",
                                    assume_yes=args.yes, resume=args.resume, output_dir=args.output_dir)
        if results is not None:
            process_results(results, metadata, args.judge)
    summarize(args)


if __name__ == "__main__":
    main()
