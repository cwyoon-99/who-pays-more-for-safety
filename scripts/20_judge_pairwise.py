"""Pairwise helpfulness judging of aligned vs. unaligned responses -> Safety Cost (Sec. 2.3, 3).

For every (model pair, benchmark, language), 100 prompts are sampled with seed 42 (the
same prompts for every language), and the judge (GPT-5-mini) compares the two responses
twice with the A/B positions swapped. Following Arena-Hard, strong preferences count
three times. Safety Cost = win rate of the unaligned model; MSC_L = Cost_L - Cost_English.

Output:
    outputs/judgments/pairwise/<benchmark>/<language>/<aligned>__<unaligned>.json
    outputs/judgments/pairwise/summary.csv

Usage:
    python scripts/20_judge_pairwise.py                     # judge everything, then summarize
    python scripts/20_judge_pairwise.py --summarize_only
    python scripts/20_judge_pairwise.py --resume outputs/batches/pairwise/<timestamp>
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
                                PAIRWISE_JUDGE, PAIRWISE_REASONING_EFFORT, generation_path, judgment_dir,
                                pairwise_path)
from safety_cost.judging import label_to_scores, parse_judgment, safety_cost
from safety_cost.openai_utils import extract_output_text, responses_task
from safety_cost.prompts import PAIRWISE_PROMPT
from safety_cost.utils import clean_response, load_json, sample_indices, save_json


def build_requests(args):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for benchmark in args.benchmarks:
            for language in args.languages:
                out_path = pairwise_path(benchmark, language, pair, args.output_dir)
                if out_path.exists() and not args.overwrite:
                    continue
                files = [generation_path(benchmark, language, pair.tag(w), args.output_dir)
                         for w in ("aligned", "unaligned")]
                if not all(f.exists() for f in files):
                    print(f"missing generations for {pair_key} {benchmark} {language}; skipped")
                    continue
                aligned, unaligned = ([r for r in load_json(f) if r.get("label") != "unsafe"] for f in files)
                indices = sample_indices(min(len(aligned), len(unaligned)), args.n_samples)

                for idx, orig_idx in enumerate(indices):
                    a_item, u_item = aligned[orig_idx], unaligned[orig_idx]
                    if a_item["prompt"] != u_item["prompt"]:
                        raise ValueError(f"Prompt mismatch at index {orig_idx} ({benchmark}/{language})")
                    a_resp = clean_response(a_item.get("generated_response", ""))
                    u_resp = clean_response(u_item.get("generated_response", ""))
                    if not a_resp or not u_resp:
                        print(f"empty response at {benchmark}/{language}/{orig_idx}; skipped")
                        continue
                    # Judgment 1: unaligned = A; judgment 2: aligned = A.
                    for judgment_num, (resp_a, resp_b, a_is_unaligned) in enumerate(
                            [(u_resp, a_resp, True), (a_resp, u_resp, False)]):
                        custom_id = f"{pair_key}:{benchmark}:{language}:{idx}:{judgment_num}"
                        content = PAIRWISE_PROMPT.format(prompt=a_item["prompt"], response_a=resp_a,
                                                         response_b=resp_b)
                        tasks.append(responses_task(custom_id, args.judge, content, args.reasoning_effort))
                        metadata[custom_id] = {
                            "save_path": str(out_path), "index": idx, "original_index": orig_idx,
                            "judgment_num": judgment_num, "a_is_unaligned": a_is_unaligned,
                            "prompt": a_item["prompt"], "prompt_original": a_item.get("prompt_original", ""),
                            "aligned_response": a_resp, "unaligned_response": u_resp,
                        }
    return tasks, metadata


def process_results(results, metadata):
    grouped = defaultdict(dict)  # save_path -> index -> item
    for result in results:
        meta = metadata.get(result.get("custom_id"))
        if meta is None:
            continue
        raw = extract_output_text(result)
        label = parse_judgment(raw) if raw is not None else "Error"
        item = grouped[meta["save_path"]].setdefault(meta["index"], {
            "index": meta["index"], "original_index": meta["original_index"],
            "prompt": meta["prompt"], "prompt_original": meta["prompt_original"],
            "aligned_response": meta["aligned_response"], "unaligned_response": meta["unaligned_response"],
            "judgment_1": "Error", "judgment_2": "Error", "raw_output_1": "", "raw_output_2": "",
            "_scores": {},
        })
        n = meta["judgment_num"] + 1
        item[f"judgment_{n}"] = label
        item[f"raw_output_{n}"] = raw or "Error"
        item["_scores"][n] = label_to_scores(label, meta["a_is_unaligned"])

    for save_path, items in grouped.items():
        rows = []
        for idx in sorted(items):
            item = items[idx]
            per_judgment = item.pop("_scores")
            scores = [s for n in sorted(per_judgment) for s in per_judgment[n]]
            item["scores"] = scores
            item["avg_score"] = sum(scores) / len(scores) if scores else None  # aligned's score
            rows.append(item)
        save_json(rows, save_path)
        print(f"saved {len(rows)} judgments -> {save_path}")


def summarize(args):
    rows = []
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for benchmark in args.benchmarks:
            costs = {}
            for language in args.languages:
                path = pairwise_path(benchmark, language, pair, args.output_dir)
                if path.exists():
                    costs[language] = safety_cost(load_json(path))
            for language, cost in costs.items():
                msc = cost - costs["english"] if "english" in costs else float("nan")
                rows.append({"family": pair.family, "benchmark": benchmark, "language": language,
                             "safety_cost": round(cost, 2), "msc": round(msc, 2)})
    if not rows:
        print("No judgment files found.")
        return
    out = judgment_dir("pairwise", args.output_dir) / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSafety Cost (%) = win rate of the unaligned model -> {out}")
    for r in rows:
        print(f"  {r['family']:<14} {r['benchmark']:<17} {r['language']:<11} "
              f"cost={r['safety_cost']:6.2f}  MSC={r['msc']:+6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--n_samples", type=int, default=N_JUDGE_SAMPLES, help="-1 = all prompts")
    parser.add_argument("--judge", default=PAIRWISE_JUDGE)
    parser.add_argument("--reasoning_effort", default=PAIRWISE_REASONING_EFFORT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_batch_args(parser)
    args = parser.parse_args()

    if not args.summarize_only:
        tasks, metadata = ([], {}) if args.resume else build_requests(args)
        results, metadata = execute(tasks, metadata, "pairwise", args.judge, "/v1/responses",
                                    assume_yes=args.yes, resume=args.resume, output_dir=args.output_dir)
        if results is not None:
            process_results(results, metadata)
    summarize(args)


if __name__ == "__main__":
    main()
