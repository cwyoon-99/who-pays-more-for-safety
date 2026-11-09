"""Unsafe-response judging on MultiJail -> Safety Gain (Sec. 3.3, Table unsafe_rate).

Every response of the aligned and unaligned models is labelled safe / unsafe / invalid
by GPT-4.1 (temperature 0). Safety Gain_L = Unsafe(M_unalg, L) - Unsafe(M_align, L).

Output:
    outputs/judgments/unsafe/multijail/<language>/<model_tag>.json
    outputs/judgments/unsafe/summary.csv     (unsafe rates and Safety Gain per language)

Usage:
    python scripts/21_judge_unsafe.py
    python scripts/21_judge_unsafe.py --summarize_only
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.batch import add_batch_args, execute
from safety_cost.config import (DEFAULT_PAIRS, MODEL_PAIRS, MULTIJAIL_LANGUAGES, UNSAFE_JUDGE,
                                generation_path, judgment_dir, unsafe_path)
from safety_cost.judging import parse_unsafe_assessment, unsafe_rate
from safety_cost.openai_utils import chat_task, extract_output_text
from safety_cost.prompts import format_unsafe_prompt
from safety_cost.utils import fix_degenerate, load_json, save_json


def preprocess(response: str) -> str:
    """Remove the reasoning trace, then cut degenerate prompt repetitions."""
    response = response or ""
    if "</think>" in response:
        response = response.split("</think>")[1]
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return fix_degenerate(response)


def build_requests(args):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for which in ("aligned", "unaligned"):
            tag = pair.tag(which)
            for language in args.languages:
                out_path = unsafe_path(language, tag, args.output_dir)
                gen_path = generation_path("multijail", language, tag, args.output_dir)
                if out_path.exists() and not args.overwrite:
                    continue
                if not gen_path.exists():
                    print(f"missing generations: {gen_path}; skipped")
                    continue
                for i, item in enumerate(load_json(gen_path)):
                    response = preprocess(item.get("generated_response", ""))
                    if not response:  # empty responses are excluded from the evaluation
                        continue
                    custom_id = f"unsafe:{tag}:{language}:{i}"
                    tasks.append(chat_task(custom_id, args.judge, format_unsafe_prompt(item["prompt"], response),
                                           temperature=0, max_tokens=10))
                    metadata[custom_id] = {
                        "save_path": str(out_path), "position": i, "id": item.get("id"),
                        "language": language, "model": tag, "prompt": item["prompt"],
                        "generated_response": response, "source": item.get("source"), "tags": item.get("tags"),
                    }
    return tasks, metadata


def process_results(results, metadata):
    grouped = defaultdict(list)
    for result in results:
        meta = metadata.get(result.get("custom_id"))
        if meta is None:
            continue
        raw = extract_output_text(result)
        assessment = parse_unsafe_assessment(raw) if raw is not None else "invalid"
        grouped[meta["save_path"]].append({**{k: v for k, v in meta.items() if k != "save_path"},
                                           "assessment": assessment})
    for save_path, items in grouped.items():
        items.sort(key=lambda x: x.pop("position"))
        save_json(items, save_path)
        print(f"saved {len(items)} assessments -> {save_path}")


def summarize(args):
    rows = []
    for pair_key in args.pairs:
        pair = MODEL_PAIRS[pair_key]
        for language in args.languages:
            rates = {}
            for which in ("aligned", "unaligned"):
                path = unsafe_path(language, pair.tag(which), args.output_dir)
                if path.exists():
                    rates[which] = unsafe_rate(load_json(path))
            if len(rates) == 2:
                rows.append({"family": pair.family, "language": language,
                             "unsafe_aligned": round(rates["aligned"], 2),
                             "unsafe_unaligned": round(rates["unaligned"], 2),
                             "safety_gain": round(rates["unaligned"] - rates["aligned"], 2)})
    if not rows:
        print("No judgment files found.")
        return
    out = judgment_dir("unsafe", args.output_dir) / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nUnsafe rate (%) and Safety Gain -> {out}")
    for r in rows:
        print(f"  {r['family']:<14} {r['language']:<11} aligned={r['unsafe_aligned']:6.2f} "
              f"unaligned={r['unsafe_unaligned']:6.2f} gain={r['safety_gain']:6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--languages", nargs="+", default=list(MULTIJAIL_LANGUAGES),
                        choices=list(MULTIJAIL_LANGUAGES))
    parser.add_argument("--judge", default=UNSAFE_JUDGE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_batch_args(parser)
    args = parser.parse_args()

    if not args.summarize_only:
        tasks, metadata = ([], {}) if args.resume else build_requests(args)
        results, metadata = execute(tasks, metadata, "unsafe", args.judge, "/v1/chat/completions",
                                    assume_yes=args.yes, resume=args.resume, output_dir=args.output_dir)
        if results is not None:
            process_results(results, metadata)
    summarize(args)


if __name__ == "__main__":
    main()
