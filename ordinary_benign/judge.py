"""Judge the ordinary benign responses (aligned vs. unaligned) and compute the ordinary Cost.

Same judge model (GPT-5-mini), pairwise prompt, verdict parsing, and Arena-Hard scoring as
the main experiment, with the settings used for this control:
  - one judgment per instruction; whether the unaligned response is shown as Assistant A
    is drawn with ``random.Random(42)`` (re-seeded for every family and language);
  - Chat Completions endpoint with ``reasoning_effort="low"``;
  - Cost_L = 100 - aligned win rate, where the win rate is the mean over all per-verdict
    scores (a strong preference contributes three scores). Failed or unparsable judgments
    contribute no score.

Input:  outputs/generations/ordinary_benign/<language>/<model_name>.json
Output: outputs/judgments/ordinary_benign/<language>/<aligned>__<unaligned>.json
        outputs/judgments/ordinary_benign/ordinary_benign_cost.json   ({pair: {language: Cost}})

Usage:
    python ordinary_benign/judge.py
    python ordinary_benign/judge.py --resume outputs/batches/ordinary_benign/<timestamp>
    python ordinary_benign/judge.py --summarize_only
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ordinary_benign.generate import BENCHMARK, LANGUAGES, model_name
from safety_cost.batch import add_batch_args, execute
from safety_cost.config import MODEL_PAIRS, PAIRWISE_JUDGE, SEED, generation_path, judgment_dir
from safety_cost.judging import label_to_scores, parse_judgment
from safety_cost.openai_utils import chat_task, extract_output_text
from safety_cost.prompts import PAIRWISE_PROMPT
from safety_cost.utils import load_json, save_json, strip_reasoning

REASONING_EFFORT = "low"


def judgment_path(language: str, pair_key: str, output_dir: Path | None) -> Path:
    return (judgment_dir(BENCHMARK, output_dir) / language
            / f"{model_name(pair_key, 'aligned')}__{model_name(pair_key, 'unaligned')}.json")


def build_requests(args):
    tasks, metadata = [], {}
    for pair_key in args.pairs:
        for language in args.languages:
            files = [generation_path(BENCHMARK, language, model_name(pair_key, w), args.output_dir)
                     for w in ("aligned", "unaligned")]
            if not all(f.exists() for f in files):
                print(f"{pair_key}/{language}: responses missing, skipped")
                continue
            if judgment_path(language, pair_key, args.output_dir).exists() and not args.overwrite:
                continue
            aligned, unaligned = ({r["id"]: r for r in load_json(f)} for f in files)
            rnd = random.Random(SEED)
            for iid in sorted(set(aligned) & set(unaligned)):
                a_is_unaligned = rnd.random() < 0.5
                a_resp, u_resp = (strip_reasoning(aligned[iid]["generated_response"]),
                                  strip_reasoning(unaligned[iid]["generated_response"]))
                resp_a, resp_b = (u_resp, a_resp) if a_is_unaligned else (a_resp, u_resp)
                content = PAIRWISE_PROMPT.format(prompt=aligned[iid]["prompt"], response_a=resp_a, response_b=resp_b)
                custom_id = f"{pair_key}:{language}:{iid}"
                tasks.append(chat_task(custom_id, args.judge, content, reasoning_effort=REASONING_EFFORT))
                metadata[custom_id] = {"pair": pair_key, "language": language, "id": iid,
                                       "a_is_unaligned": a_is_unaligned, "prompt": aligned[iid]["prompt"],
                                       "aligned_response": a_resp, "unaligned_response": u_resp}
    return tasks, metadata


def process_results(results, metadata, output_dir):
    grouped = defaultdict(list)
    for result in results:
        meta = metadata.get(result.get("custom_id"))
        if meta is None:
            continue
        raw = extract_output_text(result)
        label = parse_judgment(raw) if raw is not None else None
        grouped[(meta["language"], meta["pair"])].append({
            **{k: meta[k] for k in ("id", "prompt", "aligned_response", "unaligned_response", "a_is_unaligned")},
            "judgment": label, "raw_output": raw,
            "scores": label_to_scores(label, meta["a_is_unaligned"]) if label else [],
        })
    for (language, pair_key), rows in grouped.items():
        rows.sort(key=lambda r: r["id"])
        save_json(rows, judgment_path(language, pair_key, output_dir))
        print(f"saved {len(rows)} judgments for {pair_key}/{language}")


def summarize(args):
    costs = {}
    for pair_key in args.pairs:
        for language in args.languages:
            path = judgment_path(language, pair_key, args.output_dir)
            if not path.exists():
                continue
            scores = [s for row in load_json(path) for s in row["scores"]]
            if scores:
                costs.setdefault(pair_key, {})[language] = round(100 - 100 * sum(scores) / len(scores), 2)
    out = judgment_dir(BENCHMARK, args.output_dir) / "ordinary_benign_cost.json"
    save_json(costs, out)
    print(f"ordinary Cost (%) -> {out}")
    for pair_key, by_lang in costs.items():
        print(f"  {pair_key:<14} " + "  ".join(f"{lang}={cost:.2f}" for lang, cost in by_lang.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(MODEL_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--judge", default=PAIRWISE_JUDGE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_batch_args(parser)
    args = parser.parse_args()

    if not args.summarize_only:
        tasks, metadata = ([], {}) if args.resume else build_requests(args)
        results, metadata = execute(tasks, metadata, BENCHMARK, args.judge, "/v1/chat/completions",
                                    assume_yes=args.yes, resume=args.resume, output_dir=args.output_dir)
        if results is not None:
            process_results(results, metadata, args.output_dir)
    summarize(args)


if __name__ == "__main__":
    main()
