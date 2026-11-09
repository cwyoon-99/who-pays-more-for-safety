"""Generate responses of a matched checkpoint (c0 or c1) with vLLM.

For every (benchmark, language) the 100 prompts selected by ``sample_indices`` (seed 42,
the same instances as in the main experiment and for every language) are answered through
the Qwen3-8B chat template (the template used during SFT, user turn only). Decoding uses the
Qwen3 family defaults (temperature 0.6, top_p 0.95, top_k 20), max 2,048 new tokens, seed 42,
for both checkpoints.

Prompts are read from data/<benchmark>/<language>.json (``prompt`` field), i.e. the translated
over-refusal sets and MultiJail (scripts/00_translate_benchmarks.py, scripts/01_prepare_multijail.py).
The FalseReject set (200 test prompts sampled with seed 42 from AmazonScience/FalseReject) is built
by `python scripts/00_translate_benchmarks.py --benchmarks falsereject`.

Output: outputs/matched_checkpoint/generations/<benchmark>/<language>/<model_tag>.json
(items: id = index in the prompt file, lang, benchmark, prompt, response)

Usage:
    python matched_checkpoint/infer.py --model outputs/matched_checkpoint/models/bsft-qwen3-8b-c0 --model_tag qwen3-8b-c0
    python matched_checkpoint/infer.py --model outputs/matched_checkpoint/models/bsft-qwen3-8b-c1 --model_tag qwen3-8b-c1
    # data-parallel over 4 GPUs: run one process per GPU with --shard 0/4 ... 3/4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import N_JUDGE_SAMPLES, SEED, SYSTEM_PROMPT, generation_path, prompts_path
from safety_cost.utils import load_json, sample_indices, save_json

from common import COST_BENCHMARKS, EVAL_LANGUAGES, GAIN_BENCHMARK, MC_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Merged Hugging Face model directory.")
    parser.add_argument("--model_tag", required=True, help="Name used in output files, e.g. qwen3-8b-c0.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                        help="Tokenizer / chat template used during SFT.")
    parser.add_argument("--benchmarks", nargs="+", default=[*COST_BENCHMARKS, GAIN_BENCHMARK])
    parser.add_argument("--languages", nargs="+", default=list(EVAL_LANGUAGES))
    parser.add_argument("--n", type=int, default=N_JUDGE_SAMPLES)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--use_system_prompt", action="store_true",
                        help="Prepend the main experiment's system prompt (default: user turn only).")
    parser.add_argument("--shard", default="0/1",
                        help="i/N: only process (benchmark, language) items with index %% N == i.")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=MC_OUTPUT_DIR)
    args = parser.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=max(4096, args.max_tokens + 2048), trust_remote_code=True)
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              max_tokens=args.max_tokens, seed=args.seed)
    print(f"model={args.model} tag={args.model_tag} sampling={sampling}")

    work = [(b, lang) for b in args.benchmarks for lang in args.languages]
    work = [w for i, w in enumerate(work) if i % shard_n == shard_i]
    for benchmark, language in work:
        path = prompts_path(benchmark, language, args.data_dir)
        if not path.exists():
            print(f"  [warn] missing prompts: {path}")
            continue
        items = load_json(path)
        ids, prompts = [], []
        for idx in sample_indices(len(items), args.n):
            prompt = items[idx].get("prompt")
            if prompt:
                ids.append(idx)
                prompts.append(str(prompt))

        messages = [[{"role": "user", "content": p}] for p in prompts]
        if args.use_system_prompt:
            messages = [[{"role": "system", "content": SYSTEM_PROMPT}, *m] for m in messages]
        chats = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        outputs = llm.generate(chats, sampling)

        rows = [{"id": i, "lang": language, "benchmark": benchmark, "prompt": p, "response": o.outputs[0].text}
                for i, p, o in zip(ids, prompts, outputs)]
        out_path = generation_path(benchmark, language, args.model_tag, args.output_dir)
        save_json(rows, out_path)
        truncated = sum(len(o.outputs[0].token_ids) >= args.max_tokens for o in outputs)
        print(f"  {benchmark:18s}/{language:11s}: {len(rows):3d} saved, truncated={truncated} -> {out_path}")


if __name__ == "__main__":
    main()
