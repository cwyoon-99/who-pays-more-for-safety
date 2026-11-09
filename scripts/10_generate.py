"""Generate responses with vLLM for one model of an aligned/unaligned pair.

All (benchmark, language) prompt files are generated in a single vLLM call so that each
model is loaded once. Both models of a pair use the sampling defaults from the *aligned*
model's generation_config.json (the officially recommended settings), optionally
overridden per pair in ``safety_cost/config.py``.

Output: outputs/generations/<benchmark>/<language>/<model_tag>.json

Usage:
    # over-refusal benchmarks (Safety Cost)
    python scripts/10_generate.py --pair gemma-3-27b --which aligned
    python scripts/10_generate.py --pair gemma-3-27b --which unaligned
    # MultiJail (Safety Gain)
    python scripts/10_generate.py --pair gemma-3-27b --which aligned --benchmarks multijail
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from safety_cost.config import (BENCHMARKS, LANGUAGES, MODEL_PAIRS, MULTIJAIL_LANGUAGES, SEED,
                                SYSTEM_PROMPT, generation_path, prompts_path)
from safety_cost.utils import get_conversation_template, load_json, save_json

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p")


def aligned_sampling_defaults(aligned_id: str) -> dict:
    from transformers import GenerationConfig

    config = GenerationConfig.from_pretrained(aligned_id, trust_remote_code=True).to_dict()
    return {k: config[k] for k in SAMPLING_KEYS if config.get(k) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", required=True, choices=list(MODEL_PAIRS))
    parser.add_argument("--which", required=True, choices=["aligned", "unaligned"])
    parser.add_argument("--model_path", default=None,
                        help="Local path to load instead of the Hugging Face id in the config.")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                        choices=list(BENCHMARKS) + ["multijail"])
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Default: all languages available for each benchmark.")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Default: number of visible GPUs.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    pair = MODEL_PAIRS[args.pair]
    model_id = args.model_path or (pair.aligned if args.which == "aligned" else pair.unaligned)
    model_tag = pair.tag(args.which)
    max_tokens = args.max_tokens or pair.max_tokens

    # Collect every prompt file to generate for this model.
    jobs = []
    for benchmark in args.benchmarks:
        languages = args.languages or (list(MULTIJAIL_LANGUAGES) if benchmark == "multijail" else list(LANGUAGES))
        for language in languages:
            out_path = generation_path(benchmark, language, model_tag, args.output_dir)
            if out_path.exists() and not args.overwrite:
                print(f"skip (exists): {out_path}")
                continue
            items = load_json(prompts_path(benchmark, language, args.data_dir))
            jobs.append((out_path, items))
    if not jobs:
        return

    from transformers import AutoTokenizer
    from vllm import LLM

    tp = args.tensor_parallel_size or max(1, len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    llm = LLM(model=model_id, tensor_parallel_size=tp, gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True, generation_config="auto", max_model_len=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    sampling_params = llm.get_default_sampling_params()
    overrides = {**aligned_sampling_defaults(pair.aligned), **pair.sampling}
    for key, value in overrides.items():
        if hasattr(sampling_params, key):
            setattr(sampling_params, key, value)
    sampling_params.max_tokens = max_tokens
    print(f"model={model_id} tag={model_tag} thinking={pair.enable_thinking} sampling={overrides} "
          f"max_tokens={max_tokens}")

    prompts = []
    for _, items in jobs:
        for item in items:
            messages = [{"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": item["prompt"]}]
            prompts.append(get_conversation_template(tokenizer, messages, enable_thinking=pair.enable_thinking))

    outputs = llm.generate(prompts, sampling_params)

    cursor = 0
    for out_path, items in jobs:
        results = []
        for item in items:
            results.append({"system_prompt": args.system_prompt, **item,
                            "generated_response": outputs[cursor].outputs[0].text})
            cursor += 1
        save_json(results, out_path)
        print(f"saved {len(results)} responses -> {out_path}")


if __name__ == "__main__":
    main()
