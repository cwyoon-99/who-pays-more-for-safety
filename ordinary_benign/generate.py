"""Generate responses to the ordinary benign instructions with vLLM (one model per run).

Settings:
  - system prompt "You are a helpful assistant." (if the chat template rejects a system
    turn, it is prepended to the user turn instead);
  - each model samples with the defaults of its own generation_config.json
    (``--temperature/--top_p/--top_k`` override them), seed 42, bfloat16;
  - max_tokens 2048 (8192 for models run in thinking mode, i.e. Qwen3).

Input:  data/ordinary_benign/<language>.json
Output: outputs/generations/ordinary_benign/<language>/<model_name>.json
        ({id, lang, prompt, prompt_original, generated_response})

Usage:
    python ordinary_benign/generate.py --pair gemma-3-27b --which aligned --tp 2
    python ordinary_benign/generate.py --pair gemma-3-27b --which unaligned --tp 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import MODEL_PAIRS, SEED, generation_path, prompts_path
from safety_cost.utils import load_json, save_json

BENCHMARK = "ordinary_benign"
LANGUAGES = ("english", "chinese", "vietnamese", "arabic", "korean", "thai")
SYSTEM_PROMPT = "You are a helpful assistant."


def model_name(pair_key: str, which: str) -> str:
    """File name of a model's generations (same tag as the main pipeline)."""
    return MODEL_PAIRS[pair_key].tag(which)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", required=True, choices=list(MODEL_PAIRS))
    parser.add_argument("--which", required=True, choices=["aligned", "unaligned"])
    parser.add_argument("--model_path", default=None, help="local path instead of the Hugging Face id")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    pair = MODEL_PAIRS[args.pair]
    model_id = args.model_path or (pair.aligned if args.which == "aligned" else pair.unaligned)
    name = model_name(args.pair, args.which)
    template_kwargs = {"enable_thinking": pair.enable_thinking} if pair.enable_thinking is not None else {}
    max_tokens = args.max_tokens or (8192 if pair.enable_thinking else 2048)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def build(prompt: str) -> str:
        for messages in ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                         [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}]):
            for kwargs in (template_kwargs, {}):
                try:
                    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                         **kwargs)
                except Exception:
                    continue
        return prompt

    llm = LLM(model=model_id, tokenizer=model_id, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_tokens + 2048)
    try:
        sampling_params = llm.get_default_sampling_params()
    except Exception:
        sampling_params = SamplingParams()
    for key in ("temperature", "top_p", "top_k"):
        if getattr(args, key) is not None:
            setattr(sampling_params, key, getattr(args, key))
    sampling_params.max_tokens = max_tokens
    sampling_params.seed = SEED
    print(f"{model_id}: temperature={sampling_params.temperature} top_p={sampling_params.top_p} "
          f"top_k={sampling_params.top_k} max_tokens={sampling_params.max_tokens}", flush=True)

    for language in args.languages:
        data = load_json(prompts_path(BENCHMARK, language, args.data_dir))
        out_path = generation_path(BENCHMARK, language, name, args.output_dir)
        if out_path.exists() and len(load_json(out_path)) == len(data):
            print(f"{name} {language}: already done")
            continue
        outputs = llm.generate([build(r["prompt"]) for r in data], sampling_params)
        rows = [{"id": r["id"], "lang": language, "prompt": r["prompt"], "prompt_original": r["prompt_original"],
                 "generated_response": o.outputs[0].text} for r, o in zip(data, outputs)]
        save_json(rows, out_path)
        print(f"{name} {language}: {len(rows)} responses -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
