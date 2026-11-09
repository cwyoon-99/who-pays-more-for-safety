"""P-MMEval: Qwen3-32B vs. huihui-ai/Qwen3-32B-abliterated (thinking mode).

Run from the repository root (see capability/README.md):
    opencompass capability/configs/eval_qwen3_32b.py -w outputs/capability/opencompass/qwen3_32b
"""

from mmengine.config import read_base

from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from .pmmeval_datasets import PMMEval_paper_datasets, summarizer

datasets = PMMEval_paper_datasets

# Identical sampling for both models of the pair (same as safety_cost/config.py).
# The reasoning trace is removed before scoring.
pair_generation_kwargs = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-32b-vllm',
        path='Qwen/Qwen3-32B',
        max_out_len=8192,
        batch_size=1000,
        generation_kwargs=pair_generation_kwargs,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=16384,
        ),
        chat_template_kwargs=dict(enable_thinking=True),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-32b-abliterated-vllm',
        path='huihui-ai/Qwen3-32B-abliterated',
        max_out_len=8192,
        batch_size=1000,
        generation_kwargs=pair_generation_kwargs,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=16384,
        ),
        chat_template_kwargs=dict(enable_thinking=True),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
]
