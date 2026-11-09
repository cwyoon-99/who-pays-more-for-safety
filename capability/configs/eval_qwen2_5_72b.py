"""P-MMEval: Qwen2.5-72B-Instruct vs. huihui-ai/Qwen2.5-72B-Instruct-abliterated.

Run from the repository root (see capability/README.md):
    opencompass capability/configs/eval_qwen2_5_72b.py -w outputs/capability/opencompass/qwen2_5_72b
"""

from mmengine.config import read_base

from opencompass.models import VLLMwithChatTemplate

with read_base():
    from .pmmeval_datasets import PMMEval_paper_datasets, summarizer

datasets = PMMEval_paper_datasets

# Identical sampling for both models of the pair (the aligned model's recommended defaults).
pair_generation_kwargs = dict(repetition_penalty=1.05, temperature=0.7, top_p=0.8, top_k=20)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2_5-72b-instruct-vllm',
        path='Qwen/Qwen2.5-72B-Instruct',
        max_out_len=4096,
        batch_size=1000,
        generation_kwargs=pair_generation_kwargs,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=4096,
        ),
    ),
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2_5-72b-instruct-abliterated-vllm',
        path='huihui-ai/Qwen2.5-72B-Instruct-abliterated',
        max_out_len=4096,
        batch_size=1000,
        generation_kwargs=pair_generation_kwargs,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=4096,
        ),
    ),
]
