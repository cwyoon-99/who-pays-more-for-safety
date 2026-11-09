"""P-MMEval: Llama-3.3-70B-Instruct vs. huihui-ai/Llama-3.3-70B-Instruct-abliterated.

Run from the repository root (see capability/README.md):
    opencompass capability/configs/eval_llama3_3_70b.py -w outputs/capability/opencompass/llama3_3_70b
"""

from mmengine.config import read_base

from opencompass.models import VLLMwithChatTemplate

with read_base():
    from .pmmeval_datasets import PMMEval_paper_datasets, summarizer

datasets = PMMEval_paper_datasets

# Identical sampling for both models of the pair (the aligned model's recommended defaults).
pair_generation_kwargs = dict(temperature=0.6, top_p=0.9)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='llama3_3-70b-instruct-vllm',
        path='meta-llama/Llama-3.3-70B-Instruct',
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
        abbr='llama3_3-70b-instruct-abliterated-vllm',
        path='huihui-ai/Llama-3.3-70B-Instruct-abliterated',
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
