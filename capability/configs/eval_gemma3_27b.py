"""P-MMEval: gemma-3-27b-it vs. huihui-ai/gemma-3-27b-it-abliterated.

Run from the repository root (see capability/README.md):
    opencompass capability/configs/eval_gemma3_27b.py -w outputs/capability/opencompass/gemma3_27b
"""

from mmengine.config import read_base

from opencompass.models import VLLMwithChatTemplate

with read_base():
    from .pmmeval_datasets import PMMEval_paper_datasets, summarizer

datasets = PMMEval_paper_datasets

# Identical sampling for both models of the pair (the aligned model's recommended defaults).
pair_generation_kwargs = dict(temperature=1.0, top_p=0.95, top_k=64, min_p=0)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='gemma-3-27b-it-vllm',
        path='google/gemma-3-27b-it',
        max_out_len=4096,
        batch_size=1000,
        generation_kwargs=pair_generation_kwargs,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=4096,
            rope_scaling={'factor': 8.0, 'rope_type': 'linear'},
            # Disables CUDA graphs (does not change the sampling parameters).
            compilation_config={'level': 0, 'use_cudagraph': False},
        ),
    ),
    dict(
        type=VLLMwithChatTemplate,
        abbr='gemma-3-27b-it-abliterated-vllm',
        path='huihui-ai/gemma-3-27b-it-abliterated',
        max_out_len=4096,
        batch_size=1000,
        generation_kwargs=pair_generation_kwargs,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=4096,
            rope_scaling={'factor': 8.0, 'rope_type': 'linear'},
        ),
    ),
]
