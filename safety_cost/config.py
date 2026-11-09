"""Experiment configuration shared by every script in the repository.

Paths can be overridden with environment variables:
    SAFETY_COST_DATA_DIR    where benchmark prompts are written/read   (default: <repo>/data)
    SAFETY_COST_OUTPUT_DIR  where generations and judgments are stored (default: <repo>/outputs)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SAFETY_COST_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("SAFETY_COST_OUTPUT_DIR", REPO_ROOT / "outputs"))

SEED = 42
# Number of prompts per (benchmark, language) that are judged (Sec. 3.2).
N_JUDGE_SAMPLES = 100
# Number of OR-Bench prompts that are sampled before translation.
N_ORBENCH_PROMPTS = 200

# Universal system prompt used for every open-weight model (Appendix A.3).
SYSTEM_PROMPT = "You are a helpful assistant. Be concise and accurate."

# Over-refusal benchmarks used to measure Safety Cost.
BENCHMARKS = ("xstest", "or-bench-80k", "or-bench-hard-1k")

# English is the baseline; the remaining entries are the translated variations.
LANGUAGES = ("english", "chinese", "french", "vietnamese", "arabic", "korean", "thai", "ukrainian")
NON_ENGLISH = tuple(lang for lang in LANGUAGES if lang != "english")

# MultiJail (Deng et al., 2024) column names for the languages used for Safety Gain.
MULTIJAIL_LANGUAGES = {
    "english": "en",
    "chinese": "zh",
    "vietnamese": "vi",
    "arabic": "ar",
    "korean": "ko",
    "thai": "th",
}

# OpenAI models used for translation and LLM-as-a-judge.
TRANSLATION_MODEL = "gpt-4o"
PAIRWISE_JUDGE = "gpt-5-mini"
PAIRWISE_REASONING_EFFORT = "medium"
REFUSAL_JUDGE = "gpt-5-mini"
REFUSAL_REASONING_EFFORT = "low"
UNSAFE_JUDGE = "gpt-4.1"
META_JUDGE = "gpt-4.1"


@dataclass(frozen=True)
class ModelPair:
    """A safety-aligned model and its refusal-ablated (unaligned) counterpart."""

    family: str
    aligned: str
    unaligned: str
    # Qwen3 is run in thinking mode; the reasoning trace is stripped before judging.
    enable_thinking: bool | None = None
    # Sampling overrides. When empty, both models of the pair use the sampling
    # defaults in the *aligned* model's generation_config.json.
    sampling: dict = field(default_factory=dict)
    max_tokens: int = 8192

    def tag(self, which: str) -> str:
        """Short model name used in file names, e.g. ``Qwen3-32B-reasoning``."""
        model_id = self.aligned if which == "aligned" else self.unaligned
        name = model_id.rstrip("/").split("/")[-1]
        return f"{name}-reasoning" if self.enable_thinking else name

    @property
    def aligned_tag(self) -> str:
        return self.tag("aligned")

    @property
    def unaligned_tag(self) -> str:
        return self.tag("unaligned")


# Table 6 (model_info) of the paper.
MODEL_PAIRS = {
    "llama-3.3-70b": ModelPair(
        family="Llama-3.3-70B",
        aligned="meta-llama/Llama-3.3-70B-Instruct",
        unaligned="huihui-ai/Llama-3.3-70B-Instruct-abliterated",
    ),
    "qwen2.5-72b": ModelPair(
        family="Qwen2.5-72B",
        aligned="Qwen/Qwen2.5-72B-Instruct",
        unaligned="huihui-ai/Qwen2.5-72B-Instruct-abliterated",
    ),
    "gemma-3-27b": ModelPair(
        family="Gemma-3-27B",
        aligned="google/gemma-3-27b-it",
        unaligned="huihui-ai/gemma-3-27b-it-abliterated",
    ),
    "qwen3-32b": ModelPair(
        family="Qwen3-32B",
        aligned="Qwen/Qwen3-32B",
        unaligned="huihui-ai/Qwen3-32B-abliterated",
        enable_thinking=True,
        sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    ),
}

# Pairs used for the main results (Table win_rate, Figs. 2 and 3).
DEFAULT_PAIRS = ("llama-3.3-70b", "qwen2.5-72b", "gemma-3-27b", "qwen3-32b")


def get_pairs(keys: list[str] | None = None) -> list[ModelPair]:
    if not keys:
        return [MODEL_PAIRS[k] for k in DEFAULT_PAIRS]
    unknown = [k for k in keys if k not in MODEL_PAIRS]
    if unknown:
        raise ValueError(f"Unknown model pair(s) {unknown}; choose from {list(MODEL_PAIRS)}")
    return [MODEL_PAIRS[k] for k in keys]


# ---------------------------------------------------------------------------
# File layout
# ---------------------------------------------------------------------------
#   data/<benchmark>/<language>.json                                   prompts
#   outputs/generations/<benchmark>/<language>/<model_tag>.json         model responses
#   outputs/judgments/<kind>/<benchmark>/<language>/<file>.json         judge outputs
#   outputs/batches/<kind>/<timestamp>/                                 OpenAI Batch API files


def prompts_path(benchmark: str, language: str, data_dir: Path | None = None) -> Path:
    return Path(data_dir or DATA_DIR) / benchmark / f"{language}.json"


def generation_path(benchmark: str, language: str, model_tag: str, output_dir: Path | None = None) -> Path:
    return Path(output_dir or OUTPUT_DIR) / "generations" / benchmark / language / f"{model_tag}.json"


def judgment_dir(kind: str, output_dir: Path | None = None) -> Path:
    return Path(output_dir or OUTPUT_DIR) / "judgments" / kind


def pairwise_path(benchmark: str, language: str, pair: ModelPair, output_dir: Path | None = None) -> Path:
    return (judgment_dir("pairwise", output_dir) / benchmark / language
            / f"{pair.aligned_tag}__{pair.unaligned_tag}.json")


def refusal_path(benchmark: str, language: str, model_tag: str, output_dir: Path | None = None) -> Path:
    return judgment_dir("refusal", output_dir) / benchmark / language / f"{model_tag}.json"


def unsafe_path(language: str, model_tag: str, output_dir: Path | None = None) -> Path:
    return judgment_dir("unsafe", output_dir) / "multijail" / language / f"{model_tag}.json"


def meta_judge_path(benchmark: str, language: str, pair: ModelPair, output_dir: Path | None = None) -> Path:
    return (judgment_dir("meta_judge", output_dir) / benchmark / language
            / f"{pair.aligned_tag}__{pair.unaligned_tag}.json")
