"""Small helpers shared by generation and judging scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import SEED


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def sample_indices(total: int, n: int, seed: int = SEED) -> list[int]:
    """Indices of the judged subset.

    The seed is reset before every call so that the *same* prompts are selected for
    every language, model, and judge (aligned/unaligned files share the prompt order).
    ``n = -1`` keeps every item.
    """
    if n == -1 or n >= total:
        return list(range(total))
    random.seed(seed)
    return sorted(random.sample(range(total), n))


def strip_reasoning(response: str) -> str:
    """Drop a ``<think>...</think>`` trace so that only the final answer is judged."""
    if response and "</think>" in response:
        return response.split("</think>", 1)[1]
    return response


def fix_degenerate(response: str) -> str:
    """Cut responses in which the model starts repeating the ``prompt:`` fallback template."""
    if response and "prompt: " in response:
        return response.split("prompt: ")[0]
    return response


def clean_response(response: str) -> str:
    return strip_reasoning(fix_degenerate(response or ""))


def get_conversation_template(tokenizer, messages: list[dict], enable_thinking: bool | None = None):
    """Apply the model's chat template (``enable_thinking`` is forwarded to Qwen3 templates)."""
    if getattr(tokenizer, "chat_template", None) is None:
        return f"prompt: {messages[-1]['content']}\nresponse:"
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)
