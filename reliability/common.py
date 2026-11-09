"""Helpers shared by the reliability experiments (Appendix B).

Kept local to ``reliability/`` so that the shared ``safety_cost`` package stays minimal.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from safety_cost.config import OUTPUT_DIR, ModelPair
from safety_cost.judging import label_to_scores, parse_judgment
from safety_cost.openai_utils import extract_output_text, get_env, responses_task
from safety_cost.prompts import PAIRWISE_PROMPT

# The back-translation and cross-judge checks cover three of the four families
# (Qwen3-32B is not included).
RELIABILITY_PAIRS = ("llama-3.3-70b", "qwen2.5-72b", "gemma-3-27b")

GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
GOOGLE_LANGUAGE_CODES = {"english": "en", "chinese": "zh", "korean": "ko", "thai": "th"}


def reliability_dir(name: str, output_dir: Path | None = None) -> Path:
    """``outputs/reliability/<name>``."""
    return Path(output_dir or OUTPUT_DIR) / "reliability" / name


def cell_path(root: Path, benchmark: str, language: str, pair: ModelPair) -> Path:
    """Per-(benchmark, language, pair) file under ``root``, named like the pairwise files."""
    return Path(root) / benchmark / language / f"{pair.aligned_tag}__{pair.unaligned_tag}.json"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if (dx == 0 or dy == 0) else round(num / (dx * dy), 4)


def winner(score: float | None, dead_band: float = 0.0) -> int | None:
    """1 = aligned wins, -1 = unaligned wins, 0 = tie."""
    if score is None:
        return None
    if score > 0.5 + dead_band:
        return 1
    if score < 0.5 - dead_band:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Pairwise re-judging (cross-judge and back-translated responses)
# ---------------------------------------------------------------------------

def swap_tasks(custom_id_prefix: str, prompt: str, aligned: str, unaligned: str, judge: str,
               reasoning_effort: str | None) -> list[tuple[str, str, dict]]:
    """The two position-swapped Batch API requests for one instance.

    swap1: A = aligned, B = unaligned; swap2: A = unaligned, B = aligned.
    Returns ``[(swap, custom_id, task), ...]``.
    """
    out = []
    for swap, resp_a, resp_b in (("swap1", aligned, unaligned), ("swap2", unaligned, aligned)):
        custom_id = f"{custom_id_prefix}:{swap}"
        content = PAIRWISE_PROMPT.format(prompt=prompt, response_a=resp_a, response_b=resp_b)
        out.append((swap, custom_id, responses_task(custom_id, judge, content, reasoning_effort)))
    return out


def batch_result_text(result: dict) -> str | None:
    """Judge text of one Batch API result line; ``None`` if the request failed (non-200)."""
    response = result.get("response") or {}
    if response.get("status_code") != 200:
        return None
    return extract_output_text(result) or ""


def score_swaps(raw_1: str, raw_2: str) -> dict:
    """Arena-Hard scores of the aligned model from the two swapped judgments."""
    label_1, label_2 = parse_judgment(raw_1), parse_judgment(raw_2)
    scores = label_to_scores(label_1, a_is_unaligned=False) + label_to_scores(label_2, a_is_unaligned=True)
    return {
        "judgment_1": label_1,
        "judgment_2": label_2,
        "raw_output_1": raw_1,
        "raw_output_2": raw_2,
        "scores": scores,
        "avg_score": sum(scores) / len(scores) if scores else None,
    }


def collect_swaps(results: list[dict], metadata: dict[str, dict]) -> dict[tuple, dict[int, dict[str, str]]]:
    """Group batch outputs as ``{(benchmark, language, pair_key): {index: {swap: text}}}``."""
    grouped: dict[tuple, dict[int, dict[str, str]]] = {}
    n_errors = 0
    for result in results:
        meta = metadata.get(result.get("custom_id"))
        text = batch_result_text(result)
        if meta is None or text is None:
            n_errors += 1
            continue
        key = (meta["benchmark"], meta["language"], meta["pair"])
        grouped.setdefault(key, {}).setdefault(meta["index"], {})[meta["swap"]] = text
    print(f"{len(results)} batch results, {n_errors} errors")
    return grouped


# ---------------------------------------------------------------------------
# Google Cloud Translation API (v2, Basic)
# ---------------------------------------------------------------------------

def google_api_key() -> str:
    return get_env("GOOGLE_TRANSLATE_API_KEY")


def google_translate(text: str, source: str, target: str, key: str, retries: int = 4,
                     split_min_len: int = 200, split_chars: str = " \n\t.,!?;",
                     timeout: int = 60) -> str:
    """Translate ``text`` with the Google Cloud Translation API.

    The JSON body (rather than form data) keeps the request UTF-8 encoded. Texts that are
    rejected as too large are split near the middle (at a whitespace/punctuation
    character) and translated piecewise.
    """
    import requests

    if not text or not text.strip():
        return ""
    payload = {"q": text, "source": source, "target": target, "format": "text"}
    for attempt in range(retries):
        try:
            r = requests.post(GOOGLE_TRANSLATE_URL, params={"key": key}, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["data"]["translations"][0]["translatedText"]
        except requests.HTTPError as e:
            body = e.response.text.lower() if e.response is not None else ""
            status = e.response.status_code if e.response is not None else 0
            if status == 400 and ("too large" in body or "exceeds" in body):
                if len(text) > split_min_len:
                    mid = len(text) // 2
                    while mid < len(text) - 1 and text[mid] not in split_chars:
                        mid += 1
                    kwargs = dict(retries=retries, split_min_len=split_min_len, split_chars=split_chars,
                                  timeout=timeout)
                    left = google_translate(text[:mid], source, target, key, **kwargs)
                    right = google_translate(text[mid:], source, target, key, **kwargs)
                    return left + " " + right
                raise
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} (HTTP {status})", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)
        except Exception as e:  # network errors
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} ({type(e).__name__})", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)
    return ""


def judge_suffix(model: str) -> str:
    """Model name sanitized for use as a JSON-field suffix (``gpt-5.2`` -> ``gpt52``)."""
    return re.sub(r"[^a-zA-Z0-9]", "", model)
