"""Settings and helpers shared by the matched-checkpoint scripts (Sec. 4.4, Appendix
"Matched-Checkpoint Replication")."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import DATA_DIR, OUTPUT_DIR, SEED

# data/matched_checkpoint/: training mixtures; outputs/matched_checkpoint/: checkpoints,
# generations, judge batches and results.
MC_DATA_DIR = DATA_DIR / "matched_checkpoint"
MC_OUTPUT_DIR = OUTPUT_DIR / "matched_checkpoint"

# Default model tags used in output file names.
UNALIGNED_TAG = "qwen3-8b-c0"   # instruction-only SFT
ALIGNED_TAG = "qwen3-8b-c1"     # instruction + English safety SFT

# Evaluation languages. Korean is generated and judged, but excluded from Table
# matched_checkpoint (degenerate Korean output of the 8B checkpoints, see the appendix).
EVAL_LANGUAGES = ("english", "chinese", "vietnamese", "arabic", "korean", "thai")
TABLE_LANGUAGES = ("english", "chinese", "vietnamese", "arabic", "thai")

# Over-refusal benchmarks for the Safety Cost of the matched checkpoints, and the harmful
# prompts used for refusal / Safety Gain.
COST_BENCHMARKS = ("or-bench-hard-1k", "xstest", "falsereject")
GAIN_BENCHMARK = "multijail"

# Judge settings of the matched-checkpoint experiment (Chat Completions Batch API).
JUDGE_MODEL = "gpt-5-mini"
JUDGE_REASONING_EFFORT = "low"

# ---------------------------------------------------------------------------
# Aya instruction data (shared by build_instruction_base.py and build_c0.py)
# ---------------------------------------------------------------------------
MIN_RESP_CHARS = 50        # minimum response length, applied to every source
COLL_SCAN_CAP = 400000     # max streamed rows per language before giving up

# Canonical target language -> language names in CohereForAI/aya_dataset (human-written).
TARGET_HUMAN = {
    "English": ["English"],
    "Chinese": ["Simplified Chinese", "Traditional Chinese"],
    "Vietnamese": ["Vietnamese"],
    "Arabic": ["Standard Arabic", "Moroccan Arabic", "Egyptian Arabic"],
    "Korean": ["Korean"],
    "Thai": ["Thai"],
}
# Canonical target language -> CohereForAI/aya_collection_language_split config (backfill).
TARGET_COLLECTION_CFG = {
    "English": "english", "Chinese": "simplified_chinese", "Vietnamese": "vietnamese",
    "Arabic": "standard_arabic", "Korean": "korean", "Thai": "thai",
}
ALL_TARGET_HUMAN_NAMES = {n for v in TARGET_HUMAN.values() for n in v}


def load_aya_human():
    """aya_dataset (train) with responses shorter than MIN_RESP_CHARS removed."""
    from datasets import load_dataset

    human = load_dataset("CohereForAI/aya_dataset", split="train").to_pandas()
    before = len(human)
    human = human[human["targets"].str.len() >= MIN_RESP_CHARS].reset_index(drop=True)
    print(f"aya human: {before} rows, {len(human)} after >={MIN_RESP_CHARS}-char filter")
    return human


def collect_from_collection(cfg: str, n: int, seed: int = SEED, exclude=frozenset()) -> list[dict]:
    """Stream up to ``n`` examples from one aya_collection language config.

    Keeps responses of at least MIN_RESP_CHARS characters (drops templated short-answer QA)
    whose prompt is not in ``exclude``.
    """
    from datasets import load_dataset

    if n <= 0:
        return []
    out, scanned = [], 0
    try:
        ds = load_dataset("CohereForAI/aya_collection_language_split", cfg, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10000)
        for ex in ds:
            scanned += 1
            inp = (ex.get("inputs") or "").strip()
            tgt = (ex.get("targets") or "").strip()
            if inp and tgt and len(tgt) >= MIN_RESP_CHARS and inp not in exclude:
                out.append({"prompt": inp, "response": tgt, "language": cfg, "source": "aya_collection"})
            if len(out) >= n or scanned >= COLL_SCAN_CAP:
                break
    except Exception as e:
        print(f"  [collection {cfg}] ERROR after {len(out)} (scanned {scanned}): {type(e).__name__}: {str(e)[:120]}")
    if len(out) < n:
        print(f"  [collection {cfg}] SHORTFALL: wanted {n}, got {len(out)} after scanning {scanned}")
    return out
