"""Loaders shared by the analysis scripts (read judge outputs written by scripts/2x_*.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_cost.config import (BENCHMARKS, DEFAULT_PAIRS, LANGUAGES, MODEL_PAIRS, MULTIJAIL_LANGUAGES, REFUSAL_JUDGE,
                                pairwise_path, refusal_path, unsafe_path)
from safety_cost.judging import safety_cost, unsafe_rate
from safety_cost.utils import load_json

# Resource availability of each language (Table win_rate: H / M / L).
RESOURCE = {"english": "Base", "chinese": "H", "french": "H", "vietnamese": "H",
            "arabic": "M", "korean": "M", "thai": "L", "ukrainian": "L"}
BENCHMARK_NAMES = {"xstest": "xs-test", "or-bench-80k": "or-bench", "or-bench-hard-1k": "or-bench (hard)"}


def lang_name(language: str) -> str:
    return language.capitalize()


def load_costs(output_dir=None, pairs=None):
    """{pair_key: {benchmark: {language: Safety Cost (%)}}} for every available judgment file."""
    costs = {}
    for key in pairs or DEFAULT_PAIRS:
        pair = MODEL_PAIRS[key]
        for benchmark in BENCHMARKS:
            for language in LANGUAGES:
                path = pairwise_path(benchmark, language, pair, output_dir)
                if path.exists():
                    costs.setdefault(key, {}).setdefault(benchmark, {})[language] = safety_cost(load_json(path))
    return costs


def load_unsafe_rates(output_dir=None, pairs=None):
    """{pair_key: {language: (unsafe_aligned, unsafe_unaligned)}} on MultiJail (%)."""
    rates = {}
    for key in pairs or DEFAULT_PAIRS:
        pair = MODEL_PAIRS[key]
        for language in MULTIJAIL_LANGUAGES:
            paths = [unsafe_path(language, pair.tag(w), output_dir) for w in ("aligned", "unaligned")]
            if all(p.exists() for p in paths):
                rates.setdefault(key, {})[language] = tuple(unsafe_rate(load_json(p)) for p in paths)
    return rates


def load_refusal_rates(output_dir=None, pairs=None, judge=REFUSAL_JUDGE):
    """{pair_key: {benchmark: {language: (refusal_aligned, refusal_unaligned)}}} (%)."""
    def rate(items):
        vals = [it[f"refusal_{judge}"] for it in items if it.get(f"refusal_{judge}") in (0, 1)]
        return 100.0 * sum(vals) / len(vals) if vals else float("nan")

    rates = {}
    for key in pairs or DEFAULT_PAIRS:
        pair = MODEL_PAIRS[key]
        for benchmark in BENCHMARKS:
            for language in LANGUAGES:
                paths = [refusal_path(benchmark, language, pair.tag(w), output_dir) for w in ("aligned", "unaligned")]
                if all(p.exists() for p in paths):
                    rates.setdefault(key, {}).setdefault(benchmark, {})[language] = \
                        tuple(rate(load_json(p)) for p in paths)
    return rates


def figures_dir(output_dir=None) -> Path:
    from safety_cost.config import OUTPUT_DIR

    path = Path(output_dir or OUTPUT_DIR) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
