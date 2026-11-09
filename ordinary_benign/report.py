"""Compare the cross-lingual Cost disparity at the safety boundary and on ordinary prompts.

For every family and language, the disparity is Cost_L - Cost_English:
  - boundary: Cost averaged over the three over-refusal benchmarks (main pairwise judgments);
  - ordinary: Cost on the ordinary benign instructions (``ordinary_benign_cost.json``).
The last row is the mean over the five non-English languages (paper table "ordinary_benign",
which shows Gemma-3-27B and Qwen2.5-72B).

Usage:
    python ordinary_benign/report.py
    python ordinary_benign/report.py --cost_json path/to/ordinary_benign_cost.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ordinary_benign.generate import BENCHMARK, LANGUAGES
from safety_cost.config import BENCHMARKS, MODEL_PAIRS, judgment_dir, pairwise_path
from safety_cost.judging import safety_cost
from safety_cost.utils import load_json


def boundary_costs(pair_key: str, output_dir: Path | None) -> dict[str, float]:
    costs = {}
    for language in LANGUAGES:
        paths = [pairwise_path(b, language, MODEL_PAIRS[pair_key], output_dir) for b in BENCHMARKS]
        if all(p.exists() for p in paths):
            costs[language] = sum(safety_cost(load_json(p)) for p in paths) / len(paths)
    return costs


def disparity(costs: dict[str, float]) -> dict[str, float]:
    if "english" not in costs:
        return {}
    return {lang: costs[lang] - costs["english"] for lang in LANGUAGES if lang in costs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost_json", type=Path, default=None,
                        help="default: outputs/judgments/ordinary_benign/ordinary_benign_cost.json")
    parser.add_argument("--pairs", nargs="+", default=list(MODEL_PAIRS), choices=list(MODEL_PAIRS))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="outputs directory holding the main pairwise judgments")
    args = parser.parse_args()

    ordinary = load_json(args.cost_json or judgment_dir(BENCHMARK, args.output_dir) / "ordinary_benign_cost.json")
    for pair_key in args.pairs:
        boundary = disparity(boundary_costs(pair_key, args.output_dir))
        ordinary_disp = disparity(ordinary.get(pair_key, {}))
        if not boundary and not ordinary_disp:
            continue
        print(f"\n{MODEL_PAIRS[pair_key].family}: Cost disparity vs. English (boundary / ordinary)")
        for language in LANGUAGES:
            b, o = boundary.get(language), ordinary_disp.get(language)
            print(f"  {language:<11} {'n/a' if b is None else f'{b:+.1f}':>6} / {'n/a' if o is None else f'{o:+.1f}':>6}")
        means = []
        for d in (boundary, ordinary_disp):
            values = [v for lang, v in d.items() if lang != "english"]
            means.append(f"{sum(values) / len(values):+.1f}" if values else "n/a")
        print(f"  {'non-En mean':<11} {means[0]:>6} / {means[1]:>6}")


if __name__ == "__main__":
    main()
