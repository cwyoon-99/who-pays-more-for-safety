"""Safety Cost per benchmark, language and model (Table win_rate in the appendix).

Prints the table and writes outputs/figures/safety_cost_table.csv. The last column is
the mean over models of |Cost_L - Cost_English| (marginal Safety Cost magnitude).

Usage:
    python analysis/safety_cost_table.py [--output_dir outputs]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCHMARK_NAMES, RESOURCE, figures_dir, lang_name, load_costs

from safety_cost.config import BENCHMARKS, LANGUAGES, MODEL_PAIRS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    costs = load_costs(args.output_dir)
    pairs = [k for k in MODEL_PAIRS if k in costs]
    if not pairs:
        sys.exit("No pairwise judgments found; run scripts/20_judge_pairwise.py first.")

    header = ["benchmark", "resource", "language"] + [MODEL_PAIRS[k].family for k in pairs] + ["mean_abs_msc"]
    rows = []
    for benchmark in BENCHMARKS:
        for language in LANGUAGES:
            vals = [costs[k].get(benchmark, {}).get(language) for k in pairs]
            if any(v is None for v in vals):
                continue
            if language == "english":
                mean_abs_msc = ""
            else:
                diffs = [abs(costs[k][benchmark][language] - costs[k][benchmark]["english"]) for k in pairs]
                mean_abs_msc = f"{sum(diffs) / len(diffs):.1f}"
            rows.append([BENCHMARK_NAMES[benchmark], RESOURCE[language], lang_name(language)]
                        + [f"{v:.1f}" for v in vals] + [mean_abs_msc])

    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

    out = figures_dir(args.output_dir) / "safety_cost_table.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
