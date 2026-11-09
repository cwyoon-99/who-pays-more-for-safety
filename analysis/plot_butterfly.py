"""Figure 3: refusal-rate difference (left) vs. Safety Cost (right) per language.

* Left bars:  refusal rate of the aligned model minus that of the unaligned model (%p),
              from scripts/22_judge_refusal.py.
* Right bars: Safety Cost (%), from scripts/20_judge_pairwise.py.
Numbers next to the bars are differences from English. Symmetric patterns indicate
refusal-driven cost; asymmetric ones indicate quality degradation beyond refusals.

Usage:
    python analysis/plot_butterfly.py [--output_dir outputs]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCHMARK_NAMES, RESOURCE, figures_dir, lang_name, load_costs, load_refusal_rates  # noqa: E402

from safety_cost.config import BENCHMARKS, LANGUAGES  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output_dir", type=Path, default=None)
args = parser.parse_args()
out_dir = figures_dir(args.output_dir)

# ── Global style (match safety_utility) ───────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Georgia', 'DejaVu Serif', 'Times New Roman']

BG   = "#ffffff"
GRID = "#dddddd"
TEXT = "#000000"

# ── Data ──────────────────────────────────────────────────────────────────────
PAIR_KEYS = {'Llama-3.3': 'llama-3.3-70b', 'Qwen2.5': 'qwen2.5-72b', 'Gemma-3': 'gemma-3-27b', 'Qwen3': 'qwen3-32b'}
costs = load_costs(args.output_dir)
refusals = load_refusal_rates(args.output_dir)

data = []
for benchmark in BENCHMARKS:
    for model, key in PAIR_KEYS.items():
        for language in LANGUAGES:
            try:
                cost = costs[key][benchmark][language]
                ref_aligned, ref_unaligned = refusals[key][benchmark][language]
            except KeyError:
                continue
            data.append((BENCHMARK_NAMES[benchmark], model, lang_name(language), RESOURCE[language],
                         100 - cost, ref_aligned - ref_unaligned))
if not data:
    sys.exit("Need pairwise and refusal judgments; run scripts/20_judge_pairwise.py and 22_judge_refusal.py.")

df = pd.DataFrame(data, columns=['Dataset','Model','Language','Resource','WinRate','RefusalDiff'])
df['SafetyCost'] = 100 - df['WinRate']
df['Refusal']    = df['RefusalDiff']

# ── Palette ───────────────────────────────────────────────────────────────────
# Resource level colors (right-side bars, Safety Cost)
RES_COLORS = {
    'Base': 'black',
    'H':    '#2ca02c',
    'M':    '#1f77b4',
    'L':    '#ff7f0e',
}
REFUSAL_COLOR  = '#bdbdbd'   # light grey for left bars
REFUSAL_EDGE   = '#9e9e9e'
BASELINE_COLOR = '#555555'

# Model display names + brand colors (same as safety_utility)
MODEL_META = {
    'Llama-3.3': ('Meta',    'Llama-3.3',  '#0668E1'),
    'Qwen2.5':   ('Alibaba', 'Qwen2.5',    '#FF6A00'),
    'Gemma-3':   ('Google',  'Gemma-3',    '#4285F4'),
    'Qwen3':     ('Alibaba', 'Qwen3',      '#FF6A00'),
}

DATASETS = ['xs-test', 'or-bench', 'or-bench (hard)']
MODELS   = [m for m in ['Llama-3.3', 'Qwen2.5', 'Gemma-3', 'Qwen3'] if m in set(df['Model'])]
LANG_ORDER = ['Ukrainian', 'Thai', 'Korean', 'Arabic', 'Vietnamese', 'French', 'Chinese', 'English']
LANG_ABBREV = {'Vietnamese': 'Viet.', 'Ukrainian': 'Ukr.'}

ROW_MAX = 75   # fixed x-axis extent

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(DATASETS), len(MODELS), squeeze=False, figsize=(26, 17), sharey=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(wspace=0.06, hspace=0.18, left=0.13, right=0.97,
                    top=0.93, bottom=0.09)

for i, dataset in enumerate(DATASETS):
    row_df = df[df['Dataset'] == dataset].copy()
    row_df['LangRank'] = row_df['Language'].map(
        {lang: idx for idx, lang in enumerate(LANG_ORDER)})
    row_df = row_df.sort_values('LangRank')

    for j, model in enumerate(MODELS):
        ax = axes[i, j]
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.spines['left'].set_visible(False)

        subset = row_df[row_df['Model'] == model].reset_index(drop=True)
        eng    = subset[subset['Language'] == 'English'].iloc[0]
        eng_cost    = eng['SafetyCost']
        eng_refusal = eng['Refusal']

        y_pos = np.arange(len(subset))

        # ── Left bars: Refusal rate (grey) ────────────────────────────────
        ax.barh(y_pos, -subset['Refusal'],
                color=REFUSAL_COLOR, edgecolor=REFUSAL_EDGE,
                height=0.62, zorder=3)

        # English refusal baseline
        ax.axvline(-eng_refusal, color=BASELINE_COLOR,
                   ls='--', lw=1.4, alpha=0.7, zorder=4)

        # ── Right bars: Safety Cost (resource-colored) ────────────────────
        bar_colors = [RES_COLORS[r] for r in subset['Resource']]
        ax.barh(y_pos, subset['SafetyCost'],
                color=bar_colors, edgecolor='white',
                height=0.62, zorder=3, alpha=0.88)

        # English safety cost baseline
        ax.axvline(eng_cost, color=BASELINE_COLOR,
                   ls='--', lw=1.4, alpha=0.7, zorder=4)

        # English baseline value label (top)
        ax.text(eng_cost, len(subset) - 0.1,
                f'{eng_cost:.1f}',
                ha='center', va='bottom', fontsize=22,
                fontweight='bold', color=BASELINE_COLOR)

        # Center line
        ax.axvline(0, color='#888888', lw=1.0, zorder=5)

        # ── Diff annotations ──────────────────────────────────────────────
        for y, row in zip(y_pos, subset.itertuples()):
            if row.Language == 'English':
                continue
            # Safety Cost diff (right side)
            cost_diff = row.SafetyCost - eng_cost
            tc = '#c62828' if cost_diff >= 0 else '#2e7d32'
            ax.text(row.SafetyCost + 1.5, y,
                    f'{cost_diff:+.1f}',
                    va='center', ha='left', fontsize=24,
                    fontweight='bold', color=tc, zorder=6)
            # Refusal diff (left side)
            ref_diff = row.Refusal - eng_refusal
            tc2 = '#c62828' if ref_diff >= 0 else '#2e7d32'
            ax.text(-row.Refusal - 1.5, y,
                    f'{ref_diff:+.1f}',
                    va='center', ha='right', fontsize=24,
                    fontweight='bold', color=tc2, zorder=6)

        # ── Axes decoration ───────────────────────────────────────────────
        ax.set_xlim(-ROW_MAX, ROW_MAX)
        ax.set_ylim(-0.6, len(subset) - 0.3)

        # major_ticks = sorted(set(
            # [t for t in range(-ROW_MAX, ROW_MAX+1, 50) if t != 0] + [0]))
        major_ticks = [-50, 0, 50]
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([str(abs(t)) for t in major_ticks], fontsize=20, color=TEXT)
        ax.tick_params(axis='x', length=3, color=GRID)

        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)

        # Y-axis language labels (first column only)
        ax.set_yticks(y_pos)
        if j == 0:
            labels = [LANG_ABBREV.get(l, l) for l in subset['Language']]
            ax.set_yticklabels(labels, fontsize=22, fontweight='bold', color=TEXT)
            ax.tick_params(axis='y', length=0, pad=8)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)

        # ── Model label (title above panel) ──
        if i == 0:
            ax.set_title(model, fontsize=32, fontweight='bold',
                         color=TEXT, pad=40, fontfamily='serif')

        # ── Dataset label (left of first column) ─────────────────────────
        if j == 0:
            ax.text(-0.32, 0.5, dataset,
                    transform=ax.transAxes,
                    rotation=90, va='center', ha='center',
                    fontsize=32, fontweight='bold', color='#444444')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=REFUSAL_COLOR, edgecolor=REFUSAL_EDGE,
                   label='Refusal Rate Difference'),
    mlines.Line2D([], [], color=BASELINE_COLOR, ls='--', lw=1.5,
                  label='English Baseline'),
    mpatches.Patch(facecolor=RES_COLORS['H'], edgecolor='white',
                   label='High-resource'),
    mpatches.Patch(facecolor=RES_COLORS['M'], edgecolor='white',
                   label='Medium-resource'),
    mpatches.Patch(facecolor=RES_COLORS['L'], edgecolor='white',
                   label='Low-resource'),
]
fig.legend(handles=legend_elements,
           loc='lower center', bbox_to_anchor=(0.55, 0.01),
           ncol=5, frameon=False, fontsize=30,
           handlelength=1.6, handletextpad=0.5, columnspacing=1.2)

# ── Shared x-axis label ───────────────────────────────────────────────────────
# fig.text(0.55, -0.02,
#          r'Refusal Rate  $|$  Safety Cost  (%)',
#          ha='center', fontsize=32, color=TEXT)

for ext in ('pdf', 'png'):
    plt.savefig(out_dir / f'butterfly.{ext}', dpi=300, bbox_inches='tight')
print(f"Saved {out_dir / 'butterfly.pdf'}")
