"""Figure 5 (confound map): is Safety Cost an artifact of language deficits from ablation?

* x = Alignment Gap Difference = Delta_L - Delta_English, where
      Delta_L = Score(M_align, L) - Score(M_unalg, L) on benign P-MMEval tasks
      (capability/summarize.py writes these scores).
* y = Marginal Safety Cost = Cost_L - Cost_English (mean over the three benchmarks).

Points near x = 0 mean that ablation does not hurt a language more than English.

Usage:
    python capability/summarize.py <opencompass work dir>      # writes outputs/capability/capability_drop.csv
    python analysis/plot_confound_map.py --capability_csv outputs/capability/capability_drop.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import figures_dir, load_costs  # noqa: E402

from safety_cost.config import BENCHMARKS, DEFAULT_PAIRS, MODEL_PAIRS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--capability_csv", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, default=None)
args = parser.parse_args()
out_dir = figures_dir(args.output_dir)

# ── Global style ──────────────────────────────────────────────────────────────
rcParams['font.family']       = 'serif'
rcParams['font.serif']        = ['Georgia', 'DejaVu Serif', 'Times New Roman']

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS  = ["Llama-3.3-70B", "Qwen2.5-72B", "Gemma-3-27B", "Qwen3-32B"]
PALETTE = ['#0ea5d4', '#7b1fa2', '#ec407a', '#b388ff']
LANGS   = ["Chinese", "French", "Vietnamese", "Arabic", "Korean", "Thai", "Ukrainian"]
FAMILY_TO_KEY = {MODEL_PAIRS[k].family: k for k in DEFAULT_PAIRS}

# Benign-task alignment gap Delta_L = aligned score - unaligned score.
benign_gap = {}
with open(args.capability_csv) as f:
    for row in csv.DictReader(f):
        gap = float(row["aligned"]) - float(row["unaligned"])
        family = MODEL_PAIRS[row["pair"]].family if "pair" in row else row["family"]
        benign_gap.setdefault(family, {})[row["language"].capitalize()] = gap

costs = load_costs(args.output_dir)


def mean_cost(key, lang):
    return np.mean([costs[key][b][lang.lower()] for b in BENCHMARKS])


# Language markers and resource level
LANG_MARKER = {
    "Chinese":    ("o", "H"), "French":     ("o", "H"),
    "Vietnamese": ("o", "H"), "Arabic":     ("s", "M"),
    "Korean":     ("s", "M"), "Thai":       ("D", "L"),
    "Ukrainian":  ("D", "L"),
}

# ── Build scatter points ──────────────────────────────────────────────────────
points = []  # (x, y, model, lang, color)
for model, color in zip(MODELS, PALETTE):
    key = FAMILY_TO_KEY[model]
    if model not in benign_gap or key not in costs:
        continue
    for lang in LANGS:
        if lang not in benign_gap[model] or lang.lower() not in costs[key][BENCHMARKS[0]]:
            continue
        x = benign_gap[model][lang] - benign_gap[model]["English"]
        y = mean_cost(key, lang) - mean_cost(key, "English")
        points.append((x, y, model, lang, color))

for model in MODELS:
    xs = [abs(x) for (x, _, m, _, _) in points if m == model]
    if xs:
        print(f"{model}: mean |Alignment Gap Difference| = {np.mean(xs):.2f}")

# ── Figure ────────────────────────────────────────────────────────────────────
BG       = "#ffffff"
PANEL_BG = "#ffffff"
GRID     = "#dddddd"
TEXT     = "#000000"
ACCENT   = "#3b3b98"

fig, ax = plt.subplots(figsize=(7, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_color(GRID)

# ── Quadrant shading ──────────────────────────────────────────────────────────
ax.axhspan(0, 32, xmin=0, xmax=1, color="#6b5cd8", alpha=0.10, zorder=0)
ax.axhspan(-10, 0, xmin=0, xmax=1, color="#2196f3", alpha=0.08, zorder=0)

# Quadrant labels
ax.text(9.7, 23, "Cost > 0",
        color="#6b5cd8", fontsize=20, ha="right", va="top", style="italic")
ax.text(9.7, -4.5, "Cost < 0",
        color="#2196f3", fontsize=20, ha="right", va="bottom", style="italic")

# ── Reference lines ───────────────────────────────────────────────────────────
ax.axhline(0, color="#00000040", lw=1.0, zorder=2)
ax.axvline(0, color="#00000040", lw=1.0, zorder=2)

# ── Scatter points ────────────────────────────────────────────────────────────
for model, color in zip(MODELS, PALETTE):
    model_pts = [(x, y, lang) for (x, y, m, lang, c) in points if m == model]
    for x, y, lang in model_pts:
        marker, _ = LANG_MARKER[lang]
        ax.scatter(x, y, color=color, marker=marker, s=180,
                   alpha=0.92, zorder=5, edgecolors="#ffffff40", linewidths=0.6)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel(
    r"Alignment Gap Difference",
    color=TEXT, fontsize=20, labelpad=6)
ax.set_ylabel(
    r"Marginal Safety Cost",
    color=TEXT, fontsize=20, labelpad=6)
ax.set_xlim(-12.5, 12.5)
ax.set_ylim(-10, 25)
ax.xaxis.set_tick_params(labelcolor=TEXT, labelsize=14)
ax.yaxis.set_tick_params(labelcolor=TEXT, labelsize=14)
ax.grid(color=GRID, lw=0.6, zorder=0)

# ── Legend ────────────────────────────────────────────────────────────────────
model_handles = [
    plt.scatter([], [], color=color, marker="o", s=120,
                alpha=0.92, label=model, edgecolors="#ffffff40", linewidths=0.6)
    for model, color in zip(MODELS, PALETTE)
]

res_handles = [
    plt.scatter([], [], marker="o", color="#aaaaaa", s=80, label="High-resource"),
    plt.scatter([], [], marker="s", color="#aaaaaa", s=80, label="Mid-resource"),
    plt.scatter([], [], marker="D", color="#aaaaaa", s=80, label="Low-resource"),
]

# Legend is filled column-major: models in the first column, resource levels in the second.
dummy = plt.scatter([], [], marker="", color="none", s=0, label=" ")
combined_handles = model_handles + res_handles + [dummy]

legend = ax.legend(
    handles=combined_handles,
    ncol=2,
    fontsize=20,
    loc="lower center",
    bbox_to_anchor=(0.45, 1.02),
    framealpha=0.3,
    labelcolor=TEXT,
    facecolor=PANEL_BG,
    edgecolor=GRID,
    columnspacing=1.5,
    handletextpad=0.5,
)

plt.tight_layout(pad=0)
for ext in ("pdf", "png"):
    plt.savefig(out_dir / f"confound_map.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {out_dir / 'confound_map.pdf'}")
