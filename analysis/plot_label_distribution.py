"""Distribution of meta-judge labels (Sec. 4.2, Fig. label_distribution_by_{model,language}).

Only pairs in which *both* the aligned and the unaligned response comply (no refusal
according to scripts/22_judge_refusal.py) are kept, so the labels describe implicit
quality differences rather than refusals. Labels from both position-swapped judgments
are counted.

Usage:
    python analysis/plot_label_distribution.py [--output_dir outputs]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import figures_dir  # noqa: E402

from safety_cost.config import (BENCHMARKS, DEFAULT_PAIRS, LANGUAGES, MODEL_PAIRS, REFUSAL_JUDGE,  # noqa: E402
                                meta_judge_path, refusal_path)
from safety_cost.utils import load_json  # noqa: E402

MODEL_NAMES = {"llama-3.3-70b": "Llama-3.3-70B-IT", "qwen2.5-72b": "Qwen2.5-72B-IT",
               "gemma-3-27b": "Gemma-3-27B-IT", "qwen3-32b": "Qwen3-32B"}

# ── Global style ──────────────────────────────────────────────────────────────
rcParams['font.family']       = 'serif'
rcParams['font.serif']        = ['Georgia', 'DejaVu Serif', 'Times New Roman']
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False
rcParams['axes.spines.left']  = False
rcParams['axes.grid']         = True
rcParams['grid.color']        = '#ebebeb'
rcParams['grid.linewidth']    = 0.6

# ── Canonical label order — by category ───────────────────────────────────────
CATEGORY_LABELS = {
    "Informativeness": [
        "INFO_USABILITY",
        "COVERAGE_SCOPE",
    ],
    "Presentation Quality": [
        "CLARITY_STRUCTURE",
        "TONE_HELPFULNESS",
    ],
    "Response Integrity": [
        "CORRECTNESS_RELIABILITY",
        "CALIBRATION_NUANCE",
        "SAFETY_ETHICS",
        "OUTPUT_FAILURE",
    ],
}

CATEGORY_COLORS = {
    "Informativeness":    "#2d9e89",
    "Presentation Quality": "#c97d2a",
    "Response Integrity": "#3a5f96",
}

# Flat ordered list for bar plotting
LABEL_ORDER = [l for labels in CATEGORY_LABELS.values() for l in labels]

# ── Paper-quality palette ─────────────────────────────────────────────────────
LABEL_PALETTE = {
    # Informativeness — teal family
    "INFO_USABILITY":            "#2d9e89",
    "COVERAGE_SCOPE":            "#8fd4c8",
    # Presentation Quality — amber family
    "CLARITY_STRUCTURE":         "#c97d2a",
    "TONE_HELPFULNESS":          "#e8bc7a",
    # Response Integrity — slate-blue + red
    "CORRECTNESS_RELIABILITY":   "#3a5f96",
    "CALIBRATION_NUANCE":        "#7a9dc9",
    "SAFETY_ETHICS":             "#a8bfe0",
    "OUTPUT_FAILURE":            "#c94040",
}

LABEL_TO_CATEGORY = {
    l: cat
    for cat, labels in CATEGORY_LABELS.items()
    for l in labels
}


def _get_label_cols(df, category_col):
    """Return label columns in canonical category order."""
    raw = [
        c for c in df.columns
        if c.isupper() and not c.endswith("_pct")
        and c not in {category_col, "TOTAL_LABELS"}
    ]
    # Sort by canonical order, unknowns go last
    ordered = [l for l in LABEL_ORDER if l in raw]
    others  = [l for l in raw if l not in ordered]
    return ordered + others


def _make_grouped_legend(fig_or_ax, label_cols, *, fig_legend=False,
                         bbox=(0.5, 1.01), fontsize=10):
    """
    2-column vertical layout:
      Col 0: Informativeness header + its labels
             Presentation Quality header + its labels
      Col 1: Response Integrity header + its labels
    """
    cat_present = {}
    for cat, cat_labels in CATEGORY_LABELS.items():
        present = [l for l in cat_labels if l in label_cols]
        if present:
            cat_present[cat] = present

    if not cat_present:
        return

    def _spacer():
        return mpatches.Patch(facecolor="none", edgecolor="none", label=" ")

    def _build_col(cats_in_col):
        """Build a flat list of patches for one column (header + labels per cat)."""
        items = []
        for cat in cats_in_col:
            if cat not in cat_present:
                continue
            items.append(
                mpatches.Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="none", label=cat)
            )
            for l in cat_present[cat]:
                items.append(
                    mpatches.Patch(facecolor=LABEL_PALETTE.get(l, "#aaaaaa"),
                                   edgecolor="white", linewidth=0.4,
                                   label=f"  {l.replace('_', ' ').title()}")
                )
        return items

    # ── Assign categories to columns ────────────────────────────────────────
    all_cats = list(cat_present.keys())
    # Put first two categories in left col, rest in right col
    left_cats  = all_cats[:2]
    right_cats = all_cats[2:]

    left_col  = _build_col(left_cats)
    right_col = _build_col(right_cats)

    # Pad shorter column with spacers so both columns are same length
    n_rows = max(len(left_col), len(right_col))
    left_col  += [_spacer()] * (n_rows - len(left_col))
    right_col += [_spacer()] * (n_rows - len(right_col))

    # matplotlib fills COLUMN-MAJOR: left_col first, then right_col
    elements = left_col + right_col

    cat_names = set(cat_present.keys())

    kwargs = dict(
        handles=elements,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.97,
        edgecolor="#dddddd",
        handlelength=1.1,
        handleheight=0.95,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    if fig_legend:
        leg = fig_or_ax.legend(bbox_to_anchor=bbox, loc="lower center", **kwargs)
    else:
        leg = fig_or_ax.legend(loc="upper right", **kwargs)

    for text in leg.get_texts():
        if text.get_text().strip() in cat_names:
            text.set_fontweight("bold")


def _style_ax(ax, label_cols, df_sub, display_col, cats):
    """Draw grouped bars with canonical label order and apply styling."""
    n_cats   = len(cats)
    n_labels = len(label_cols)
    x        = np.arange(n_cats)
    w        = 0.75 / n_labels

    for i, label in enumerate(label_cols):
        color  = LABEL_PALETTE.get(label, "#aaaaaa")
        vals   = df_sub.set_index(display_col).loc[cats, label].values.astype(float)
        offset = (i - n_labels / 2 + 0.5) * w
        ax.bar(x + offset, vals, width=w * 0.90,
               color=color, edgecolor="white", linewidth=0.5,
               zorder=3, alpha=0.93)

    # Thin vertical separators between language groups
    for xi in x[1:]:
        ax.axvline(xi - 0.5, color="#e0e0e0", lw=0.6, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=28, color="#222222")
    ax.set_ylabel("Count", fontsize=20, color="#444444", labelpad=6)
    ax.yaxis.set_tick_params(labelsize=20, labelcolor="#555555")
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_axisbelow(True)


def plot_label_distribution_two_row(df, category_col: str, title_prefix: str, save_path: str = None):
    """Two-row grouped bar — drop-in replacement for language breakdown."""
    if df is None or df.empty:
        print(f"[plot_label_distribution_two_row] Empty DataFrame for {title_prefix}, skipping.")
        return

    label_cols = _get_label_cols(df, category_col)
    if not label_cols:
        print(f"[plot_label_distribution_two_row] No label columns found, skipping.")
        return

    display_col = f"_display_{category_col}"
    df = df.copy()
    df[display_col] = df[category_col].apply(lambda v: "English" if v == "all" else v)

    all_cats = df[display_col].tolist()
    mid      = (len(all_cats) + 1) // 2
    rows     = [all_cats[:mid], all_cats[mid:]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                             facecolor="white", sharey=False)
    fig.subplots_adjust(hspace=0.18, top=0.82)

    for ax, cats in zip(axes, rows):
        ax.set_facecolor("white")
        _style_ax(ax, label_cols, df, display_col, cats)

    # Shared grouped legend above both panels
    _make_grouped_legend(fig, label_cols, fig_legend=True,
                         bbox=(0.5, 0.85), fontsize=24)

    # fig.suptitle(title_prefix, fontsize=13, fontweight="bold", color="#1a1a1a", y=1.04)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close(fig)



def collect_label_counts(output_dir, pair_keys, judge=REFUSAL_JUDGE):
    """Label counts for both-compliant pairs, grouped by model and by language."""
    by_model = defaultdict(lambda: defaultdict(int))
    by_language = defaultdict(lambda: defaultdict(int))
    for key in pair_keys:
        pair = MODEL_PAIRS[key]
        for benchmark in BENCHMARKS:
            for language in LANGUAGES:
                mj_path = meta_judge_path(benchmark, language, pair, output_dir)
                ref_paths = [refusal_path(benchmark, language, pair.tag(w), output_dir)
                             for w in ("aligned", "unaligned")]
                if not mj_path.exists() or not all(p.exists() for p in ref_paths):
                    continue
                # Match refusal labels to meta-judged pairs by prompt text.
                aligned_ref, unaligned_ref = ({it["prompt"]: it.get(f"refusal_{judge}") for it in load_json(p)}
                                              for p in ref_paths)
                for entry in load_json(mj_path):
                    prompt = entry.get("prompt") or entry.get("prompt_original")
                    if aligned_ref.get(prompt) != 0 or unaligned_ref.get(prompt) != 0:
                        continue
                    for label in entry.get("meta_judge_labels_1", []) + entry.get("meta_judge_labels_2", []):
                        by_model[MODEL_NAMES[key]][label] += 1
                        by_language[language.capitalize()][label] += 1
    return by_model, by_language


def counts_to_df(counts, index_name, order):
    labels = sorted({label for c in counts.values() for label in c})
    rows = []
    for category in [o for o in order if o in counts]:
        row = {index_name: category, "total_labels": sum(counts[category].values())}
        row.update({label: counts[category].get(label, 0) for label in labels})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), choices=list(MODEL_PAIRS))
    args = parser.parse_args()
    out_dir = figures_dir(args.output_dir)

    by_model, by_language = collect_label_counts(args.output_dir, args.pairs)
    if not by_model:
        sys.exit("Need meta-judge and refusal judgments; run scripts/22_judge_refusal.py and 23_meta_judge.py.")
    df_model = counts_to_df(by_model, "model", [MODEL_NAMES[k] for k in args.pairs])
    df_language = counts_to_df(by_language, "language", [l.capitalize() for l in LANGUAGES])
    df_model.to_csv(out_dir / "label_distribution_by_model.csv", index=False)
    df_language.to_csv(out_dir / "label_distribution_by_language.csv", index=False)

    plot_label_distribution_two_row(df_model, category_col="model", title_prefix="",
                                    save_path=out_dir / "label_distribution_by_model.pdf")
    plot_label_distribution_two_row(df_language, category_col="language", title_prefix="",
                                    save_path=out_dir / "label_distribution_by_language.pdf")
