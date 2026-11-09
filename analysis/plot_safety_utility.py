"""Figure 2: Safety Gain (x) vs. Safety Cost (y) per language, relative to English.

* Safety Cost  = mean over the three over-refusal benchmarks of the unaligned model's
                 pairwise win rate (scripts/20_judge_pairwise.py).
* Safety Gain  = Unsafe(M_unalg, L) - Unsafe(M_align, L) on MultiJail
                 (scripts/21_judge_unsafe.py).

Label positions are hand-tuned (``label_offsets``); adjust them if labels overlap.

Usage:
    python analysis/plot_safety_utility.py [--output_dir outputs]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox, AnnotationBbox, HPacker, TextArea, VPacker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import figures_dir, load_costs, load_unsafe_rates  # noqa: E402

from safety_cost.config import BENCHMARKS, MODEL_PAIRS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output_dir", type=Path, default=None)
args = parser.parse_args()
out_dir = figures_dir(args.output_dir)

# ── Global style ───────────────────────────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Georgia', 'DejaVu Serif', 'Times New Roman']

BG       = "#ffffff"
GRID     = "#dddddd"
TEXT     = "#000000"

# ── Data ──────────────────────────────────────────────────────────────────────
languages = ['English', 'Chinese', 'Vietnamese', 'Arabic', 'Korean', 'Thai']
PAIR_KEYS = {'Llama-3.3-70B': 'llama-3.3-70b', 'Qwen2.5-72B': 'qwen2.5-72b',
             'Gemma-3-27B': 'gemma-3-27b', 'Qwen3-32B': 'qwen3-32b'}

costs = load_costs(args.output_dir)
unsafe = load_unsafe_rates(args.output_dir)

# Aligned-model win rates per benchmark (Safety Cost = 100 - win rate).
win_rates = {m: {l: [100 - costs[k][b][l.lower()] for b in BENCHMARKS] for l in languages}
             for m, k in PAIR_KEYS.items() if k in costs and k in unsafe}
# (aligned unsafe %, unaligned unsafe %) on MultiJail.
unsafe_rates = {m: {l: unsafe[k][l.lower()] for l in languages}
                for m, k in PAIR_KEYS.items() if m in win_rates}
if not win_rates:
    sys.exit("Need pairwise and unsafe judgments; run scripts/20_judge_pairwise.py and 21_judge_unsafe.py.")

lang_colors = {
    'English':    '#2c2c2c',
    'Chinese':    '#c62828',
    'Arabic':     '#2e7d32',
    'Korean':     '#003478',
    'Vietnamese': '#b8860b',
    'Thai':       '#8e24aa',
}

models = [m for m in ['Llama-3.3-70B', 'Qwen2.5-72B', 'Gemma-3-27B', 'Qwen3-32B'] if m in win_rates]

label_offsets = {
    'Llama-3.3-70B': {
        'English':    (-20, -25),
        'Chinese':    (12, 8),
        'Vietnamese': (-60, 12),
        'Arabic':     (12, 4),
        'Korean':     (-90, 0),
        'Thai':       (-58, -6),
    },
    'Qwen2.5-72B': {
        'English':    (12, 8),
        'Chinese':    (12, 8),
        'Vietnamese': (-60, 35),
        'Arabic':     (-88, -4),
        'Korean':     (2, 14),
        'Thai':       (10, -20),
    },
    'Gemma-3-27B': {
        'English':    (14, -25),
        'Chinese':    (-120, -24),
        'Vietnamese': (-22, 8),
        'Arabic':     (10, -24),
        'Korean':     (-95, -4),
        'Thai':       (-60, 8),
    },
    'Qwen3-32B': {
        'English':    (12, -18),
        'Chinese':    (12, 8),
        'Vietnamese': (-145, -6),
        'Arabic':     (12, 8),
        'Korean':     (-95, -6),
        'Thai':       (-65, -18),
    },
}

# ── Quadrant config ────────────────────────────────────────────────────────────
QUAD_COLORS = {
    'top_right':    ('#e0e0e0', 0.25),
    'bottom_left':  ('#e0e0e0', 0.25),
    'bottom_right': ('#c8e6c9', 0.55),
    'top_left':     ('#ffcdd2', 0.35),
}

# Colored arrow quadrant labels
G    = '#2e7d32'   # green  (good: ↑ safety, ↓ tax)
R    = '#c62828'   # red    (bad:  ↓ safety, ↑ tax)
GRAY = '#555555'

# QUAD_DEFS = {
#     'top_right':    ('upper right', [
#         # [('Trade-off', GRAY)],
#         [('(', GRAY), ('↑', G), (' safety, ', GRAY), ('↑', R), (' Tax)', GRAY)],
#     ]),
#     'top_left':     ('upper left', [
#         [('Double Penalty', GRAY)],
#         [('(', GRAY), ('↓', R), (' safety, ', GRAY), ('↑', R), (' Tax)', GRAY)],
#     ]),
#     'bottom_right': ('lower right', [
#         [('Improvement', GRAY)],
#         [('(', GRAY), ('↑', G), (' safety, ', GRAY), ('↓', G), (' Tax)', GRAY)],
#     ]),
#     'bottom_left':  ('lower left', [
#         # [('Trade-off', GRAY)],
#         [('(', GRAY), ('↓', R), (' safety, ', GRAY), ('↓', G), (' Tax)', GRAY)],
#     ]),
# }

QUAD_DEFS = {
    'top_right':    ('upper right', [
        # [('Trade-off', GRAY)],
        [('↑', G), (' safety, ', GRAY), ('↑', R), (' Cost', GRAY)],
    ]),
    'top_left':     ('upper left', [
        # [('Double Penalty', GRAY)],
        [('↓', R), (' safety, ', GRAY), ('↑', R), (' Cost', GRAY)],
    ]),
    'bottom_right': ('lower right', [
        # [('Improvement', GRAY)],
        [('↑', G), (' safety, ', GRAY), ('↓', G), (' Cost', GRAY)],
    ]),
    'bottom_left':  ('lower left', [
        # [('Trade-off', GRAY)],
        [('↓', R), (' safety, ', GRAY), ('↓', G), (' Cost', GRAY)],
    ]),
}

# ── Precompute spans ───────────────────────────────────────────────────────────
all_x_spans, all_y_spans = [], []
for model in models:
    model_xs, model_ys = [], []
    for lang in languages:
        aligned_unsafe, ablit_unsafe = unsafe_rates[model][lang]
        model_xs.append(ablit_unsafe - aligned_unsafe)
        model_ys.append(100 - np.mean(win_rates[model][lang]))
    all_x_spans.append(max(model_xs) - min(model_xs))
    all_y_spans.append(max(model_ys) - min(model_ys))

global_x_span = max(all_x_spans) + 8
global_y_span = max(all_y_spans) + 6

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor(BG)

for idx, model in enumerate(models):
    row, col = divmod(idx, 2)
    ax = axes[row][col]
    ax.set_facecolor('#fafafa')
    for spine in ax.spines.values():
        spine.set_color(GRID)

    eng_aligned, eng_ablit = unsafe_rates[model]['English']
    eng_sg = eng_ablit - eng_aligned
    eng_ug = 100 - np.mean(win_rates[model]['English'])

    xs_all, ys_all = [], []
    for lang in languages:
        al, ab = unsafe_rates[model][lang]
        xs_all.append(ab - al)
        ys_all.append(100 - np.mean(win_rates[model][lang]))

    x_center = (min(xs_all) + max(xs_all)) / 2
    y_center = (min(ys_all) + max(ys_all)) / 2
    if model == 'Gemma-3-27B':
        y_center = y_center

    xlim = (x_center - global_x_span / 2, x_center + global_x_span / 2)
    ylim = (y_center - global_y_span / 2, y_center + global_y_span / 2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    res = 400
    xi = np.linspace(*xlim, res)
    yi = np.linspace(*ylim, res)
    XX, YY = np.meshgrid(xi, yi)
    dx = XX - eng_sg
    dy = YY - eng_ug

    max_dist = max(
        np.sqrt((xlim[1] - eng_sg)**2 + (ylim[1] - eng_ug)**2),
        np.sqrt((xlim[0] - eng_sg)**2 + (ylim[0] - eng_ug)**2),
        np.sqrt((xlim[1] - eng_sg)**2 + (ylim[0] - eng_ug)**2),
        np.sqrt((xlim[0] - eng_sg)**2 + (ylim[1] - eng_ug)**2),
    )
    dist = np.sqrt(dx**2 + dy**2)
    norm_dist = np.clip(dist / max_dist, 0, 1)

    img = np.ones((res, res, 4))
    alpha_max = 0.70

    for qname, mask_fn in [
        ('top_right',    lambda dx, dy: (dx >= 0) & (dy >= 0)),
        ('bottom_left',  lambda dx, dy: (dx <  0) & (dy <  0)),
        ('bottom_right', lambda dx, dy: (dx >= 0) & (dy <  0)),
        ('top_left',     lambda dx, dy: (dx <  0) & (dy >= 0)),
    ]:
        mask = mask_fn(dx, dy)
        r, g, b = hex_to_rgb(QUAD_COLORS[qname][0])
        alpha = norm_dist * alpha_max
        img[:, :, 0] = np.where(mask, 1.0 * (1 - alpha) + r * alpha, img[:, :, 0])
        img[:, :, 1] = np.where(mask, 1.0 * (1 - alpha) + g * alpha, img[:, :, 1])
        img[:, :, 2] = np.where(mask, 1.0 * (1 - alpha) + b * alpha, img[:, :, 2])
        img[:, :, 3] = np.where(mask, 1.0, img[:, :, 3])

    ax.imshow(img, extent=[*xlim, *ylim],
              origin='lower', aspect='auto', zorder=0, interpolation='bilinear')

    # ── Quadrant label annotations (colored arrows) ──────────────────────────
    xmin, xmax = xlim
    ymin, ymax = ylim
    ts = dict(fontsize=18, fontstyle='italic', fontfamily='DejaVu Sans')
    pad = 0.5
    corners = {
        'upper right': (xmax - pad, ymax - pad - 3),
        'upper left':  (xmin + pad, ymax - pad),
        'lower right': (xmax - pad, ymin + pad),
        'lower left':  (xmin + pad, ymin + pad),
    }
    box_align = {
        'upper right': (1, 1),
        'upper left':  (0, 1),
        'lower right': (1, 0),
        'lower left':  (0, 0),
    }
    for qname, (loc, lines) in QUAD_DEFS.items():
        row_boxes = [
            HPacker(children=[TextArea(t, textprops=dict(color=c, **ts))
                               for t, c in line],
                    align='baseline', pad=0, sep=0)
            for line in lines
        ]
        packed = VPacker(children=row_boxes,
                         align='left' if 'left' in loc else 'right',
                         pad=0, sep=2)
        cx, cy = corners[loc]
        ab = AnnotationBbox(packed, (cx, cy), xycoords='data',
                            box_alignment=box_align[loc],
                            frameon=False, zorder=1, pad=0.3)
        ax.add_artist(ab)

    # ── Reference lines through English ────────────────────────────────────
    ref_style = dict(color='#555555', lw=1.2, ls='--', alpha=0.55, zorder=2)
    ax.axhline(y=eng_ug, **ref_style)
    ax.axvline(x=eng_sg, **ref_style)

    # ── Arrows from English ─────────────────────────────────────────────────
    for lang in languages:
        if lang == 'English':
            continue
        al, ab = unsafe_rates[model][lang]
        sg = ab - al
        ug = 100 - np.mean(win_rates[model][lang])
        ax.annotate('',
                    xy=(sg, ug), xytext=(eng_sg, eng_ug),
                    arrowprops=dict(arrowstyle='->', color=lang_colors[lang],
                                    alpha=0.45, lw=1.8, mutation_scale=22),
                    zorder=3)

    # ── Scatter + labels ────────────────────────────────────────────────────
    for lang in languages:
        al, ab = unsafe_rates[model][lang]
        sg = ab - al
        ug = 100 - np.mean(win_rates[model][lang])

        marker = '*' if lang == 'English' else 'o'
        size   = 700 if lang == 'English' else 180

        ax.scatter(sg, ug,
                   c=lang_colors[lang], marker=marker, s=size,
                   alpha=0.92, edgecolors='#ffffff80', linewidths=0.6,
                   zorder=5)

        ox, oy = label_offsets[model][lang]
        ax.annotate(lang, (sg, ug),
                    textcoords='offset points', xytext=(ox, oy),
                    fontsize=20, color=lang_colors[lang], fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='#fafafa')],
                    zorder=6)

    # ── Model label ─────────────────────────────────────────────────────────
    model_display = {
        'Llama-3.3-70B': ('Meta',    'Llama-3.3-70B', '#0668E1'),
        'Qwen2.5-72B':   ('Alibaba', 'Qwen2.5-72B',   '#FF6A00'),
        'Gemma-3-27B':   ('Google',  'Gemma-3-27B',   '#4285F4'),
        'Qwen3-32B':     ('Alibaba', 'Qwen3-32B',      '#FF6A00'),
    }
    _, short_name, _ = model_display[model]
    t_model = TextArea(short_name, textprops=dict(
        fontsize=28, color='#666666', fontweight='bold', fontfamily='serif'))
    box = HPacker(children=[t_model], align='baseline', pad=0, sep=1)
    ab_box = AnchoredOffsetbox(loc='upper right', child=box, pad=0.4,
                               borderpad=0.5, frameon=False)
    ab_box.patch.set_facecolor('white')
    ab_box.patch.set_edgecolor('#bbbbbb')
    ab_box.patch.set_alpha(0.92)
    ax.add_artist(ab_box)

    ax.grid(color=GRID, lw=0.5, zorder=0, alpha=0.6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(labelsize=15, colors=TEXT)
    ax.set_xlabel('')
    ax.set_ylabel('')

# ── Shared axis labels ────────────────────────────────────────────────────────
fig.text(0.5, 0.03,
         r'Safety Gain: Increase in Safe Responses',
         ha='center', fontsize=24, color=TEXT)

fig.text(0.03, 0.5,
         r'Safety Cost: Reduction in Utility',
         va='center', rotation='vertical', fontsize=24, color=TEXT)

plt.tight_layout(rect=[0.04, 0.04, 1.0, 1.0], pad=2.0)
for ext in ('pdf', 'png'):
    plt.savefig(out_dir / f'safety_utility.{ext}', dpi=300, bbox_inches='tight')
print(f"Saved {out_dir / 'safety_utility.pdf'}")
