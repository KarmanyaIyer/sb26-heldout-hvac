"""Paper figure: per-scenario change vs the public chain on both simulators (30-day val16).

python src/figures.py  -> paper/figures/flip.pdf
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SERIES = [("v2a", "Ours (v2a): 2 h hold", "#2a78d6", "o"), ("v2b", "Ours, 72 h hold (v2b)", "#eb6834", "D")]
CLIMATE = ["fairbanks", "minneapolis", "denver", "atlanta", "phoenix", "singapore"]


def scen(name):
    d = json.loads((RES / f"{name}.json").read_text())
    return {r["scenario"]: -r["mean"] for r in d["scenarios"]}


def rel(ref, run):
    a, b = scen(ref), scen(run)
    assert set(a) == set(b)
    return {k: 100 * (b[k] - a[k]) / a[k] for k in a}


order = sorted(scen("penguin_val_fixed"), key=lambda k: (CLIMATE.index(k.split("/")[1].split("_")[0]), k), reverse=True)
labels = [f"{k.split('/')[1].replace('_', ' ')} ({k.split('/')[0].replace('building_', 'b')})" for k in order]

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.size": 0,
                     "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (sim, title) in zip(axes, [("fixed", "Corrected simulator"), ("shipped", "Shipped simulator")]):
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for (key, lab, col, mk), off in zip(SERIES, (0.17, -0.17)):
        d = rel(f"penguin_val_{sim}", f"{key}_val_{sim}")
        xs = [d[k] for k in order]
        ax.scatter(xs, [i + off for i in range(len(order))], s=16, marker=mk, color=col, edgecolor="white", linewidth=0.6,
                   zorder=3, label=lab)
    ax.set_title(title, fontsize=8, color=INK, pad=4)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Score change vs. public chain (%), left is better", color=INK)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(labels, fontsize=7)
for ax in axes:
    for i in range(len(order)):
        if i % 2:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", zorder=0, lw=0)
axes[0].set_ylim(-0.7, len(order) - 0.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.56, 1.04), handletextpad=0.3,
           columnspacing=1.5)
fig.savefig(OUT / "flip.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "flip.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print(OUT / "flip.pdf")
