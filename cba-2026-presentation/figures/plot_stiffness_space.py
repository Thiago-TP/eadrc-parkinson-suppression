"""
Appendix figure: the stiffness space of the Monte Carlo runs.

Plots the table of stiffness samples in configs.yaml (the same 99 samples are
used by every control strategy and scenario), the sampling intervals and the
nominal values.

Run from the repository root:
    uv run cba-2026-presentation/figures/plot_stiffness_space.py
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml

OUT = "cba-2026-presentation/figures/stiffness_space.pdf"

# Slide theme colors (cba2026.sty)
PURPLE, ORANGE, PETROL, GRAPHITE = "#7040B8", "#F28C38", "#245B73", "#303030"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{sfmath}",
        "font.size": 14,
    }
)

with open("configs.yaml") as f:
    params = yaml.safe_load(f)["parameters"]

nominal = np.array([params["k1"], params["k2"], params["k3"], params["k4"]])
samples = np.array(params["stiffness_samples"])

keys = ["k1", "k2", "k3", "k4"]
labels = [
    r"$k_1$ (ombro)",
    r"$k_2$ (cotovelo)",
    r"$k_3$ (b\'iceps)",
    r"$k_4$ (punho)",
]
bounds = [params["stiffness_intervals"][k] for k in keys]
n = len(keys) - 1  # grid of the 6 pairs: rows k2..k4, columns k1..k3

fig, axes = plt.subplots(n, n, figsize=(6.2, 4.6))
for r in range(n):
    for c in range(n):
        ax = axes[r, c]
        if c > r:
            ax.axis("off")
            continue

        i, j = r + 1, c  # y and x parameter indices
        (lo_x, hi_x), (lo_y, hi_y) = bounds[j], bounds[i]

        # Sampling interval as a box around the samples
        ax.add_patch(
            plt.Rectangle(
                (lo_x, lo_y),
                hi_x - lo_x,
                hi_y - lo_y,
                fill=False,
                edgecolor=PETROL,
                linewidth=1.4,
            )
        )
        ax.scatter(
            samples[:, j], samples[:, i], s=16, color=PURPLE, alpha=0.7, linewidths=0
        )
        ax.scatter(
            nominal[j],
            nominal[i],
            s=190,
            marker="*",
            color=ORANGE,
            edgecolor=GRAPHITE,
            linewidths=0.8,
            zorder=5,
        )
        ax.set_xlim(lo_x - 0.1 * (hi_x - lo_x), hi_x + 0.1 * (hi_x - lo_x))
        ax.set_ylim(lo_y - 0.1 * (hi_y - lo_y), hi_y + 0.1 * (hi_y - lo_y))

        # Ticks at the interval limits and at the nominal value
        ax.set_xticks([lo_x, nominal[j], hi_x])
        ax.set_yticks([lo_y, nominal[i], hi_y])
        ax.tick_params(length=2, labelsize=11, colors=GRAPHITE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if r == n - 1:
            ax.set_xlabel(labels[j], color=GRAPHITE)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(labels[i], color=GRAPHITE)
        else:
            ax.set_yticklabels([])

# Legend in the empty upper triangle
handles = [
    plt.Line2D([], [], marker="o", ls="", color=PURPLE, alpha=0.7, ms=6,
               label=r"Amostras ($99$ execu\c{c}\~oes)"),
    plt.Line2D([], [], marker="*", ls="", color=ORANGE, mec=GRAPHITE, ms=15,
               label="Modelo nominal"),
    plt.Line2D([], [], marker="s", ls="", mfc="none", mec=PETROL, ms=10,
               label=r"Intervalo de amostragem"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.97),
           frameon=False, fontsize=13, labelcolor=GRAPHITE)

fig.align_ylabels(axes[:, 0])
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {OUT} with {len(samples)} samples")
