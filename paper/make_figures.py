"""Regenerate the paper's figures as column-width vector PDFs.

Reads only files under results/ (tables and raw simulation output) and writes
paper/figures/*.pdf. Run from anywhere:

    .venv/Scripts/python.exe paper/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
    }
)

# Okabe-Ito palette (colour-blind safe); marker and line style vary as well.
STYLE = {
    "mahalanobis": ("Mahalanobis", "#0072B2", "o", "-"),
    "knn": ("k-NN", "#009E73", "s", "-"),
    "autoencoder": ("Autoencoder", "#D55E00", "^", "-"),
    "iforest": ("Isolation Forest", "#CC79A7", "D", "-"),
    "gaussian_z": ("Gaussian z", "#555555", "v", "--"),
    "robust_z": ("Robust z", "#E69F00", "x", "--"),
}
W, H = 3.45, 2.45  # IEEE single-column width in inches


def curves(ax, df, order, band=True):
    for m in order:
        label, color, marker, ls = STYLE[m]
        g = df[df.model == m].sort_values("history_size")
        ax.plot(g.history_size, g["mean"], color=color, marker=marker, ls=ls, label=label)
        if band:
            ax.fill_between(g.history_size, g.ci_lo, g.ci_hi, color=color, alpha=0.13, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Personal training transactions $n$")
    ax.set_ylabel("PR-AUC")
    ax.grid(alpha=0.25, lw=0.4)


def finish(fig, ax, name, ncol=2, loc="upper left"):
    ax.legend(frameon=False, ncol=ncol, loc=loc, handlelength=1.8, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name)
    plt.close(fig)


order = ["mahalanobis", "knn", "autoencoder", "iforest", "gaussian_z", "robust_z"]

# Fig. 1: Berka data-efficiency curves.
berka = pd.read_csv(RES / "tables" / "e1_curves_personal.csv")
fig, ax = plt.subplots(figsize=(W, H))
curves(ax, berka, order)
ax.axhline(0.10, color="k", lw=0.6, ls=":")
ax.text(250, 0.105, "prevalence 0.10", fontsize=5.5, va="bottom", ha="right")
ax.set_ylim(0.05, 0.84)
finish(fig, ax, "fig_berka_curves.pdf")

# Fig. 2: personal vs global vs shrinkage hybrid, Mahalanobis, Berka.
gm = pd.read_csv(RES / "tables" / "e1_global_means.csv").set_index("model").pr_auc["mahalanobis"]
hyb = pd.read_csv(RES / "tables" / "e1_personal_vs_hybrid_shrinkage_mahalanobis.csv")
fig, ax = plt.subplots(figsize=(W, 2.1))
ax.plot(hyb.history_size, hyb.personal_mean, color="#0072B2", marker="o", label="Personal")
ax.plot(hyb.history_size, hyb.other_mean, color="#D55E00", marker="^", label="Shrinkage hybrid")
ax.axhline(gm, color="#009E73", ls="--", lw=1.1, label="Global (no personal data)")
ax.set_xscale("log")
ax.set_xlabel("Personal training transactions $n$")
ax.set_ylabel("PR-AUC")
ax.grid(alpha=0.25, lw=0.4)
finish(fig, ax, "fig_arms.pdf", ncol=1, loc="lower right")

# Fig. 3: TabFormer (synthetic) curves.
tab = pd.read_csv(RES / "tables" / "e7_curves_personal.csv")
tgm = pd.read_csv(RES / "tables" / "e7_global_means.csv").set_index("model").pr_auc["mahalanobis"]
fig, ax = plt.subplots(figsize=(W, H))
curves(ax, tab, order)
ax.axhline(tgm, color="#009E73", ls="-.", lw=0.9)
ax.text(1.05, tgm + 0.012, "global Mahalanobis", fontsize=5.5, color="#009E73", va="bottom")
ax.set_ylim(0.05, 0.9)
finish(fig, ax, "fig_tabformer_curves.pdf")

# Fig. 4: simulation, personal Mahalanobis by archetype (mean +/- 1.96 SEM over users).
sim = pd.read_csv(RES / "raw" / "b1_simulation.csv")
sim = sim[(sim.arm == "personal") & (sim.model == "mahalanobis")]
arch_style = {
    "regular": ("#0072B2", "o"),
    "moderate": ("#56B4E9", "s"),
    "burst": ("#E69F00", "^"),
    "seasonal": ("#009E73", "D"),
    "traveller": ("#D55E00", "v"),
    "variable": ("#7A1F1F", "x"),
}
fig, ax = plt.subplots(figsize=(W, H))
for arch, (color, marker) in arch_style.items():
    c = sim[sim.archetype == arch].groupby("history_size")["pr_auc"].agg(["mean", "sem"]).sort_index()
    ax.plot(c.index, c["mean"], color=color, marker=marker, label=arch.capitalize())
    ax.fill_between(c.index, c["mean"] - 1.96 * c["sem"], c["mean"] + 1.96 * c["sem"], color=color, alpha=0.12, lw=0)
ax.set_xscale("log")
ax.set_xlabel("Personal training transactions $n$")
ax.set_ylabel("PR-AUC")
ax.set_ylim(0.15, 1.08)
ax.grid(alpha=0.25, lw=0.4)
finish(fig, ax, "fig_sim_archetypes.pdf", ncol=3, loc="upper left")

print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
