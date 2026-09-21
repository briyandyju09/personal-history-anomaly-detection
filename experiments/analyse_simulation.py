"""Analysis and figures for Experiment B (controlled simulation)."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stats import fit_saturation, holm_correct, relative_threshold

RAW, FIG, TAB = "results/raw", "results/figures", "results/tables"
COLORS = {
    "regular": "#1b4965",
    "moderate": "#5fa8d3",
    "seasonal": "#62b6cb",
    "burst": "#d69f7e",
    "traveller": "#c1666b",
    "variable": "#8a3033",
}


def main():
    df = pd.read_csv(f"{RAW}/b1_simulation.csv")
    comp = pd.read_csv(f"{RAW}/b3_sim_complexity.csv")
    cls = pd.read_csv(f"{RAW}/b2_anomaly_classes.csv")

    sub = df[(df.arm == "personal") & (df.model == "mahalanobis")]

    # ---- Figure: curves by archetype ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch, g in sub.groupby("archetype"):
        c = g.groupby("history_size")["pr_auc"].agg(["mean", "sem"]).sort_index()
        ax.plot(c.index, c["mean"], marker="o", ms=4, color=COLORS[arch], label=arch)
        ax.fill_between(
            c.index,
            c["mean"] - 1.96 * c["sem"],
            c["mean"] + 1.96 * c["sem"],
            color=COLORS[arch],
            alpha=0.15,
        )
    ax.set_xscale("log")
    ax.set_xlabel("personal training transactions (log scale)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Simulation: data efficiency by behavioural archetype")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_simulation_archetypes.png", dpi=160)
    plt.close(fig)

    # ---- Saturation fits per archetype ----
    fits = []
    for arch, g in sub.groupby("archetype"):
        c = g.groupby("history_size")["pr_auc"].mean().sort_index()
        f = fit_saturation(c.index.to_numpy(), c.to_numpy())
        f["archetype"] = arch
        fits.append(f)
    pd.DataFrame(fits).to_csv(f"{TAB}/b1_saturation_by_archetype.csv", index=False)

    # ---- Per-user threshold vs measured complexity ----
    rows = []
    for uid, g in sub.groupby("user_id"):
        c = g.groupby("history_size")["pr_auc"].mean().sort_index()
        r = relative_threshold(c.index.to_numpy(), c.to_numpy(), 0.90)
        rows.append(
            {"user_id": uid, "threshold": r.get("threshold"), "best": r.get("reference")}
        )
    thr = pd.DataFrame(rows).merge(comp, on="user_id")
    thr.to_csv(f"{TAB}/b3_per_user_thresholds.csv", index=False)

    descriptors = [
        "log_amount_sd",
        "merchant_entropy",
        "category_entropy",
        "location_entropy",
        "gap_cv",
        "n_unique_merchants",
    ]
    out, pv = [], {}
    for d in descriptors:
        ok = thr[["threshold", d, "best"]].dropna()
        rho, p = sps.spearmanr(ok[d], ok["threshold"])
        rb, pb = sps.spearmanr(ok[d], ok["best"])
        pv[d] = p
        out.append(
            {
                "descriptor": d,
                "rho_vs_threshold": float(rho),
                "p_vs_threshold": float(p),
                "rho_vs_best": float(rb),
                "p_vs_best": float(pb),
                "n": int(len(ok)),
            }
        )
    res = pd.DataFrame(out)
    res["p_holm"] = res.descriptor.map(holm_correct(pv))
    res.to_csv(f"{TAB}/b3_complexity_correlations.csv", index=False)
    print("=== B3 complexity vs threshold (simulation) ===")
    print(res.round(4).to_string(index=False))
    print(f"\nusers with undefined threshold: {thr.threshold.isna().sum()} of {len(thr)}")

    # ---- Anomaly class detectability ----
    piv = cls.pivot_table(
        index="history_size", columns="anomaly_type", values="mean_score_percentile"
    )
    piv.round(3).to_csv(f"{TAB}/b2_anomaly_class_detectability.csv")
    print("\n=== B2 mean score percentile by anomaly class (1.0 = ranked most anomalous) ===")
    print(piv.round(3))

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in piv.columns:
        ax.plot(piv.index, piv[col], marker="o", ms=4, label=col)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("personal training transactions")
    ax.set_ylabel("mean anomaly-score percentile")
    ax.set_title("Simulation: detectability by anomaly class (Mahalanobis)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_anomaly_classes.png", dpi=160)
    plt.close(fig)

    # ---- Arms figure ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, c in [
        ("personal", "#1b4965"),
        ("global", "#2a9d8f"),
        ("hybrid_shrinkage", "#e07a5f"),
    ]:
        g = df[(df.model == "mahalanobis") & (df.arm == arm)]
        s = g.groupby("history_size")["pr_auc"].mean().sort_index()
        ax.plot(s.index, s.values, marker="o", ms=4, color=c, label=arm.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("personal training transactions")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Simulation: personal, global and hybrid arms")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_simulation_arms.png", dpi=160)
    plt.close(fig)

    print("\n=== Arms (simulation, mahalanobis) ===")
    print(
        df[df.model == "mahalanobis"]
        .pivot_table(index="history_size", columns="arm", values="pr_auc")
        .round(3)
    )


if __name__ == "__main__":
    main()
