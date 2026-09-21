"""
Analysis and figures for Experiment E7 (TabFormer, SYNTHETIC card-structured
replication layer). Mirrors experiments/analyse.py's structure applied to
results/raw/e7_tabformer_impostor.csv. Results here are never merged with
the Berka (real-data) tables produced by experiments/analyse.py, and every
figure/table is titled to say "TabFormer / synthetic" explicitly.

Usage:
    python -m experiments.analyse_tabformer
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stats import (
    bootstrap_mean_ci,
    fit_saturation,
    holm_correct,
    marginal_gains,
    paired_bootstrap_diff,
    relative_threshold,
    wilcoxon_paired,
)

RAW, FIG, TAB = "results/raw", "results/figures", "results/tables"
METRIC = "pr_auc"
PALETTE = {
    "mahalanobis": "#1b4965", "knn": "#5fa8d3", "iforest": "#c1666b",
    "autoencoder": "#d69f7e", "gaussian_z": "#8d99ae", "robust_z": "#bcc3d0",
}


def curve_table(df, arm, metric=METRIC):
    rows = []
    sub = df[df.arm == arm]
    for (n, m), g in sub.groupby(["history_size", "model"]):
        mean, lo, hi, k = bootstrap_mean_ci(g[metric].to_numpy(), seed=7)
        rows.append({"history_size": n, "model": m, "mean": mean, "ci_lo": lo,
                      "ci_hi": hi, "n_entities": k,
                      "sd": float(np.nanstd(g[metric].to_numpy(), ddof=1))})
    return pd.DataFrame(rows).sort_values(["model", "history_size"])


def main():
    path = f"{RAW}/e7_tabformer_impostor.csv"
    if not os.path.exists(path):
        print(f"{path} not found; run experiments.run_tabformer first.")
        return
    df = pd.read_csv(path)
    print(f"E7 TabFormer (SYNTHETIC): {len(df)} rows, "
          f"{df.entity_id.nunique()} entities", flush=True)

    curves = curve_table(df, "personal")
    curves.to_csv(f"{TAB}/e7_curves_personal.csv", index=False)
    gmeans = df[df.arm == "global"].groupby("model")[METRIC].mean()
    gmeans.to_csv(f"{TAB}/e7_global_means.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in curves.groupby("model"):
        g = g.sort_values("history_size")
        c = PALETTE.get(model, "#444444")
        ax.plot(g.history_size, g["mean"], marker="o", ms=4, color=c, label=model)
        ax.fill_between(g.history_size, g.ci_lo, g.ci_hi, color=c, alpha=0.15)
    ax.axhline(gmeans.get("mahalanobis", np.nan), ls="--", lw=1.2, color="#2a9d8f",
               label="global Mahalanobis (no personal data)")
    ax.set_xscale("log")
    ax.set_xlabel("personal training transactions (log scale)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("E7: data efficiency on TabFormer (SYNTHETIC card data)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_tabformer_data_efficiency.png", dpi=160)
    plt.close(fig)

    # Saturation + marginal gains
    sat, marg = [], []
    for model, g in curves.groupby("model"):
        g = g.sort_values("history_size")
        f = fit_saturation(g.history_size.to_numpy(), g["mean"].to_numpy())
        f["model"] = model
        sat.append(f)
        for row in marginal_gains(g.history_size.to_numpy(), g["mean"].to_numpy()):
            row["model"] = model
            marg.append(row)
    pd.DataFrame(sat).to_csv(f"{TAB}/e7_saturation_fits.csv", index=False)
    pd.DataFrame(marg).to_csv(f"{TAB}/e7_marginal_gains.csv", index=False)

    # Thresholds
    rows = []
    for model, g in curves.groupby("model"):
        g = g.sort_values("history_size")
        r90 = relative_threshold(g.history_size.to_numpy(), g["mean"].to_numpy(), 0.90)
        rows.append({"model": model, "best_observed": r90.get("reference"),
                      "rel90_threshold": r90.get("threshold")})
    pd.DataFrame(rows).to_csv(f"{TAB}/e7_thresholds.csv", index=False)

    # Personal vs global, Mahalanobis (paired over entities)
    pvals = {}
    comp_rows = []
    for n, g in df[df.model == "mahalanobis"].groupby("history_size"):
        p = g[g.arm == "personal"].set_index("entity_id")[METRIC]
        o = g[g.arm == "global"].set_index("entity_id")[METRIC]
        common = p.index.intersection(o.index)
        a, b = p.loc[common].to_numpy(), o.loc[common].to_numpy()
        boot = paired_bootstrap_diff(a, b, seed=11)
        w = wilcoxon_paired(a, b)
        pvals[n] = w["p"]
        comp_rows.append({"history_size": n, "personal_mean": float(np.nanmean(a)),
                           "global_mean": float(np.nanmean(b)), "paired_diff": boot["diff"],
                           "ci_lo": boot["lo"], "ci_hi": boot["hi"],
                           "wilcoxon_p": w["p"], "effect": w["effect"], "n_pairs": boot["n"]})
    comp = pd.DataFrame(comp_rows)
    comp["p_holm"] = comp.history_size.map(holm_correct(pvals))
    comp.to_csv(f"{TAB}/e7_personal_vs_global_mahalanobis.csv", index=False)

    # Cost
    cost = (df[df.arm == "personal"].groupby(["history_size", "model"])["cost_ms"]
            .median().unstack())
    cost.to_csv(f"{TAB}/e7_cost.csv")

    # Isolation Forest vs Mahalanobis at small n -- the notable divergence
    # from Berka (where IForest is worst at every size). Report explicitly.
    iforest_curve = curves[curves.model == "iforest"].set_index("history_size")["mean"]
    maha_curve = curves[curves.model == "mahalanobis"].set_index("history_size")["mean"]
    iforest_better_at = sorted(set(iforest_curve.index) &
                                set(iforest_curve[iforest_curve > maha_curve].index))

    print("\n=== E7 personal arm, mean PR-AUC ===")
    print(curves.pivot_table(index="history_size", columns="model", values="mean").round(3))
    print(f"\nHistory sizes where Isolation Forest personal-arm PR-AUC EXCEEDS "
          f"Mahalanobis (TabFormer only; never happens on Berka): {iforest_better_at}")
    print("\n=== E7 thresholds ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n=== E7 personal vs global (Mahalanobis), paired ===")
    print(comp.round(4).to_string(index=False))

    summary = {
        "n_entities": int(df.entity_id.nunique()),
        "iforest_beats_mahalanobis_at_n": iforest_better_at,
    }
    with open(f"{TAB}/e7_analysis_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nE7 analysis complete", flush=True)


if __name__ == "__main__":
    main()
