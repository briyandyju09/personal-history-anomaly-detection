"""
Analysis of experimental output: data-efficiency curves, personalisation
thresholds, paired comparisons, diminishing returns and figures.

Usage:
    python -m experiments.analyse
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
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stats import (
    bootstrap_mean_ci,
    fit_saturation,
    holm_correct,
    marginal_gains,
    paired_bootstrap_diff,
    relative_threshold,
    stability_threshold,
    wilcoxon_paired,
)

RAW = "results/raw"
FIG = "results/figures"
TAB = "results/tables"
METRIC = "pr_auc"
PALETTE = {
    "mahalanobis": "#1b4965",
    "knn": "#5fa8d3",
    "iforest": "#c1666b",
    "autoencoder": "#d69f7e",
    "gaussian_z": "#8d99ae",
    "robust_z": "#bcc3d0",
}


def curve_table(df: pd.DataFrame, arm: str, metric: str = METRIC) -> pd.DataFrame:
    """Mean and bootstrap CI of a metric by history size and model."""
    rows = []
    sub = df[df.arm == arm]
    for (n, m), g in sub.groupby(["history_size", "model"]):
        mean, lo, hi, k = bootstrap_mean_ci(g[metric].to_numpy(), seed=7)
        rows.append(
            {
                "history_size": n,
                "model": m,
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_accounts": k,
                "sd": float(np.nanstd(g[metric].to_numpy(), ddof=1)),
                "degenerate_rate": float(g["degenerate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "history_size"])


def thresholds_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Apply the three threshold definitions to each model's curve."""
    rows = []
    for model, g in curves.groupby("model"):
        g = g.sort_values("history_size")
        sizes = g.history_size.to_numpy()
        vals = g["mean"].to_numpy()
        spreads = g["sd"].to_numpy()
        rel = relative_threshold(sizes, vals, 0.90)
        rel95 = relative_threshold(sizes, vals, 0.95)
        stab = stability_threshold(sizes, spreads, float(np.nanmedian(spreads)))
        rows.append(
            {
                "model": model,
                "best_observed": rel.get("reference"),
                "rel90_threshold": rel.get("threshold"),
                "rel95_threshold": rel95.get("threshold"),
                "stability_threshold": stab.get("threshold"),
                "stability_target_sd": stab.get("max_spread"),
            }
        )
    return pd.DataFrame(rows)


def personal_vs_other(df: pd.DataFrame, model: str, other_arm: str) -> pd.DataFrame:
    """Paired comparison of the personal arm against another arm, by size."""
    rows, pvals = [], {}
    for n, g in df[df.model == model].groupby("history_size"):
        p = g[g.arm == "personal"].set_index("account_id")[METRIC]
        o = g[g.arm == other_arm].set_index("account_id")[METRIC]
        common = p.index.intersection(o.index)
        a, b = p.loc[common].to_numpy(), o.loc[common].to_numpy()
        boot = paired_bootstrap_diff(a, b, seed=11)
        w = wilcoxon_paired(a, b)
        pvals[n] = w["p"]
        rows.append(
            {
                "history_size": n,
                "personal_mean": float(np.nanmean(a)),
                "other_mean": float(np.nanmean(b)),
                "paired_diff": boot["diff"],
                "ci_lo": boot["lo"],
                "ci_hi": boot["hi"],
                "wilcoxon_p": w["p"],
                "effect_rank_biserial": w["effect"],
                "n_pairs": boot["n"],
            }
        )
    out = pd.DataFrame(rows)
    adj = holm_correct(pvals)
    out["p_holm"] = out.history_size.map(adj)
    return out


def fig_curves(curves: pd.DataFrame, global_means: pd.Series, path: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in curves.groupby("model"):
        g = g.sort_values("history_size")
        c = PALETTE.get(model, "#444444")
        ax.plot(g.history_size, g["mean"], marker="o", ms=4, color=c, label=model)
        ax.fill_between(g.history_size, g.ci_lo, g.ci_hi, color=c, alpha=0.15)
    if global_means is not None and "knn" in global_means:
        ax.axhline(
            global_means["knn"],
            ls="--",
            lw=1.2,
            color="#2a9d8f",
            label="global kNN (no personal data)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("personal training transactions (log scale)")
    ax.set_ylabel("PR-AUC")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_arms(df: pd.DataFrame, path: str):
    """Personal vs global vs hybrid for the two strongest model families."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    specs = [
        ("mahalanobis", ["personal", "global", "hybrid_shrinkage"], "Mahalanobis"),
        ("knn", ["personal", "global", "hybrid_fusion"], "k-NN distance"),
    ]
    colors = {
        "personal": "#1b4965",
        "global": "#2a9d8f",
        "hybrid_shrinkage": "#e07a5f",
        "hybrid_fusion": "#e07a5f",
    }
    for ax, (model, arms, label) in zip(axes, specs):
        for arm in arms:
            sub = df[(df.model == model) & (df.arm == arm)]
            if sub.empty:
                continue
            g = sub.groupby("history_size")[METRIC].mean().sort_index()
            ax.plot(
                g.index,
                g.values,
                marker="o",
                ms=4,
                color=colors[arm],
                label=arm.replace("_", " "),
            )
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel("personal training transactions")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("PR-AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def complexity_analysis(e1: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """
    Per-account personalisation threshold versus behavioural descriptors.

    The per-account threshold is the smallest evaluated history size at which
    that account's PR-AUC reaches 90% of its own best observed value and stays
    there. Accounts whose curve never stabilises receive NaN and are excluded
    from the correlation, which is recorded.
    """
    sub = e1[(e1.arm == "personal") & (e1.model == "mahalanobis")]
    rows = []
    for acc, g in sub.groupby("account_id"):
        g = g.sort_values("history_size")
        r = relative_threshold(g.history_size.to_numpy(), g[METRIC].to_numpy(), 0.90)
        rows.append(
            {
                "account_id": acc,
                "threshold": r.get("threshold"),
                "best": r.get("reference"),
            }
        )
    thr = pd.DataFrame(rows).merge(comp, on="account_id", how="inner")

    descriptors = [
        "amount_cv",
        "amount_rcv",
        "log_amount_iqr",
        "operation_entropy",
        "ksymbol_entropy",
        "operation_diversity",
        "gap_cv",
        "credit_share",
    ]
    out, pvals = [], {}
    for d in descriptors:
        ok = thr[["threshold", d, "best"]].dropna()
        if len(ok) < 20:
            continue
        rho, p = sps.spearmanr(ok[d], ok["threshold"])
        rho_b, p_b = sps.spearmanr(ok[d], ok["best"])
        pvals[d] = p
        out.append(
            {
                "descriptor": d,
                "spearman_rho_vs_threshold": float(rho),
                "p_vs_threshold": float(p),
                "spearman_rho_vs_best_prauc": float(rho_b),
                "p_vs_best_prauc": float(p_b),
                "n": int(len(ok)),
            }
        )
    res = pd.DataFrame(out)
    adj = holm_correct(pvals)
    res["p_holm_vs_threshold"] = res.descriptor.map(adj)
    thr.to_csv(f"{TAB}/per_account_thresholds.csv", index=False)
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    summary = {}

    e1 = pd.read_csv(f"{RAW}/e1_impostor.csv")
    curves = curve_table(e1, "personal")
    curves.to_csv(f"{TAB}/e1_curves_personal.csv", index=False)

    gmeans = e1[e1.arm == "global"].groupby("model")[METRIC].mean()
    gmeans.to_csv(f"{TAB}/e1_global_means.csv")
    fig_curves(
        curves,
        gmeans,
        f"{FIG}/fig1_data_efficiency_impostor.png",
        "Data efficiency, impostor protocol (Berka, real histories)",
    )
    fig_arms(e1, f"{FIG}/fig2_arms.png")

    thr = thresholds_table(curves)
    thr.to_csv(f"{TAB}/e1_thresholds.csv", index=False)

    # Saturation fits and marginal gains
    sat, marg = [], []
    for model, g in curves.groupby("model"):
        g = g.sort_values("history_size")
        f = fit_saturation(g.history_size.to_numpy(), g["mean"].to_numpy())
        f["model"] = model
        sat.append(f)
        for row in marginal_gains(g.history_size.to_numpy(), g["mean"].to_numpy()):
            row["model"] = model
            marg.append(row)
    pd.DataFrame(sat).to_csv(f"{TAB}/e1_saturation_fits.csv", index=False)
    pd.DataFrame(marg).to_csv(f"{TAB}/e1_marginal_gains.csv", index=False)

    # Arm comparisons
    for model, other in [
        ("mahalanobis", "global"),
        ("mahalanobis", "hybrid_shrinkage"),
        ("knn", "global"),
        ("knn", "hybrid_fusion"),
    ]:
        t = personal_vs_other(e1, model, other)
        t.to_csv(f"{TAB}/e1_personal_vs_{other}_{model}.csv", index=False)

    # Parametric protocol
    if os.path.exists(f"{RAW}/e2_parametric_amount.csv"):
        e2 = pd.read_csv(f"{RAW}/e2_parametric_amount.csv")
        c2 = curve_table(e2, "personal")
        c2.to_csv(f"{TAB}/e2_curves_personal.csv", index=False)
        fig_curves(
            c2,
            e2[e2.arm == "global"].groupby("model")[METRIC].mean(),
            f"{FIG}/fig3_data_efficiency_parametric.png",
            "Data efficiency, parametric amount-shift protocol",
        )
        thresholds_table(c2).to_csv(f"{TAB}/e2_thresholds.csv", index=False)

    # Temporal coverage
    if os.path.exists(f"{RAW}/e3_temporal_coverage.csv"):
        e3 = pd.read_csv(f"{RAW}/e3_temporal_coverage.csv")
        agg = (
            e3.groupby(["model", "stride"])
            .agg(
                pr_auc=("pr_auc", "mean"),
                span_days=("span_days", "median"),
                n=("account_id", "nunique"),
            )
            .reset_index()
        )
        agg.to_csv(f"{TAB}/e3_temporal_coverage.csv", index=False)
        rows = []
        for model in e3.model.unique():
            sub = e3[e3.model == model]
            base = sub[sub.stride == 1].set_index("account_id")["pr_auc"]
            for s in sorted(sub.stride.unique()):
                if s == 1:
                    continue
                arm = sub[sub.stride == s].set_index("account_id")["pr_auc"]
                common = base.index.intersection(arm.index)
                b = paired_bootstrap_diff(
                    arm.loc[common].to_numpy(), base.loc[common].to_numpy(), seed=13
                )
                w = wilcoxon_paired(
                    arm.loc[common].to_numpy(), base.loc[common].to_numpy()
                )
                rows.append(
                    {
                        "model": model,
                        "stride": s,
                        "diff_vs_stride1": b["diff"],
                        "ci_lo": b["lo"],
                        "ci_hi": b["hi"],
                        "p": w["p"],
                        "effect": w["effect"],
                        "n": b["n"],
                    }
                )
        pd.DataFrame(rows).to_csv(f"{TAB}/e3_temporal_paired.csv", index=False)

    # Behavioural complexity
    if os.path.exists(f"{RAW}/e4_complexity.csv"):
        comp = pd.read_csv(f"{RAW}/e4_complexity.csv")
        ca = complexity_analysis(e1, comp)
        ca.to_csv(f"{TAB}/e4_complexity_correlations.csv", index=False)
        summary["complexity_rows"] = int(len(ca))

    summary["n_accounts_e1"] = int(e1.account_id.nunique())
    with open(f"{TAB}/analysis_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("analysis complete")


if __name__ == "__main__":
    main()
