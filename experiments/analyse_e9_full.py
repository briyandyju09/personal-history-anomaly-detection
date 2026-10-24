"""
Summaries of the full E9 run (results/raw/e9_loo_full.csv).

Writes
  results/tables/e9_full_summary.csv   mean metrics by (arm, model, calib, n), pooled
                                       over seeds, plus the across-seed SD of the
                                       per-seed means (cohort-selection stability)
  results/tables/e9_full_paired.csv    paired in-sample minus LOO FPR / flag rate per
                                       (arm, model, n); Wilcoxon over account-seed rows
Usage (cwd = research/):  python -m experiments.analyse_e9_full

Nothing here defines a "usable operating point" criterion: that would be a
post-hoc choice. The tables report mean FPR, flag rate, recall and precision
against the nominal alpha = 0.05 and leave the reading to the paper text.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SRC = "results/raw/e9_loo_full.csv"
METRICS = ["flag_rate", "fpr", "recall", "precision", "f1", "pr_auc"]


def main():
    df = pd.read_csv(SRC)
    keys = ["arm", "model", "calib", "history_size"]

    per_seed = df.groupby(keys + ["cohort_seed"])[METRICS].mean().reset_index()
    pooled = df.groupby(keys)[METRICS].mean()
    sd = per_seed.groupby(keys)[METRICS].std().add_suffix("_seed_sd")
    summ = pooled.join(sd).reset_index()
    summ["n_account_seed_rows"] = df.groupby(keys).size().to_numpy()
    summ.round(4).to_csv("results/tables/e9_full_summary.csv", index=False)

    ins = df[df.calib == "insample"].set_index(
        ["arm", "model", "history_size", "cohort_seed", "account_id"])
    loo = df[df.calib == "loo"].set_index(ins.index.names)
    rows = []
    for (arm, model, n), _ in ins.groupby(level=[0, 1, 2]):
        if n < 2:
            continue
        a = ins.loc[(arm, model, n)]
        b = loo.loc[(arm, model, n)]
        for c in ["fpr", "flag_rate"]:
            d = (a[c] - b[c]).dropna()
            p = wilcoxon(d).pvalue if (d != 0).any() else float("nan")
            rows.append(dict(arm=arm, model=model, history_size=n, metric=c,
                             mean_insample=a[c].mean(), mean_loo=b[c].mean(),
                             mean_diff=d.mean(), wilcoxon_p=p, n_pairs=len(d)))
    pd.DataFrame(rows).round(6).to_csv("results/tables/e9_full_paired.csv", index=False)

    show = summ[(summ.history_size.isin([5, 10, 25, 50, 100, 250]))]
    for model in ["gaussian_z", "robust_z", "mahalanobis", "knn", "iforest", "autoencoder"]:
        s = show[(show.arm == "personal") & (show.model == model)]
        print(f"\n== {model} (personal)")
        print(s.pivot(index="history_size", columns="calib",
                      values=["flag_rate", "fpr", "recall"]).round(3).to_string())
    s = show[(show.arm == "hybrid_shrinkage")]
    print("\n== hybrid_shrinkage (mahalanobis)")
    print(s.pivot(index="history_size", columns="calib",
                  values=["flag_rate", "fpr", "recall"]).round(3).to_string())
    m = summ[(summ.arm == "personal") & (summ.model == "mahalanobis") & (summ.calib == "loo")]
    print("\nMahalanobis LOO: across-seed SD of per-seed mean FPR")
    print(m[["history_size", "fpr", "fpr_seed_sd", "flag_rate_seed_sd"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
