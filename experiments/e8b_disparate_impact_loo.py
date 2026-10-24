"""
E8b: E8 disparate-impact audit repeated with leave-one-out calibrated FPR.

E8 used the in-sample-calibrated FPR from E1. E9 showed that threshold is
biased low for small n (at n=10 it flags ~96% of everything for every
account), so the n=10 audit ran against a ceiling. E8b repeats the audit on
the E9 leave-one-out FPR at n in {25, 50, 100}.

Usage (cwd = research/):
    python -m experiments.e9_loo_calibration     # first, writes the input
    python -m experiments.e8b_disparate_impact_loo
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.e8_disparate_impact import load_account_district, load_district_table
from src.stats import holm_correct

SIZES = [25, 50, 100]
VARS = ["avg_salary", "unemployment_96", "urban_ratio"]


def main():
    e9 = pd.read_csv("results/raw/e9_loo_calibration.csv")
    linked = load_account_district().merge(load_district_table(), on="district_id", how="left")
    rows = []
    for n in SIZES:
        sub = e9[(e9.model == "mahalanobis") & (e9.calib == "loo") & (e9.n == n)]
        m = sub[["account_id", "fpr"]].merge(linked, on="account_id", how="left").dropna(
            subset=["fpr"] + VARS
        )
        kw = scipy_stats.kruskal(
            *[g["fpr"].to_numpy() for _, g in m.groupby("region") if len(g) >= 3]
        )
        pv, rhos = {}, {}
        for v in VARS:
            rho, p = scipy_stats.spearmanr(m[v], m["fpr"])
            pv[v], rhos[v] = p, rho
        adj = holm_correct(pv)
        for v in VARS:
            rows.append(dict(history_size=n, variable=v, spearman_rho=rhos[v],
                             p_raw=pv[v], p_holm=adj[v], n=len(m),
                             kruskal_region_H=kw.statistic, kruskal_region_p=kw.pvalue,
                             mean_fpr=m["fpr"].mean()))
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/e8b_fpr_district_correlation_loo.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
