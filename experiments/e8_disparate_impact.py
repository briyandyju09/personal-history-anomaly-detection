"""
E8: disparate-impact audit.

PAPER.md.reference Sec. 10 argues that sparse-history users are
disproportionately drawn from populations already at risk of financial
exclusion, and that a personal model without a history floor concentrates
false positives on them -- but the reference paper's own ethics section
states this without measuring it against any demographic axis ("We did not
audit for disparate impact, and no system of this kind should be deployed
without such an audit", Sec. 10). E8 runs that audit on the one demographic
signal Berka actually provides: each account's district.

Design
------
Berka's `account` table gives each account a `district_id`; `district`
(district.asc / A1-A16) gives, per district: region, population,
urbanisation ratio, average salary, unemployment rate ('95, '96) and
committed-crime counts (column mapping independently confirmed against two
descriptions of the published schema, not re-derived from the raw file
alone -- see docs/DECISION_LOG.md). This is district-of-account
demographic/economic context, NOT individual-level demographics (Berka
carries no age, gender, income or ethnicity field), so this is a coarse,
ecological audit, not an individual-level fairness audit. That limitation is
stated plainly in the results, not glossed over.

For each of the 300 evaluated accounts we take the operating-point false
positive rate (personal arm, Mahalanobis, training-quantile threshold --
exactly the FPR figures already in results/raw/e1_impostor.csv, Sec. 7.2)
at two history sizes: n=10 (the "flags almost everything" regime) and n=50
(the paper's own "tolerable" threshold), and test whether FPR varies with
district-level average salary, unemployment rate and urbanisation.

Usage:
    python -m experiments.e8_disparate_impact
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stats import bootstrap_mean_ci, holm_correct

OUT = "results/raw"
TAB = "results/tables"

DISTRICT_COLS = [
    "district_id", "district_name", "region", "n_inhabitants",
    "n_muni_lt499", "n_muni_500_1999", "n_muni_2000_9999", "n_muni_gt10000",
    "n_cities", "urban_ratio", "avg_salary", "unemployment_95",
    "unemployment_96", "entrepreneurs_per_1000", "crimes_95", "crimes_96",
]
AUDIT_HISTORY_SIZES = [10, 50]
AUDIT_MODEL = "mahalanobis"


def load_district_table() -> pd.DataFrame:
    df = pd.read_csv("data/raw/district.csv", sep=";", header=0, names=DISTRICT_COLS)
    for c in ["unemployment_95", "unemployment_96"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_account_district() -> pd.DataFrame:
    acc = pd.read_csv(
        "data/raw/account.csv", sep=";",
        names=["account_id", "district_id", "frequency", "date"], header=0,
    )
    return acc[["account_id", "district_id"]]


def tercile(s: pd.Series) -> pd.Series:
    return pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop")


def main():
    e1_path = f"{OUT}/e1_impostor.csv"
    if not os.path.exists(e1_path):
        print(f"ERROR: {e1_path} not found. Run experiments.run_all first.")
        sys.exit(1)
    e1 = pd.read_csv(e1_path)

    district = load_district_table()
    acc_dist = load_account_district()
    linked = acc_dist.merge(district, on="district_id", how="left")

    all_results = []
    corr_rows = []
    for n in AUDIT_HISTORY_SIZES:
        sub = e1[
            (e1.arm == "personal") & (e1.model == AUDIT_MODEL) & (e1.history_size == n)
        ][["account_id", "fpr", "precision", "recall", "flag_rate"]]
        merged = sub.merge(linked, on="account_id", how="left")
        merged = merged.dropna(subset=["fpr", "avg_salary", "unemployment_96", "urban_ratio"])
        print(f"\n=== E8 @ n={n}: {len(merged)} accounts with district linkage ===")

        # By region
        by_region = (
            merged.groupby("region")["fpr"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
        by_region["history_size"] = n
        print("\nFPR by region:")
        print(by_region.round(3))
        by_region.to_csv(f"{TAB}/e8_fpr_by_region_n{n}.csv")

        kw = scipy_stats.kruskal(
            *[g["fpr"].to_numpy() for _, g in merged.groupby("region") if len(g) >= 3]
        )
        print(f"Kruskal-Wallis across regions: H={kw.statistic:.3f}, p={kw.pvalue:.4f}")

        # Continuous district variables: Spearman correlation with FPR
        pvals = {}
        for var in ["avg_salary", "unemployment_96", "urban_ratio"]:
            rho, p = scipy_stats.spearmanr(merged[var], merged["fpr"])
            pvals[var] = p
            corr_rows.append({
                "history_size": n, "variable": var, "spearman_rho": rho,
                "p_raw": p, "n": len(merged),
            })
        adj = holm_correct(pvals)
        for row in corr_rows:
            if row["history_size"] == n:
                row["p_holm"] = adj[row["variable"]]

        # Tercile comparison (low vs high) for interpretability
        for var in ["avg_salary", "unemployment_96", "urban_ratio"]:
            merged[f"{var}_tercile"] = tercile(merged[var])
            g = merged.groupby(f"{var}_tercile", observed=True)["fpr"]
            means = g.apply(lambda s: bootstrap_mean_ci(s.to_numpy(), seed=7))
            for lvl, (mean, lo, hi, nn) in means.items():
                all_results.append({
                    "history_size": n, "variable": var, "tercile": lvl,
                    "mean_fpr": mean, "ci_lo": lo, "ci_hi": hi, "n": nn,
                })

    tercile_df = pd.DataFrame(all_results)
    tercile_df.to_csv(f"{TAB}/e8_fpr_by_district_tercile.csv", index=False)
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(f"{TAB}/e8_fpr_district_correlation.csv", index=False)

    print("\n=== E8: FPR by district-variable tercile ===")
    print(tercile_df.round(4).to_string(index=False))
    print("\n=== E8: Spearman correlation, FPR vs. district variables (Holm-corrected) ===")
    print(corr_df.round(4).to_string(index=False))
    print(
        "\nNOTE: this is an ECOLOGICAL (district-level) audit, not an "
        "individual-level fairness audit -- Berka carries no per-person "
        "demographic field. See docs/DECISION_LOG.md and the revised "
        "PAPER.md Sec. 10 for the limitation this implies.",
        flush=True,
    )
    print("\nE8 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
