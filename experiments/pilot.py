"""
Pilot run. Validates the pipeline on a small account sample before committing
to the full experiment. Checks for leakage symptoms, degenerate regimes and
implausible metric values.

Usage:
    python -m experiments.pilot
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import CohortSpec, eligible_accounts, load_transactions
from src.experiment_engine import EVAL_WINDOW, HISTORY_SIZES, run_history_size_experiment

N_PILOT = 60
N_DONORS = 200
SEED = 20260920


def main():
    trans = load_transactions()
    spec = CohortSpec(min_history=max(HISTORY_SIZES), eval_window=EVAL_WINDOW)
    pool = eligible_accounts(trans, spec)
    print(f"eligible accounts (>= {spec.min_transactions} txns): {len(pool)}")

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(pool, size=N_PILOT + N_DONORS, replace=False)
    accounts, donors = chosen[:N_PILOT], chosen[N_PILOT:]
    assert len(set(accounts) & set(donors)) == 0, "evaluated and donor sets overlap"

    print(f"evaluating {len(accounts)} accounts, {len(donors)} donor accounts")
    res = run_history_size_experiment(
        trans, accounts, donors, protocol="impostor", seed=SEED, verbose=True
    )

    os.makedirs("results/raw", exist_ok=True)
    res.to_csv("results/raw/pilot_impostor.csv", index=False)
    print(f"\nrows: {len(res)}")

    piv = (
        res[res.arm == "personal"]
        .pivot_table(index="history_size", columns="model", values="pr_auc")
        .round(3)
    )
    print("\nPersonal arm, mean PR-AUC by history size:")
    print(piv)

    base = res.n_anomalies.mean() / res.window.mean()
    print(f"\nanomaly prevalence in eval window: {base:.3f}")

    glob = (
        res[res.arm == "global"]
        .groupby("model")["pr_auc"]
        .mean()
        .round(3)
    )
    print("\nGlobal arm mean PR-AUC (constant across n by construction):")
    print(glob)

    hyb = (
        res[res.arm.str.startswith("hybrid")]
        .pivot_table(index="history_size", columns="arm", values="pr_auc")
        .round(3)
    )
    print("\nHybrid arms, mean PR-AUC by history size:")
    print(hyb)

    print("\nDegenerate fit rate by model and history size (personal arm):")
    print(
        res[res.arm == "personal"]
        .pivot_table(index="history_size", columns="model", values="degenerate")
        .round(2)
    )

    # Sanity checks
    print("\n--- SANITY CHECKS ---")
    gvar = res[res.arm == "global"].groupby(["account_id", "model"])["pr_auc"].nunique()
    print(f"global arm varies with n (should be 1 everywhere): max={gvar.max()}")
    print(f"PR-AUC in [0,1]: {res.pr_auc.min():.3f} to {res.pr_auc.max():.3f}")
    perfect = (res.pr_auc > 0.999).mean()
    print(f"fraction of near-perfect PR-AUC (leakage smell if high): {perfect:.4f}")


if __name__ == "__main__":
    main()
