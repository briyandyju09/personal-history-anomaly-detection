"""
E6: direct measurement of behavioural drift.

The reference paper (PAPER.md.reference Sec. 7.6, 8.2) *interprets* two
observations -- the negative 100->250 marginal PR-AUC gain in Sec. 7.1, and
the cost of reaching further back in Sec. 7.6 -- as consequences of
behavioural drift ("transactions from three years ago describe a person who
no longer exists"), but explicitly states this was never measured directly
("We did not measure drift directly, so this is an interpretation, not a
finding", Sec. 8.2). E6 measures it.

Design
------
For each of the same 300 evaluated accounts (identical cohort draw, seed
20260920, as experiments/run_all.py's e1_e2 -- deterministic re-derivation,
not a re-run of the detector experiment):

  recent_block = the 100 transactions immediately preceding the evaluation
                 window (exactly what the n=100 history condition trains on)
  delta_block  = the 150 transactions immediately before THAT (positions
                 101-250 back) -- exactly the transactions that are added
                 when history size grows from 100 to 250, and nothing else

Both blocks are converted to the same causal feature representation used by
every detector (src.feature_engineering.build_features), standardised by a
scaler FITTED ON recent_block ONLY (so "drift" is measured in the coordinate
system of the account's current behaviour, not an arbitrary global one).
Drift is the Mahalanobis distance between the two blocks' mean feature
vectors, using recent_block's own ridge-regularised covariance -- i.e. "how
many of my own recent standard deviations away is my older behaviour",
which is dimensionally the same quantity the Mahalanobis detector itself
uses, but computed here purely descriptively: no detector score, no
anomaly label, no circularity with the main evaluation.

This directly tests, rather than assumes, whether the size of an account's
drift predicts the direction and size of its own 100->250 marginal gain
(read from the already-produced results/raw/e1_impostor.csv, personal arm,
Mahalanobis and kNN). Spearman correlation, Holm-corrected across the two
models, matching the RQ4 methodology in src/stats.py / feature_engineering.

A secondary lag-sweep (drift as a function of how far back the comparison
block sits) is also computed, to show whether drift accumulates gradually
or is concentrated at some particular depth.

Usage:
    python -m experiments.e6_drift
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import CohortSpec, eligible_accounts, load_transactions
from src.experiment_engine import EVAL_WINDOW, HISTORY_SIZES
from src.feature_engineering import FEATURE_NAMES, build_features, fit_scaler
from src.stats import holm_correct

SEED = 20260920
N_EVAL = 300
N_DONORS = 400
RECENT_N = 100
DELTA_N = 150  # positions 101-250 back: exactly what n=250 adds over n=100
LAG_BLOCK = 25
LAG_OFFSETS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
OUT = "results/raw"
TAB = "results/tables"


def _split_accounts(pool, n_eval, n_donors, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_eval + n_donors, replace=False)
    return chosen[:n_eval], chosen[n_eval:]


def _ridge_cov(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    n, d = X.shape
    S = np.cov(X, rowvar=False, ddof=1) if n > 1 else np.zeros((d, d))
    S = S + ridge * np.eye(d) * max(1.0, float(np.trace(S)) / max(d, 1))
    return S


def mahalanobis_drift(block_a: np.ndarray, block_b_ref: np.ndarray) -> float:
    """
    Mahalanobis distance between mean(block_a) and mean(block_b_ref), using
    block_b_ref's own ridge-regularised covariance. block_b_ref is the
    "current behaviour" reference; block_a is the block being compared
    against it.
    """
    mu_ref = block_b_ref.mean(axis=0)
    cov_ref = _ridge_cov(block_b_ref)
    prec = np.linalg.pinv(cov_ref)
    mu_a = block_a.mean(axis=0)
    delta = mu_a - mu_ref
    return float(np.sqrt(max(delta @ prec @ delta, 0.0)))


def main():
    trans = load_transactions()
    spec = CohortSpec(min_history=max(HISTORY_SIZES), eval_window=EVAL_WINDOW)
    pool = eligible_accounts(trans, spec)
    accounts, _donors = _split_accounts(pool, N_EVAL, N_DONORS, SEED)
    by_account = {a: g for a, g in trans.groupby("account_id", sort=False)}

    drift_rows = []
    lag_rows = []
    per_feature_shift = []

    for acc in accounts:
        g = by_account[int(acc)].reset_index(drop=True)
        history_all = g.iloc[:-EVAL_WINDOW].reset_index(drop=True)
        H = len(history_all)
        if H < RECENT_N + DELTA_N:
            continue

        recent_block = history_all.iloc[-RECENT_N:].reset_index(drop=True)
        delta_block = history_all.iloc[-(RECENT_N + DELTA_N):-RECENT_N].reset_index(drop=True)

        X_recent_raw = build_features(recent_block)
        X_delta_raw = build_features(delta_block)
        scaler = fit_scaler(X_recent_raw)
        X_recent = scaler.transform(X_recent_raw)
        X_delta = scaler.transform(X_delta_raw)

        drift = mahalanobis_drift(X_delta, X_recent)
        drift_rows.append({"account_id": int(acc), "drift_100_250": drift})

        # per-feature standardised mean shift (diagnostic, not a test)
        shift = X_delta.mean(axis=0) - X_recent.mean(axis=0)
        per_feature_shift.append(shift)

        # lag sweep: compare 25-txn blocks at increasing distance back
        # against the most-recent 25-txn block, all within the SAME
        # recent_block scaling for comparability across lags.
        ref25_raw = build_features(history_all.iloc[-LAG_BLOCK:].reset_index(drop=True))
        ref25 = scaler.transform(ref25_raw)
        for L in LAG_OFFSETS:
            start = H - LAG_BLOCK - L
            end = H - L
            if start < 0:
                continue
            blk = history_all.iloc[start:end].reset_index(drop=True)
            if len(blk) < LAG_BLOCK:
                continue
            Xb = scaler.transform(build_features(blk))
            d = mahalanobis_drift(Xb, ref25) if L > 0 else 0.0
            lag_rows.append({"account_id": int(acc), "lag": L, "drift": d})

    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(f"{OUT}/e6_drift_per_account.csv", index=False)

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(f"{OUT}/e6_drift_lag_sweep.csv", index=False)

    shift_arr = np.array(per_feature_shift)
    shift_summary = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_standardised_shift": shift_arr.mean(axis=0),
        "mean_abs_standardised_shift": np.abs(shift_arr).mean(axis=0),
    }).sort_values("mean_abs_standardised_shift", ascending=False)
    shift_summary.to_csv(f"{TAB}/e6_per_feature_shift.csv", index=False)

    # --- join with e1_impostor.csv to test drift vs. 100->250 marginal gain ---
    e1_path = f"{OUT}/e1_impostor.csv"
    if not os.path.exists(e1_path):
        print(f"WARNING: {e1_path} not found; skipping drift-vs-gain "
              "correlation (run experiments.run_all first). Drift measures "
              "themselves are still saved.", flush=True)
        print(drift_df.describe().round(3))
        return

    e1 = pd.read_csv(e1_path)
    corr_rows = []
    pvals = {}
    for model in ["mahalanobis", "knn"]:
        piv = (
            e1[(e1.arm == "personal") & (e1.model == model)]
            .pivot_table(index="account_id", columns="history_size", values="pr_auc")
        )
        if 100 not in piv.columns or 250 not in piv.columns:
            continue
        gain = (piv[250] - piv[100]).rename("gain_100_250")
        merged = drift_df.set_index("account_id").join(gain, how="inner").dropna()
        if len(merged) < 10:
            continue
        rho, p = scipy_stats.spearmanr(merged["drift_100_250"], merged["gain_100_250"])
        corr_rows.append({
            "model": model, "n": len(merged), "spearman_rho": rho, "p_raw": p,
            "mean_gain": merged["gain_100_250"].mean(),
            "mean_drift": merged["drift_100_250"].mean(),
        })
        pvals[model] = p

    corr_df = pd.DataFrame(corr_rows)
    if len(corr_df):
        adj = holm_correct(pvals)
        corr_df["p_holm"] = corr_df["model"].map(adj)
    corr_df.to_csv(f"{TAB}/e6_drift_vs_gain_correlation.csv", index=False)

    print("=== E6: drift (100->250 delta block vs. recent-100 reference) ===")
    print(drift_df["drift_100_250"].describe().round(3))
    print("\n=== E6: drift vs. 100->250 marginal PR-AUC gain, per account ===")
    print(corr_df.round(4).to_string(index=False))
    print("\n=== E6: per-feature mean standardised shift (top 8) ===")
    print(shift_summary.head(8).round(3).to_string(index=False))
    print("\n=== E6: lag sweep, mean drift across accounts ===")
    print(lag_df.groupby("lag")["drift"].agg(["mean", "std", "count"]).round(3))
    print("\nE6 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
