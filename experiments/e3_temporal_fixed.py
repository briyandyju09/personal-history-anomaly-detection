"""
E3 (corrected): transaction count held fixed, temporal coverage varied.

Flaw in the first version
-------------------------
Temporal coverage was widened by stride-sampling the history. Stride sampling
changes the inter-arrival gap feature: under stride 2 the gaps computed on the
thinned sequence are roughly twice the account's true gaps, so the training
distribution of that feature no longer matched the evaluation window's. Part of
the measured degradation was therefore a feature artefact rather than an effect
of temporal coverage.

Correction
----------
Gap features for selected transactions are recomputed against the account's
TRUE full timeline, that is, each selected transaction's real gap from its
actual immediate predecessor in the unthinned history. Stride then changes only
which transactions are used and how far back they reach, not what any
individual transaction's features mean.

Control arm
-----------
A "recent-thinned" control holds both the count (50) and the span roughly at
the stride-1 span while still thinning, by stride-sampling within a recent
block. Comparing it against stride 1 isolates the cost of thinning alone;
comparing the reach-back arms against it isolates the cost of reaching further
back. Without this control, thinning and reach-back remain confounded.

Usage:
    python -m experiments.e3_temporal_fixed
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_injection import build_impostor_pool, inject_impostor
from src.data_loader import CohortSpec, eligible_accounts, load_transactions
from src.evaluation import evaluate
from src.experiment_engine import EVAL_WINDOW, INJECTION_RATE
from src.feature_engineering import build_features, fit_scaler
from src.models import KNNDistance, ShrinkageMahalanobis
from src.stats import paired_bootstrap_diff, wilcoxon_paired

SEED = 20260920
N_FIXED = 50
STRIDES = [1, 2, 4]
OUT = "results/raw"
TAB = "results/tables"


def features_true_timeline(history_all: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """
    Build features for the transactions at positions ``idx`` of the full
    history, with the gap feature taken from the true full timeline.
    """
    sel = history_all.iloc[idx].reset_index(drop=True)
    X = build_features(sel)

    dates = pd.to_datetime(history_all["date"]).values.astype(
        "datetime64[D]"
    ).astype(np.int64)
    true_gap = np.empty(len(idx), dtype=float)
    for j, p in enumerate(idx):
        true_gap[j] = float(dates[p] - dates[p - 1]) if p > 0 else np.nan
    med = np.nanmedian(true_gap) if np.isfinite(true_gap).any() else 1.0
    true_gap = np.where(np.isnan(true_gap), med, true_gap)
    X[:, 6] = np.log1p(np.clip(true_gap, 0.0, None))  # log_gap_days column
    return X


def main():
    trans = load_transactions()
    spec = CohortSpec(min_history=N_FIXED * max(STRIDES), eval_window=EVAL_WINDOW)
    pool = eligible_accounts(trans, spec)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(pool, size=min(700, len(pool)), replace=False)
    accounts, donors = chosen[:300], chosen[300:]
    print(f"eligible {len(pool)}, evaluating {len(accounts)}", flush=True)

    by_account = {a: g for a, g in trans.groupby("account_id", sort=False)}
    rows = []

    for acc in accounts:
        g = by_account[acc].reset_index(drop=True)
        r = np.random.default_rng(SEED * 100003 + int(acc))
        window = g.iloc[-EVAL_WINDOW:].reset_index(drop=True)
        history_all = g.iloc[:-EVAL_WINDOW].reset_index(drop=True)

        p = build_impostor_pool(trans, acc, donors, size=EVAL_WINDOW, rng=r)
        inj = inject_impostor(window, p, INJECTION_RATE, r)
        if inj.labels.sum() == 0:
            continue
        X_eval_raw = build_features(inj.frame)
        H = len(history_all)

        arms = {}
        for s in STRIDES:
            idx = np.arange(H - 1, -1, -s)[:N_FIXED][::-1]
            if len(idx) < N_FIXED:
                continue
            arms[f"stride{s}"] = idx
        # Control: thinned by 4 but confined to the most recent block, so the
        # span stays close to the stride-1 span.
        recent_block = np.arange(H - N_FIXED * 4, H)
        if recent_block[0] >= 0:
            arms["thin4_recent"] = recent_block[::4][:N_FIXED]

        for arm, idx in arms.items():
            if len(idx) < N_FIXED:
                continue
            dsel = pd.to_datetime(history_all.iloc[idx]["date"])
            span = (dsel.max() - dsel.min()).days
            recency = (
                pd.to_datetime(history_all["date"]).iloc[-1] - dsel.min()
            ).days

            X_hist_raw = features_true_timeline(history_all, idx)
            sc = fit_scaler(X_hist_raw)
            X_hist, X_eval = sc.transform(X_hist_raw), sc.transform(X_eval_raw)

            for mname, model in [
                ("mahalanobis", ShrinkageMahalanobis(kappa=0.0)),
                ("knn", KNNDistance(k=5)),
            ]:
                model.fit(X_hist)
                met = evaluate(inj.labels, model.score(X_eval), model.score(X_hist))
                met.update(
                    account_id=int(acc),
                    arm=arm,
                    span_days=int(span),
                    oldest_lookback_days=int(recency),
                    n_train=N_FIXED,
                    model=mname,
                )
                rows.append(met)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e3_temporal_fixed.csv", index=False)

    agg = (
        df.groupby(["model", "arm"])
        .agg(
            pr_auc=("pr_auc", "mean"),
            span_days=("span_days", "median"),
            lookback=("oldest_lookback_days", "median"),
            n=("account_id", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(f"{TAB}/e3_temporal_fixed.csv", index=False)
    print(agg.round(4).to_string(index=False))

    comp = []
    for model in df.model.unique():
        sub = df[df.model == model]
        for ref, arm in [
            ("stride1", "stride2"),
            ("stride1", "stride4"),
            ("stride1", "thin4_recent"),
            ("thin4_recent", "stride4"),
        ]:
            a = sub[sub.arm == arm].set_index("account_id")["pr_auc"]
            b = sub[sub.arm == ref].set_index("account_id")["pr_auc"]
            c = a.index.intersection(b.index)
            if len(c) < 20:
                continue
            bt = paired_bootstrap_diff(a.loc[c].to_numpy(), b.loc[c].to_numpy(), seed=13)
            w = wilcoxon_paired(a.loc[c].to_numpy(), b.loc[c].to_numpy())
            comp.append(
                {
                    "model": model,
                    "arm": arm,
                    "reference": ref,
                    "diff": bt["diff"],
                    "ci_lo": bt["lo"],
                    "ci_hi": bt["hi"],
                    "p": w["p"],
                    "effect": w["effect"],
                    "n": bt["n"],
                }
            )
    cdf = pd.DataFrame(comp)
    cdf.to_csv(f"{TAB}/e3_temporal_fixed_paired.csv", index=False)
    print()
    print(cdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
