"""
Full experiment suite on the Berka dataset.

Experiments:
    E1  history-size curves, impostor protocol (primary)
    E2  history-size curves, parametric amount-shift protocol
    E3  temporal coverage at fixed transaction count
    E4  behavioural complexity descriptors per account

Usage:
    python -m experiments.run_all
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_injection import build_impostor_pool, inject_impostor
from src.data_loader import CohortSpec, account_summary, eligible_accounts, load_transactions
from src.evaluation import evaluate
from src.experiment_engine import (
    EVAL_WINDOW,
    HISTORY_SIZES,
    INJECTION_RATE,
    build_global_prior,
    run_history_size_experiment,
)
from src.feature_engineering import behavioural_descriptors, build_features, fit_scaler
from src.models import KNNDistance, ShrinkageMahalanobis

SEED = 20260920
N_EVAL = 300
N_DONORS = 400
OUT = "results/raw"


def _split_accounts(pool, n_eval, n_donors, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_eval + n_donors, replace=False)
    return chosen[:n_eval], chosen[n_eval:]


def e1_e2(trans):
    spec = CohortSpec(min_history=max(HISTORY_SIZES), eval_window=EVAL_WINDOW)
    pool = eligible_accounts(trans, spec)
    accounts, donors = _split_accounts(pool, N_EVAL, N_DONORS, SEED)
    meta = {
        "eligible_accounts": int(len(pool)),
        "min_transactions_required": spec.min_transactions,
        "n_evaluated": int(len(accounts)),
        "n_donors": int(len(donors)),
        "eval_window": EVAL_WINDOW,
        "injection_rate": INJECTION_RATE,
        "history_sizes": HISTORY_SIZES,
        "seed": SEED,
    }
    print("E1: impostor protocol", flush=True)
    r1 = run_history_size_experiment(
        trans, accounts, donors, protocol="impostor", seed=SEED
    )
    r1.to_csv(f"{OUT}/e1_impostor.csv", index=False)

    print("E2: parametric amount-shift protocol", flush=True)
    r2 = run_history_size_experiment(
        trans,
        accounts,
        donors,
        protocol="parametric",
        param_kind="amount",
        param_magnitude=5.0,
        seed=SEED,
    )
    r2.to_csv(f"{OUT}/e2_parametric_amount.csv", index=False)

    with open(f"{OUT}/e1_e2_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return accounts, donors


def e3_temporal_coverage(trans, accounts, donors):
    """
    Hold the training transaction count fixed at n=50 and vary how far back in
    time those 50 transactions are drawn from, by systematic stride sampling.

    stride 1: the 50 most recent transactions before the window (compact span)
    stride 2: every 2nd transaction going back (roughly double the span)
    stride 4: every 4th transaction going back (roughly quadruple the span)

    Transaction count is identical across arms; only temporal coverage and
    recency change. The arms are paired within account.
    """
    n_fixed, strides = 50, [1, 2, 4]
    need = n_fixed * max(strides) + EVAL_WINDOW
    counts = trans.groupby("account_id").size()
    usable = [a for a in accounts if counts.get(a, 0) >= need]
    print(f"E3: temporal coverage, {len(usable)} usable accounts", flush=True)

    prior = build_global_prior(trans, donors, 40, np.random.default_rng(SEED))
    by_account = {a: g for a, g in trans.groupby("account_id", sort=False)}
    rows = []

    for acc in usable:
        g = by_account[acc].reset_index(drop=True)
        rng = np.random.default_rng(SEED * 100003 + int(acc))
        window = g.iloc[-EVAL_WINDOW:].reset_index(drop=True)
        history_all = g.iloc[:-EVAL_WINDOW].reset_index(drop=True)

        pool = build_impostor_pool(trans, acc, donors, size=EVAL_WINDOW, rng=rng)
        inj = inject_impostor(window, pool, INJECTION_RATE, rng)
        if inj.labels.sum() == 0:
            continue
        X_eval_raw = build_features(inj.frame)

        for s in strides:
            idx = np.arange(len(history_all) - 1, -1, -s)[:n_fixed][::-1]
            hist = history_all.iloc[idx].reset_index(drop=True)
            if len(hist) < n_fixed:
                continue
            span = (
                pd.to_datetime(hist["date"]).max() - pd.to_datetime(hist["date"]).min()
            ).days

            X_hist_raw = build_features(hist)
            scaler = fit_scaler(X_hist_raw)
            X_hist, X_eval = scaler.transform(X_hist_raw), scaler.transform(X_eval_raw)

            for mname, model in [
                ("mahalanobis", ShrinkageMahalanobis(kappa=0.0)),
                ("knn", KNNDistance(k=5)),
            ]:
                model.fit(X_hist)
                met = evaluate(
                    inj.labels, model.score(X_eval), model.score(X_hist)
                )
                met.update(
                    account_id=int(acc),
                    stride=s,
                    span_days=int(span),
                    n_train=n_fixed,
                    model=mname,
                )
                rows.append(met)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e3_temporal_coverage.csv", index=False)
    print(f"E3 rows: {len(df)}", flush=True)


def e4_complexity(trans, accounts):
    """Behavioural descriptors computed on each account's training history."""
    by_account = {a: g for a, g in trans.groupby("account_id", sort=False)}
    rows = []
    for acc in accounts:
        g = by_account[acc].reset_index(drop=True)
        hist = g.iloc[:-EVAL_WINDOW]
        # Descriptors are computed on the 250 transactions preceding the window,
        # i.e. the largest training history any arm sees, so they describe the
        # same behaviour the models are asked to learn.
        d = behavioural_descriptors(hist.iloc[-250:])
        d["account_id"] = int(acc)
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e4_complexity.csv", index=False)
    print(f"E4 rows: {len(df)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    trans = load_transactions()
    account_summary(trans).to_csv(f"{OUT}/account_summary.csv", index=False)
    accounts, donors = e1_e2(trans)
    e3_temporal_coverage(trans, accounts, donors)
    e4_complexity(trans, accounts)
    print("ALL EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
