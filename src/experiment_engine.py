"""
Experiment engine for the history-size curves on real data.

Design
------
For each eligible account we fix a single evaluation window: the last W
transactions of that account's chronological history. Training history for
size n is the n transactions immediately preceding that window.

    [ .... earlier history .... | n training txns | W evaluation txns ]
                                 <--------------->  <--------------->
                                     varies              fixed

Fixing the evaluation window across all n has two consequences that matter.
First, no transaction is ever both trained on and tested on, at any n. Second,
every history size is scored on exactly the same held-out transactions for a
given account, so comparisons across n are paired at the account level and the
across-account variation in window difficulty cancels in paired tests.

The global model is fitted on training transactions pooled from accounts that
are disjoint from the evaluated accounts, so population information never
includes the evaluated account's own future or past.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomaly_injection import build_impostor_pool, inject_impostor, inject_parametric
from .evaluation import evaluate
from .feature_engineering import FEATURE_NAMES, build_features, fit_scaler
from .models import (
    GaussianZ,
    IForest,
    KNNDistance,
    RobustZ,
    ScoreFusionHybrid,
    ShrinkageMahalanobis,
    TinyAutoencoder,
)

HISTORY_SIZES = [1, 5, 10, 25, 50, 100, 250]
EVAL_WINDOW = 60
INJECTION_RATE = 0.10
ALPHA = 0.05
HYBRID_KAPPA = 25.0


@dataclass
class GlobalPrior:
    """Population-level information fitted on donor accounts only."""

    mu: np.ndarray
    cov: np.ndarray
    scaler: object
    raw_matrix: np.ndarray
    donor_accounts: np.ndarray


def build_global_prior(
    trans: pd.DataFrame,
    donor_accounts: np.ndarray,
    per_account: int,
    rng: np.random.Generator,
) -> GlobalPrior:
    """
    Pool the EARLY transactions of donor accounts into a population sample.

    We take each donor's earliest ``per_account`` transactions so the pooled
    sample cannot contain anything from the period the evaluated accounts are
    tested on, and so the global model is not advantaged by seeing later data.
    """
    chunks = []
    for acc, g in trans[trans["account_id"].isin(donor_accounts)].groupby(
        "account_id", sort=True
    ):
        chunks.append(g.head(per_account))
    pooled = pd.concat(chunks, ignore_index=True)
    X = build_features(pooled)
    scaler = fit_scaler(X)
    Xs = scaler.transform(X)
    return GlobalPrior(
        mu=Xs.mean(axis=0),
        cov=np.cov(Xs, rowvar=False, ddof=1),
        scaler=scaler,
        raw_matrix=X,
        donor_accounts=donor_accounts,
    )


def _fit_global_models(prior: GlobalPrior, seed: int, subsample: int = 4000):
    """Fit the population-level detectors once and reuse them for all accounts."""
    Xs = prior.scaler.transform(prior.raw_matrix)
    rng = np.random.default_rng(seed)
    if len(Xs) > subsample:
        Xs = Xs[rng.choice(len(Xs), size=subsample, replace=False)]
    models = {
        "gaussian_z": GaussianZ(0).fit(Xs),
        "robust_z": RobustZ(0).fit(Xs),
        "mahalanobis": ShrinkageMahalanobis(kappa=0.0).fit(Xs),
        "knn": KNNDistance(k=5).fit(Xs[: min(len(Xs), 1500)]),
        "iforest": IForest(seed=seed).fit(Xs),
        "autoencoder": TinyAutoencoder(seed=seed).fit(Xs),
    }
    return models, Xs


def run_history_size_experiment(
    trans: pd.DataFrame,
    accounts: np.ndarray,
    donor_accounts: np.ndarray,
    protocol: str = "impostor",
    history_sizes: list | None = None,
    eval_window: int = EVAL_WINDOW,
    injection_rate: float = INJECTION_RATE,
    seed: int = 0,
    param_kind: str = "amount",
    param_magnitude: float = 5.0,
    global_per_account: int = 40,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full personal / global / hybrid comparison across history sizes.

    Returns one row per (account, history size, model, arm).
    """
    history_sizes = history_sizes or HISTORY_SIZES
    rng_master = np.random.default_rng(seed)

    prior = build_global_prior(trans, donor_accounts, global_per_account, rng_master)
    global_models, global_train = _fit_global_models(prior, seed)
    global_train_scores = {
        k: m.score(global_train) for k, m in global_models.items()
    }

    by_account = {a: g for a, g in trans.groupby("account_id", sort=False)}
    rows = []

    for ai, acc in enumerate(accounts):
        g = by_account[acc].reset_index(drop=True)
        rng = np.random.default_rng(seed * 100003 + int(acc))

        window = g.iloc[-eval_window:].reset_index(drop=True)
        history_all = g.iloc[:-eval_window].reset_index(drop=True)

        # ---- inject anomalies once per account, shared across all n ----
        if protocol == "impostor":
            pool = build_impostor_pool(
                trans, acc, donor_accounts, size=eval_window, rng=rng
            )
            inj = inject_impostor(window, pool, injection_rate, rng)
        else:
            inj = inject_parametric(
                window,
                injection_rate,
                rng,
                kind=param_kind,
                magnitude=param_magnitude,
            )
        labels = inj.labels
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        X_eval_raw = build_features(inj.frame)

        for n in history_sizes:
            if n > len(history_all):
                continue
            hist = history_all.iloc[-n:].reset_index(drop=True)
            X_hist_raw = build_features(hist)

            scaler = fit_scaler(X_hist_raw)
            X_hist = scaler.transform(X_hist_raw)
            X_eval = scaler.transform(X_eval_raw)

            # Global arm uses the global scaler so it is not handed the
            # personal scale as a hidden form of personalisation.
            X_eval_global = prior.scaler.transform(X_eval_raw)

            personal_models = {
                "gaussian_z": GaussianZ(0),
                "robust_z": RobustZ(0),
                "mahalanobis": ShrinkageMahalanobis(kappa=0.0),
                "knn": KNNDistance(k=5),
                "iforest": IForest(seed=seed),
                "autoencoder": TinyAutoencoder(seed=seed),
            }

            for mname, model in personal_models.items():
                t0 = time.perf_counter()
                model.fit(X_hist)
                s_eval = model.score(X_eval)
                s_train = model.score(X_hist)
                fit_ms = (time.perf_counter() - t0) * 1000.0

                met = evaluate(labels, s_eval, s_train, alpha=ALPHA)
                met.update(
                    account_id=int(acc),
                    history_size=n,
                    model=mname,
                    arm="personal",
                    protocol=inj.protocol,
                    degenerate=bool(getattr(model, "degenerate", False)),
                    cost_ms=fit_ms,
                    n_anomalies=int(labels.sum()),
                    window=len(labels),
                )
                rows.append(met)

            # ---- global arm (independent of n, recorded at each n for pairing)
            for mname, gm in global_models.items():
                t0 = time.perf_counter()
                s_eval = gm.score(X_eval_global)
                fit_ms = (time.perf_counter() - t0) * 1000.0
                met = evaluate(
                    labels, s_eval, global_train_scores[mname], alpha=ALPHA
                )
                met.update(
                    account_id=int(acc),
                    history_size=n,
                    model=mname,
                    arm="global",
                    protocol=inj.protocol,
                    degenerate=False,
                    cost_ms=fit_ms,
                    n_anomalies=int(labels.sum()),
                    window=len(labels),
                )
                rows.append(met)

            # ---- hybrid arm: shrinkage Mahalanobis ----
            t0 = time.perf_counter()
            hyb = ShrinkageMahalanobis(kappa=HYBRID_KAPPA)
            hyb.set_prior(prior.mu, prior.cov)
            # The prior lives in global-scaler space, so the hybrid is fitted
            # and scored in that same space for internal consistency.
            X_hist_g = prior.scaler.transform(X_hist_raw)
            hyb.fit(X_hist_g)
            s_eval = hyb.score(X_eval_global)
            s_train = hyb.score(X_hist_g)
            fit_ms = (time.perf_counter() - t0) * 1000.0
            met = evaluate(labels, s_eval, s_train, alpha=ALPHA)
            met.update(
                account_id=int(acc),
                history_size=n,
                model="mahalanobis",
                arm="hybrid_shrinkage",
                protocol=inj.protocol,
                degenerate=False,
                cost_ms=fit_ms,
                n_anomalies=int(labels.sum()),
                window=len(labels),
            )
            rows.append(met)

            # ---- hybrid arm: score fusion over kNN ----
            t0 = time.perf_counter()
            fusion = ScoreFusionHybrid(
                personal=KNNDistance(k=5),
                global_model=global_models["knn"],
                kappa=HYBRID_KAPPA,
            )
            fusion.fit(X_hist)
            # Fusion scores the same rows through both members; the personal
            # member sees personal-scaled data, the global member global-scaled.
            sp = fusion._to_rank(fusion.personal.score(X_eval))
            sg = fusion._to_rank(global_models["knn"].score(X_eval_global))
            w = n / (n + HYBRID_KAPPA)
            s_eval = w * sp + (1 - w) * sg
            spt = fusion._to_rank(fusion.personal.score(X_hist))
            sgt = fusion._to_rank(global_models["knn"].score(X_hist_g))
            s_train = w * spt + (1 - w) * sgt
            fit_ms = (time.perf_counter() - t0) * 1000.0
            met = evaluate(labels, s_eval, s_train, alpha=ALPHA)
            met.update(
                account_id=int(acc),
                history_size=n,
                model="knn",
                arm="hybrid_fusion",
                protocol=inj.protocol,
                degenerate=False,
                cost_ms=fit_ms,
                n_anomalies=int(labels.sum()),
                window=len(labels),
            )
            rows.append(met)

        if verbose and (ai + 1) % 50 == 0:
            print(f"  {ai + 1}/{len(accounts)} accounts", flush=True)

    return pd.DataFrame(rows)
