"""
Experiment engine for E7 (TabFormer card-structured replication).

Same split design and pairing logic as src/experiment_engine.py's
run_history_size_experiment: a fixed evaluation window per entity, training
history of size n immediately preceding it, no transaction ever both
trained on and tested on, the global model fitted only on donor entities'
early transactions, and the same shrinkage-hybrid mechanism (kappa=25) at
score and parameter level. Reuses src/models.py and src/evaluation.py
UNCHANGED -- both operate on plain np.ndarray and have no Berka-specific
assumptions.

Why a separate engine rather than parameterising experiment_engine.py
-----------------------------------------------------------------------
TabFormer's schema (User/Card/MCC/Merchant Name/Merchant State/Use Chip) is
structurally different from Berka's (account_id/operation/k_symbol), and the
novelty-based feature encoding in tabformer_features.py has no Berka
equivalent. Threading both schemas through one generic function would need
either a wide abstraction layer (risking subtle bugs in the already-verified
Berka path) or per-dataset callback plumbing that adds complexity without
real reuse, since the two feature-building and injection steps do not share
code today. This engine instead reuses everything that IS
dataset-independent (models.py, evaluation.py, feature_engineering.fit_scaler,
the split design itself) and reimplements only the dataset-specific glue,
which is the smaller and lower-risk surface. Documented as a deviation from
the original plan's "parameterise experiment_engine.py"; see
docs/DECISION_LOG.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import evaluate
from .feature_engineering import fit_scaler
from .models import (
    GaussianZ,
    IForest,
    KNNDistance,
    RobustZ,
    ScoreFusionHybrid,
    ShrinkageMahalanobis,
    TinyAutoencoder,
)
from .tabformer_features import build_features
from .tabformer_injection import build_impostor_pool, inject_impostor

HISTORY_SIZES = [1, 5, 10, 25, 50, 100, 250, 500]
EVAL_WINDOW = 60
INJECTION_RATE = 0.10
ALPHA = 0.05
HYBRID_KAPPA = 25.0


@dataclass
class GlobalPrior:
    mu: np.ndarray
    cov: np.ndarray
    scaler: object
    raw_matrix: np.ndarray
    donor_entities: np.ndarray


def build_global_prior(
    trans: pd.DataFrame,
    donor_entities: np.ndarray,
    per_entity: int,
    rng: np.random.Generator,
) -> GlobalPrior:
    chunks = []
    for eid, g in trans[trans["entity_id"].isin(donor_entities)].groupby(
        "entity_id", sort=True
    ):
        chunks.append(g.head(per_entity))
    pooled = pd.concat(chunks, ignore_index=True)
    X = build_features(pooled)
    scaler = fit_scaler(X)
    Xs = scaler.transform(X)
    return GlobalPrior(
        mu=Xs.mean(axis=0),
        cov=np.cov(Xs, rowvar=False, ddof=1),
        scaler=scaler,
        raw_matrix=X,
        donor_entities=donor_entities,
    )


def _fit_global_models(prior: GlobalPrior, seed: int, subsample: int = 4000):
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
    entities: np.ndarray,
    donor_entities: np.ndarray,
    history_sizes: list | None = None,
    eval_window: int = EVAL_WINDOW,
    injection_rate: float = INJECTION_RATE,
    seed: int = 0,
    global_per_entity: int = 40,
    verbose: bool = True,
) -> pd.DataFrame:
    """Returns one row per (entity, history size, model, arm), plus the
    fraud rate observed in that entity's evaluation window (Protocol F,
    exploratory; see tabformer_injection.py docstring)."""
    history_sizes = history_sizes or HISTORY_SIZES
    rng_master = np.random.default_rng(seed)

    prior = build_global_prior(trans, donor_entities, global_per_entity, rng_master)
    global_models, global_train = _fit_global_models(prior, seed)
    global_train_scores = {k: m.score(global_train) for k, m in global_models.items()}

    by_entity = {e: g for e, g in trans.groupby("entity_id", sort=False)}
    rows = []

    for ai, ent in enumerate(entities):
        g = by_entity[ent].reset_index(drop=True)
        # Stable integer seed derived from the entity string, same role as
        # Berka's `seed * 100003 + account_id`.
        ent_seed = seed * 100003 + (abs(hash(ent)) % 1_000_000_007)
        rng = np.random.default_rng(ent_seed)

        window = g.iloc[-eval_window:].reset_index(drop=True)
        history_all = g.iloc[:-eval_window].reset_index(drop=True)

        pool = build_impostor_pool(trans, ent, donor_entities, size=eval_window, rng=rng)
        inj = inject_impostor(window, pool, injection_rate, rng)
        labels = inj.labels
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        # Protocol F exploratory signal: genuine (non-injected) fraud rows
        # in this entity's own evaluation window, before injection overwrote
        # anything. Recorded but not used to select or weight anything.
        genuine_fraud = int(window["is_fraud"].sum())

        X_eval_raw = build_features(inj.frame)

        for n in history_sizes:
            if n > len(history_all):
                continue
            hist = history_all.iloc[-n:].reset_index(drop=True)
            X_hist_raw = build_features(hist)

            scaler = fit_scaler(X_hist_raw)
            X_hist = scaler.transform(X_hist_raw)
            X_eval = scaler.transform(X_eval_raw)
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
                    entity_id=ent, history_size=n, model=mname, arm="personal",
                    protocol=inj.protocol,
                    degenerate=bool(getattr(model, "degenerate", False)),
                    cost_ms=fit_ms, n_anomalies=int(labels.sum()), window=len(labels),
                    genuine_fraud_in_window=genuine_fraud,
                )
                rows.append(met)

            for mname, gm in global_models.items():
                t0 = time.perf_counter()
                s_eval = gm.score(X_eval_global)
                fit_ms = (time.perf_counter() - t0) * 1000.0
                met = evaluate(labels, s_eval, global_train_scores[mname], alpha=ALPHA)
                met.update(
                    entity_id=ent, history_size=n, model=mname, arm="global",
                    protocol=inj.protocol, degenerate=False, cost_ms=fit_ms,
                    n_anomalies=int(labels.sum()), window=len(labels),
                    genuine_fraud_in_window=genuine_fraud,
                )
                rows.append(met)

            t0 = time.perf_counter()
            hyb = ShrinkageMahalanobis(kappa=HYBRID_KAPPA)
            hyb.set_prior(prior.mu, prior.cov)
            X_hist_g = prior.scaler.transform(X_hist_raw)
            hyb.fit(X_hist_g)
            s_eval = hyb.score(X_eval_global)
            s_train = hyb.score(X_hist_g)
            fit_ms = (time.perf_counter() - t0) * 1000.0
            met = evaluate(labels, s_eval, s_train, alpha=ALPHA)
            met.update(
                entity_id=ent, history_size=n, model="mahalanobis", arm="hybrid_shrinkage",
                protocol=inj.protocol, degenerate=False, cost_ms=fit_ms,
                n_anomalies=int(labels.sum()), window=len(labels),
                genuine_fraud_in_window=genuine_fraud,
            )
            rows.append(met)

            t0 = time.perf_counter()
            fusion = ScoreFusionHybrid(
                personal=KNNDistance(k=5), global_model=global_models["knn"], kappa=HYBRID_KAPPA
            )
            fusion.fit(X_hist)
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
                entity_id=ent, history_size=n, model="knn", arm="hybrid_fusion",
                protocol=inj.protocol, degenerate=False, cost_ms=fit_ms,
                n_anomalies=int(labels.sum()), window=len(labels),
                genuine_fraud_in_window=genuine_fraud,
            )
            rows.append(met)

        if verbose and (ai + 1) % 50 == 0:
            print(f"  {ai + 1}/{len(entities)} entities", flush=True)

    return pd.DataFrame(rows)
