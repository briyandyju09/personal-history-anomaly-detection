"""
Experiment B: controlled simulation.

B1  history-size curves per behavioural archetype
B2  detection by anomaly class and shift magnitude
B3  parameter-controlled behavioural complexity vs required history

Usage:
    python -m experiments.run_simulation
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate
from src.feature_engineering import fit_scaler
from src.models import IForest, KNNDistance, ShrinkageMahalanobis, TinyAutoencoder
from src.simulation import (
    ARCHETYPES,
    generate_normal,
    make_profile,
    observed_complexity,
    sim_features,
    simulate_user,
)
from src.stats import relative_threshold

SEED = 20260920
HISTORY_SIZES = [1, 5, 10, 25, 50, 100, 250, 500]
N_EVAL = 100
ANOMALY_RATE = 0.10
DELTA = 3.0
N_USERS_PER_ARCHETYPE = 40
OUT = "results/raw"


def build_global_pool(rng, n_users=120, n_each=60):
    """Population sample from simulated users disjoint from the evaluated set."""
    frames = []
    arch = list(ARCHETYPES)
    for u in range(10_000, 10_000 + n_users):
        p = make_profile(u, arch[u % len(arch)], rng)
        frames.append(generate_normal(p, n_each, rng))
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    gpool = build_global_pool(rng)
    Xg = sim_features(gpool, reference=gpool)
    gscaler = fit_scaler(Xg)
    Xg_s = gscaler.transform(Xg)
    global_models = {
        "mahalanobis": ShrinkageMahalanobis(kappa=0.0).fit(Xg_s),
        "knn": KNNDistance(k=5).fit(Xg_s[:1500]),
        "iforest": IForest(seed=SEED).fit(Xg_s),
        "autoencoder": TinyAutoencoder(seed=SEED).fit(Xg_s),
    }
    gtrain = {k: m.score(Xg_s) for k, m in global_models.items()}
    gprior_mu, gprior_cov = Xg_s.mean(axis=0), np.cov(Xg_s, rowvar=False, ddof=1)

    rows, comp_rows, cls_rows = [], [], []
    uid = 0
    for arch in ARCHETYPES:
        for _ in range(N_USERS_PER_ARCHETYPE):
            uid += 1
            r = np.random.default_rng(SEED * 7919 + uid)
            prof = make_profile(uid, arch, r)
            hist_full, win = simulate_user(
                prof, max(HISTORY_SIZES), N_EVAL, ANOMALY_RATE, DELTA, r
            )
            labels = win["is_anomaly"].to_numpy()
            comp = observed_complexity(hist_full)
            comp.update(user_id=uid, archetype=arch)
            comp_rows.append(comp)

            for n in HISTORY_SIZES:
                hist = hist_full.iloc[-n:].reset_index(drop=True)
                Xh = sim_features(hist, reference=hist)
                Xe = sim_features(win, reference=hist)
                sc = fit_scaler(Xh)
                Xh_s, Xe_s = sc.transform(Xh), sc.transform(Xe)

                models = {
                    "mahalanobis": ShrinkageMahalanobis(kappa=0.0),
                    "knn": KNNDistance(k=5),
                    "iforest": IForest(seed=SEED),
                    "autoencoder": TinyAutoencoder(seed=SEED),
                }
                for mn, m in models.items():
                    m.fit(Xh_s)
                    s_e, s_t = m.score(Xe_s), m.score(Xh_s)
                    met = evaluate(labels, s_e, s_t)
                    met.update(
                        user_id=uid, archetype=arch, history_size=n,
                        model=mn, arm="personal",
                        degenerate=bool(getattr(m, "degenerate", False)),
                    )
                    rows.append(met)

                    if mn == "mahalanobis":
                        # per-class detection: rank of each anomaly class
                        order = np.argsort(np.argsort(s_e)) / max(len(s_e) - 1, 1)
                        for kind in win.loc[labels == 1, "anomaly_type"].unique():
                            sel = (win["anomaly_type"] == kind).to_numpy()
                            cls_rows.append(
                                {
                                    "user_id": uid, "archetype": arch,
                                    "history_size": n, "anomaly_type": kind,
                                    "mean_score_percentile": float(order[sel].mean()),
                                    "n": int(sel.sum()),
                                }
                            )

                # global + hybrid arms
                Xe_g = gscaler.transform(sim_features(win, reference=gpool))
                for mn, gm in global_models.items():
                    met = evaluate(labels, gm.score(Xe_g), gtrain[mn])
                    met.update(
                        user_id=uid, archetype=arch, history_size=n,
                        model=mn, arm="global", degenerate=False,
                    )
                    rows.append(met)

                Xh_g = gscaler.transform(sim_features(hist, reference=gpool))
                hyb = ShrinkageMahalanobis(kappa=25.0)
                hyb.set_prior(gprior_mu, gprior_cov)
                hyb.fit(Xh_g)
                met = evaluate(labels, hyb.score(Xe_g), hyb.score(Xh_g))
                met.update(
                    user_id=uid, archetype=arch, history_size=n,
                    model="mahalanobis", arm="hybrid_shrinkage", degenerate=False,
                )
                rows.append(met)
        print(f"  archetype {arch} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/b1_simulation.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(f"{OUT}/b3_sim_complexity.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(f"{OUT}/b2_anomaly_classes.csv", index=False)

    print("\n=== B1 personal PR-AUC by archetype and history size (mahalanobis) ===")
    print(
        df[(df.arm == "personal") & (df.model == "mahalanobis")]
        .pivot_table(index="history_size", columns="archetype", values="pr_auc")
        .round(3)
    )

    print("\n=== B3 per-archetype relative-90 threshold (mahalanobis) ===")
    out = []
    for arch, g in df[
        (df.arm == "personal") & (df.model == "mahalanobis")
    ].groupby("archetype"):
        cur = g.groupby("history_size")["pr_auc"].mean().sort_index()
        r = relative_threshold(cur.index.to_numpy(), cur.to_numpy(), 0.90)
        out.append(
            {"archetype": arch, "threshold": r.get("threshold"), "best": r.get("reference")}
        )
    t = pd.DataFrame(out)
    t.to_csv("results/tables/b3_archetype_thresholds.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
