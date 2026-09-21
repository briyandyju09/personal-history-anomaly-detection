"""
E5: multi-seed replication of the primary curves.

Closes the reference paper's own top-listed limitation (PAPER.md.reference
Sec. 9, item 13: "Single seed for cohort selection... the first item of
future work") and Appendix A.5's "Multi-seed repetition of cohort selection
(single seed used)".

Re-runs, at 5 independent seeds each with its own account/donor cohort draw
(Berka) or its own per-user profile draws (simulation):

  - E1 (Protocol I, impostor substitution, Berka, N_EVAL=300/N_DONORS=400,
    identical design to experiments/run_all.py's e1_e2, impostor arm only)
  - Experiment B1 (controlled simulation, identical design to
    experiments/run_simulation.py's main loop)

Scope decision, documented rather than silently applied: E2 (parametric
protocol) is NOT replicated across seeds here. E1 is the paper's primary
protocol and the one every headline number in Sec. 7.1-7.5 is drawn from;
E2 exists as the Sec. 7.7 robustness check against a different anomaly
definition, already run once. Five-seed E2 replication is recorded as
"not yet evaluated" (see the revised Appendix A.5) rather than run under
time pressure and reported thinly.

Usage:
    python -m experiments.e5_multiseed
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import CohortSpec, eligible_accounts, load_transactions
from src.evaluation import evaluate
from src.experiment_engine import EVAL_WINDOW, HISTORY_SIZES, run_history_size_experiment
from src.feature_engineering import fit_scaler
from src.models import IForest, KNNDistance, ShrinkageMahalanobis, TinyAutoencoder
from src.simulation import ARCHETYPES, generate_normal, make_profile, sim_features, simulate_user
from src.stats import relative_threshold

MULTISEED_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
N_EVAL_BERKA = 300
N_DONORS_BERKA = 400
OUT = "results/raw"
TAB = "results/tables"

# Simulation parameters, identical to experiments/run_simulation.py
SIM_HISTORY_SIZES = [1, 5, 10, 25, 50, 100, 250, 500]
SIM_N_EVAL = 100
SIM_ANOMALY_RATE = 0.10
SIM_DELTA = 3.0
SIM_N_USERS_PER_ARCHETYPE = 40


def _split_accounts(pool, n_eval, n_donors, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_eval + n_donors, replace=False)
    return chosen[:n_eval], chosen[n_eval:]


def run_e1_multiseed(trans):
    spec = CohortSpec(min_history=max(HISTORY_SIZES), eval_window=EVAL_WINDOW)
    pool = eligible_accounts(trans, spec)
    all_rows = []
    for seed in MULTISEED_SEEDS:
        t0 = time.time()
        accounts, donors = _split_accounts(pool, N_EVAL_BERKA, N_DONORS_BERKA, seed)
        print(f"[E5-E1] seed={seed}: {len(accounts)} accounts, "
              f"{len(donors)} donors", flush=True)
        r = run_history_size_experiment(
            trans, accounts, donors, protocol="impostor", seed=seed, verbose=False
        )
        r["cohort_seed"] = seed
        all_rows.append(r)
        print(f"[E5-E1] seed={seed} done in {time.time()-t0:.0f}s, "
              f"{len(r)} rows", flush=True)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(f"{OUT}/e5_multiseed_e1.csv", index=False)
    return df


def build_global_pool(rng, n_users=120, n_each=60):
    frames = []
    arch = list(ARCHETYPES)
    for u in range(10_000, 10_000 + n_users):
        p = make_profile(u, arch[u % len(arch)], rng)
        frames.append(generate_normal(p, n_each, rng))
    return pd.concat(frames, ignore_index=True)


def run_simulation_multiseed():
    all_rows = []
    for seed in MULTISEED_SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        gpool = build_global_pool(rng)
        Xg = sim_features(gpool, reference=gpool)
        gscaler = fit_scaler(Xg)
        Xg_s = gscaler.transform(Xg)
        global_models = {
            "mahalanobis": ShrinkageMahalanobis(kappa=0.0).fit(Xg_s),
            "knn": KNNDistance(k=5).fit(Xg_s[:1500]),
        }
        gtrain = {k: m.score(Xg_s) for k, m in global_models.items()}

        rows = []
        uid = 0
        for arch in ARCHETYPES:
            for _ in range(SIM_N_USERS_PER_ARCHETYPE):
                uid += 1
                r = np.random.default_rng(seed * 7919 + uid)
                prof = make_profile(uid, arch, r)
                hist_full, win = simulate_user(
                    prof, max(SIM_HISTORY_SIZES), SIM_N_EVAL, SIM_ANOMALY_RATE,
                    SIM_DELTA, r,
                )
                labels = win["is_anomaly"].to_numpy()

                for n in SIM_HISTORY_SIZES:
                    hist = hist_full.iloc[-n:].reset_index(drop=True)
                    Xh = sim_features(hist, reference=hist)
                    Xe = sim_features(win, reference=hist)
                    sc = fit_scaler(Xh)
                    Xh_s, Xe_s = sc.transform(Xh), sc.transform(Xe)

                    # Mahalanobis and kNN only, to keep 5-seed replication
                    # tractable; these are the models the reference paper's
                    # headline thresholds and crossovers are computed from
                    # (Sec. 7.1, 7.3, 7.4). Isolation Forest and the
                    # autoencoder are not replicated across seeds here; this
                    # is recorded in the revised Appendix A.5.
                    m_maha = ShrinkageMahalanobis(kappa=0.0).fit(Xh_s)
                    s_e, s_t = m_maha.score(Xe_s), m_maha.score(Xh_s)
                    met = evaluate(labels, s_e, s_t)
                    met.update(
                        user_id=uid, archetype=arch, history_size=n,
                        model="mahalanobis", arm="personal", cohort_seed=seed,
                    )
                    rows.append(met)

                    m_knn = KNNDistance(k=5).fit(Xh_s)
                    s_e, s_t = m_knn.score(Xe_s), m_knn.score(Xh_s)
                    met = evaluate(labels, s_e, s_t)
                    met.update(
                        user_id=uid, archetype=arch, history_size=n,
                        model="knn", arm="personal", cohort_seed=seed,
                    )
                    rows.append(met)
        df_seed = pd.DataFrame(rows)
        all_rows.append(df_seed)
        print(f"[E5-B1] seed={seed} done in {time.time()-t0:.0f}s, "
              f"{len(df_seed)} rows", flush=True)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(f"{OUT}/e5_multiseed_simulation.csv", index=False)
    return df


def summarise(df_e1, df_sim):
    os.makedirs(TAB, exist_ok=True)

    # Per-seed relative-90 threshold for Mahalanobis, personal arm, Berka
    rows = []
    for seed, g in df_e1[
        (df_e1.arm == "personal") & (df_e1.model == "mahalanobis")
    ].groupby("cohort_seed"):
        cur = g.groupby("history_size")["pr_auc"].mean().sort_index()
        r = relative_threshold(cur.index.to_numpy(), cur.to_numpy(), 0.90)
        rows.append({"dataset": "berka", "model": "mahalanobis",
                      "cohort_seed": seed, "threshold_90": r.get("threshold"),
                      "reference_best": r.get("reference")})
    for seed, g in df_sim[df_sim.model == "mahalanobis"].groupby("cohort_seed"):
        cur = g.groupby("history_size")["pr_auc"].mean().sort_index()
        r = relative_threshold(cur.index.to_numpy(), cur.to_numpy(), 0.90)
        rows.append({"dataset": "simulation", "model": "mahalanobis",
                      "cohort_seed": seed, "threshold_90": r.get("threshold"),
                      "reference_best": r.get("reference")})
    thr = pd.DataFrame(rows)
    thr.to_csv(f"{TAB}/e5_threshold_seed_stability.csv", index=False)
    print("\n=== Per-seed relative-90 threshold, Mahalanobis ===")
    print(thr.round(3).to_string(index=False))

    # Mean +/- across-seed SD of PR-AUC by history size (Berka, personal, Mahalanobis)
    curve = (
        df_e1[(df_e1.arm == "personal") & (df_e1.model == "mahalanobis")]
        .groupby(["cohort_seed", "history_size"])["pr_auc"].mean()
        .unstack("cohort_seed")
    )
    curve["mean"] = curve.mean(axis=1)
    curve["sd_across_seeds"] = curve[MULTISEED_SEEDS].std(axis=1, ddof=1)
    curve.to_csv(f"{TAB}/e5_curve_seed_variation_berka_mahalanobis.csv")
    print("\n=== Berka Mahalanobis PR-AUC: mean and across-seed SD ===")
    print(curve.round(4).to_string())

    # Global vs personal crossover, per seed (Mahalanobis)
    cross_rows = []
    for seed, g in df_e1[df_e1.model == "mahalanobis"].groupby("cohort_seed"):
        personal = g[g.arm == "personal"].groupby("history_size")["pr_auc"].mean()
        glob = g[g.arm == "global"]["pr_auc"].mean()
        crossing = personal[personal > glob]
        first_n = int(crossing.index.min()) if len(crossing) else None
        cross_rows.append({"cohort_seed": seed, "global_mean": glob,
                            "first_n_personal_beats_global": first_n})
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(f"{TAB}/e5_crossover_seed_stability.csv", index=False)
    print("\n=== Per-seed global<->personal crossover (Mahalanobis) ===")
    print(cross.round(3).to_string(index=False))


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"E5 multi-seed replication, seeds={MULTISEED_SEEDS}", flush=True)
    trans = load_transactions()

    df_e1 = run_e1_multiseed(trans)
    df_sim = run_simulation_multiseed()
    summarise(df_e1, df_sim)

    with open(f"{OUT}/e5_meta.json", "w") as fh:
        json.dump({"seeds": MULTISEED_SEEDS, "n_eval_berka": N_EVAL_BERKA,
                    "n_donors_berka": N_DONORS_BERKA,
                    "note": "E1 (impostor) and simulation B1 (Mahalanobis+kNN "
                            "only) replicated across seeds. E2 and the full "
                            "B1 model set (iforest, autoencoder) not "
                            "replicated across seeds; single-seed results "
                            "for those remain in e1/e2/b1 tables."},
                   fh, indent=2)
    print("E5 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
