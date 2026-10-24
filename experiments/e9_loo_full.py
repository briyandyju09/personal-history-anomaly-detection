"""
E9 (full): in-sample vs leave-one-out threshold calibration for every personal
detector and the shrinkage-hybrid arm, at all five E5 cohort seeds.

experiments/e9_loo_calibration.py covered Mahalanobis and k-NN on the primary
seed only. This extends it to gaussian_z, robust_z, mahalanobis, knn, iforest,
autoencoder and hybrid_shrinkage (Mahalanobis, kappa=25), on the exact cohorts
E5 uses (same seeds, same account/donor draws, same injected labels, same
detector seeds).

Calibration rules compared, per account / history size / detector:
  insample  threshold = 95th pct of scores on the training points the model was
            fitted on (what experiment_engine.py does)
  loo       threshold = 95th pct of leave-one-out scores: each training point is
            scored by a model refitted on the other n-1 points. Exact refit for
            every detector, no shortcuts. Undefined at n=1 (nothing is flagged).

NOT covered: the global arm (its threshold comes from the model's scores on its
own large pooled fitting sample; a leave-one-out version would refit on ~4000
points per detector) and the score-fusion k-NN hybrid.

Validation built in: the insample rows are checked against the existing
results/raw/e5_multiseed_e1.csv (and e1_impostor.csv for the primary seed). The
script reports the maximum absolute difference in PR-AUC, flag rate and FPR; if
the in-sample rows do not reproduce, the leave-one-out rows should not be
trusted.

Usage (cwd = research/):
    python -m experiments.e9_loo_full                 # all 5 seeds, all cores
    python -m experiments.e9_loo_full --limit 3 --seeds 20260920   # smoke test
Writes results/raw/e9_full_seed<seed>.csv per seed (resumable: finished seeds are
skipped) and results/raw/e9_loo_full.csv when all requested seeds are done.
"""

import os

# One BLAS thread per worker, set before numpy is imported anywhere.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.anomaly_injection import build_impostor_pool, inject_impostor
from src.data_loader import CohortSpec, eligible_accounts, load_transactions
from src.evaluation import evaluate
from src.experiment_engine import (
    EVAL_WINDOW,
    HISTORY_SIZES,
    HYBRID_KAPPA,
    INJECTION_RATE,
    build_global_prior,
)
from src.feature_engineering import build_features, fit_scaler
from src.models import (
    GaussianZ,
    IForest,
    KNNDistance,
    RobustZ,
    ShrinkageMahalanobis,
    TinyAutoencoder,
)

SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
N_EVAL, N_DONORS = 300, 400
OUT = "results/raw"

_G = {}  # per-process cache: trans, by_account, per-seed prior


def _init():
    trans = load_transactions()
    _G["trans"] = trans
    _G["by_account"] = {a: g for a, g in trans.groupby("account_id", sort=False)}
    _G["prior"] = {}


def _prior(seed, donors):
    if seed not in _G["prior"]:
        _G["prior"][seed] = build_global_prior(
            _G["trans"], donors, 40, np.random.default_rng(seed)
        )
    return _G["prior"][seed]


def _personal_models(seed):
    return {
        "gaussian_z": lambda: GaussianZ(0),
        "robust_z": lambda: RobustZ(0),
        "mahalanobis": lambda: ShrinkageMahalanobis(kappa=0.0),
        "knn": lambda: KNNDistance(k=5),
        "iforest": lambda: IForest(seed=seed),
        "autoencoder": lambda: TinyAutoencoder(seed=seed),
    }


def _loo_scores(make, X):
    out = np.empty(len(X))
    for i in range(len(X)):
        out[i] = make().fit(np.delete(X, i, axis=0)).score(X[i:i + 1])[0]
    return out


def _rows(base, arm, model, labels, s_eval, s_in, s_loo):
    rows = []
    for calib, s_train in (("insample", s_in), ("loo", s_loo)):
        m = evaluate(labels, s_eval, s_train, alpha=0.05)
        m.update(base, arm=arm, model=model, calib=calib)
        rows.append(m)
    return rows


def run_account(task):
    seed, acc, donors = task
    trans, by_account = _G["trans"], _G["by_account"]
    g = by_account[acc].reset_index(drop=True)
    rng = np.random.default_rng(seed * 100003 + int(acc))
    window = g.iloc[-EVAL_WINDOW:].reset_index(drop=True)
    hist_all = g.iloc[:-EVAL_WINDOW].reset_index(drop=True)
    pool = build_impostor_pool(trans, acc, donors, size=EVAL_WINDOW, rng=rng)
    inj = inject_impostor(window, pool, INJECTION_RATE, rng)
    labels = inj.labels
    if labels.sum() == 0 or labels.sum() == len(labels):
        return []
    prior = _prior(seed, donors)
    X_eval_raw = build_features(inj.frame)
    X_eval_g = prior.scaler.transform(X_eval_raw)
    makers = _personal_models(seed)
    rows = []
    for n in HISTORY_SIZES:
        if n > len(hist_all):
            continue
        hist = hist_all.iloc[-n:].reset_index(drop=True)
        Xh_raw = build_features(hist)
        sc = fit_scaler(Xh_raw)
        Xh, Xe = sc.transform(Xh_raw), sc.transform(X_eval_raw)
        base = dict(cohort_seed=seed, account_id=int(acc), history_size=n)
        for name, make in makers.items():
            m = make().fit(Xh)
            s_eval, s_in = m.score(Xe), m.score(Xh)
            s_loo = _loo_scores(make, Xh) if n >= 2 else np.array([])
            rows += _rows(base, "personal", name, labels, s_eval, s_in, s_loo)
        # hybrid shrinkage Mahalanobis, fitted/scored in global-scaler space
        Xh_g = prior.scaler.transform(Xh_raw)

        def mk_h():
            h = ShrinkageMahalanobis(kappa=HYBRID_KAPPA)
            h.set_prior(prior.mu, prior.cov)
            return h

        hyb = mk_h().fit(Xh_g)
        s_loo = _loo_scores(mk_h, Xh_g) if n >= 2 else np.array([])
        rows += _rows(base, "hybrid_shrinkage", "mahalanobis", labels,
                      hyb.score(X_eval_g), hyb.score(Xh_g), s_loo)
    return rows


def validate(df, seed):
    """In-sample rows must reproduce the E5 (and, for the primary seed, E1) rows."""
    ref = pd.read_csv(f"{OUT}/e5_multiseed_e1.csv")
    ref = ref[(ref.cohort_seed == seed) & (ref.arm.isin(["personal", "hybrid_shrinkage"]))]
    mine = df[(df.calib == "insample") & (df.cohort_seed == seed)]
    keys = ["account_id", "history_size", "model", "arm"]
    j = mine.merge(ref, on=keys, suffixes=("_mine", "_ref"))
    out = {"seed": seed, "matched_rows": len(j), "my_rows": len(mine)}
    for c in ["pr_auc", "flag_rate", "fpr", "recall"]:
        d = (j[f"{c}_mine"] - j[f"{c}_ref"]).abs()
        out[f"max_abs_diff_{c}"] = float(np.nanmax(d.to_numpy())) if len(d) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--limit", type=int, default=None, help="first N accounts per seed (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    trans = load_transactions()
    spec = CohortSpec(min_history=max(HISTORY_SIZES), eval_window=EVAL_WINDOW)
    pool_acc = eligible_accounts(trans, spec)
    del trans

    reports, frames = [], []
    with Pool(a.workers, initializer=_init) as pool:
        for seed in a.seeds:
            path = f"{OUT}/e9_full_seed{seed}.csv" + (".smoke" if a.limit else "")
            if os.path.exists(path):
                print(f"seed {seed}: found {path}, skipping", flush=True)
                df = pd.read_csv(path)
            else:
                chosen = np.random.default_rng(seed).choice(
                    pool_acc, size=N_EVAL + N_DONORS, replace=False)
                accounts, donors = chosen[:N_EVAL], chosen[N_EVAL:]
                if a.limit:
                    accounts = accounts[: a.limit]
                tasks = [(seed, int(x), donors) for x in accounts]
                t0, rows = time.time(), []
                for k, r in enumerate(pool.imap_unordered(run_account, tasks), 1):
                    rows += r
                    if k % 10 == 0 or k == len(tasks):
                        print(f"seed {seed}: {k}/{len(tasks)} accounts, "
                              f"{time.time() - t0:.0f}s", flush=True)
                df = pd.DataFrame(rows)
                df.to_csv(path, index=False)
            rep = validate(df, seed)
            print("validation:", rep, flush=True)
            reports.append(rep)
            frames.append(df)

    if not a.limit and len(a.seeds) == len(SEEDS):
        pd.concat(frames, ignore_index=True).to_csv(f"{OUT}/e9_loo_full.csv", index=False)
        pd.DataFrame(reports).to_csv("results/tables/e9_full_validation.csv", index=False)
        print("wrote results/raw/e9_loo_full.csv", flush=True)


if __name__ == "__main__":
    main()
