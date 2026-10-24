"""
E9: is the 'calibration is expensive' result (paper 7.2) caused by
in-sample training scores?  Compare thresholds from
  (a) in-sample train scores   (what the paper does)
  (b) leave-one-out train scores (honest out-of-sample estimate)
for personal Mahalanobis and k-NN on the exact E1 cohort/labels.
Usage (cwd = research/):  python -m experiments.e9_loo_calibration
Writes results/raw/e9_loo_calibration.csv. Same cohort, seed and labels as E1.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd

from src.anomaly_injection import build_impostor_pool, inject_impostor
from src.data_loader import CohortSpec, eligible_accounts, load_transactions
from src.experiment_engine import EVAL_WINDOW, HISTORY_SIZES, INJECTION_RATE
from src.feature_engineering import build_features, fit_scaler
from src.models import KNNDistance, ShrinkageMahalanobis
from src.evaluation import operating_point_metrics, training_quantile_threshold

SEED, N_EVAL, N_DONORS = 20260920, 300, 400
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/raw/e9_loo_calibration.csv"

trans = load_transactions()
spec = CohortSpec(min_history=max(HISTORY_SIZES), eval_window=EVAL_WINDOW)
pool = eligible_accounts(trans, spec)
rng0 = np.random.default_rng(SEED)
chosen = rng0.choice(pool, size=N_EVAL + N_DONORS, replace=False)
accounts, donors = chosen[:N_EVAL], chosen[N_EVAL:]

# NOTE: the engine builds the global prior with rng_master before the loop but
# that rng is not used per-account (per-account rng is re-seeded), so the
# per-account RNG stream below matches run_history_size_experiment exactly.
by_account = {a: g for a, g in trans.groupby("account_id", sort=False)}


def loo_scores(model_cls, X, **kw):
    n = len(X)
    out = np.empty(n)
    for i in range(n):
        m = model_cls(**kw).fit(np.delete(X, i, axis=0))
        out[i] = m.score(X[i:i + 1])[0]
    return out


rows = []
for ai, acc in enumerate(accounts):
    g = by_account[acc].reset_index(drop=True)
    rng = np.random.default_rng(SEED * 100003 + int(acc))
    window = g.iloc[-EVAL_WINDOW:].reset_index(drop=True)
    hist_all = g.iloc[:-EVAL_WINDOW].reset_index(drop=True)
    p = build_impostor_pool(trans, acc, donors, size=EVAL_WINDOW, rng=rng)
    inj = inject_impostor(window, p, INJECTION_RATE, rng)
    labels = inj.labels
    if labels.sum() == 0 or labels.sum() == len(labels):
        continue
    X_eval_raw = build_features(inj.frame)
    for n in HISTORY_SIZES:
        if n > len(hist_all):
            continue
        hist = hist_all.iloc[-n:].reset_index(drop=True)
        Xh_raw = build_features(hist)
        sc = fit_scaler(Xh_raw)
        Xh, Xe = sc.transform(Xh_raw), sc.transform(X_eval_raw)
        for mname, cls, kw in [("mahalanobis", ShrinkageMahalanobis, dict(kappa=0.0)),
                               ("knn", KNNDistance, dict(k=5))]:
            m = cls(**kw).fit(Xh)
            s_eval = m.score(Xe)
            s_in = m.score(Xh)
            thr_in = training_quantile_threshold(s_in)
            if n >= 2:
                s_loo = loo_scores(cls, Xh, **kw)
                thr_loo = training_quantile_threshold(s_loo)
            else:
                thr_loo = np.inf
            for tag, thr in [("insample", thr_in), ("loo", thr_loo)]:
                r = operating_point_metrics(labels, s_eval, thr)
                r.update(account_id=int(acc), n=n, model=mname, calib=tag, thr=thr)
                rows.append(r)
    if ai % 50 == 0:
        print("account", ai, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
agg = (df.groupby(["model", "calib", "n"])[["flag_rate", "fpr", "recall", "precision", "f1"]]
         .mean().round(3))
print(agg.to_string())
