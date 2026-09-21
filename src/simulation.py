"""
Experiment B: controlled behavioural simulation.

Purpose
-------
The Berka dataset cannot support three things this study needs: merchant,
category and geographic behaviour channels; deliberately controlled levels of
behavioural variability; and a known generative truth against which "behavioural
complexity" can be manipulated rather than merely observed. The simulation
supplies those, at the cost of being synthetic.

Nothing produced here is a real-world observation. Results from this layer are
reported separately throughout and are never pooled with the real-data results.

Generative model
----------------
An individual i is defined by a parameter vector, not by a label. Archetypes are
named presets over that vector, so "highly variable user" is a setting of
measurable parameters rather than a story:

    amount          log-normal, log-amount mean mu_i, dispersion sigma_i
    merchant        categorical over M_i merchants, Zipf-shaped with exponent z_i
    category        categorical over C_i categories, Dirichlet concentration a_i
    location        categorical over L_i locations, home-dominant with weight h_i
    timing          inter-arrival Gamma(shape k_i, scale determined by rate r_i)
    hour of day     von Mises around preferred hour with concentration c_i

Drift (used for the seasonal and mixed archetypes) rotates the category and
merchant distributions at a controlled rate d_i, so old history stops
representing current behaviour by construction.

Anomaly classes
---------------
Anomalies are generated from deliberately shifted distributions, independent of
every detector. The shift is applied to the raw generative parameter, never to a
score:

    amount      log-amount shifted by delta * sigma_i
    merchant    merchant drawn from outside the individual's merchant set
    category    category drawn from the individual's rarest categories
    geographic  location drawn from outside the individual's location set
    temporal    hour drawn from the antipode of the preferred hour
    burst       several transactions compressed into a short interval
    multivariate amount, merchant and location shifted jointly

Ground truth is the generating branch, which no detector can observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_MERCHANTS_GLOBAL = 400
N_CATEGORIES_GLOBAL = 20
N_LOCATIONS_GLOBAL = 50


@dataclass
class Profile:
    """Measurable behavioural parameters for one simulated individual."""

    user_id: int
    archetype: str
    log_amount_mu: float
    log_amount_sigma: float
    merchant_ids: np.ndarray
    merchant_p: np.ndarray
    category_ids: np.ndarray
    category_p: np.ndarray
    location_ids: np.ndarray
    location_p: np.ndarray
    gap_shape: float
    gap_scale: float
    hour_mu: float
    hour_kappa: float
    drift_rate: float = 0.0
    regimes: list = field(default_factory=list)


ARCHETYPES = {
    # sigma, n_merchants, zipf, n_categories, n_locations, home_weight,
    # gap_shape, mean_gap_days, hour_kappa, drift_rate
    "regular": dict(
        sigma=0.30, n_merch=5, zipf=2.0, n_cat=3, n_loc=1,
        home=0.98, gap_shape=8.0, gap_mean=3.0, hour_kappa=6.0, drift=0.0,
    ),
    "moderate": dict(
        sigma=0.65, n_merch=20, zipf=1.2, n_cat=7, n_loc=3,
        home=0.85, gap_shape=3.0, gap_mean=3.0, hour_kappa=2.5, drift=0.0,
    ),
    "variable": dict(
        sigma=1.10, n_merch=60, zipf=0.6, n_cat=14, n_loc=8,
        home=0.55, gap_shape=1.2, gap_mean=3.0, hour_kappa=0.6, drift=0.0,
    ),
    "traveller": dict(
        sigma=0.85, n_merch=45, zipf=0.8, n_cat=9, n_loc=15,
        home=0.35, gap_shape=1.5, gap_mean=3.0, hour_kappa=1.0, drift=0.0,
    ),
    "seasonal": dict(
        sigma=0.70, n_merch=25, zipf=1.0, n_cat=10, n_loc=3,
        home=0.85, gap_shape=2.5, gap_mean=3.0, hour_kappa=2.0, drift=0.004,
    ),
    "burst": dict(
        sigma=0.75, n_merch=25, zipf=1.0, n_cat=8, n_loc=3,
        home=0.85, gap_shape=0.35, gap_mean=3.0, hour_kappa=2.0, drift=0.0,
    ),
}


def _zipf_probs(n: int, exponent: float, rng) -> np.ndarray:
    w = 1.0 / np.power(np.arange(1, n + 1), exponent)
    rng.shuffle(w)
    return w / w.sum()


def make_profile(user_id: int, archetype: str, rng: np.random.Generator) -> Profile:
    cfg = ARCHETYPES[archetype]
    merch = rng.choice(N_MERCHANTS_GLOBAL, size=cfg["n_merch"], replace=False)
    cats = rng.choice(N_CATEGORIES_GLOBAL, size=cfg["n_cat"], replace=False)
    locs = rng.choice(N_LOCATIONS_GLOBAL, size=cfg["n_loc"], replace=False)

    loc_p = np.full(len(locs), (1.0 - cfg["home"]) / max(len(locs) - 1, 1))
    loc_p[0] = cfg["home"]
    loc_p = loc_p / loc_p.sum()

    return Profile(
        user_id=user_id,
        archetype=archetype,
        log_amount_mu=float(rng.uniform(2.5, 4.5)),
        log_amount_sigma=cfg["sigma"],
        merchant_ids=merch,
        merchant_p=_zipf_probs(len(merch), cfg["zipf"], rng),
        category_ids=cats,
        category_p=_zipf_probs(len(cats), 1.0, rng),
        location_ids=locs,
        location_p=loc_p,
        gap_shape=cfg["gap_shape"],
        gap_scale=cfg["gap_mean"] / cfg["gap_shape"],
        hour_mu=float(rng.uniform(0, 2 * np.pi)),
        hour_kappa=cfg["hour_kappa"],
        drift_rate=cfg["drift"],
    )


def _drifted(p: np.ndarray, t: int, rate: float, rng) -> np.ndarray:
    """Rotate a categorical distribution slowly over time."""
    if rate <= 0:
        return p
    shift = int((rate * t) % len(p))
    return np.roll(p, shift)


def generate_normal(profile: Profile, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n transactions from the individual's own behavioural process."""
    gaps = rng.gamma(profile.gap_shape, profile.gap_scale, size=n)
    t_days = np.cumsum(np.maximum(gaps, 0.01))

    rows = []
    for i in range(n):
        mp = _drifted(profile.merchant_p, i, profile.drift_rate, rng)
        cp = _drifted(profile.category_p, i, profile.drift_rate, rng)
        hour = (
            rng.vonmises(profile.hour_mu, profile.hour_kappa) + np.pi
        ) / (2 * np.pi) * 24.0
        rows.append(
            {
                "user_id": profile.user_id,
                "t_days": t_days[i],
                "amount": float(
                    np.exp(
                        rng.normal(profile.log_amount_mu, profile.log_amount_sigma)
                    )
                ),
                "merchant": int(rng.choice(profile.merchant_ids, p=mp)),
                "category": int(rng.choice(profile.category_ids, p=cp)),
                "location": int(rng.choice(profile.location_ids, p=profile.location_p)),
                "hour": float(hour % 24.0),
                "is_anomaly": 0,
                "anomaly_type": "none",
            }
        )
    return pd.DataFrame(rows)


ANOMALY_TYPES = [
    "amount",
    "merchant",
    "category",
    "geographic",
    "temporal",
    "multivariate",
]


def make_anomaly(
    profile: Profile,
    base: dict,
    kind: str,
    delta: float,
    rng: np.random.Generator,
) -> dict:
    """
    Replace a genuine transaction with one generated from a shifted
    distribution. ``delta`` is the shift magnitude in units natural to the
    channel, fixed in advance.
    """
    row = dict(base)
    unseen_m = np.setdiff1d(np.arange(N_MERCHANTS_GLOBAL), profile.merchant_ids)
    unseen_l = np.setdiff1d(np.arange(N_LOCATIONS_GLOBAL), profile.location_ids)
    unseen_c = np.setdiff1d(np.arange(N_CATEGORIES_GLOBAL), profile.category_ids)

    if kind in ("amount", "multivariate"):
        row["amount"] = float(
            np.exp(
                profile.log_amount_mu + delta * profile.log_amount_sigma
                + rng.normal(0, profile.log_amount_sigma * 0.25)
            )
        )
    if kind in ("merchant", "multivariate"):
        row["merchant"] = int(rng.choice(unseen_m))
    if kind == "category":
        row["category"] = int(
            rng.choice(unseen_c) if len(unseen_c) else rng.choice(profile.category_ids)
        )
    if kind in ("geographic", "multivariate"):
        row["location"] = int(
            rng.choice(unseen_l) if len(unseen_l) else rng.choice(profile.location_ids)
        )
    if kind == "temporal":
        row["hour"] = float(
            (
                (profile.hour_mu + np.pi) / (2 * np.pi) * 24.0 + 12.0
            ) % 24.0
        )
    row["is_anomaly"] = 1
    row["anomaly_type"] = kind
    return row


def simulate_user(
    profile: Profile,
    n_history: int,
    n_eval: int,
    anomaly_rate: float,
    delta: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Produce a chronologically ordered history and an evaluation window with
    injected anomalies. History and window never share a transaction.
    """
    full = generate_normal(profile, n_history + n_eval, rng)
    hist = full.iloc[:n_history].reset_index(drop=True)
    win = full.iloc[n_history:].reset_index(drop=True)

    k = max(1, int(round(anomaly_rate * len(win))))
    slots = rng.choice(len(win), size=k, replace=False)
    recs = win.to_dict("records")
    for j, s in enumerate(slots):
        kind = ANOMALY_TYPES[j % len(ANOMALY_TYPES)]
        recs[s] = make_anomaly(profile, recs[s], kind, delta, rng)
    return hist, pd.DataFrame(recs)


SIM_FEATURES = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "log_gap",
    "merchant_seen",
    "category_seen",
    "location_seen",
    "merchant_freq",
    "category_freq",
    "location_freq",
]


def sim_features(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> np.ndarray:
    """
    Feature map for simulated transactions.

    The "seen" and "freq" columns are computed against a reference history,
    which at evaluation time is the TRAINING history only. This is the
    mechanism by which a personal model can recognise an unfamiliar merchant,
    and it uses no information from the evaluation window.
    """
    ref = reference if reference is not None else df
    n = len(df)
    X = np.zeros((n, len(SIM_FEATURES)))
    X[:, 0] = np.log1p(df["amount"].to_numpy())
    h = df["hour"].to_numpy() / 24.0 * 2 * np.pi
    X[:, 1] = np.sin(h)
    X[:, 2] = np.cos(h)
    gaps = np.diff(df["t_days"].to_numpy(), prepend=df["t_days"].iloc[0])
    X[:, 3] = np.log1p(np.clip(gaps, 0, None))

    for j, col in enumerate(["merchant", "category", "location"]):
        counts = ref[col].value_counts()
        total = max(counts.sum(), 1)
        vals = df[col].to_numpy()
        X[:, 4 + j] = np.isin(vals, counts.index.to_numpy()).astype(float)
        X[:, 7 + j] = [counts.get(v, 0) / total for v in vals]
    return X


def observed_complexity(hist: pd.DataFrame) -> dict:
    """Descriptors measured on generated history, for validation against parameters."""
    amt = hist["amount"].to_numpy()

    def ent(s):
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log(p)).sum()) if len(p) > 1 else 0.0

    gaps = np.diff(hist["t_days"].to_numpy())
    return {
        "log_amount_sd": float(np.std(np.log1p(amt), ddof=1)),
        "merchant_entropy": ent(hist["merchant"]),
        "category_entropy": ent(hist["category"]),
        "location_entropy": ent(hist["location"]),
        "gap_cv": float(np.std(gaps, ddof=1) / np.mean(gaps)) if len(gaps) > 1 else np.nan,
        "n_unique_merchants": int(hist["merchant"].nunique()),
    }
