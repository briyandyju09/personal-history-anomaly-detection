"""
Feature construction for per-account behavioural modelling.

Design constraints, all of which exist to keep the evaluation honest:

1. Causality. Every feature for transaction t depends only on t itself and on
   transactions strictly earlier in the same account. No feature uses future
   information, and no feature uses population statistics estimated on the
   evaluation window.

2. No account-state variables. `balance` is deliberately excluded. Balance is
   an accumulated state variable rather than a behavioural choice, and under
   the impostor injection protocol (see anomaly_injection.py) an injected row
   would otherwise carry a balance discontinuity that reveals its provenance
   for reasons unrelated to behaviour. Including it would inflate every
   detector's score through an artefact of the protocol, not through genuine
   behavioural discrimination.

3. Fixed global vocabulary. Categorical levels come from the dataset schema
   published by the primary source, not from the data split, so a one-hot
   vector has the same meaning for every account and every history size.

4. Scaling is fitted on training history only. See `fit_scaler`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Levels documented on the primary source page.
OPERATIONS = [
    "VYBER KARTOU",  # credit card withdrawal
    "VKLAD",  # credit in cash
    "PREVOD Z UCTU",  # collection from another bank
    "VYBER",  # withdrawal in cash
    "PREVOD NA UCET",  # remittance to another bank
]
K_SYMBOLS = [
    "POJISTNE",  # insurance payment
    "SLUZBY",  # payment for statement
    "UROK",  # interest credited
    "SANKC. UROK",  # sanction interest
    "SIPO",  # household
    "DUCHOD",  # old-age pension
    "UVER",  # loan payment
]
CREDIT_TYPES = {"PRIJEM"}

NUMERIC_FEATURES = [
    "log_amount",
    "is_credit",
    "dom_sin",
    "dom_cos",
    "dow_sin",
    "dow_cos",
    "log_gap_days",
]
CATEGORICAL_FEATURES = (
    [f"op_{o}" for o in OPERATIONS]
    + ["op_missing"]
    + [f"ks_{k}" for k in K_SYMBOLS]
    + ["ks_missing"]
)
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Map a chronologically ordered transaction frame for a single account to a
    float matrix of shape (len(df), len(FEATURE_NAMES)).

    `df` must contain: date, type, operation, amount, k_symbol.
    The inter-arrival gap for the first row is imputed as the median gap of the
    remaining rows (or 1 day when the account has a single transaction), so the
    feature is defined without looking forward beyond the first observation.
    """
    n = len(df)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)

    amount = df["amount"].to_numpy(dtype=np.float64)
    out[:, 0] = np.log1p(np.clip(amount, 0.0, None))
    out[:, 1] = df["type"].isin(CREDIT_TYPES).to_numpy(dtype=np.float64)

    dates = pd.to_datetime(df["date"])
    dom = dates.dt.day.to_numpy(dtype=np.float64)
    dow = dates.dt.dayofweek.to_numpy(dtype=np.float64)
    out[:, 2] = np.sin(2 * np.pi * dom / 31.0)
    out[:, 3] = np.cos(2 * np.pi * dom / 31.0)
    out[:, 4] = np.sin(2 * np.pi * dow / 7.0)
    out[:, 5] = np.cos(2 * np.pi * dow / 7.0)

    day_num = dates.values.astype("datetime64[D]").astype(np.int64)
    gaps = np.diff(day_num).astype(np.float64)
    if n > 1:
        first_gap = float(np.median(gaps)) if len(gaps) else 1.0
        gap_full = np.concatenate([[first_gap], gaps])
    else:
        gap_full = np.array([1.0])
    out[:, 6] = np.log1p(np.clip(gap_full, 0.0, None))

    col = len(NUMERIC_FEATURES)
    op = df["operation"].astype("object").to_numpy()
    for o in OPERATIONS:
        out[:, col] = (op == o).astype(np.float64)
        col += 1
    out[:, col] = pd.isna(pd.Series(op)).to_numpy(dtype=np.float64)
    col += 1

    ks = df["k_symbol"].astype("object").to_numpy()
    for k in K_SYMBOLS:
        out[:, col] = (ks == k).astype(np.float64)
        col += 1
    out[:, col] = pd.isna(pd.Series(ks)).to_numpy(dtype=np.float64)
    return out


@dataclass
class Scaler:
    """Standardiser fitted on training history only."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def fit_scaler(X_train: np.ndarray, eps: float = 1e-6) -> Scaler:
    """
    Fit a standardiser on training rows only.

    With n = 1 the standard deviation is undefined, and with small n many
    one-hot columns are constant. Both cases collapse to unit scale, which
    keeps the transform defined without inventing dispersion that the training
    history does not contain.
    """
    mean = X_train.mean(axis=0)
    if X_train.shape[0] < 2:
        scale = np.ones_like(mean)
    else:
        scale = X_train.std(axis=0, ddof=1)
        scale = np.where(scale < eps, 1.0, scale)
    return Scaler(mean=mean, scale=scale)


# ----------------------------------------------------------------------------
# Behavioural complexity descriptors (RQ4)
# ----------------------------------------------------------------------------


def _entropy(values: pd.Series) -> float:
    """Shannon entropy in nats of a categorical series, NaN treated as a level."""
    counts = values.fillna("__missing__").value_counts()
    if len(counts) <= 1:
        return 0.0
    p = counts.to_numpy(dtype=np.float64)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def behavioural_descriptors(df: pd.DataFrame) -> dict:
    """
    Measurable descriptors of an account's behaviour, computed on a given
    window of transactions. Reported individually. We deliberately do not
    collapse them into a single composite index: no weighting scheme is
    defensible a priori, and the components are not measured on a common scale.

    Returned keys:
        amount_cv            coefficient of variation of raw amounts
        amount_rcv           robust dispersion, MAD / |median|
        log_amount_iqr       interquartile range of log1p(amount)
        operation_entropy    entropy of the operation field (nats)
        ksymbol_entropy      entropy of the k_symbol field (nats)
        operation_diversity  unique operations / transactions
        gap_cv               coefficient of variation of inter-arrival gaps
        credit_share         fraction of credit transactions
        n                    window size the descriptors were computed on
    """
    amount = df["amount"].to_numpy(dtype=np.float64)
    med = float(np.median(amount))
    mad = float(np.median(np.abs(amount - med)))
    mean = float(np.mean(amount))
    std = float(np.std(amount, ddof=1)) if len(amount) > 1 else 0.0

    dates = pd.to_datetime(df["date"]).values.astype("datetime64[D]").astype(np.int64)
    gaps = np.diff(dates).astype(np.float64)
    gap_cv = (
        float(np.std(gaps, ddof=1) / np.mean(gaps))
        if len(gaps) > 1 and np.mean(gaps) > 0
        else np.nan
    )

    logamt = np.log1p(np.clip(amount, 0.0, None))
    return {
        "amount_cv": std / mean if mean > 0 else np.nan,
        "amount_rcv": (1.4826 * mad / abs(med)) if med != 0 else np.nan,
        "log_amount_iqr": float(np.subtract(*np.percentile(logamt, [75, 25]))),
        "operation_entropy": _entropy(df["operation"]),
        "ksymbol_entropy": _entropy(df["k_symbol"]),
        "operation_diversity": float(df["operation"].nunique(dropna=False)) / len(df),
        "gap_cv": gap_cv,
        "credit_share": float(df["type"].isin(CREDIT_TYPES).mean()),
        "n": int(len(df)),
    }
