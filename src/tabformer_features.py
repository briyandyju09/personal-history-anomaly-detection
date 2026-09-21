"""
Feature construction for the TabFormer replication layer (E7).

Follows the same causality discipline as src/feature_engineering.py: every
feature for transaction t depends only on t and on transactions strictly
earlier in the same entity's (User, Card) history. Scaling is fitted on
training history only (reuses feature_engineering.fit_scaler).

Design differences from Berka, and why
---------------------------------------
Berka has no merchant, category or geographic fields (data limitation
documented in PAPER.md Sec. 5.3). TabFormer has all three, which is the
whole point of running E7: it lets us test the three anomaly channels the
real-data experiment could not reach.

* Category  -> MCC (merchant category code), a fixed one-hot over the
  corpus-wide top-12 codes (src.tabformer_loader.TOP_MCC) plus "other",
  exactly mirroring Berka's fixed OPERATIONS/K_SYMBOLS vocabulary.
* Merchant  -> "merchant novelty": has this entity transacted with this
  exact Merchant Name before, computed causally against strictly-prior
  transactions in the entity's OWN training history (never the population).
* Geography -> Merchant State. Raw state is 223-valued, too sparse for a
  fixed one-hot at small personal-history sizes, so we use "state novelty"
  (has this entity been billed in this state before) and "is_home_state"
  (does this transaction's state match the entity's most frequent
  prior state), both causal, both binary.

No `balance`-equivalent field exists in TabFormer (it has no running account
balance column), so the exclusion reason in feature_engineering.py's
docstring (point 2) does not apply here; nothing analogous was excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tabformer_loader import TOP_MCC

NUMERIC_FEATURES = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_gap_days",
    "has_error",
    "is_chip",
    "is_online",
    "merchant_novel",
    "mcc_novel",
    "state_novel",
    "is_home_state",
]
MCC_FEATURES = [f"mcc_{m}" for m in TOP_MCC] + ["mcc_other"]
FEATURE_NAMES = NUMERIC_FEATURES + MCC_FEATURES


def _causal_novelty(values: np.ndarray) -> np.ndarray:
    """
    For each position i, 1.0 if values[i] has not appeared in values[:i],
    else 0.0. Position 0 is always novel (no prior history). O(n) via a
    running set; only depends on strictly earlier entries in this same
    array, which is itself built from a single entity's chronological rows.
    """
    seen = set()
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        out[i] = 0.0 if v in seen else 1.0
        seen.add(v)
    return out


def _causal_home_state(states: np.ndarray) -> np.ndarray:
    """
    is_home_state[i] = 1 if states[i] equals the mode of states[:i]
    (ties broken by first-seen). Position 0 has no prior state -> 0.
    """
    from collections import Counter
    counts = Counter()
    out = np.zeros(len(states), dtype=np.float64)
    mode = None
    for i, s in enumerate(states):
        if mode is not None:
            out[i] = 1.0 if s == mode else 0.0
        counts[s] += 1
        mode = counts.most_common(1)[0][0]
    return out


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Map a chronologically ordered transaction frame for a single (User, Card)
    entity to a float matrix of shape (len(df), len(FEATURE_NAMES)).

    `df` must contain: date, hour, amount, has_error, Use Chip, Merchant Name,
    Merchant State, MCC, all already restricted to one entity and sorted
    chronologically (src.tabformer_loader.load_cohort guarantees this).
    """
    n = len(df)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)

    amount = df["amount"].to_numpy(dtype=np.float64)
    out[:, 0] = np.log1p(np.clip(amount, 0.0, None))

    hour = df["hour"].to_numpy(dtype=np.float64)
    out[:, 1] = np.sin(2 * np.pi * hour / 24.0)
    out[:, 2] = np.cos(2 * np.pi * hour / 24.0)

    dates = pd.to_datetime(df["date"])
    dow = dates.dt.dayofweek.to_numpy(dtype=np.float64)
    out[:, 3] = np.sin(2 * np.pi * dow / 7.0)
    out[:, 4] = np.cos(2 * np.pi * dow / 7.0)

    day_num = dates.values.astype("datetime64[D]").astype(np.int64)
    gaps = np.diff(day_num).astype(np.float64)
    if n > 1:
        first_gap = float(np.median(gaps)) if len(gaps) else 1.0
        gap_full = np.concatenate([[first_gap], gaps])
    else:
        gap_full = np.array([1.0])
    out[:, 5] = np.log1p(np.clip(gap_full, 0.0, None))

    out[:, 6] = df["has_error"].to_numpy(dtype=np.float64)

    chip = df["Use Chip"].astype("object").to_numpy()
    out[:, 7] = (chip == "Chip Transaction").astype(np.float64)
    out[:, 8] = (chip == "Online Transaction").astype(np.float64)

    merchant = df["Merchant Name"].astype("object").to_numpy()
    out[:, 9] = _causal_novelty(merchant)

    mcc = df["MCC"].astype("object").to_numpy()
    out[:, 10] = _causal_novelty(mcc)

    state = df["Merchant State"].astype("object").fillna("__missing__").to_numpy()
    out[:, 11] = _causal_novelty(state)
    out[:, 12] = _causal_home_state(state)

    col = len(NUMERIC_FEATURES)
    for m in TOP_MCC:
        out[:, col] = (mcc == m).astype(np.float64)
        col += 1
    known = set(TOP_MCC)
    out[:, col] = np.array([0.0 if m in known else 1.0 for m in mcc])

    return out


def behavioural_descriptors(df: pd.DataFrame) -> dict:
    """
    Descriptors analogous to feature_engineering.behavioural_descriptors,
    extended with the merchant/category/geographic diversity channels
    Berka cannot supply. Reported individually, not combined into an index,
    for the same reason given in feature_engineering.py.
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

    def ent(s: pd.Series) -> float:
        counts = s.fillna("__missing__").value_counts()
        if len(counts) <= 1:
            return 0.0
        p = counts.to_numpy(dtype=np.float64)
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    return {
        "amount_cv": std / mean if mean > 0 else np.nan,
        "amount_rcv": (1.4826 * mad / abs(med)) if med != 0 else np.nan,
        "merchant_entropy": ent(df["Merchant Name"]),
        "mcc_entropy": ent(df["MCC"]),
        "state_entropy": ent(df["Merchant State"]),
        "merchant_diversity": float(df["Merchant Name"].nunique()) / len(df),
        "gap_cv": gap_cv,
        "n": int(len(df)),
    }
