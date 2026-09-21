"""
Ground-truth anomaly protocol for the TabFormer replication layer (E7).

Protocol I (impostor substitution), adapted from src/anomaly_injection.py to
TabFormer's schema. Identical logic: a held-out transaction of the target
entity is replaced by a real transaction drawn from a different entity's
history, keeping the target's timeline slot (date, hour) but taking the
donor's behavioural attributes. Ground truth is provenance, independent of
any detector's score, for the same reasons given in anomaly_injection.py.

BEHAVIOURAL_COLS differs from Berka's because the schema differs: amount,
merchant, category (MCC) and use-chip channel travel with the donor row;
the timeline slot does not, so (as in Protocol I on Berka) timing is not a
discriminative signal under this protocol.

Protocol F (secondary, exploratory): TabFormer's generator-produced
"Is Fraud?" column is used as an independent secondary label. This is
GENERATOR-SIMULATED fraud, not adjudicated real-world fraud, and it is never
described as "confirmed fraud" in this project. Its prevalence (0.122% of
all transactions, independently measured this session) is far below the
10% injection rate used in Protocol I, so per-account pairing is usually
infeasible; Protocol F results are pooled across accounts rather than
paired, and that is a stated limitation of Protocol F specifically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BEHAVIOURAL_COLS = ["amount", "Use Chip", "MCC", "Merchant Name", "Merchant State", "has_error"]


@dataclass
class InjectionResult:
    frame: pd.DataFrame
    labels: np.ndarray
    n_injected: int
    protocol: str
    detail: dict


def inject_impostor(
    target_window: pd.DataFrame,
    impostor_pool: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> InjectionResult:
    win = target_window.reset_index(drop=True).copy()
    n = len(win)
    k = int(round(rate * n))
    k = max(1, min(k, n - 1)) if n > 1 else 0
    labels = np.zeros(n, dtype=int)
    if k == 0 or len(impostor_pool) == 0:
        return InjectionResult(win, labels, 0, "impostor", {"rate": rate})

    slots = rng.choice(n, size=k, replace=False)
    donors = rng.choice(len(impostor_pool), size=k, replace=False)
    donor_rows = impostor_pool.iloc[donors]

    for col in BEHAVIOURAL_COLS:
        win.loc[slots, col] = donor_rows[col].to_numpy()
    labels[slots] = 1

    return InjectionResult(
        win, labels, k, "impostor",
        {"rate": rate, "donor_entities": donor_rows["entity_id"].tolist()},
    )


def build_impostor_pool(
    trans: pd.DataFrame,
    exclude_entity: str,
    candidate_entities: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    others = candidate_entities[candidate_entities != exclude_entity]
    donors = rng.choice(others, size=min(size, len(others)), replace=False)
    pool = trans[trans["entity_id"].isin(donors)]
    if len(pool) > size:
        idx = rng.choice(len(pool), size=size, replace=False)
        pool = pool.iloc[idx]
    return pool.reset_index(drop=True)
