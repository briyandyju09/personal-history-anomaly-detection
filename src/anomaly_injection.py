"""
Ground-truth anomaly protocols for the real-data experiments.

The Berka dataset carries no fraud labels and no anomaly labels. Rather than
invent labels, we construct ground truth that is defined independently of every
detector we evaluate. Two protocols are used.

PROTOCOL I: impostor substitution
    A held-out transaction of the target account is replaced by a real
    transaction drawn from a different real account. The replacement keeps the
    target's timeline slot (date), and takes the impostor's behavioural
    attributes: amount, credit/debit type, operation and k_symbol.

    Ground truth is provenance: the row was produced by a different person's
    real financial behaviour. No statistical criterion is involved, so the
    label cannot be circular with respect to any detector's score.

    Substitution rather than insertion is deliberate. If impostor rows were
    inserted, the surrounding inter-arrival gaps would change, and detectors
    could exploit that artefact instead of behaviour. Substitution holds the
    timeline fixed. The cost is that timing ceases to be a discriminative
    signal under this protocol; timing anomalies are studied under Protocol P
    and in the controlled simulation instead.

PROTOCOL P: parametric perturbation
    A held-out transaction of the target account is replaced by a copy of
    itself with one attribute deliberately shifted by a controlled magnitude:
    amount scaled by a factor, operation swapped to a level the account uses
    rarely or never, or the timeline slot moved.

    The perturbation is defined on the raw attribute, parameterised by a
    magnitude we set in advance. It is not defined using any detector's score
    or decision rule. The magnitude parameter lets us measure detection as a
    function of deviation size rather than reporting a single number.

IMPORTANT: neither protocol produces fraud. A Protocol I row is one real
person's ordinary transaction relocated into another person's history; a
Protocol P row is a deliberately shifted version of a real transaction. Both
are behavioural anomalies by construction. Neither is evidence about how a
detector would perform against an adversary who is actively trying to imitate
the account holder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BEHAVIOURAL_COLS = ["amount", "type", "operation", "k_symbol"]


@dataclass
class InjectionResult:
    """Evaluation window after injection, with provenance labels."""

    frame: pd.DataFrame  # the evaluation window, same length as the original
    labels: np.ndarray  # 1 = injected anomaly, 0 = genuine
    n_injected: int
    protocol: str
    detail: dict


def inject_impostor(
    target_window: pd.DataFrame,
    impostor_pool: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """
    Protocol I. Replace a fraction ``rate`` of the target's evaluation window
    with behavioural attributes drawn from ``impostor_pool``.

    ``impostor_pool`` must contain transactions from accounts other than the
    target; the caller is responsible for that exclusion.
    """
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
        win,
        labels,
        k,
        "impostor",
        {"rate": rate, "donor_accounts": donor_rows["account_id"].tolist()},
    )


def inject_parametric(
    target_window: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
    kind: str = "amount",
    magnitude: float = 5.0,
    rare_operations: list | None = None,
) -> InjectionResult:
    """
    Protocol P. Replace a fraction ``rate`` of the evaluation window with
    deliberately shifted copies of the account's own transactions.

    kind:
        "amount"    multiply the amount by ``magnitude``
        "operation" swap the operation to a level the account does not use
        "timing"    move the transaction by ``magnitude`` days within the slot
    """
    win = target_window.reset_index(drop=True).copy()
    n = len(win)
    k = int(round(rate * n))
    k = max(1, min(k, n - 1)) if n > 1 else 0
    labels = np.zeros(n, dtype=int)
    if k == 0:
        return InjectionResult(win, labels, 0, f"param_{kind}", {"rate": rate})

    slots = rng.choice(n, size=k, replace=False)

    if kind == "amount":
        win.loc[slots, "amount"] = (
            win.loc[slots, "amount"].to_numpy(dtype=float) * magnitude
        )
    elif kind == "operation":
        used = set(pd.Series(win["operation"]).dropna().unique())
        pool = rare_operations or [
            "VYBER KARTOU",
            "PREVOD NA UCET",
            "VKLAD",
            "PREVOD Z UCTU",
            "VYBER",
        ]
        unused = [o for o in pool if o not in used]
        if not unused:
            unused = pool
        win.loc[slots, "operation"] = rng.choice(unused, size=k, replace=True)
    elif kind == "timing":
        shifted = pd.to_datetime(win.loc[slots, "date"]) + pd.to_timedelta(
            int(magnitude), unit="D"
        )
        win.loc[slots, "date"] = shifted
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")

    labels[slots] = 1
    return InjectionResult(
        win,
        labels,
        k,
        f"param_{kind}",
        {"rate": rate, "magnitude": magnitude},
    )


def build_impostor_pool(
    trans: pd.DataFrame,
    exclude_account: int,
    candidate_accounts: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample donor transactions from accounts other than the target."""
    others = candidate_accounts[candidate_accounts != exclude_account]
    donors = rng.choice(others, size=min(size, len(others)), replace=False)
    pool = trans[trans["account_id"].isin(donors)]
    if len(pool) > size:
        idx = rng.choice(len(pool), size=size, replace=False)
        pool = pool.iloc[idx]
    return pool.reset_index(drop=True)
