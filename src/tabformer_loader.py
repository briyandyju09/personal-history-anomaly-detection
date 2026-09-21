"""
Loading for the IBM TabFormer synthetic credit-card transaction dataset
(Experiment E7 -- card-structured replication layer).

Source
------
    https://github.com/IBM/TabFormer  (Apache-2.0)
    Padhi, I. et al. (2021). Tabular Transformers for Modeling Multivariate
    Time Series. ICASSP 2021.
    File: data/credit_card/transactions.tgz, fetched via the repository's
    Git-LFS media endpoint (no authentication required), extracting to
    ``card_transaction.v1.csv``.

THIS DATASET IS SYNTHETIC. It is a generator-produced simulation of card
transactions, including its "Is Fraud?" column, which is generator output and
NOT an adjudicated fraud label. TabFormer results are never pooled with the
real-data (Berka) results or with the controlled behavioural simulation
(src/simulation.py). They exist to test whether the *shape* of the real-data
data-efficiency curves is an artefact of one 1990s retail-banking dataset, on
a schema that (unlike Berka) has merchant, category and geographic channels.

Verification
------------
We independently observed (this session, chunked read of the full file):
    total rows            24,386,900  (matches the repository's stated
                                        "24 million records")
    unique users           2,000      (matches "2,000 users")
    unique (user, card)     6,139
    year range              1991-2020
This is a single-source dataset: TabFormer's GitHub repository is the only
distribution point we used (the README also lists an IBM Box mirror of the
identical archive, not fetched separately here). This is weaker provenance
than the Berka dataset's two-independent-mirror SHA-256 cross-check in
src/data_loader.py, and is stated as a limitation.

Entity definition
------------------
We use (User, Card) as the personal history unit, not User alone. A person
can hold several cards with materially different usage (e.g. a primary card
vs. a rarely used one), and per-card behavioural profiling is the unit used
in the card-fraud sequence-modelling literature (Jurgovsky et al., 2018,
treat detection as sequence classification over a single card's transaction
stream). This mirrors Berka's account_id as "one behavioural stream", not
"one legal person" (a Berka client can also hold multiple accounts).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw_tabformer", "card_transaction.v1.csv",
)

EXPECTED_ROWS = 24_386_900
EXPECTED_USERS = 2000

USECOLS = [
    "User", "Card", "Year", "Month", "Day", "Time", "Amount",
    "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
    "MCC", "Errors?", "Is Fraud?",
]

# Top-12 MCC codes by corpus-wide frequency (chunked full-file count, this
# session), covering 75.6% of all transactions. This is dataset-level
# structure fixed in advance of any experiment split -- analogous to Berka's
# hardcoded OPERATIONS / K_SYMBOLS vocabulary in feature_engineering.py --
# not something fitted per account or per history size. MCC itself is a
# standardised (ISO 18245) merchant-category code, not a label the detectors
# are trying to predict.
TOP_MCC = [5411, 5499, 5541, 5812, 5912, 4829, 4784, 5300, 4121, 7538, 5814, 5311]


class DataIntegrityError(RuntimeError):
    pass


def entity_id(user: pd.Series, card: pd.Series) -> pd.Series:
    return user.astype(str) + "_" + card.astype(str)


@dataclass(frozen=True)
class TabformerCohort:
    """Result of the two-pass load: chosen accounts, donors, transactions."""

    trans: pd.DataFrame
    accounts: np.ndarray
    donors: np.ndarray
    n_eligible: int


def _chunks(usecols=None, chunksize=3_000_000):
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} missing. Download and extract transactions.tgz from "
            "https://github.com/IBM/TabFormer (data/credit_card/) first."
        )
    return pd.read_csv(RAW_PATH, usecols=usecols or USECOLS, chunksize=chunksize)


def entity_depth_counts() -> pd.Series:
    """One pass over the full file: transaction count per (User, Card)."""
    from collections import Counter
    c = Counter()
    n_rows = 0
    for ch in _chunks(usecols=["User", "Card"]):
        eid = entity_id(ch["User"], ch["Card"])
        c.update(eid.tolist())
        n_rows += len(ch)
    if n_rows != EXPECTED_ROWS:
        raise DataIntegrityError(
            f"card_transaction.v1.csv has {n_rows:,} rows; expected "
            f"{EXPECTED_ROWS:,} (the count documented and independently "
            "observed for this file). Refusing to use this copy."
        )
    return pd.Series(c).sort_values(ascending=False)


def load_cohort(
    min_transactions: int,
    n_eval: int,
    n_donors: int,
    seed: int,
) -> TabformerCohort:
    """
    Select ``n_eval`` + ``n_donors`` eligible (User, Card) entities and load
    only their transactions, sorted chronologically within each entity.

    Two passes over the 2.3 GB file: pass 1 computes per-entity depth (see
    ``entity_depth_counts``); pass 2 extracts rows for the chosen entities
    only, keeping memory bounded regardless of the file's total size.
    """
    depth = entity_depth_counts()
    eligible = depth[depth >= min_transactions].index.to_numpy()
    n_eligible = len(eligible)
    if n_eligible < n_eval + n_donors:
        raise ValueError(
            f"Only {n_eligible} entities have >= {min_transactions} "
            f"transactions; need {n_eval + n_donors}."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n_eval + n_donors, replace=False)
    chosen_set = set(chosen.tolist())
    accounts, donors = chosen[:n_eval], chosen[n_eval:]

    rows = []
    for ch in _chunks():
        eid = entity_id(ch["User"], ch["Card"])
        mask = eid.isin(chosen_set)
        if mask.any():
            sub = ch.loc[mask].copy()
            sub["entity_id"] = eid.loc[mask].to_numpy()
            rows.append(sub)
    trans = pd.concat(rows, ignore_index=True)

    trans["amount"] = (
        trans["Amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    )
    trans["date"] = pd.to_datetime(
        dict(year=trans["Year"], month=trans["Month"], day=trans["Day"])
    )
    hm = trans["Time"].str.split(":", expand=True).astype(int)
    trans["hour"] = hm[0] + hm[1] / 60.0
    trans["is_fraud"] = (trans["Is Fraud?"] == "Yes").astype(int)
    trans["has_error"] = trans["Errors?"].notna().astype(int)

    # Deterministic chronological order within entity; original row order
    # (index of appearance in the file, which is itself chronological in the
    # TabFormer generator's output) breaks same-timestamp ties.
    trans = trans.reset_index(names="_orig_order")
    trans = trans.sort_values(
        ["entity_id", "date", "hour", "_orig_order"], kind="mergesort"
    ).drop(columns="_orig_order").reset_index(drop=True)

    return TabformerCohort(trans=trans, accounts=accounts, donors=donors, n_eligible=n_eligible)


if __name__ == "__main__":
    depth = entity_depth_counts()
    print(f"entities (User,Card pairs): {len(depth):,}")
    print(f"total transactions: {depth.sum():,}  (expected {EXPECTED_ROWS:,})")
    for thr in (310, 560, 1000):
        print(f"  entities with >= {thr} txns: {(depth >= thr).sum():,}")
