"""
Loading and integrity verification for the PKDD'99 Financial ("Berka") dataset.

Primary source description:
    https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm
    Prepared by Petr Berka and Marta Sochorova, PKDD'99 Discovery Challenge.

The primary host was not resolvable from our execution environment, so files
are retrieved from a GitHub mirror. We do not trust the mirror on faith: the
loader verifies every table's record count against the counts published on the
primary source page. A mismatch raises, rather than silently proceeding.

Published counts (primary source):
    account 4500, client 5369, disp 5369, order 6471,
    trans 1056320, loan 682, card 892, district 77
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIRROR_BASE = (
    "https://raw.githubusercontent.com/Kusainov/"
    "czech-banking-fin-analysis/master"
)

# Record counts published by the primary source (Berka & Sochorova, PKDD'99).
EXPECTED_ROWS = {
    "account": 4500,
    "client": 5369,
    "disp": 5369,
    "order": 6471,
    "trans": 1056320,
    "loan": 682,
    "card": 892,
    "district": 77,
}

DEFAULT_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


class DataIntegrityError(RuntimeError):
    """Raised when a downloaded table does not match the primary source."""


def download_raw(raw_dir: str = DEFAULT_RAW_DIR, force: bool = False) -> None:
    """Fetch the eight Berka tables from the mirror into ``raw_dir``."""
    os.makedirs(raw_dir, exist_ok=True)
    for name in EXPECTED_ROWS:
        path = os.path.join(raw_dir, f"{name}.csv")
        if os.path.exists(path) and not force:
            continue
        url = f"{MIRROR_BASE}/{name}.csv"
        with urllib.request.urlopen(url) as resp:
            payload = resp.read()
        with open(path, "wb") as fh:
            fh.write(payload)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_table(name: str, raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load one Berka table and verify its row count against the primary source."""
    path = os.path.join(raw_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run src.data_loader.download_raw() first."
        )
    df = pd.read_csv(path, sep=";", low_memory=False)
    expected = EXPECTED_ROWS[name]
    if len(df) != expected:
        raise DataIntegrityError(
            f"Table '{name}' has {len(df)} rows; primary source specifies "
            f"{expected}. Refusing to use this copy."
        )
    return df


def load_transactions(raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Load the transaction table with parsed dates, sorted chronologically
    within each account.

    Columns returned:
        trans_id, account_id, date (datetime64), type, operation,
        amount, balance, k_symbol, bank, account
    """
    df = load_table("trans", raw_dir)
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.zfill(6), format="%y%m%d"
    )
    # Blank strings in the source encode "no value"; normalise to NaN.
    for col in ["operation", "k_symbol", "bank"]:
        df[col] = df[col].replace("", np.nan)
        df[col] = df[col].where(df[col].notna(), None)
    # Deterministic chronological order; trans_id breaks same-day ties.
    df = df.sort_values(["account_id", "date", "trans_id"], kind="mergesort")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class CohortSpec:
    """Eligibility rule for including an account in an experiment."""

    min_history: int  # largest history size the experiment must supply
    eval_window: int  # number of held-out genuine transactions per account

    @property
    def min_transactions(self) -> int:
        return self.min_history + self.eval_window


def eligible_accounts(trans: pd.DataFrame, spec: CohortSpec) -> np.ndarray:
    """Account ids with enough transactions to supply history + eval window."""
    counts = trans.groupby("account_id").size()
    return np.sort(counts[counts >= spec.min_transactions].index.to_numpy())


def account_summary(trans: pd.DataFrame) -> pd.DataFrame:
    """Per-account descriptive summary used for the dataset audit table."""
    g = trans.groupby("account_id")
    out = pd.DataFrame(
        {
            "n_transactions": g.size(),
            "first_date": g["date"].min(),
            "last_date": g["date"].max(),
            "n_operations": g["operation"].nunique(dropna=True),
            "n_ksymbols": g["k_symbol"].nunique(dropna=True),
            "amount_median": g["amount"].median(),
        }
    )
    out["span_days"] = (out["last_date"] - out["first_date"]).dt.days
    out["txn_per_month"] = out["n_transactions"] / (
        out["span_days"].clip(lower=1) / 30.44
    )
    return out.reset_index()


if __name__ == "__main__":
    download_raw()
    t = load_transactions()
    print(f"transactions: {len(t):,}  accounts: {t.account_id.nunique():,}")
    print(f"date range: {t.date.min().date()} to {t.date.max().date()}")
    for name in EXPECTED_ROWS:
        p = os.path.join(DEFAULT_RAW_DIR, f"{name}.csv")
        print(f"{name:9s} sha256={file_sha256(p)[:16]}  rows={EXPECTED_ROWS[name]}")
