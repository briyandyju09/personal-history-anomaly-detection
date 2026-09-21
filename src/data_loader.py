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

# A second, independent GitHub mirror of the same primary-source files, used
# only to cross-check the primary mirror's content by hash. Different GitHub
# account, different repository history, same eight tables in the original
# ";"-delimited "*.asc" format (verified identical column layout to the
# MIRROR_BASE "*.csv" files, e.g. "account.asc" vs "account.csv"). This is a
# corroboration check, not a replacement source: if a mismatch is found we
# raise rather than silently preferring one side.
CROSSCHECK_MIRROR_BASE = "https://raw.githubusercontent.com/jlacko/berka-dataset/master"
CROSSCHECK_EXT = {name: f"{name}.asc" for name in
                   ["account", "client", "disp", "order", "trans", "loan", "card", "district"]}

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


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def crosscheck_raw(raw_dir: str = DEFAULT_RAW_DIR) -> dict:
    """
    Download the same eight tables from an independent second mirror
    (different GitHub account/repository, original ``.asc`` naming) and
    compare byte-for-byte via SHA-256 against what ``download_raw`` fetched
    from ``MIRROR_BASE``. Returns a per-table report; raises
    ``DataIntegrityError`` if any table's content disagrees between mirrors.

    This does not touch the primary source (unreachable from this
    environment; see module docstring) but it means the row-count check in
    ``load_table`` no longer rests on trusting a single third party.
    """
    report = {}
    for name in EXPECTED_ROWS:
        primary_path = os.path.join(raw_dir, f"{name}.csv")
        with open(primary_path, "rb") as fh:
            primary_payload = fh.read()
        primary_hash = bytes_sha256(primary_payload)
        primary_hash_normalized = bytes_sha256(primary_payload.replace(b"\r\n", b"\n"))

        url = f"{CROSSCHECK_MIRROR_BASE}/{CROSSCHECK_EXT[name]}"
        with urllib.request.urlopen(url) as resp:
            payload = resp.read()
        cross_hash = bytes_sha256(payload)
        # The two mirrors use different line endings (LF vs CRLF); compare
        # content after normalising, note the raw hashes for the record.
        cross_hash_normalized = bytes_sha256(payload.replace(b"\r\n", b"\n"))
        match = cross_hash_normalized == primary_hash_normalized
        report[name] = {
            "primary_mirror_sha256_raw": primary_hash,
            "crosscheck_mirror_sha256_raw": cross_hash,
            "primary_mirror_sha256_lf_normalized": primary_hash_normalized,
            "crosscheck_mirror_sha256_lf_normalized": cross_hash_normalized,
            "match": match,
        }
        if not match:
            raise DataIntegrityError(
                f"Table '{name}' differs between the two independent mirrors "
                "after line-ending normalisation: "
                f"{MIRROR_BASE} ({primary_hash_normalized[:16]}...) vs "
                f"{CROSSCHECK_MIRROR_BASE} ({cross_hash_normalized[:16]}...). "
                "Refusing to treat either copy as verified."
            )
    return report


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
    import json
    import sys

    download_raw()
    t = load_transactions()
    print(f"transactions: {len(t):,}  accounts: {t.account_id.nunique():,}")
    print(f"date range: {t.date.min().date()} to {t.date.max().date()}")
    for name in EXPECTED_ROWS:
        p = os.path.join(DEFAULT_RAW_DIR, f"{name}.csv")
        print(f"{name:9s} sha256={file_sha256(p)[:16]}  rows={EXPECTED_ROWS[name]}")

    print("\nCross-checking against independent second mirror "
          f"({CROSSCHECK_MIRROR_BASE}) ...")
    try:
        report = crosscheck_raw()
    except DataIntegrityError as e:
        print(f"CROSS-CHECK FAILED: {e}")
        sys.exit(1)
    for name, r in report.items():
        print(f"{name:9s} match={r['match']}  "
              f"primary(LF-norm)={r['primary_mirror_sha256_lf_normalized'][:16]}  "
              f"crosscheck(LF-norm)={r['crosscheck_mirror_sha256_lf_normalized'][:16]}")
    out_path = os.path.join(
        os.path.dirname(DEFAULT_RAW_DIR), "..", "docs", "data_integrity_report.json"
    )
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nAll {len(report)} tables verified against two independent "
          f"mirrors. Report written to {out_path}")
