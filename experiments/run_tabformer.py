"""
E7: card-structured replication on IBM TabFormer (synthetic).

Runs Protocol I (impostor substitution, primary) at history sizes up to 500,
which real Berka data cannot support (Sec. 6.3 of PAPER.md). Also reports a
Protocol F exploratory pooled analysis against the dataset's own
generator-produced "Is Fraud?" column.

THIS ENTIRE EXPERIMENT USES SYNTHETIC DATA. Results are reported in their own
section, never pooled with the real-data (Berka) results or the controlled
behavioural simulation (src/simulation.py). See src/tabformer_loader.py for
dataset provenance and src/tabformer_engine.py for the split design (which is
identical in structure to src/experiment_engine.py's, applied to a different
schema).

Usage:
    python -m experiments.run_tabformer
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabformer_engine import EVAL_WINDOW, HISTORY_SIZES, run_history_size_experiment
from src.tabformer_features import behavioural_descriptors
from src.tabformer_loader import EXPECTED_ROWS, EXPECTED_USERS, load_cohort

SEED = 20260920
N_EVAL = 300
N_DONORS = 400
OUT = "results/raw"
TAB = "results/tables"


def main():
    t0 = time.time()
    min_transactions = max(HISTORY_SIZES) + EVAL_WINDOW
    print(f"loading cohort: min_transactions={min_transactions}, "
          f"n_eval={N_EVAL}, n_donors={N_DONORS}, seed={SEED}", flush=True)
    cohort = load_cohort(min_transactions, N_EVAL, N_DONORS, SEED)
    trans = cohort.trans
    print(f"eligible entities: {cohort.n_eligible:,}  "
          f"loaded rows: {len(trans):,}  "
          f"(loader verified {EXPECTED_ROWS:,} total rows, "
          f"{EXPECTED_USERS:,} users in the source file)", flush=True)

    meta = {
        "dataset": "IBM TabFormer credit_card (SYNTHETIC)",
        "eligible_entities": int(cohort.n_eligible),
        "min_transactions_required": min_transactions,
        "n_evaluated": int(len(cohort.accounts)),
        "n_donors": int(len(cohort.donors)),
        "eval_window": EVAL_WINDOW,
        "history_sizes": HISTORY_SIZES,
        "seed": SEED,
    }
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    with open(f"{OUT}/e7_tabformer_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)

    res = run_history_size_experiment(
        trans, cohort.accounts, cohort.donors,
        history_sizes=HISTORY_SIZES, seed=SEED, verbose=True,
    )
    res.to_csv(f"{OUT}/e7_tabformer_impostor.csv", index=False)
    print(f"\nrows: {len(res)}  elapsed: {time.time()-t0:.0f}s", flush=True)

    piv = (
        res[res.arm == "personal"]
        .pivot_table(index="history_size", columns="model", values="pr_auc")
        .round(3)
    )
    print("\nPersonal arm, mean PR-AUC by history size (Protocol I, SYNTHETIC data):")
    print(piv)

    # --- Protocol F, exploratory, pooled (not paired -- see tabformer_injection.py) ---
    total_windows = res[res.arm == "personal"].groupby(
        ["history_size", "model"]
    )["window"].first().groupby("history_size").first()
    fraud_summary = (
        res[res.arm == "personal"]
        .groupby("history_size")["genuine_fraud_in_window"]
        .agg(["sum", "count"])
    )
    fraud_summary.columns = ["genuine_fraud_txns_seen", "account_model_rows"]
    print("\nProtocol F exploratory: genuine (generator-labelled) fraud rows "
          "present in evaluation windows, by history size (pooled across the "
          "300 evaluated entities; each entity's window is counted once per "
          "history size regardless of model, so divide by n_models to get "
          "entity-level totals):")
    print(fraud_summary)
    fraud_summary.to_csv(f"{TAB}/e7_protocol_f_fraud_prevalence.csv")

    # Behavioural descriptors for the evaluated cohort (for E4-style analysis)
    desc_rows = []
    by_entity = {e: g for e, g in trans[trans.entity_id.isin(cohort.accounts)].groupby("entity_id")}
    for ent, g in by_entity.items():
        d = behavioural_descriptors(g.tail(250))
        d["entity_id"] = ent
        desc_rows.append(d)
    pd.DataFrame(desc_rows).to_csv(f"{OUT}/e7_tabformer_complexity.csv", index=False)

    print(f"\ndone in {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
