"""
Assemble web/data.json from committed result files under results/raw and
results/tables. This is the single source of truth for the interactive
site: web/index.html embeds this JSON literally (as a
<script type="application/json"> block, parsed at load time) so the page
works when opened directly from disk (file://), without a server.

No number in web/index.html is hand-typed; everything traces to a file
this script reads, which in turn traces to executed experiment code.

Usage:
    python -m web.build_data
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.md2html import convert as md_to_html

RAW = "results/raw"
TAB = "results/tables"
OUT = "web/data.json"


def read_csv(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def curve_dict(df, arm, metric="pr_auc"):
    """{model: {n:[...], mean:[...]}} for a metric, from a raw per-row CSV."""
    out = {}
    sub = df[df.arm == arm]
    for model, g in sub.groupby("model"):
        g = g.groupby("history_size")[metric].mean().sort_index()
        out[model] = {"n": [int(x) for x in g.index], "mean": [round(float(x), 4) for x in g.values]}
    return out


def main():
    D = {}

    # ---- E1/E2 curves with CI (reuse results/tables if analyse.py has run) ----
    e1c = read_csv(f"{TAB}/e1_curves_personal.csv")
    if e1c is not None:
        e1 = {}
        for model, g in e1c.groupby("model"):
            g = g.sort_values("history_size")
            e1[model] = {
                "n": [int(x) for x in g.history_size],
                "mean": [round(float(x), 4) for x in g["mean"]],
                "lo": [round(float(x), 4) for x in g.ci_lo],
                "hi": [round(float(x), 4) for x in g.ci_hi],
                "sd": [round(float(x), 4) for x in g.sd],
            }
        D["e1"] = e1
    e1g = read_csv(f"{TAB}/e1_global_means.csv")
    if e1g is not None:
        D["e1_global"] = dict(zip(e1g.iloc[:, 0], e1g.iloc[:, 1].round(4)))

    e2c = read_csv(f"{TAB}/e2_curves_personal.csv")
    if e2c is not None:
        e2 = {}
        for model, g in e2c.groupby("model"):
            g = g.sort_values("history_size")
            e2[model] = {
                "n": [int(x) for x in g.history_size],
                "mean": [round(float(x), 4) for x in g["mean"]],
                "lo": [round(float(x), 4) for x in g.ci_lo],
                "hi": [round(float(x), 4) for x in g.ci_hi],
                "sd": [round(float(x), 4) for x in g.sd],
            }
        D["e2"] = e2
    e2g = read_csv(f"{TAB}/e2_thresholds.csv")  # placeholder if global not saved separately

    # ---- Per-metric curves for the metric toggle (Main Experiment) ----
    e1_raw = read_csv(f"{RAW}/e1_impostor.csv")
    if e1_raw is not None:
        metrics = {}
        for metric in ["precision", "recall", "f1", "pr_auc", "fpr"]:
            metrics[metric] = curve_dict(e1_raw, "personal", metric)
        D["metrics_personal"] = metrics
        D["metrics_global"] = {
            metric: {
                m: round(float(v), 4)
                for m, v in e1_raw[e1_raw.arm == "global"].groupby("model")[metric].mean().items()
            }
            for metric in ["precision", "recall", "f1", "pr_auc", "fpr"]
        }

    # ---- arms, op table, cost, saturation, thresholds (unchanged shape) ----
    if e1_raw is not None:
        arms = {}
        for model, g in e1_raw.groupby("model"):
            arm_d = {}
            for arm, gg in g.groupby("arm"):
                gg2 = gg.groupby("history_size")["pr_auc"].mean().sort_index()
                arm_d[arm] = {"n": [int(x) for x in gg2.index], "mean": [round(float(x), 4) for x in gg2.values]}
            arms[model] = arm_d
        D["arms"] = arms

        top = e1_raw[(e1_raw.arm == "personal") & (e1_raw.model == "mahalanobis")]
        opg = top.groupby("history_size").agg(
            precision=("precision", "mean"), recall=("recall", "mean"),
            f1=("f1", "mean"), fpr=("fpr", "mean"), flag_rate=("flag_rate", "mean"),
        ).sort_index()
        D["op"] = {
            "n": [int(x) for x in opg.index],
            **{c: [round(float(v), 3) for v in opg[c]] for c in ["precision", "recall", "f1", "fpr", "flag_rate"]},
        }

        cost = e1_raw[e1_raw.arm == "personal"].groupby(["model", "history_size"])["cost_ms"].median()
        D["cost"] = {
            model: {str(int(n)): round(float(v), 2) for n, v in g.droplevel(0).items()}
            for model, g in cost.groupby(level=0)
        }

    sat = read_csv(f"{TAB}/e1_saturation_fits.csv")
    if sat is not None:
        D["sat"] = sat.round(3).to_dict("records")

    thr = read_csv(f"{TAB}/e1_thresholds.csv")
    if thr is not None:
        D["thresholds_e1"] = thr.round(3).to_dict("records")

    pvg = read_csv(f"{TAB}/e1_personal_vs_global_mahalanobis.csv")
    if pvg is not None:
        D["pvg"] = pvg.round(4).to_dict("records")
    pvh = read_csv(f"{TAB}/e1_personal_vs_hybrid_shrinkage_mahalanobis.csv")
    if pvh is not None:
        D["pvh"] = pvh.round(4).to_dict("records")

    # ---- Simulation ----
    b1 = read_csv(f"{RAW}/b1_simulation.csv")
    if b1 is not None:
        sim = {}
        sub = b1[(b1.arm == "personal") & (b1.model == "mahalanobis")]
        for arch, g in sub.groupby("archetype"):
            g = g.groupby("history_size")["pr_auc"].mean().sort_index()
            sim[arch] = {"n": [int(x) for x in g.index], "mean": [round(float(x), 4) for x in g.values]}
        D["sim"] = sim

    archthr = read_csv(f"{TAB}/b1_saturation_by_archetype.csv")
    b3thr = read_csv(f"{TAB}/b3_archetype_thresholds.csv")
    if b3thr is not None:
        D["archthr"] = b3thr.round(3).to_dict("records")

    cls = read_csv(f"{TAB}/b2_anomaly_class_detectability.csv")
    if cls is not None:
        cls = cls.set_index(cls.columns[0])
        D["classes"] = {"n": [int(x) for x in cls.index], **{c: [round(float(v), 3) for v in cls[c]] for c in cls.columns}}

    comp_real = read_csv(f"{TAB}/e4_complexity_correlations.csv")
    if comp_real is not None:
        D["complex_real"] = comp_real.round(4).to_dict("records")
    comp_sim = read_csv(f"{TAB}/b3_complexity_correlations.csv")
    if comp_sim is not None:
        D["complex_sim"] = comp_sim.round(4).to_dict("records")

    temporal = read_csv(f"{TAB}/e3_temporal_fixed.csv")
    if temporal is not None:
        D["temporal"] = temporal.round(4).to_dict("records")

    # ---- E5: multi-seed ----
    e5_thr = read_csv(f"{TAB}/e5_threshold_seed_stability.csv")
    if e5_thr is not None:
        D["e5_threshold_stability"] = e5_thr.round(3).to_dict("records")
    e5_curve = read_csv(f"{TAB}/e5_curve_seed_variation_berka_mahalanobis.csv")
    if e5_curve is not None:
        D["e5_curve_variation"] = e5_curve.round(4).to_dict("records")
    e5_cross = read_csv(f"{TAB}/e5_crossover_seed_stability.csv")
    if e5_cross is not None:
        D["e5_crossover"] = e5_cross.round(3).to_dict("records")

    # ---- E6: drift ----
    e6_drift = read_csv(f"{RAW}/e6_drift_per_account.csv")
    if e6_drift is not None:
        D["e6_drift_summary"] = {
            "n": int(len(e6_drift)),
            "mean": round(float(e6_drift.drift_100_250.mean()), 3),
            "median": round(float(e6_drift.drift_100_250.median()), 3),
            "std": round(float(e6_drift.drift_100_250.std()), 3),
        }
    e6_corr = read_csv(f"{TAB}/e6_drift_vs_gain_correlation.csv")
    if e6_corr is not None:
        D["e6_corr"] = e6_corr.round(4).to_dict("records")
    e6_lag = read_csv(f"{RAW}/e6_drift_lag_sweep.csv")
    if e6_lag is not None:
        lag = e6_lag.groupby("lag")["drift"].mean().sort_index()
        D["e6_lag"] = {"lag": [int(x) for x in lag.index], "mean": [round(float(x), 3) for x in lag.values]}

    # ---- E7: TabFormer ----
    e7c = read_csv(f"{TAB}/e7_curves_personal.csv")
    if e7c is not None:
        e7 = {}
        for model, g in e7c.groupby("model"):
            g = g.sort_values("history_size")
            e7[model] = {
                "n": [int(x) for x in g.history_size],
                "mean": [round(float(x), 4) for x in g["mean"]],
                "lo": [round(float(x), 4) for x in g.ci_lo],
                "hi": [round(float(x), 4) for x in g.ci_hi],
            }
        D["e7"] = e7
    e7g = read_csv(f"{TAB}/e7_global_means.csv")
    if e7g is not None:
        D["e7_global"] = dict(zip(e7g.iloc[:, 0], e7g.iloc[:, 1].round(4)))
    e7thr = read_csv(f"{TAB}/e7_thresholds.csv")
    if e7thr is not None:
        D["e7_thresholds"] = e7thr.round(3).to_dict("records")
    e7pvg = read_csv(f"{TAB}/e7_personal_vs_global_mahalanobis.csv")
    if e7pvg is not None:
        D["e7_pvg"] = e7pvg.round(4).to_dict("records")
    e7comp = read_csv(f"{TAB}/e7_complexity_correlations.csv")
    if e7comp is not None:
        D["e7_complexity"] = e7comp.round(4).to_dict("records")

    # ---- E8: disparate impact ----
    e8_tercile = read_csv(f"{TAB}/e8_fpr_by_district_tercile.csv")
    if e8_tercile is not None:
        D["e8_tercile"] = e8_tercile.round(4).to_dict("records")
    e8_corr = read_csv(f"{TAB}/e8_fpr_district_correlation.csv")
    if e8_corr is not None:
        D["e8_corr"] = e8_corr.round(4).to_dict("records")

    # ---- Dataset facts (Explore the Data / Dataset section) ----
    acc = read_csv(f"{RAW}/account_summary.csv")
    if acc is not None:
        n = acc["n_transactions"]
        hist, edges = np.histogram(np.log10(n.clip(lower=1)), bins=24)
        D["dataset_depth_hist"] = {
            "edges": [round(float(10**e), 1) for e in edges],
            "counts": [int(x) for x in hist],
        }
        D["dataset_facts"] = {
            "n_accounts": int(len(acc)),
            "median_txns": float(n.median()),
            "mean_txns": round(float(n.mean()), 1),
            "max_txns": int(n.max()),
            "ge100": int((n >= 100).sum()),
            "ge250": int((n >= 250).sum()),
            "ge500": int((n >= 500).sum()),
        }

    meta = read_csv(f"{RAW}/e1_e2_meta.json") if False else None
    meta_path = f"{RAW}/e1_e2_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            D["e1_meta"] = json.load(fh)
    e7meta_path = f"{RAW}/e7_tabformer_meta.json"
    if os.path.exists(e7meta_path):
        with open(e7meta_path) as fh:
            D["e7_meta"] = json.load(fh)

    # ---- Full paper, converted to HTML for the "Paper" section ----
    paper_path = "PAPER.md"
    if os.path.exists(paper_path):
        with open(paper_path, encoding="utf-8") as fh:
            paper_md = fh.read()
        D["paper_html"] = md_to_html(paper_md)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(D, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {OUT}, {os.path.getsize(OUT):,} bytes, {len(D)} top-level keys")
    print(sorted(D.keys()))


if __name__ == "__main__":
    main()
