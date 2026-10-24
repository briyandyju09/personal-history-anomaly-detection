# How Much Personal History Is Enough?

Measuring the data requirement for individualised financial behavioural anomaly detection.

**Status: first-draft research project, independently re-verified and extended.** Every number in `PAPER.md` was produced by the code here. A second session independently re-executed the entire pipeline from a clean environment before adding anything new — see `docs/VERIFICATION.md` for the number-by-number comparison (bit-exact match). Anything not executed is listed as not yet evaluated in Appendix A.5 rather than estimated.

---

## What this project asks

How many of a person's own transactions does an unsupervised anomaly detector need before their personal behavioural baseline is worth using?

Deployed systems assume a sufficiency point exists and filter users below it. We measure where it is, whether it exists, and whether it depends on the individual — on real bank data, on a controlled simulation, and (to test whether the finding is a quirk of one dataset) on a second, structurally different synthetic dataset.

## What it is not

Not a fraud detection study. Not a new algorithm. Not evidence that any of this is production-ready. An anomaly is a deviation from a behavioural baseline; it is not fraud.

---

## Headline results

On 300 real bank accounts, best model (shrinkage-regularised Mahalanobis):

| Question | Answer |
|---|---|
| Where does ranking performance saturate? | 90% of best observed at **25** transactions, saturated by 50 (fit R² = 0.970). **Identical at all 5 independent cohort-selection seeds tested (E5).** |
| Where does it get a usable operating point? | **Depends on the threshold rule.** In-sample calibration (first draft): 50 transactions, and at 10 it flags 96% of everything. Leave-one-out calibration (E9, corrected): about **25** transactions for Mahalanobis, the autoencoder and the hybrid (FPR 0.066 / 0.049 / 0.040 vs 0.05 nominal), about 50 for k-NN, not reached within 250 for Isolation Forest; at 10 Mahalanobis flags only 1.2% (recall 0.055). The "96%" figure was an in-sample-scoring artefact, but the bias is not uniform in direction (Isolation Forest at n=5 goes the other way). All six detectors plus the hybrid, five seeds, exact leave-one-out. |
| Does across-individual performance ever stabilise? | **No**, at any size we evaluated |
| Do sophisticated models help under scarcity? | No **on this dataset** — Isolation Forest is worst at every size and 340× more expensive. **But this reverses on our second dataset** (TabFormer, synthetic): Isolation Forest is competitive-to-best at small history sizes there. The claim is dataset-dependent, not universal. |
| Is personalisation always better than a population model? | No. Global wins below ~10 transactions (Mahalanobis), ~50 (k-NN) on Berka; below ~25-100 on TabFormer, whose global model is stronger |
| Does a hybrid help? | Yes on real data, never worse than either parent. Fails in simulation when personal signal lives in the representation |
| Does behavioural variability set the requirement? | Yes under controlled manipulation (50 to 500 transactions across archetypes). Null on real Berka data, but a real (Holm-significant) effect on TabFormer's geographic descriptor — mild evidence the real-data null reflects missing channels |
| Does a negative 100→250 result mean older history is "stale" (drift)? | We tested this directly (E6): drift is real but does **not** predict which accounts see the effect. The mechanism is an open question again |
| Is false-positive burden concentrated in poorer/rural districts? | No Holm-significant relationship (E8 in-sample FPR; E8b leave-one-out FPR). Ecological audit, 300 accounts, cannot exclude weak effects; an uncorrected weak negative salary trend is reported, not claimed |

---

## Repository layout

```
research/
├── PAPER.md                       paper (independently re-verified + extended)
├── PAPER.md.reference              the prior session's original first draft, kept for comparison
├── README.md
├── requirements.txt
├── docs/
│   ├── DECISION_LOG.md             every consequential decision, with reasons
│   ├── ENVIRONMENT.md              exact environment this session verified against
│   └── VERIFICATION.md             number-by-number re-verification result
├── data/
│   ├── README.md                  acquisition and licence
│   ├── raw/                       Berka, downloaded, not redistributed
│   └── raw_tabformer/             TabFormer (E7), downloaded, not redistributed
├── src/
│   ├── data_loader.py             download + dual-mirror integrity verification
│   ├── feature_engineering.py     causal features, complexity descriptors (Berka)
│   ├── anomaly_injection.py       Protocol I and Protocol P (Berka)
│   ├── models.py                  detectors, global and hybrid arms (shared, dataset-agnostic)
│   ├── evaluation.py              imbalance-appropriate metrics (shared)
│   ├── stats.py                   bootstrap, tests, saturation, thresholds (shared)
│   ├── simulation.py              controlled behavioural simulation
│   ├── experiment_engine.py       split design and experiment loop (Berka)
│   └── tabformer_*.py             loader/features/injection/engine for E7 (TabFormer)
├── experiments/
│   ├── pilot.py                   pipeline validation
│   ├── run_all.py                 E1-E4 on real data
│   ├── e3_temporal_fixed.py       corrected RQ5
│   ├── run_simulation.py          Experiment B
│   ├── analyse.py                 real-data analysis and figures
│   ├── analyse_simulation.py      simulation analysis and figures
│   ├── e5_multiseed.py            E5: 5-seed replication
│   ├── e6_drift.py                E6: direct drift measurement
│   ├── run_tabformer.py           E7: TabFormer replication
│   ├── analyse_tabformer.py       E7 tables and figures
│   ├── e8_disparate_impact.py     E8: disparate-impact audit (in-sample FPR)
│   ├── e9_loo_calibration.py      E9: in-sample vs leave-one-out threshold calibration (Mahalanobis, k-NN, primary seed)
│   ├── e9_loo_full.py             E9 full: every detector + hybrid, five seeds, exact LOO (~4 h, resumable)
│   ├── analyse_e9_full.py         E9 full: summary and paired tables
│   └── e8b_disparate_impact_loo.py  E8b: E8 repeated on the leave-one-out FPR
├── results/
│   ├── raw/                       per-account/entity per-model records
│   ├── tables/                    analysis tables
│   ├── figures/                   figures
│   └── _prior_run/                snapshot of the prior session's results, kept for comparison
└── web/
    ├── index.html                 interactive research site (built, not hand-edited)
    ├── index_template.html        page template — edit this, not index.html
    ├── data.json                  data the page embeds, rebuilt from results/
    ├── build_data.py              assembles data.json from results/
    ├── build_site.py              stitches index_template.html + data.json -> index.html
    └── md2html.py                 minimal Markdown->HTML converter for the embedded paper
```

---

## Reproducing

```bash
pip install -r requirements.txt

# Real-data pipeline (Berka)
python -m src.data_loader               # download + dual-mirror verify + print SHA-256
python -m experiments.pilot             # pipeline validation
python -m experiments.run_all           # E1-E4, real data
python -m experiments.e3_temporal_fixed # corrected RQ5
python -m experiments.run_simulation    # Experiment B (controlled simulation)
python -m experiments.analyse           # real-data tables/figures
python -m experiments.analyse_simulation

# Extensions added in the independent re-verification session
python -m experiments.e5_multiseed        # E5: 5-seed replication of E1 + simulation
python -m experiments.e6_drift            # E6: direct drift measurement (needs e1_impostor.csv)
python -m experiments.run_tabformer       # E7: download TabFormer first, see src/tabformer_loader.py
python -m experiments.analyse_tabformer
python -m experiments.e8_disparate_impact # E8: disparate-impact audit (needs e1_impostor.csv)
python -m experiments.e9_loo_calibration  # E9: in-sample vs leave-one-out calibration
python -m experiments.e9_loo_full         # E9 full: all detectors + hybrid, 5 seeds (~4 h on 20 workers)
python -m experiments.analyse_e9_full     # E9 full: summary tables
python -m experiments.e8b_disparate_impact_loo  # E8b: E8 on LOO FPR (needs e9 output)

# Rebuild the interactive site from the results above
python -m web.build_data
python -m web.build_site
```

Master seed `20260920` throughout, except E5, which sweeps `{20260920, 20260921, 20260922, 20260923, 20260924}` by design. Per-account seeds are `seed * 100003 + account_id` (Berka) or a stable hash of the entity string (TabFormer), so any single account/entity can be re-run in isolation.

The Berka loader verifies every downloaded table against the record counts published by the primary source **and** cross-checks byte content against a second, independent GitHub mirror; it refuses to proceed on either mismatch.

Wall-clock times are intentionally not given here — they are hardware- and load-dependent (our re-verification session ran several of these concurrently) and belong in a run log, not a reproducibility claim.

---

## Design choices that keep the evaluation honest

**No leakage.** The evaluation window is the last 60 transactions of each account/entity. Training history is the *n* transactions immediately before it. Nothing is ever both trained on and tested on. The window is identical across all history sizes, so comparisons are paired within account.

**Non-circular ground truth.** Anomalies are defined by provenance, not by a statistical rule. A held-out transaction is replaced by a real transaction from a different real account. No detector's score is involved in producing the label.

**No test-label threshold tuning, but the first-draft rule was biased.** Operating points in the main pipeline come from the 95th percentile of the model's scores on its own training data. That is in-sample and biased low at small n (E9 showed the apparent "flags 96% at n=10" result is largely this artefact). The corrected leave-one-out rule now covers the six personal detectors and the shrinkage hybrid on Berka at five seeds; the global arm, score-fusion hybrid, parametric protocol, simulation and TabFormer still use the in-sample rule, which is biased low for most detectors at small n but not an upper bound for all of them. PR-AUC results do not depend on any threshold rule.

**Degenerate regimes are recorded, not hidden.** At n = 1 dispersion is undefined. The affected models fall back to documented behaviour and every row carries a `degenerate` flag.

**Account-level resampling.** Bootstrap and paired tests resample accounts, the independent unit. Resampling transactions would understate uncertainty.

**Interpretations get tested, not left standing.** When the first draft offered an untested explanation (behavioural drift, §7.6), the second session built a direct test for it (E6) rather than letting the plausible-sounding story stand unchecked. It failed the test, and the paper says so.

---

## Known limitations

Read `PAPER.md` §9 in full before citing anything here. The largest:

- One **real** dataset, one bank, 1990s Czech retail banking. A second, synthetic, structurally different dataset (TabFormer, E7) probes but cannot close this — and itself surfaced a genuine divergence (Isolation Forest's ranking reverses), not a clean confirmation.
- Day-resolution timestamps and no merchant/category/geography fields on real data.
- Real-data history sizes stop at 250 because only 119 of 4,500 accounts have 500+ transactions.
- The evaluated cohort is the more active half of the dataset, and so is least representative of the sparse-history users the study is most about.
- Neither injection protocol models an adversary trying to imitate the account holder.
- 10% injection rate is far above realistic fraud prevalence. Curve *shape* transfers; absolute values do not.
- Cohort selection was repeated at 5 seeds (E5) for the primary curve, not for everything — E2, Isolation Forest/autoencoder in the simulation, and E3/E4/E7/E8 remain single-seed.
- The disparate-impact audit (E8) is ecological (district-level); Berka has no individual-level demographic field.

---

## Ethics

Sparse-history users are disproportionately new customers, young people and recent immigrants, and they are the users with the thinnest personal baseline. The first draft said a personal model flags almost everything for them; that came from the biased in-sample threshold and is withdrawn (with leave-one-out calibration a 10-transaction detector is nearly silent instead, and from about 25 transactions FPR is 6-8% against a 5% target). The concern about deploying personalisation without a history floor stands, in a less clear-cut form. A district-level disparate-impact audit (E8, E8b) found no Holm-significant relationship between false-positive rate and a district's salary, unemployment, urbanisation or region — which narrows but does not clear this concern, since individual-level axes (age, tenure, immigration status, disability) remain untested. See `PAPER.md` §10.
