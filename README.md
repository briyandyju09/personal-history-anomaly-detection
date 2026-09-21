# How Much Personal History Is Enough?

Measuring the data requirement for individualised financial behavioural anomaly detection.

**Status: first-draft research project.** Every number in `PAPER.md` was produced by the code here. Anything not executed is listed as not yet evaluated in Appendix A.5 rather than estimated.

---

## What this project asks

How many of a person's own transactions does an unsupervised anomaly detector need before their personal behavioural baseline is worth using?

Deployed systems assume a sufficiency point exists and filter users below it. We measure where it is, whether it exists, and whether it depends on the individual.

## What it is not

Not a fraud detection study. Not a new algorithm. Not evidence that any of this is production-ready. An anomaly is a deviation from a behavioural baseline; it is not fraud.

---

## Headline results

On 300 real bank accounts, best model (shrinkage-regularised Mahalanobis):

| Question | Answer |
|---|---|
| Where does ranking performance saturate? | 90% of best observed at **25** transactions, saturated by 50 (fit R² = 0.970) |
| Where does it become deployable? | **50** transactions. At 10 it flags 96% of all transactions |
| Does across-individual performance ever stabilise? | **No**, at any size we evaluated |
| Do sophisticated models help under scarcity? | No. Isolation Forest is worst at every size and 340× more expensive |
| Is personalisation always better than a population model? | No. Global wins below ~10 transactions (Mahalanobis), ~50 (k-NN) |
| Does a hybrid help? | Yes on real data, never worse than either parent. Fails in simulation when personal signal lives in the representation |
| Does behavioural variability set the requirement? | Yes under controlled manipulation (50 to 500 transactions across archetypes). **Null on real data** |

---

## Repository layout

```
research/
├── PAPER.md                       first-draft paper
├── README.md
├── requirements.txt
├── docs/
│   └── DECISION_LOG.md            every consequential decision, with reasons
├── data/
│   ├── README.md                  acquisition and licence
│   └── raw/                       downloaded, not redistributed
├── src/
│   ├── data_loader.py             download + integrity verification
│   ├── feature_engineering.py     causal features, complexity descriptors
│   ├── anomaly_injection.py       Protocol I and Protocol P
│   ├── models.py                  detectors, global and hybrid arms
│   ├── evaluation.py              imbalance-appropriate metrics
│   ├── stats.py                   bootstrap, tests, saturation, thresholds
│   ├── simulation.py              controlled behavioural simulation
│   └── experiment_engine.py       split design and experiment loop
├── experiments/
│   ├── pilot.py                   pipeline validation
│   ├── run_all.py                 E1-E4 on real data
│   ├── e3_temporal_fixed.py       corrected RQ5
│   ├── run_simulation.py          Experiment B
│   ├── analyse.py                 real-data analysis and figures
│   └── analyse_simulation.py      simulation analysis and figures
├── results/
│   ├── raw/                       per-account per-model records
│   ├── tables/                    analysis tables
│   └── figures/                   figures
└── web/
    └── index.html                 interactive research site
```

---

## Reproducing

```bash
pip install -r requirements.txt

python -m src.data_loader               # download + verify + print SHA-256
python -m experiments.pilot             # pipeline validation      ~4 min
python -m experiments.run_all           # E1-E4, real data        ~25 min
python -m experiments.e3_temporal_fixed # corrected RQ5            ~4 min
python -m experiments.run_simulation    # Experiment B             ~7 min
python -m experiments.analyse           # real-data tables/figures
python -m experiments.analyse_simulation
```

Master seed `20260920`. Per-account seeds are `seed * 100003 + account_id`, so any single account can be re-run in isolation.

The loader will refuse to proceed if any downloaded table does not match the record counts published by the primary source.

---

## Design choices that keep the evaluation honest

**No leakage.** The evaluation window is the last 60 transactions of each account. Training history is the *n* transactions immediately before it. Nothing is ever both trained on and tested on. The window is identical across all history sizes, so comparisons are paired within account.

**Non-circular ground truth.** Anomalies are defined by provenance, not by a statistical rule. A held-out transaction is replaced by a real transaction from a different real account. No detector's score is involved in producing the label.

**No test-label threshold tuning.** Operating points come from the 95th percentile of the model's scores on its own training data, which is what a deployed system could actually do, and which visibly breaks when history is too short.

**Degenerate regimes are recorded, not hidden.** At n = 1 dispersion is undefined. The affected models fall back to documented behaviour and every row carries a `degenerate` flag.

**Account-level resampling.** Bootstrap and paired tests resample accounts, the independent unit. Resampling transactions would understate uncertainty.

---

## Known limitations

Read `PAPER.md` §9 in full before citing anything here. The largest:

- One dataset, one bank, 1990s Czech retail banking. Day-resolution timestamps, no merchant, category or geography.
- Real-data history sizes stop at 250 because only 119 of 4,500 accounts have 500+ transactions.
- The evaluated cohort is the more active half of the dataset, and so is least representative of the sparse-history users the study is most about.
- Neither injection protocol models an adversary trying to imitate the account holder.
- 10% injection rate is far above realistic fraud prevalence. Curve *shape* transfers; absolute values do not.
- Single seed for cohort selection.

---

## Ethics

Sparse-history users are disproportionately new customers, young people and recent immigrants, and they are exactly the users for whom a personal model flags almost everything. Deploying personalisation without a history floor concentrates false positives on populations already more likely to face financial exclusion. See `PAPER.md` §10.
