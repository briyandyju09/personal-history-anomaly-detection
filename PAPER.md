# How Much Personal History Is Enough? Measuring the Data Requirement for Individualised Financial Behavioural Anomaly Detection

**Status: first draft, independently re-verified and extended. All reported numbers were produced by the code in this repository, executed in the environment recorded in Appendix A.1. Nothing is estimated, extrapolated or illustrative. This version adds four experiments (§6.8) to a first draft completed 2026-09-20: 5-seed replication (E5), direct drift measurement (E6), a second structurally different dataset (E7, synthetic), and a disparate-impact audit (E8). **A further experiment, E9 (2026-10-23, from an automated review loop), found that the first draft's operating-point calibration was biased at small n and withdrew its headline "flags 96% at n = 10" result (§7.2); E8 was repeated as E8b under corrected calibration. E9 was first run on the primary seed for Mahalanobis and k-NN, and was extended on 2026-10-24 to all six detectors plus the shrinkage-hybrid arm at all five E5 seeds (§7.2c); the global arm, the score-fusion hybrid and E7 (TabFormer) still use the in-sample rule for their operating-point metrics.** Findings that changed as a result are flagged inline; findings that did not are unchanged.**

---

## Abstract

Systems that detect unusual financial behaviour by comparing a transaction against an individual's own history must decide when a person has transacted enough for their personal baseline to be worth using. Deployed systems and the patent literature assume such a sufficiency point exists and filter out users below it, but we could not find a study that measures where it lies, or whether it exists. We measure it.

We construct data-efficiency curves for unsupervised behavioural anomaly detection as a function of the number of transactions in one person's own history, on 300 accounts from the PKDD'99 Financial ("Berka") dataset of real Czech bank transactions, on a controlled simulation of 240 individuals with deliberately varied behavioural parameters, and — to test whether the shape found on Berka is dataset-specific — on 300 (User, Card) entities from IBM's synthetic TabFormer card-transaction dataset. Ground truth on real data is defined by provenance rather than by any statistical rule: a held-out transaction is replaced by a real transaction from a different real account, so labels cannot be circular with respect to any detector. Every history size is evaluated on an identical held-out window per account, making comparisons across sizes paired.

Seven findings. First, a threshold exists and is earlier than expected: shrinkage-regularised Mahalanobis distance reaches 90% of its best observed ranking performance at 25 personal transactions (saturating-exponential fit R² = 0.970, τ ≈ 11.5), and gains nothing measurable beyond 100; the same shape (steep rise, early plateau) reappears on the synthetic card dataset, at a later threshold (250) driven by a stronger global baseline. Second, ranking quality and operating-point behaviour need separate treatment, and our first draft got this wrong. At 10 transactions the same model attains PR-AUC 0.441 against a 0.10 prevalence baseline, and with a threshold calibrated on its own in-sample training scores it flags 96% of all transactions. A direct test (E9) showed that figure is an artefact of in-sample scoring, not of data scarcity: with leave-one-out calibration the detector flags 1.2% at 10 transactions (recall 0.058, i.e. silent rather than noisy), and the false-positive rate is within 1.6 to 3 points of its nominal 5% from 25 transactions, the same place the ranking threshold falls. The operating point is highly sensitive to the calibration rule, which we vary in only one way. Extending the check to all six detectors and the hybrid at five seeds shows the artefact is not specific to Mahalanobis (the autoencoder's in-sample flag rate at 10 transactions is 0.69 against 0.025 out-of-sample), is stable across cohorts, and is not uniform in direction (Isolation Forest flags more, not less, under the out-of-sample rule at 5 transactions). The shrinkage hybrid has the best small-history operating behaviour we measured. Third, model sophistication under scarcity is **dataset-dependent, not universal**: Isolation Forest is the weakest detector at every history size on Berka (τ ≈ 71) but is competitive-to-best at small history sizes on the synthetic card dataset, reversing the ranking; univariate z-score baselines on transaction amount never learn at all on either dataset. Fourth, a population model outperforms personalisation below a model- and dataset-dependent crossover (roughly 10 transactions for Mahalanobis and 50 for k-NN on Berka; between 25 and 100 on the synthetic dataset, whose global model is stronger), and a shrinkage hybrid is never worse than either parent at any size on Berka. Fifth, behavioural variability predicts the data requirement strongly under controlled manipulation (Spearman ρ = 0.48 to 0.71, all p < 0.001 after Holm correction; per-archetype thresholds range from 50 to 500 transactions); none of eight descriptors measured on real Berka accounts predicted that account's threshold after correction, but one geographic descriptor computed on the synthetic card dataset did (ρ = 0.19, Holm p = 0.028), which is mild evidence that the real-data null reflects missing channels rather than a genuinely absent effect. Sixth, an interpretation offered in our own first draft — that a negative 100→250 marginal-gain step reflects behavioural drift — was tested directly and did not survive the test: drift accumulates measurably with lag, but per-account drift magnitude does not predict that account's own marginal loss (Holm p = 0.32 for both models tested). Seventh, a district-level disparate-impact audit found no Holm-significant relationship between an account's local false-positive rate and its district's average salary, unemployment rate, urbanisation or region, under either calibration rule (it cannot exclude weak effects at n = 300; a weak, uncorrected negative salary trend is reported, not claimed). All headline curves and thresholds were additionally checked for cohort-selection stability across 5 independent seeds: the relative-90 threshold (25 transactions) and the global/personal crossover (10 transactions) for Mahalanobis were identical at every seed.

We emphasise a distinction the deployment literature often blurs: an anomaly is a deviation from a behavioural baseline. It is not fraud, and this study produces no evidence about fraud.

---

## 1. Introduction

A personalised anomaly detector compares each new transaction against a model of one person's own past behaviour. The appeal is obvious: a 900-unit purchase is unremarkable for one person and extraordinary for another, and a single population-wide rule serves both badly.

The approach has an equally obvious dependency. The personal model has to be estimated from the person's own transactions, and at the moment an account is opened there are none. Somewhere between zero transactions and a lifetime of them, the personal baseline becomes worth using. Production systems handle this with a cutoff. One patent family describes filtering out "unseasoned users, i.e. users with insufficient history" (US 11025653 and continuations). The cutoff is treated as a known engineering parameter.

We could not find published work measuring it. That is the gap this paper addresses.

### 1.1 What this paper contributes

1. **A measurement, not an algorithm.** Data-efficiency curves for per-user behavioural anomaly detection across history sizes from 1 to 500 transactions, on real transaction histories and on a controlled simulation. Every detector we use is standard and off the shelf. The contribution is the measurement protocol and the resulting quantities.

2. **A non-circular ground-truth protocol for unlabelled transaction data.** Provenance-based impostor substitution, which defines anomaly without reference to any statistical score.

3. **Three operational definitions of a personalisation threshold**, applied to the same curves, with their disagreements reported rather than resolved by preference.

4. **A paired global / personal / hybrid comparison at every history size**, with the crossover points located empirically.

5. **A negative result** on whether observed behavioural complexity predicts an individual's data requirement in real data, contrasted with a strong positive result under controlled manipulation.

6. **A second, structurally different dataset used to interrogate, not just confirm, the real-data findings.** Card-structured, synthetic, with merchant/category/geography channels the real dataset lacks (§5.4, §7.10). It surfaced a genuine contradiction (Isolation Forest's relative ranking reverses) rather than a clean replication, which we treat as itself informative about how far the Berka-specific conclusions generalise.

7. **A demonstration that the paper's own untested interpretations get tested, not left standing.** An interpretation offered for a puzzling result in an earlier version of this paper (behavioural drift explaining a negative marginal-gain step, §7.6) was converted into a direct measurement (§6.8, E6) and did not survive it. We report the retraction rather than the more comfortable original story.

### 1.2 What this paper is not

It is not a fraud detection study. It is not a proposal for a new detector. It is not evidence that any of this would work in production, on modern card data, or against an adversary who is trying to imitate the account holder.

---

## 2. Terminology

The brief for this work insisted on a distinction we maintain throughout, because conflating these terms is how anomaly detection acquires unearned authority.

- **Behavioural anomaly.** A transaction that deviates from a behavioural pattern established from prior observations of the same entity. This is what we measure.
- **Statistical outlier.** A point in the tail of a fitted distribution. A model artefact, dependent on the model.
- **Suspicious transaction.** An institutional judgement that a transaction warrants review. Involves policy, cost and context that we do not model.
- **Confirmed fraud.** An adjudicated finding that a transaction was unauthorised. Requires evidence external to the transaction record.

These form no implication chain. Most behavioural anomalies are benign: a holiday, a house move, a new job, a large one-off purchase. **An anomaly detector identifies deviations from a behavioural baseline; it does not establish fraud or wrongdoing.**

---

## 3. Related work and the research gap

### 3.1 Established knowledge

**Per-user behavioural profiling for card fraud is mature.** Dal Pozzolo et al. (2018) formalise realistic evaluation for card fraud detection including feature augmentation from cardholder profiles and the alert-feedback interaction. Bahnsen et al. (2016) aggregate raw transactions into cardholder spending-behaviour features. Jurgovsky et al. (2018) treat detection as sequence classification over a card's transaction stream with LSTMs. Lucas and Jurgovsky (2020) survey the field's characteristic problems: class imbalance, dataset shift and evolving behaviour.

Personal-baseline methods also appear in granted patents. US 11005864 and US 11706234 build per-user profiles from median, median absolute deviation and a "median of abnormality" statistic, scoring new activity by relative deviation. US 11025653 and its continuations construct separate per-user models and explicitly filter out users with insufficient history.

**Cold start is studied, but at the wrong granularity for our question.** The industrial transfer-learning literature addresses cold start when a *processor or market* lacks history, adapting models from data-rich source domains. Graph-based approaches such as SparseFraudNet (ACM TOIS) address new users through relational aggregation. Both route around the per-user data requirement rather than quantifying it.

**Learning-curve methodology is well developed.** Viering and Loog (2023) review learning-curve shapes and establish that no universal functional form exists, that curves are commonly power-law or exponential but can be ill-behaved, and that both rate and saturation level are task-dependent. Mohr and van Rijn (2022) survey their use in decision making. Per-user cold-start sweeps have been run in recommender systems, where SPMC (Cai et al., 2017) caps per-user interaction counts and reports performance across the cap.

### 3.2 The gap

Synthesising the above: the field knows how to build personal behavioural models, knows that users with thin histories are a problem, and has general-purpose tooling for measuring performance against training-set size. What is missing is the application of the third to the first two. Specifically we found no study that reports, for unsupervised behavioural anomaly detection on financial transactions, performance as a function of the number of transactions in the individual's own history, with an operational sufficiency criterion and a matched comparison against population and hybrid models at each size.

### 3.3 Honest limits of this gap analysis

The audit used open web sources. Paywalled venues (IEEE Xplore, ScienceDirect, Springer, ACM DL) were reachable only through abstracts and citing-paper summaries. Industrial work of this kind is frequently unpublished. A closely related study may exist that we did not find. We claim the gap as the result of a documented search, not as a proof of absence.

We also state plainly that the contribution is empirical and methodological. Claiming algorithmic novelty would be unsupportable, since every model here is standard.

---

## 4. Research questions and hypothesis

**RQ1.** How does personalised detection performance change as personal history grows?
**RQ2.** Does performance plateau, and where?
**RQ3.** Can a defensible sufficiency threshold be identified, and do different definitions agree?
**RQ4.** Does the requirement depend on the individual's behavioural variability?
**RQ5.** Is transaction count or temporal coverage the binding constraint?
**RQ6.** Under scarcity, do sophisticated models beat simple statistics?
**RQ7.** How does personalisation compare with a population model at each history size?
**RQ8.** Can combining population and personal information reduce the personal data requirement?

**Central hypothesis (H1).** *Personalised detection performance improves as personal history increases, and the amount required for reliable detection varies with the individual's behavioural complexity and variability.*

H1 is falsifiable in this design. It fails if curves are flat or non-monotone, if personalisation never beats the population model, or if variability does not predict the requirement. All three failure modes occur somewhere in our results.

---

## 5. Data

### 5.1 Datasets considered

Every dataset assessed against the requirements of per-entity identity, timestamps, sufficient per-entity depth, permissive licensing and real behaviour.

| Dataset | Source | Real? | Entity ID | Timestamps | Labels | Verdict |
|---|---|---|---|---|---|---|
| PKDD'99 Financial (Berka) | vse.cz, PKDD'99 Challenge | Yes | Account, 4,500 | Daily, 1993-1998 | None | **Selected** |
| Credit Card Fraud (ULB) | Kaggle mlg-ulb, DbCL-1.0 | Yes | **None** | Seconds, 2 days | Fraud | Rejected: no cardholder ID, so per-user histories are impossible. Fatal. |
| IEEE-CIS Fraud Detection | Kaggle, 2019 | Yes | None ground-truth | Relative | Fraud | Rejected: the commonly used "UID" is a reverse-engineered competition artefact, not an entity key. |
| IBM TabFormer credit card | github.com/IBM/TabFormer | **No** | 2,000 users | Yes | Fraud | Rejected as the real-data layer: synthetic. **Used as a third, clearly-separated replication layer, §5.4.** |
| Sparkov simulated | Kaggle kartik2112, CC0 | **No** | 1,000 customers | Yes | Fraud | Rejected: synthetic. |
| PaySim / BankSim | Kaggle | No | Accounts | Steps | Fraud | Rejected: synthetic, shallow per-entity histories. |
| IBM AML | github.com/IBM/AML-Data, CDLA-Sharing-1.0 | No | Accounts | Yes | Laundering | Rejected: synthetic, and the task is network-level. |
| Bank Account Fraud | NeurIPS 2022, Feedzai | Partly | Applications | No sequence | Fraud | Rejected: application records, not transaction sequences. |

The field's most-cited transaction dataset is unusable here for a simple structural reason: the ULB set's PCA anonymisation removed the cardholder identifier. That fact alone constrains what the literature has been able to ask about personalisation.

### 5.2 Selected dataset

**PKDD'99 Financial Dataset ("Berka").** Prepared by Petr Berka and Marta Sochorova for the PKDD'99 Discovery Challenge. Primary description: <https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm>

- Real, anonymised retail banking transactions from a Czech bank.
- 1,056,320 transactions across 4,500 accounts, 1993-01-01 to 1998-12-31.
- Per-account depth: median 208, mean 234.7, maximum 675. 3,883 accounts have ≥ 100; 1,835 have ≥ 250; only 119 have ≥ 500.
- Transaction fields: `account_id`, `date` (day resolution), `type` (credit/debit), `operation` (5 modes), `amount`, `balance`, `k_symbol` (7 characterisations), partner `bank` and `account`.
- No fraud labels, no anomaly labels.

**Acquisition and verification.** The primary host did not resolve from our execution environment. Files were retrieved from a GitHub mirror (`Kusainov/czech-banking-fin-analysis`). We did not accept the mirror on trust. `src/data_loader.py` checks every one of the eight tables against the record counts published on the primary page and raises on mismatch. All eight matched exactly. SHA-256 digests are in the reproducibility appendix.

**Licensing.** Released for an open research challenge; the primary page carries no machine-readable licence. We treat it as research-use data, use it only for research, and redistribute no raw records. Reproducers download it themselves.

### 5.3 Limitations of the selected data, accepted deliberately

1. **No fraud labels.** The reason this study is about behavioural anomalies and makes no fraud claim.
2. **Day-resolution timestamps.** No hour-of-day modelling is possible.
3. **No merchant, category or geography fields.** Four of the anomaly classes in the original brief cannot be studied on this dataset. They are moved to the simulation, and, partially, to §5.4.
4. **1990s Czech retail banking.** Standing orders, cash withdrawals and pension credits dominate. Generalisation to modern card spending is an open question, not an assumption.
5. **Depth ceiling.** Discussed in §6.3.

### 5.4 A third layer: card-structured replication on IBM TabFormer (SYNTHETIC)

Limitations 2, 3 and 4 above motivated adding a third, clearly separated layer (Experiment E7) rather than accepting them silently. **IBM TabFormer's credit-card dataset is SYNTHETIC** (Padhi et al., 2021) — a generator-produced simulation, not real behaviour — so it is never pooled with the real Berka results or with the controlled behavioural simulation of §6.7. Its purpose is narrow: test whether the real-data curve *shape* (steep early rise, plateau, a global/personal crossover) is an artefact specific to one 1990s retail-banking dataset, on a schema that has the merchant, category, geographic and sub-daily-timing channels Berka lacks.

- Source: <https://github.com/IBM/TabFormer> (Apache-2.0), file `data/credit_card/transactions.tgz`, fetched via the repository's Git-LFS media endpoint, no authentication required.
- 24,386,900 transactions, 2,000 users — independently counted this session by a full chunked pass over the downloaded file, matching the repository's stated "24 million records" and "2,000 users".
- Entity unit: **(User, Card)**, not User alone (a person can hold several cards with materially different usage; per-card profiling is the unit used in card-fraud sequence modelling, e.g. Jurgovsky et al., 2018). 6,139 distinct (User, Card) pairs; 4,538 have ≥ 560 transactions, comfortably supporting history sizes up to **500**, which real Berka cannot (§6.3).
- Fields used: `Amount`, `Use Chip` (swipe/chip/online), `Merchant Name`, `Merchant City`/`State`, `MCC` (merchant category code, 109 distinct values), `Errors?`, date and time to the minute, and the generator's own `Is Fraud?` label (0.122% prevalence, independently measured).
- **Single-source verification.** Unlike Berka's two-independent-mirror SHA-256 cross-check (§5.2), only the primary GitHub LFS endpoint was fetched here (the README also lists an IBM Box mirror of the same archive, not separately downloaded). This is weaker provenance, stated as a limitation of this layer specifically.

---

## 6. Methodology

### 6.1 Split design

For each account, the **evaluation window is the last 60 transactions**. Training history at size *n* is the *n* transactions **immediately preceding** that window.

```
[ ....... earlier history ....... | n training txns | 60 evaluation txns ]
                                   <-------------->   <---------------->
                                        varies              fixed
```

Two consequences matter. No transaction is ever both trained on and tested on, at any *n*. And because the window is identical across all *n* for a given account, comparisons across history sizes are **paired at the account level**, so across-account differences in window difficulty cancel in paired tests.

The population model is fitted on the **earliest** transactions of donor accounts that are disjoint from the evaluated accounts, so it never sees an evaluated account's data at any point in time, and is not advantaged by seeing later data.

### 6.2 Ground truth

The brief warned against circular evaluation, and this was the single most consequential design decision. Labelling anomalies with a statistical rule and then evaluating detectors computing something similar measures agreement between two versions of one rule.

**Protocol I, impostor substitution (primary).** A held-out transaction of the target account is replaced by a real transaction from a different real account. The replacement keeps the target's timeline slot and takes the impostor's `amount`, `type`, `operation` and `k_symbol`. Ground truth is **provenance**: a different person's real behaviour produced this row. No statistical criterion is involved.

Substitution rather than insertion, because insertion would perturb the surrounding inter-arrival gaps and let detectors exploit that artefact instead of behaviour. The cost is real: **timing carries no discriminative signal under Protocol I**, because the injected row inherits the target's true date.

**Protocol P, parametric perturbation (secondary).** A held-out transaction is replaced by a copy of itself with one raw attribute shifted by a magnitude fixed in advance. We report the amount-shift variant at ×5.

Injection rate is 10% in both protocols, giving 6 anomalies in a 60-transaction window. Injection is performed **once per account** and shared across all history sizes, preserving the pairing.

### 6.3 Deviations from the original brief, with reasons

**History sizes stop at 250 on real data, not 500.** Supporting *n* = 500 plus a 60-transaction window requires 560 transactions, which about 60 accounts have, and those are the most active accounts in the dataset. The *n* = 500 point would be computed on a biased cohort a fifth the size of every other point, breaking the paired design and making the curve non-comparable along its own x-axis. We therefore evaluate *n* ∈ {1, 5, 10, 25, 50, 100, 250} on a single fixed cohort of accounts with ≥ 310 transactions (1,319 eligible; 300 evaluated, 400 donors). The simulation extends to 500, where depth is not a constraint.

**`balance` excluded from features.** It is accumulated account state, not a behavioural choice, and under Protocol I a substituted row would carry a balance discontinuity revealing provenance for reasons unrelated to behaviour. Including it would inflate every detector through a protocol artefact.

**Clustering replaced by k-NN distance.** With *n* as small as 1 to 25 in a 21-dimensional, one-hot-heavy space, K-Means cluster counts and DBSCAN density parameters cannot be estimated without tuning on data we do not have. Distance to the k-th nearest training point is the representation-appropriate member of that family, requires no cluster count and degrades gracefully. Using all three regardless would have been variety for its own sake.

**No composite complexity index.** We could not justify a weighting across quantities on different scales. Descriptors are reported and tested individually with Holm correction.

### 6.4 Features

Twenty-one dimensions, all causal (depending only on the transaction and strictly earlier transactions of the same account): log amount; credit indicator; day-of-month and day-of-week as sine/cosine pairs; log inter-arrival gap; one-hot `operation` (5 + missing); one-hot `k_symbol` (7 + missing).

Standardisation is **fitted on training history only**. At *n* = 1 dispersion is undefined and scales collapse to unity, which is recorded as a degenerate regime rather than concealed.

### 6.5 Models

| Model | Form | Role |
|---|---|---|
| Gaussian z | \|z\| on log amount, mean/SD | Simple statistical baseline |
| Robust z | \|z\| on log amount, median/MAD | Robust statistical baseline |
| Mahalanobis | Shrinkage-regularised, ridge-stabilised | Simple multivariate |
| k-NN | Distance to 5th nearest training point | Non-parametric |
| Isolation Forest | 200 trees, seeded | Standard ML detector |
| Autoencoder | 21-4-21 tanh, full-batch Adam, NumPy | Nonlinear representation |

**Hybrid mechanism (RQ8).** Personal location and scatter are shrunk toward population estimates with a weight determined solely by history size:

    μ_hybrid = (n·μ_personal + κ·μ_global) / (n + κ)
    Σ_hybrid = (n·S_personal + κ·S_global) / (n + κ) + εI

κ = 25, fixed in advance, not tuned on test data. At *n* = 0 this is exactly the global model; as *n* grows it converges to the personal model. A score-fusion hybrid with weight n/(n+κ) over rank-normalised scores is also evaluated.

### 6.6 Evaluation

**Why not accuracy.** At 10% prevalence, a detector flagging nothing scores 90% accuracy while detecting nothing. Accuracy cannot separate a useful detector from an inert one here, and changes with prevalence without any change in the detector. We report **PR-AUC as primary** (both precision and recall condition on the minority class) alongside ROC-AUC.

**Threshold without leakage.** Operating-point metrics need a threshold, and choosing one on test labels would leak. We use the 95th percentile of the scores the fitted model assigns to **its own training data** — a rule a deployed system could actually apply. **Correction (E9, §7.2):** that rule is in-sample and is biased low at small *n*, so its "degradation" is largely a property of the rule, not of history length. Operating-point results are therefore reported under both this rule and a leave-one-out rule, and only the latter should be read as a statement about a deployable detector.

**Statistics.** Resampling is over **accounts**, the independent unit; resampling transactions would understate uncertainty. Paired bootstrap (2,000 resamples) and Wilcoxon signed-rank for arm comparisons, with rank-biserial effect size reported alongside every p value, and Holm correction within each family. Saturation curves are fitted **with R² reported**, so poor fits are visible.

### 6.7 Experiment B: controlled simulation

The real data cannot supply merchant, category or geographic channels, and cannot supply *controlled* variability. We simulate 240 individuals across six archetypes (40 each), each defined by a measurable parameter vector rather than a label: log-amount dispersion, merchant count and Zipf exponent, category count, location count and home-location weight, inter-arrival Gamma shape, hour-of-day von Mises concentration, and a drift rate that rotates merchant and category distributions over time.

Archetypes: **regular** (σ = 0.30, 5 merchants, 1 location), **moderate**, **variable** (σ = 1.10, 60 merchants, 8 locations), **traveller** (15 locations, home weight 0.35), **seasonal** (drift 0.004), **burst** (Gamma shape 0.35, producing inactivity punctuated by clusters).

Six anomaly classes generated from deliberately shifted distributions, independent of any detector: amount, merchant, category, geographic, temporal, and multivariate. Ground truth is the generating branch, which no detector observes.

**Nothing from the simulation is a real-world observation, and simulation results are never pooled with real-data results.**

### 6.8 Four additional experiments

Four experiments were added after independently re-verifying the results above (§A.1-A.2), to close gaps this paper's own first draft left explicitly open.

**E5, multi-seed replication.** E1 (Protocol I) and Experiment B's B1 curves were re-run at 5 independent cohort-selection seeds (Berka: a fresh 300-account/400-donor draw per seed; simulation: fresh per-user profile draws per seed), rather than the single seed used throughout the rest of this paper. Scope was narrowed to keep runtime tractable: E2 (parametric protocol) was not replicated across seeds, and the simulation replicate uses only Mahalanobis and k-NN. Both narrowings are stated wherever E5 results are reported.

**E6, direct drift measurement.** §7.6 and §8.2 below *interpret* two observations — the negative 100→250 marginal PR-AUC gain, and the cost of reaching further back in history — as consequences of behavioural drift, without measuring drift directly. E6 tests this. For each evaluated account, `delta_block` is the 150 transactions added when history grows from 100 to 250 (positions 101-250 back), and `recent_block` is the 100 transactions already present at n = 100. Both are converted to the same causal feature representation used by every detector, standardised by a scaler fitted on `recent_block` only, and drift is the Mahalanobis distance between the two blocks' mean feature vectors using `recent_block`'s own ridge-regularised covariance — "how many of my own recent standard deviations away is my older behaviour." This is a descriptive distance computed independently of any detector's score or any anomaly label. We then test, by Spearman correlation (Holm-corrected across the two models tested), whether an account's own drift magnitude predicts that account's own 100→250 marginal gain.

**E7, TabFormer replication.** Described in §5.4. The same split design as §6.1 (fixed 60-transaction evaluation window, training history of size *n* immediately preceding it, donor entities disjoint from evaluated ones) and the same Protocol I injection logic (§6.2) are applied to TabFormer's schema, at history sizes up to 500. A secondary, exploratory **Protocol F** evaluates against the dataset's own generator-produced `Is Fraud?` label — simulated fraud, never described as confirmed fraud — pooled rather than paired across accounts because of its low prevalence.

**E8, disparate-impact audit.** §10 of this paper's first draft asserted a disparate-impact concern about sparse-history users without auditing for it. E8 joins the 300 evaluated Berka accounts to their district's economic attributes (`district.asc`: average salary, unemployment rate, urbanisation, region) and tests, at n = 10 and n = 50, whether the personal Mahalanobis detector's false-positive rate varies with these district variables (Kruskal-Wallis across region; Spearman, Holm-corrected, for the three continuous variables). This is an **ecological** (district-level) audit, not an individual-level fairness audit — Berka carries no per-person demographic field.

**E9, calibration check (added 2026-10-23).** §6.6 specified the operating threshold as the 95th percentile of the scores a fitted model assigns to *its own training data*. E9 tests whether that rule is biased at small *n*: for personal Mahalanobis and k-NN, on the identical E1 cohort, seed and injected labels, it recomputes the operating-point metrics using the 95th percentile of **leave-one-out** training scores (each training point scored by a model fitted on the remaining *n*−1; undefined at *n* = 1). In-sample rows are regenerated alongside and match the E1 table to three decimals. **E8b** repeats E8 with the leave-one-out FPR at *n* = 25, 50, 100.

**E9 full run (2026-10-24).** `experiments/e9_loo_full.py` repeats E9 for all six personal detectors and the shrinkage-hybrid arm (κ = 25) on the five E5 cohorts, with exact leave-one-out refits throughout (no shortcuts or K-fold approximations), using the same cohorts, injected labels and detector seeds as E5. Each seed's in-sample rows are checked against `results/raw/e5_multiseed_e1.csv` and the maximum absolute differences are recorded in `results/tables/e9_full_validation.csv`. `experiments/analyse_e9_full.py` was written before the results existed and defines no "usable operating point" criterion; the tables report FPR, flag rate, recall and precision against the nominal 0.05.

---

## 7. Results

All figures in `results/figures/`, all tables in `results/tables/`. Real-data results are from 300 accounts, 400 donor accounts, seed 20260920.

### 7.1 RQ1 and RQ2: data-efficiency curves on real data

Mean PR-AUC, personal arm, Protocol I. Prevalence baseline is 0.100.

| n | robust_z | gaussian_z | iforest | autoencoder | k-NN | Mahalanobis |
|---|---|---|---|---|---|---|
| 1 | 0.161 | 0.161 | 0.100 | 0.161 | 0.161 | 0.161 |
| 5 | 0.162 | 0.177 | 0.149 | 0.218 | 0.147 | 0.210 |
| 10 | 0.159 | 0.174 | 0.136 | 0.360 | 0.132 | 0.441 |
| 25 | 0.156 | 0.170 | 0.181 | 0.502 | 0.204 | 0.625 |
| 50 | 0.153 | 0.168 | 0.230 | 0.503 | 0.523 | 0.647 |
| 100 | 0.153 | 0.166 | 0.292 | 0.517 | 0.563 | **0.657** |
| 250 | 0.153 | 0.167 | 0.343 | 0.511 | 0.609 | 0.642 |

**Observation.** Curves separate into three regimes. Mahalanobis and the autoencoder rise steeply and saturate early. k-NN is flat or declining to *n* = 25, then rises sharply between 25 and 50 (+0.320) and continues gaining. Both univariate z-scores are flat throughout.

**Saturation fits** (saturating exponential y = y_max − (y_max − y₀)e^(−n/τ)):

| Model | y₀ | y_max | τ | R² |
|---|---|---|---|---|
| Mahalanobis | 0.078 | 0.658 | 11.5 | 0.970 |
| Autoencoder | 0.106 | 0.518 | 11.1 | 0.980 |
| k-NN | 0.089 | 0.633 | 50.9 | 0.912 |
| Isolation Forest | 0.111 | 0.350 | 71.2 | 0.985 |
| Robust z | 0.162 | 0.153 | 21.7 | 0.924 |
| Gaussian z | −0.370 | 0.170 | 0.25 | **0.469** |

**Interpretation.** Four of six curves are well described by a saturating exponential. The Gaussian z fit has R² = 0.469 and a nonsensical y₀ = −0.370; we report the fit and decline to interpret it, since the underlying curve is essentially flat and a saturation model is not the right description of flatness. This is why we report R² rather than the parameters alone.

**Marginal gains, Mahalanobis:** 1→5 +0.049, 5→10 **+0.231**, 10→25 +0.184, 25→50 +0.022, 50→100 +0.010, 100→250 **−0.016**.

**Answer to RQ1.** Performance improves with personal history for multivariate detectors, steeply between 5 and 25 transactions. It does not improve for univariate amount-only detectors at any size.

**Answer to RQ2.** Yes, performance plateaus, and early. For Mahalanobis, 98% of the total observed gain occurs by *n* = 50. The 100→250 step is negative (−0.016), which is the "additional history can hurt" outcome; §7.6 discusses it, including a direct test (E6) of the mechanism we originally proposed, which did not survive that test.

**Is this curve an artefact of the one cohort we happened to draw? (Experiment E5, §6.8.)** We re-ran E1 at 5 independent cohort-selection seeds (fresh 300-account/400-donor draws each time). The relative-90 threshold for Mahalanobis is **exactly 25 transactions at every one of the 5 seeds** (best-observed PR-AUC ranging 0.656-0.664 across seeds). The full curve is similarly stable: across-seed standard deviation of mean PR-AUC ranges from 0.003 (at *n* = 50) to 0.010 (at *n* = 1), small relative to the curve's own rise of roughly 0.5 over that range. **The headline threshold is not an artefact of one cohort draw.**

### 7.2 The gap between ranking quality and deployability

PR-AUC measures ranking. A deployed detector needs an operating point. **This section was materially revised after Experiment E9 (§6.8) found that the first-draft calibration rule was biased at small n; the original result and the corrected one are both reported.**

**(a) First-draft rule: in-sample calibration.** Mahalanobis, personal arm, threshold at the 95th percentile of the scores the fitted model assigns to **the same training points it was fitted on**:

| n | Precision | Recall | F1 | FPR | Flag rate |
|---|---|---|---|---|---|
| 1 | 0.100 | 1.000 | 0.182 | 0.999 | **0.999** |
| 5 | 0.101 | 1.000 | 0.184 | 0.988 | **0.989** |
| 10 | 0.104 | 0.997 | 0.189 | 0.956 | **0.960** |
| 25 | 0.256 | 0.899 | 0.391 | 0.335 | 0.391 |
| 50 | 0.357 | 0.811 | 0.483 | 0.189 | 0.251 |
| 100 | 0.417 | 0.753 | 0.523 | 0.134 | 0.196 |
| 250 | 0.460 | 0.687 | 0.536 | 0.104 | 0.162 |

At *n* = 10, PR-AUC is 0.441, more than four times the prevalence baseline, yet this operating point flags 96% of all transactions.

**What this table does and does not show (correction to the first draft).** The first draft read this as "the model cannot estimate the scale of its own score distribution from 10 points, so calibration is more expensive than ranking." That interpretation was not tested and, on testing, is mostly wrong. The threshold here is computed from **in-sample** scores, and in-sample scores are biased low *by construction* whenever a model is evaluated on the points it was fitted to: a personal Mahalanobis fit on *n* ≤ *d* = 21 points has a singular covariance, so every training point lies in the span of the fit and scores near zero; a k-NN training point counts itself as a neighbour at distance 0. The 95th percentile of those scores is therefore systematically, not randomly, too low. This is a bias that more data removes only slowly, not a noisy-quantile effect.

**(b) Corrected rule: leave-one-out calibration (Experiment E9).** Same accounts, same seed, same injected labels as E1 (and the in-sample rows reproduce the table above exactly, which validates the script). The threshold is the 95th percentile of **leave-one-out** training scores: each training point is scored by a model fitted on the other *n*−1 points, which is an out-of-sample estimate a deployed system can also compute. Mahalanobis, personal arm, 300 accounts, nominal FPR = 0.05:

| n | In-sample flag rate | In-sample FPR | LOO flag rate | LOO FPR | LOO recall | LOO precision |
|---|---|---|---|---|---|---|
| 5 | 0.989 | 0.988 | 0.152 | 0.151 | 0.163 | 0.110 |
| 10 | **0.960** | 0.956 | **0.012** | 0.007 | 0.058 | 0.667 |
| 25 | 0.391 | 0.335 | 0.110 | 0.066 | 0.503 | 0.577 |
| 50 | 0.251 | 0.189 | 0.126 | 0.073 | 0.612 | 0.558 |
| 100 | 0.196 | 0.134 | 0.134 | 0.076 | 0.658 | 0.533 |
| 250 | 0.162 | 0.104 | 0.136 | 0.080 | 0.647 | 0.516 |

(n = 1 has no leave-one-out score; the rule flags nothing.) Paired in-sample-minus-LOO FPR differences are large and significant at every size (Wilcoxon p < 10⁻³⁶; mean difference 0.949 at *n* = 10, 0.268 at 25, 0.116 at 50). The k-NN detector shows the same direction with a smaller magnitude (`results/tables/e9_loo_calibration_summary.csv`).

**Revised reading.** The "flags 96% at *n* = 10" headline is **an artefact of in-sample scoring, not a finding about data scarcity**, and we withdraw it as a result about detectors. Under out-of-sample calibration the *n* = 10 failure is the opposite one: the detector flags almost nothing (flag rate 0.012, recall 0.058), because 10 leave-one-out scores give a threshold dominated by the single most extreme of them. From *n* = 25 the false-positive rate is near-nominal (0.066 against 0.05, rising slowly to 0.080 at *n* = 250) and recall is about 0.5-0.65. The calibration-versus-ranking gap that the first draft called "the most practically important result" therefore shrinks to a much smaller one: usable operating behaviour appears at about *n* = 25, the same place as the ranking threshold, rather than at 50. What survives is a narrower point: **PR-AUC at *n* = 10 (0.441) still does not translate to a usable operating point**, only now because the detector is silent, not because it is noisy, and only for as long as the threshold must be derived from the person's own 10 points.

Two caveats on the corrected result. First, the nominal 5% is still missed by 1.6-3 percentage points at every size ≥ 25 (training points and evaluation-window points are not exchangeable, since the window is later in time; we did not diagnose the cause, and §7.6 reports that the drift explanation we had expected to apply elsewhere did not survive testing), so "near-nominal" should not be read as "well calibrated". Second, leave-one-out is one of several out-of-sample rules (split-half and a population-derived threshold are others); we did not compare them, so the corrected numbers describe this rule, not out-of-sample calibration in general.

**(c) Extension: every detector, five seeds (E9 full run, 2026-10-24).** The first E9 covered Mahalanobis and k-NN on one seed. We repeated it with exact leave-one-out (one refit per training point, no shortcuts) for all six personal detectors and the shrinkage-hybrid arm, on the five E5 cohorts (300 accounts each; 1,500 account-seed rows per cell). The in-sample rows reproduce the E5 results on all 14,700 rows per seed to within 1.1 × 10⁻¹⁶, and the primary-seed Mahalanobis and k-NN rows reproduce the first E9 exactly (8,400 of 8,400 rows). Mean FPR under leave-one-out, with the in-sample value in parentheses (nominal 0.05):

| Detector | n=5 | n=10 | n=25 | n=50 | n=100 | n=250 |
|---|---|---|---|---|---|---|
| Gaussian z | 0.049 (0.263) | 0.012 (0.108) | 0.005 (0.017) | 0.006 (0.007) | 0.007 (0.007) | 0.006 (0.007) |
| Robust z | 0.045 (0.237) | 0.048 (0.104) | 0.017 (0.016) | 0.013 (0.006) | 0.012 (0.006) | 0.009 (0.006) |
| Mahalanobis | 0.154 (0.989) | 0.012 (0.958) | 0.066 (0.331) | 0.068 (0.185) | 0.073 (0.132) | 0.083 (0.107) |
| k-NN | 0.492 (0.492) | 0.132 (0.194) | 0.041 (0.098) | 0.054 (0.104) | 0.071 (0.105) | 0.082 (0.111) |
| Isolation Forest | 0.535 (0.236) | 0.205 (0.199) | 0.009 (0.143) | 0.025 (0.113) | 0.042 (0.090) | 0.055 (0.075) |
| Autoencoder | 0.191 (0.984) | 0.021 (0.667) | 0.049 (0.205) | 0.060 (0.127) | 0.068 (0.100) | 0.077 (0.091) |
| Hybrid (shrinkage Mahalanobis) | 0.097 (0.395) | 0.028 (0.266) | 0.040 (0.184) | 0.060 (0.138) | 0.070 (0.114) | 0.080 (0.102) |

Recall under leave-one-out:

| Detector | n=5 | n=10 | n=25 | n=50 | n=100 | n=250 |
|---|---|---|---|---|---|---|
| Gaussian z | 0.070 | 0.038 | 0.047 | 0.053 | 0.054 | 0.055 |
| Robust z | 0.057 | 0.063 | 0.037 | 0.035 | 0.037 | 0.035 |
| Mahalanobis | 0.168 | 0.055 | 0.499 | 0.596 | 0.642 | 0.640 |
| k-NN | 0.455 | 0.093 | 0.038 | 0.386 | 0.545 | 0.602 |
| Isolation Forest | 0.519 | 0.167 | 0.018 | 0.076 | 0.164 | 0.236 |
| Autoencoder | 0.226 | 0.064 | 0.366 | 0.445 | 0.480 | 0.483 |
| Hybrid (shrinkage Mahalanobis) | 0.272 | 0.254 | 0.468 | 0.586 | 0.633 | 0.635 |

What the extension shows:

- **The Mahalanobis correction is stable across cohorts.** Pooled over five seeds, FPR is 0.012 at *n* = 10 and 0.066 at *n* = 25; the across-seed SD of the per-seed mean FPR is at most 0.008 for Mahalanobis at any size ≥ 5.
- **The artefact is not specific to Mahalanobis.** The autoencoder's in-sample flag rate is 0.691 at *n* = 10 against 0.025 under leave-one-out; the hybrid's is 0.296 against 0.051. In-sample and leave-one-out FPR differ significantly (paired Wilcoxon p < 10⁻⁴) in 39 of the 42 detector-by-size cells: the in-sample FPR is higher in 35 of them and lower in 4 (Isolation Forest at *n* = 5; robust z at *n* = 50, 100, 250). The three cells without a detectable difference are Isolation Forest at *n* = 10, robust z at *n* = 25, and k-NN at *n* = 5 (identical, see below).
- **"Usable operating point at about 25" holds only for the detectors that also rank well.** At *n* = 25 the Mahalanobis (0.066), autoencoder (0.049) and hybrid (0.040) FPRs are within about 1.6 points of 0.05, with recall 0.37 to 0.50. k-NN's FPR is 0.041 at *n* = 25 but its recall is 0.038: it is silent until about 50 (recall 0.386). Isolation Forest's FPR and recall at *n* = 25 are 0.009 and 0.018 and its recall reaches only 0.236 at 250, so no usable operating point appears within the sizes tested, consistent with its PR-AUC curve. Both z-score detectors are conservative under leave-one-out and keep recall at 0.035 to 0.055 at every *n* ≥ 25, consistent with their flat PR-AUC curves.
- **The direction of the bias is not uniform.** For Isolation Forest the leave-one-out rule flags *more* than the in-sample rule at *n* = 5 (flag rate 0.533 against 0.232), the two rules are indistinguishable at *n* = 10 (paired p = 0.7), and for robust z the in-sample FPR is slightly *lower* at *n* ≥ 25 (by 0.001 to 0.007). The in-sample rule is therefore not an upper bound on false-positive burden for every detector, and limitation 15 (§9) is revised accordingly. k-NN at *n* = 5 is identical under both rules on all 1,500 rows; this is an identity, not a coincidence: with *k* = *n* = 5 the in-sample 5th-nearest-neighbour distance (counting the point itself at distance 0) equals the maximum distance to the other four points, which is exactly the leave-one-out score.
- **A caveat on Isolation Forest at the smallest sizes.** The leave-one-out forest is fitted on *n*−1 points, and an isolation forest's score normalisation depends on its subsample size, so the leave-one-out scores and the scores of the *n*-point forest applied to the evaluation window are not on exactly the same scale. We did not test whether this explains Isolation Forest's behaviour at *n* ≤ 10, and that behaviour should not be read as a property of the detector.
- **The hybrid has the best small-*n* operating behaviour.** At *n* = 10 its leave-one-out FPR is 0.028 and recall 0.254, against 0.012 and 0.055 for personal Mahalanobis; at *n* = 5, 0.097 and 0.272 against 0.154 and 0.168. This extends the RQ8 result from ranking (§7.5) to the operating point. It is an observation from the table, not a pre-specified or significance-tested comparison, and it holds under one calibration rule and one anomaly protocol.

Still uncovered: the global arm (its threshold comes from the model's scores on its own pooled sample of about 4,000 points, so it would need a different design), the score-fusion k-NN hybrid, and TabFormer (E7).

**Consequence for RQ3.** A threshold defined on PR-AUC and a threshold defined on false-positive rate can give different answers on the same curves, and **a measured calibration requirement depends on how the threshold is calibrated**. Under the first-draft in-sample rule the answers were 25 (ranking) and 50 (FPR); under leave-one-out they are 25 and about 25 for Mahalanobis (and the autoencoder and hybrid), but about 50 for k-NN and not reached within 250 for Isolation Forest (§7.2c). We therefore no longer claim a ranking-versus-deployability gap of that size, and any single-number "N transactions is enough" claim conceals both the criterion and the calibration rule.

### 7.3 RQ3: personalisation thresholds under three definitions

| Model | Best observed | Relative-90 | Relative-95 | Stability |
|---|---|---|---|---|
| Mahalanobis | 0.657 | **25** | 25 | undefined |
| Autoencoder | 0.517 | 25 | 25 | undefined |
| k-NN | 0.609 | 100 | 250 | undefined |
| Isolation Forest | 0.343 | 250 | 250 | undefined |
| Robust z | 0.162 | 1 | undefined | 25 |
| Gaussian z | 0.177 | 1 | undefined | 50 |

**Three definitions, three answers, and their disagreements are informative.**

*Relative-90* gives 25 for the best model. But it is defined against the best **observed** value, not a true asymptote. If a curve were still rising at 250 the reference would be a lower bound and the threshold optimistic. For Mahalanobis the curve is not still rising, so the reference is sound here.

*Relative-90 is degenerate for flat curves.* Robust z gets a threshold of 1 transaction because it reaches 90% of its (worthless) best immediately. A relative criterion answers "when does this model stop improving", not "when is this model any good". Read alone it rewards models that never learn.

*Stability* is **undefined for every model that actually works.** Across-account standard deviation never falls below its own median for Mahalanobis, k-NN, the autoencoder or Isolation Forest, at any size. It is defined only for the two flat baselines, which are stably useless. This is the clearest evidence in the paper that **the mean curve hides enormous individual variation that does not shrink with more data**.

*Absolute* thresholds require a policy level we decline to invent, since it should follow from the cost of a false positive in a deployment we have not modelled.

**Answer to RQ3.** A threshold can be identified, but only conditional on the definition. On the best real-data model: 25 transactions for ranking performance; for a usable false-positive rate, 50 under the first-draft in-sample calibration but about 25 under leave-one-out calibration (§7.2, E9), so that figure depends on the calibration rule as much as on the data; and **no size at all** for across-individual stability. All three are read off a coarse grid ({1, 5, 10, 25, 50, 100, 250}): "25" means the first grid point at which the criterion is met, and the true crossing lies somewhere between 10 and 25 (at *n* = 10 Mahalanobis is at 67% of its best, at 25 at 95%).

*Statistical power note.* The real-data complexity correlations in §7.8 against the per-account threshold use the 183 accounts with a defined threshold. At that sample size a two-sided Spearman test at α = 0.05 reaches about 80% power only for |ρ| ≳ 0.21 (before Holm correction, which raises the bar further), so the null (all |ρ| ≤ 0.12) rules out moderate-to-strong effects but not weak ones. The same applies to the E8 audit (§7.11, 300 accounts, ≈ 80% power for |ρ| ≳ 0.16).

### 7.4 RQ6 and RQ7: model complexity and global versus personal

**Model complexity.** Isolation Forest is the **worst** detector at every history size, from 0.100 at *n* = 1 (exactly the prevalence baseline, i.e. no signal) to 0.343 at 250, never approaching Mahalanobis. Its τ = 71.2 means it is still far from saturation at the largest size we can evaluate. The autoencoder is competitive but saturates below Mahalanobis (0.518 vs 0.658).

Computation matches: at *n* = 250 the median fit cost is 0.46 ms for Mahalanobis, 5.0 ms for k-NN, 22.9 ms for the autoencoder, and 156.3 ms for Isolation Forest, a 340-fold difference for the worst-performing model.

**Answer to RQ6, on this dataset.** No. Under scarcity the sophisticated models are strictly worse and far more expensive. Isolation Forest's subsampling-based isolation mechanism has nothing to work with when the training set is smaller than its own default subsample. The best detector in this study is a shrinkage-regularised covariance estimate. **This answer does not generalise**: §7.10 (Experiment E7) finds Isolation Forest competitive-to-best under scarcity on a second, structurally different dataset. Read this section's answer as a finding about Berka, not a general law about model complexity under data scarcity.

**Global versus personal, paired over 300 accounts:**

Mahalanobis (global constant at 0.330):

| n | Personal | Paired diff | 95% CI | Effect (rank-biserial) | p (Holm) |
|---|---|---|---|---|---|
| 1 | 0.161 | −0.169 | [−0.196, −0.142] | −0.686 | < 0.001 |
| 5 | 0.210 | −0.120 | [−0.144, −0.096] | −0.544 | < 0.001 |
| 10 | 0.441 | **+0.111** | [+0.082, +0.140] | +0.465 | < 0.001 |
| 25 | 0.625 | +0.295 | [+0.266, +0.324] | +0.882 | < 0.001 |
| 100 | 0.657 | +0.327 | [+0.299, +0.355] | +0.940 | < 0.001 |

k-NN (global constant at 0.465): personal is **worse** through *n* = 25 (−0.261, effect −0.883) and better from *n* = 50 (+0.059) onward, reaching +0.145 at 250.

**Is the crossover stable across cohorts? (Experiment E5.)** At each of the 5 independent seeds, personal Mahalanobis first exceeds the global model at **exactly *n* = 10** — identical across every seed, even though the global model's own mean PR-AUC varies with the donor draw (0.262 to 0.330 across seeds). The crossover location is a property of the personal curve's shape relative to a globally-competitive baseline, not a coincidence of one donor sample.

**Answer to RQ7.** Personalisation is actively harmful below a crossover that depends on the model: between 5 and 10 transactions for Mahalanobis, between 25 and 50 for k-NN. Below it, a population model built from other people's data beats the person's own history, by large effect sizes. The common assumption that personalisation is a safe default is wrong in the sparse regime.

### 7.5 RQ8: hybrid modelling

Shrinkage-Mahalanobis hybrid versus personal Mahalanobis, paired:

| n | Personal | Hybrid | Paired diff | p (Holm) |
|---|---|---|---|---|
| 1 | 0.161 | **0.329** | −0.168 | < 0.001 |
| 5 | 0.210 | **0.398** | −0.188 | < 0.001 |
| 10 | 0.441 | 0.474 | −0.033 | 0.078 |
| 25 | 0.625 | 0.609 | +0.016 | 0.795 |
| 50 | 0.647 | 0.648 | −0.001 | 1.000 |
| 100 | 0.657 | 0.654 | +0.003 | 1.000 |
| 250 | 0.642 | 0.639 | +0.002 | 1.000 |

**Answer to RQ8, on real data: yes, decisively.** The hybrid is far better than the personal model when history is scarce (+0.188 at *n* = 5), and from 25 onward there is no detectable difference (Holm-adjusted p ≥ 0.795, paired differences ≤ 0.016). It is never worse than either parent at any size. The shrinkage weight n/(n+κ) removes the cold-start penalty at no measurable large-*n* cost, which makes the hybrid the dominant choice rather than a compromise.

**A contrary result from the simulation, reported because it qualifies the above.** In the simulation the hybrid plateaus at 0.401 while the personal model reaches 0.924. The reason is mechanical: the simulation's strongest features are personal novelty indicators (is this merchant one *you* have used), computed against a reference history. The hybrid computes them against the **population** reference, which destroys exactly the signal that matters. Parameter shrinkage cannot recover information the feature representation has already discarded.

**We therefore narrow the claim.** Shrinkage hybrids reduce the personal data requirement when the personal signal lives in the *parameters* of a shared representation. They do not help, and can hurt badly, when the personal signal lives in the *representation itself*. The Berka feature set is of the first kind; the simulation's is of the second. Which kind a production system resembles is an empirical question about its features, not a property of hybridisation.

### 7.6 RQ5: transaction count versus temporal coverage

**A methodological correction we made, and its consequence.** Our first version widened temporal coverage by stride-sampling history. Stride sampling changes the inter-arrival gap feature: under stride 2 the gaps computed on the thinned sequence are roughly double the account's true gaps, so the training distribution of that feature no longer matched the evaluation window's. It reported a large degradation (Mahalanobis 0.647 → 0.435 at stride 4). **Most of that was a feature artefact.** We recomputed gap features against each transaction's true position in the unthinned timeline and re-ran.

Corrected results, count fixed at 50, 300 accounts:

| Arm | Median span | Mahalanobis | k-NN |
|---|---|---|---|
| stride 1 (50 most recent) | 224 days | 0.663 | 0.540 |
| stride 2 | 457 days | 0.660 | 0.478 |
| stride 4 | 946 days | 0.627 | 0.430 |

Paired against stride 1: Mahalanobis stride 2 −0.004 (p = 0.304, not detectable), stride 4 −0.036 (p < 0.001, effect −0.300). k-NN stride 2 −0.062, stride 4 −0.110 (both p < 0.001).

**A control arm that failed, reported as such.** We intended a "recent-thinned" control to separate the cost of thinning from the cost of reaching further back. It failed: with count held fixed, thinning by *k* **necessarily** reaches back *k* times further, so the control arm had the same span as stride 4 (950 vs 946 days) and produced near-identical results. This is not a bug but a logical property of the design: **temporal coverage and sampling density cannot be varied independently at fixed count.** RQ5 as originally posed is therefore not fully separable, and we say so rather than presenting the confounded comparison as clean.

**What can still be concluded.** Combining with §7.1: increasing count from 50 to 250 at natural density changes Mahalanobis PR-AUC from 0.647 to 0.642, essentially nothing. Holding count at 50 while drawing from a four-times-longer span costs 0.036. So within this dataset, **neither more transactions nor more time helps once about 50 recent transactions are available, and reaching further back is mildly harmful.** Recency dominates volume.

**Interpretation, offered as one possible explanation.** This is consistent with behavioural drift: transactions from three years ago describe a person who no longer exists, and including them widens the estimated personal distribution so that genuinely unusual transactions no longer fall in its tail. This also offers a mechanism for the negative 100→250 marginal gain in §7.1. In the first draft of this paper we did not measure drift directly, so this was an interpretation, not a finding. We have since tested it (Experiment E6, below), and it does not survive direct measurement.

**Experiment E6: testing the drift interpretation directly.** For each of the 300 evaluated accounts, drift is the Mahalanobis distance between the 150 transactions added when history grows from 100 to 250 and the 100 transactions already present at n = 100, in the account's own current-behaviour coordinate system (§6.8). Drift is real and substantial: median 0.69 (of the account's own recent standard deviations), mean 1.00, ranging up to 12.6. A lag sweep confirms it accumulates with distance into the past — mean drift rises from 0 at lag 0 to 1.72 at lag 25, and plateaus around 3.2-3.6 by 150-200 transactions back — so **accounts' older behaviour genuinely does look different from their recent behaviour**, on average, as the drift interpretation assumed.

But per-account drift magnitude does **not** predict that account's own 100→250 marginal PR-AUC gain: Spearman ρ = −0.081 for Mahalanobis (p = 0.160, Holm-adjusted p = 0.32) and ρ = +0.070 for k-NN (p = 0.226, Holm-adjusted p = 0.32), both n = 300. **The drift interpretation, tested directly, is not supported.** Accounts whose older transactions look most different from their recent ones are not the accounts whose PR-AUC drops most between n = 100 and n = 250. Drift accumulates with lag; it just does not explain which accounts lose performance when older history is added. The mechanism behind the negative 100→250 step, and behind the cost of reaching further back described earlier in this section, remains genuinely open — we no longer offer behavioural drift, specifically in the form we measured it, as the explanation.

### 7.7 Protocol P: does the finding survive a different anomaly definition?

Mean PR-AUC under the ×5 amount-shift protocol:

| n | gaussian_z | Mahalanobis | k-NN | iforest |
|---|---|---|---|---|
| 1 | 0.278 | 0.258 | 0.258 | 0.100 |
| 25 | 0.292 | 0.423 | 0.153 | 0.163 |
| 100 | 0.294 | 0.455 | 0.212 | 0.194 |
| 250 | 0.292 | 0.453 | 0.217 | 0.211 |

**Observation.** The ordering changes. Univariate Gaussian z, useless under Protocol I (0.166 at *n* = 100), becomes competitive under Protocol P (0.294), which is expected since the perturbation is applied to amount alone. Mahalanobis remains best but at a lower level (0.455 vs 0.657), and its relative-90 threshold is again **25**.

**Interpretation.** The threshold estimate of roughly 25 transactions is stable across two structurally different anomaly definitions, which is mild evidence that it reflects something about the estimation problem rather than about one protocol. The *level* of achievable performance is strongly protocol-dependent and should not be transferred between them.

### 7.8 RQ4: behavioural complexity

**Real data: a null result.** We computed eight descriptors on each account's 250-transaction history and correlated each against that account's own relative-90 threshold (Spearman, Holm-corrected over the family of eight).

| Descriptor | ρ vs threshold | p (Holm) | ρ vs best PR-AUC | p (uncorr.) |
|---|---|---|---|---|
| amount_cv | −0.119 | 0.867 | −0.136 | 0.066 |
| operation_entropy | +0.113 | 0.898 | +0.209 | 0.005 |
| ksymbol_entropy | −0.010 | 1.000 | +0.196 | 0.008 |
| operation_diversity | −0.053 | 1.000 | +0.167 | 0.024 |
| gap_cv | +0.081 | 1.000 | −0.036 | 0.631 |
| amount_rcv | −0.049 | 1.000 | −0.034 | 0.647 |
| log_amount_iqr | −0.033 | 1.000 | +0.005 | 0.946 |
| credit_share | +0.090 | 1.000 | −0.017 | 0.822 |

**No descriptor predicts an account's data requirement.** All Holm-adjusted p ≥ 0.867, all |ρ| ≤ 0.12. We report this null rather than selecting an uncorrected result.

Two caveats limit how strongly the null should be read. First, **117 of 300 accounts (39%) had no defined threshold**, because their curve never reached and held 90% of their own best value. The correlation is computed on the 183 that did, which is a non-random subset of the better-behaved accounts. Second, the descriptors correlate positively with *best achievable* performance, not with the requirement: accounts with more diverse `operation` and `k_symbol` usage are easier to protect (ρ ≈ 0.20, p ≈ 0.005), plausibly because a more distinctive profile is easier to distinguish from an impostor's.

**Simulation: a strong positive result.** With variability *manipulated* rather than observed, every descriptor predicts the requirement:

| Descriptor | ρ vs threshold | p (Holm) | ρ vs best |
|---|---|---|---|
| n_unique_merchants | **+0.708** | < 0.001 | −0.259 |
| category_entropy | +0.700 | < 0.001 | −0.284 |
| merchant_entropy | +0.700 | < 0.001 | −0.288 |
| log_amount_sd | +0.673 | < 0.001 | −0.255 |
| location_entropy | +0.617 | < 0.001 | −0.172 |
| gap_cv | +0.484 | < 0.001 | −0.189 |

Per-archetype relative-90 thresholds (Mahalanobis):

| Archetype | Threshold | Best PR-AUC |
|---|---|---|
| regular | **50** | 0.939 |
| moderate | 250 | 0.929 |
| seasonal | 250 | 0.923 |
| burst | 250 | 0.927 |
| traveller | **500** | 0.928 |
| variable | **500** | 0.900 |

**Answer to RQ4.** H1's complexity clause is **supported under controlled manipulation and unsupported in observational real data**. The archetypes differ tenfold in data requirement (50 to 500) while converging to similar asymptotes (0.900 to 0.939), so variability changes the *rate* of learning far more than the *ceiling*.

**Interpretation of the discrepancy.** Three explanations are consistent with our evidence and we cannot distinguish them. The Berka descriptors may be too thin to capture variability, since the dataset has no merchant, category or location channels and only two categorical fields, precisely the channels that carry the simulation's effect. The real cohort's variability range may be too narrow, since all accounts are ordinary retail accounts at one bank. Or observed variability in real data may be confounded with account type, tenure and product mix in ways the simulation excludes by construction. **The simulation result is not evidence that the effect exists in real data; it is evidence that the effect exists when the generative parameters are what the theory says they are.**

**A partial test of the first explanation (Experiment E7, §5.4, §7.10).** TabFormer's schema has exactly the merchant/category/geography channels Berka lacks, even though the underlying data are synthetic. Repeating this correlation on the 300 TabFormer entities (§7.10) finds `state_entropy` **does** significantly predict the entity's relative-90 threshold after Holm correction (ρ = +0.19, p_holm = 0.028; `merchant_diversity` is marginal at p_holm = 0.148), unlike every descriptor on Berka. This is consistent with, though it does not prove, the first explanation: channel availability, not just "being real data", may be what the real-data null was missing. It is a synthetic-data result and cannot upgrade the real-data null to a positive finding; it narrows which of the three explanations looks most likely.

### 7.9 Anomaly class detectability (simulation)

Mean anomaly-score percentile, Mahalanobis (1.0 = ranked most anomalous):

| n | amount | temporal | merchant | category | geographic | multivariate |
|---|---|---|---|---|---|---|
| 1 | 0.789 | 0.721 | 0.545 | 0.547 | 0.642 | 0.896 |
| 10 | 0.576 | 0.666 | 0.670 | 0.745 | 0.811 | 0.932 |
| 50 | 0.744 | 0.738 | 0.838 | 0.892 | 0.913 | 0.999 |
| 250 | 0.861 | 0.845 | 0.947 | 0.957 | 0.969 | 1.000 |
| 500 | 0.873 | 0.855 | 0.955 | 0.964 | 0.974 | 1.000 |

**Observation.** Classes have different data requirements. Categorical-novelty anomalies (merchant, category, geographic) start near chance and improve steeply, because recognising an unfamiliar merchant requires having observed the familiar ones. Amount and temporal anomalies start well above chance at *n* = 1 and improve least, because a single observation already locates a distribution's centre roughly. Multivariate anomalies are effectively saturated by *n* = 50.

Amount detectability **declines** from 0.789 at *n* = 1 to 0.576 at *n* = 10 before recovering. Interpretation: with one observation the model has an implausibly narrow implied scale and flags any deviation; as a few more arrive the scale estimate becomes wide but still unstable. Apparent competence at *n* = 1 is an artefact of degenerate variance, not detection.

### 7.10 Experiment E7: does the curve shape survive a different, card-structured dataset? (SYNTHETIC)

**Every number in this subsection is from TabFormer (§5.4), a synthetic dataset. Nothing here is a real-data observation, and nothing here is pooled with §7.1-§7.9.** Its purpose is to test whether the qualitative shape found on Berka — steep early rise, plateau, a global/personal crossover, an early-and-stable relative-90 threshold — is specific to one dataset, or reappears on a structurally different schema with merchant, category, geographic and sub-daily-timing channels.

Mean PR-AUC, personal arm, Protocol I, 300 evaluated (User, Card) entities:

| n | robust_z | gaussian_z | iforest | autoencoder | knn | mahalanobis |
|---|---|---|---|---|---|---|
| 1 | 0.212 | 0.212 | 0.100 | 0.149 | 0.149 | 0.149 |
| 5 | 0.215 | 0.210 | 0.362 | 0.194 | 0.187 | 0.206 |
| 10 | 0.216 | 0.212 | 0.403 | 0.299 | 0.329 | 0.339 |
| 25 | 0.217 | 0.215 | 0.515 | 0.426 | 0.456 | 0.465 |
| 50 | 0.221 | 0.217 | 0.567 | 0.469 | 0.489 | 0.548 |
| 100 | 0.220 | 0.219 | 0.585 | 0.500 | 0.513 | 0.630 |
| 250 | 0.221 | 0.218 | 0.597 | 0.577 | 0.590 | 0.721 |
| 500 | 0.221 | 0.220 | 0.576 | 0.602 | 0.630 | **0.748** |

**The shape replicates.** Mahalanobis again rises steeply and reaches the highest ceiling (0.748 at *n* = 500). Both univariate z-scores are again flat throughout (0.21-0.22, no learning at any size). The relative-90 threshold for Mahalanobis is **250**, later than Berka's 25 — plausibly because TabFormer's 26-dimensional feature space (13 numeric/binary plus a 13-level MCC one-hot, versus Berka's 21) needs more observations to stop being singular for a *n* ≤ *d* covariance estimate (§6.5), and because the global prior here is itself much stronger (below), raising the bar the personal curve must clear to look "saturated."

**One result does NOT replicate, and we report the divergence plainly.** On Berka, Isolation Forest is the *worst* detector at every history size (§7.4). On TabFormer it is competitive-to-best at small *n*: it is the single best detector at *n* = 5, 10, 25 and 50 (e.g. 0.362 vs Mahalanobis's 0.206 at *n* = 5), and only falls behind Mahalanobis from *n* = 100 onward. We do not have a confirmed mechanism for this reversal; a plausible candidate is that TabFormer's higher-dimensional, more one-hot-heavy feature space gives isolation-based splitting more structure to exploit at small sample sizes than Berka's, but we did not test this directly and offer it as a hypothesis, not a finding. **The headline RQ6 conclusion ("sophisticated models are worse under scarcity") is therefore dataset-dependent, not universal, and this paper's own second dataset contradicts it at small history sizes.**

**Personal vs. global, Mahalanobis, paired over 300 entities:**

| n | Personal | Global (0.524, constant) | Paired diff | p (Holm) |
|---|---|---|---|---|
| 1 | 0.149 | 0.524 | −0.375 | < 0.001 |
| 25 | 0.465 | 0.524 | −0.059 | < 0.001 |
| 50 | 0.548 | 0.524 | +0.023 | 0.094 |
| 100 | 0.630 | 0.524 | +0.105 | < 0.001 |
| 500 | 0.748 | 0.524 | +0.223 | < 0.001 |

The crossover (personalisation stops being harmful) falls between *n* = 25 and *n* = 100 here, later than Berka's 5-10, because TabFormer's global model is itself much stronger (PR-AUC 0.524 vs Berka's 0.330) — a stronger population baseline raises the bar personalisation must clear, exactly as the hybrid mechanism's logic (§6.5) predicts. **The qualitative finding survives: personalisation is harmful below a model- and dataset-dependent crossover, never absent, never universal.**

**Protocol F (exploratory, generator-labelled fraud) could not be evaluated at this sample size.** Zero of the 300 evaluated entities' 60-transaction evaluation windows contained a genuine (generator-labelled) fraud transaction, consistent with the corpus-wide 0.122% fraud prevalence against a 60-row window (expected count ≈ 0.07 per window). This is reported as an honest null result of the attempt, not worked around by resampling toward fraud-containing entities post hoc; a targeted follow-up that deliberately selects entities with historical fraud is listed in Appendix A.5 as not yet evaluated.

### 7.11 Experiment E8: disparate-impact audit (ecological, district-level)

§10 of this paper's original draft argued that sparse-history users bear a disproportionate cost without auditing for it. E8 tests, at *n* = 10 (the "flags almost everything" regime, §7.2) and *n* = 50 (the "tolerable" regime), whether the personal Mahalanobis detector's false-positive rate varies with the evaluated account's district-level average salary, unemployment rate, urbanisation or region (`district.asc`, 300 accounts linked). **This is an ecological audit — district-of-account economic context, not an individual-level demographic field, which Berka does not have.**

| n | Kruskal-Wallis across 8 regions | ρ (FPR, avg. salary) | ρ (FPR, unemployment) | ρ (FPR, urban ratio) |
|---|---|---|---|---|
| 10 | H = 10.6, p = 0.159 | +0.053, p_holm = 1.00 | +0.056, p_holm = 1.00 | +0.048, p_holm = 1.00 |
| 50 | H = 2.2, p = 0.948 | −0.057, p_holm = 0.75 | −0.009, p_holm = 0.87 | −0.067, p_holm = 0.75 |

**Caveat added after E9 (§7.2): the n = 10 half of this audit is uninformative.** The FPR used above comes from the in-sample calibration rule, which at *n* = 10 flags ≈ 96% of everything for every account, so FPR sits at a ceiling with almost no between-account variation to correlate with district variables. We therefore repeated the audit (E8b, `results/tables/e8b_fpr_district_correlation_loo.csv`) with the leave-one-out FPR at *n* = 25, 50 and 100, where the rate is informative (mean FPR 0.066-0.076):

| n | ρ (salary) | p_holm | ρ (unemployment) | p_holm | ρ (urban ratio) | p_holm | Kruskal-Wallis, region |
|---|---|---|---|---|---|---|---|
| 25 | −0.115 | 0.137 | +0.012 | 0.838 | −0.110 | 0.137 | p = 0.70 |
| 50 | −0.053 | 0.719 | −0.026 | 0.719 | −0.077 | 0.554 | p = 0.53 |
| 100 | −0.130 | 0.073 | +0.104 | 0.145 | −0.073 | 0.211 | p = 0.49 |

Still no Holm-significant relationship. One pattern is worth flagging rather than hiding: the salary correlation is negative at all three sizes (higher-salary districts have somewhat lower FPR), with unadjusted p = 0.046 and 0.024 at *n* = 25 and 100. These are not corrected for the nine tests across sizes and variables, they are small (|ρ| ≤ 0.13, at or below the effect this sample can reliably detect), and we do not claim an effect; but the audit does not rule out a weak salary gradient either.

**Original E8 result (in-sample FPR).** No detectable relationship between any district economic variable and false-positive rate, at either history size. All |ρ| ≤ 0.07, no Holm-adjusted p below 0.75. This is a null result on the specific axis tested, and it is uniform across districts under the in-sample rule (mean FPR ≈ 0.95 at *n* = 10 and ≈ 0.19 at *n* = 50 in every tercile; `results/tables/e8_fpr_by_district_tercile.csv`), but that uniformity is of limited interest now that E9 shows the in-sample rule inflates FPR at small *n* for everyone. What this audit does **not** find is that any FPR burden concentrates in lower-income, higher-unemployment or rural districts specifically. Whether it concentrates by age, tenure, immigration status or other axes Berka cannot supply remains untested, and §10 is revised accordingly.

---

## 8. Discussion

### 8.1 What we take to be established by this work

Within the stated scope, the following are experimental observations:

1. Personalised behavioural anomaly detection on real transaction histories improves steeply between 5 and 25 personal transactions and is essentially saturated by 50.
2. Ranking quality and operating-point behaviour reach adequacy at about the same history size once the threshold is calibrated out-of-sample (leave-one-out, E9: FPR 0.066 at *n* = 25 against 0.05 nominal). The first draft's claim that calibration lags ranking by a wide margin came from calibrating on in-sample scores and is withdrawn. What remains: at *n* = 10 a 0.441 PR-AUC does not yield a usable operating point under either rule (in-sample flags 96%; leave-one-out flags 1.2%, recall 0.058).
3. Isolation Forest is the worst performer at every history size **on Berka**, and the most expensive by two orders of magnitude — but this is dataset-dependent, not universal: it is competitive-to-best at small *n* on TabFormer (§7.10). The mechanism-level claim ("sophisticated models cannot help under scarcity") does not survive replication on a second dataset; the narrower, supported claim is that model ranking under scarcity is itself dataset-dependent, which is a finding of its own.
4. Personalisation is worse than a population model below a model- **and dataset-**dependent crossover (§7.4, §7.10 both show it, at different sizes).
5. A history-weighted shrinkage hybrid dominates both parents when personal signal lives in the parameters of a shared representation, and fails when it lives in the representation.
6. Behavioural variability governs the data requirement under controlled manipulation; we could not detect that relationship in real Berka accounts, but a weaker, Holm-significant version of it appears on TabFormer's `state_entropy` (§7.10), a dataset that (unlike Berka) actually carries a geographic channel.
7. **(New, E5.)** The primary curves and thresholds are not an artefact of one cohort draw: re-running E1 at 5 independent seeds shows the qualitative shape is stable, with seed-to-seed detail in §7.1 as revised.
8. **(New, E6.)** Direct measurement rules out the specific drift mechanism this paper's first draft offered for the negative 100→250 step (§7.6): drift accumulates with lag as expected, but per-account drift magnitude does not predict that account's own marginal loss.
9. **(New, E8/E8b.)** An ecological (district-level) disparate-impact audit found no Holm-significant relationship between false-positive rate and a district's average salary, unemployment rate, urbanisation or region, under either the first-draft in-sample calibration (*n* = 10, 50) or leave-one-out calibration (*n* = 25, 50, 100; §7.11). A weak, uncorrected negative salary trend at two sizes is reported but not claimed.
10. **(New, E9.)** The first draft's headline that a detector "flags 96% of everything at *n* = 10" is an artefact of calibrating the threshold on in-sample training scores. Under leave-one-out calibration it flags 1.2% at *n* = 10, the FPR is within 1.6-3 points of nominal from *n* = 25, and the ranking-versus-deployability gap shrinks accordingly (§7.2). The full run (§7.2c: six detectors, shrinkage hybrid, five seeds) confirms the Mahalanobis numbers are stable across cohorts, shows the artefact also affects the autoencoder and hybrid, and shows it is not uniform in direction (Isolation Forest, robust z).

### 8.2 What remains interpretation

That the real-data null on complexity reflects thin descriptors rather than a genuinely absent effect is interpretation, though §7.10's TabFormer result (a real, Holm-significant `state_entropy` effect on a dataset that has a geographic channel) makes the "thin descriptors" explanation somewhat more credible than the "genuinely no effect in real retail banking" alternative — without resolving it, since TabFormer is synthetic.

The mechanism behind the negative 100→250 marginal gain and the cost of reaching further back (§7.6) is now **more open, not less**, than the first draft of this paper suggested. We directly tested the drift explanation (E6, §7.6) and it did not hold at the account level, despite drift itself being real and lag-dependent. We do not have a replacement mechanism to offer; this is now reported as an unresolved question rather than a plausible-but-untested interpretation.

### 8.3 Practical implication, stated carefully

If a practitioner wanted one number, this study's answer for its best real-data model would be: **about 25 transactions for ranking, about 25 (leave-one-out calibration; 50 under our first-draft in-sample rule) before the false-positive rate is near its nominal level, and no number at all that guarantees a given individual is well served.** The third clause matters most. The across-account spread never stabilised at any history size we evaluated, so a system meeting a target on average will still fail badly for a substantial minority of users. Population averages are the wrong unit for a decision that is applied per person.

---

## 9. Limitations

**Data.**
1. One **real** dataset, one bank, one country, 1993-1998 retail banking. A second, structurally different dataset (TabFormer, card-structured, merchant/category/geography/sub-daily timing) was added as Experiment E7 (§5.4, §7.10) specifically to probe this limitation, but TabFormer is synthetic, so generalisation to modern real card payments remains untested. E7's own headline finding — that Isolation Forest's relative ranking reverses between the two datasets — is itself evidence that this limitation matters in practice, not just in principle.
2. Day-resolution timestamps preclude hour-of-day behaviour on **real** data. TabFormer (synthetic) has minute-resolution timestamps and was used to include an hour-of-day feature (§5.4), but this does not close the gap for real data.
3. No merchant, category or geographic channels on **real** data. The three anomaly classes that showed the steepest data requirements in simulation could not be tested on real behaviour. E7 (synthetic) partially probes this (§7.10's `state_entropy` result), but cannot close it for real data.
4. Real-data history sizes stop at 250 because of dataset depth. We cannot rule out changes beyond that point. (TabFormer, synthetic, supports sizes to 500 and its curves are still rising at 250 for several models, §7.10 — a hint, not evidence, about what a deeper real dataset might show.)
5. The evaluated cohort (≥ 310 transactions) is the more active half of the dataset and is not representative of all account holders, least of all the sparse-history users the study is most about.

**Protocol.**
6. Neither injection protocol produces fraud, and neither models an adversary attempting to imitate the account holder. A capable adversary would choose transactions near the victim's behavioural centre, which is the hardest case and the one we cannot evaluate.
7. Protocol I renders timing non-discriminative by construction.
8. The 10% injection rate is far above realistic fraud prevalence. Absolute PR-AUC values are not transferable to a realistic base rate; the *shape* of the curves is the transferable quantity.
9. RQ5 is not fully separable: coverage and sampling density are mathematically coupled at fixed count.

**Modelling.**
10. κ = 25 was fixed in advance, not tuned. A tuned κ would likely improve the hybrid, and tuning it on our data would invalidate the comparison.
11. The autoencoder is deliberately small; larger networks were not evaluated and would be expected to fare worse under scarcity, not better.
12. Threshold definitions are evaluated on mean curves; the per-account threshold distribution is heavy-tailed with 39% undefined.

**Statistical.**
13. Cohort selection was repeated at 5 seeds (Experiment E5, §6.8), not the single seed of the first draft, for E1 and for Experiment B's Mahalanobis/k-NN curves specifically. E2 and Isolation Forest/autoencoder in the simulation remain single-seed, and the E5 cohort-selection replication does not extend to E3, E4, E7 or E8, each of which remains a single realised sample. See §7.1 for what the 5-seed replication found.
14. Wilcoxon p values over 300 paired accounts are extremely small for effects of any size, which is why effect sizes are reported alongside. Several differences that are statistically unambiguous (the hybrid at *n* = 10, −0.033) are practically negligible.
15. **(New, E9.)** Operating-point metrics for the six personal detectors and the shrinkage hybrid on Berka are reported under both the in-sample and the leave-one-out rule (§7.2c). Every other operating-point figure (the global arm, the score-fusion hybrid, the E2 protocol, E3/E4, the simulation, TabFormer and the E8 tables using the in-sample FPR) still uses the in-sample rule, which §7.2c shows is biased low for most detectors at small *n* but is *not* an upper bound for every detector (Isolation Forest at *n* = 5 and robust z at *n* ≥ 50 go the other way), so those figures cannot be read as a worst case either. Leave-one-out is one of several out-of-sample rules and was the only one tried; for Isolation Forest at *n* ≤ 10 its scale mismatch with the *n*-point forest is an untested confound. The relative-90 ranking thresholds (PR-AUC) do not depend on any threshold rule and are unaffected.
16. **(New.)** The TabFormer replication layer (E7) verified only via a single download source (§5.4); the drift measurement (E6) and disparate-impact audit (E8) are each a single realised analysis, not repeated across seeds.

---

## 10. Ethical considerations

**An anomaly detector identifies deviations from a behavioural baseline; it does not establish fraud or wrongdoing.** Every ethical concern below follows from how easily that sentence is forgotten.

**Treating unusual behaviour as suspicious behaviour.** The transactions our detectors flag are, by construction, transactions that differ from a person's past. In real life those are holidays, house moves, medical costs, a new job, a death in the family, leaving an abusive relationship. A system that escalates deviation to suspicion punishes people for changing.

**False positives are not symmetric with false negatives.** A missed anomaly costs an institution money. A false positive can mean a declined card in a foreign country, a frozen account during an emergency, or an account-holder treated as a suspect. Our best real-data model at 50 transactions has a 7.3% false-positive rate under leave-one-out calibration (19% under the in-sample rule of our first draft, which we now regard as biased, §7.2), against a nominal 5%. At realistic transaction volumes even 7% means a very large number of wrongly flagged legitimate transactions.

**Sparse-history users bear the cost.** The people with the least history are new customers, young people, recent immigrants, people re-entering the formal banking system. §7.2 shows these are exactly the users for whom a personal model flags almost everything. Deploying personalisation without a history floor concentrates false positives on populations already more likely to face financial exclusion. The finding that a global or hybrid model outperforms personalisation in the sparse regime is directly relevant: for these users, personalisation is not merely useless, it is harmful.

**Discrimination.** Behavioural features are proxies for life circumstances. Irregular income, cash reliance and geographic mobility are correlated with class, occupation, immigration status and disability. A detector calibrated on regular salaried behaviour will treat irregular lives as anomalous by default. The first draft of this paper stated that no disparate-impact audit had been run. We have since run one (Experiment E8, §7.11): a district-level (ecological) test of whether the personal detector's false-positive rate varies with average salary, unemployment rate, urbanisation or region. **It found no Holm-significant relationship on any of these four axes**, under the first-draft in-sample calibration (*n* = 10, 50; all |ρ| ≤ 0.07) or, after we found that rule biased (E9), under leave-one-out calibration (E8b, *n* = 25, 50, 100; all Holm p ≥ 0.07, with an uncorrected, non-claimed negative salary trend of |ρ| ≤ 0.13). This does not clear the concern; it narrows it, and with 300 accounts it cannot exclude weak effects (§7.3 power note). Berka carries no per-person demographic field, so E8 cannot test age, tenure, immigration status or disability directly, and the concern raised in this paragraph remains untested on those axes. What E8/E8b do show is that false-positive burden is not, in this dataset, visibly concentrated by district-level economic disadvantage. The thin-history risk itself is now less clear-cut than the first draft said: under in-sample calibration it is a flood of false positives, under leave-one-out calibration at *n* = 10 it is a silent detector. No system of this kind should be deployed without an individual-level audit on the axes we could not test.

**Privacy and surveillance.** Personal transaction histories are among the most revealing datasets about a person. Per-user behavioural modelling requires retaining and processing them continuously. That §7.6 finds recent history to be sufficient and older history mildly harmful is a rare case of a privacy interest and a performance interest aligning: shorter retention appears to cost nothing here.

**Explainability.** Mahalanobis distance yields a scalar with no account of which behaviour was unusual. Any deployment affecting a person's access to their money should be able to state the reason, and should support contest and correction.

**Our own data handling.** The Berka dataset was anonymised by its publisher before release. We redistribute no raw records, work only at account-identifier level, and make no attempt at re-identification.

**Misuse.** This work should not be cited as evidence that N transactions make a detector safe to deploy, nor that flagged transactions warrant adverse action against a customer.

---

## 11. Conclusion

We measured how much of a person's own transaction history an unsupervised behavioural anomaly detector needs, on real bank histories, under controlled simulation, and — after independently re-verifying this paper's own first draft and extending it with four further experiments (§6.8) — on a second, structurally different (synthetic) dataset, using a ground-truth protocol defined by provenance rather than by any statistical rule.

A sufficiency point exists and is earlier than the framing of the question suggests: roughly 25 transactions for ranking quality on our best real-data model, saturating by 50, a shape that replicates qualitatively (with a later threshold) on a card-structured synthetic dataset with a much stronger global baseline (§7.10). But the single number is the least useful part of the answer. The first draft's strongest practical claim, that ranking quality and calibration have very different requirements because a model that ranks well at 10 transactions flags 96% of everything, did not survive our own follow-up (E9): the 96% was produced by calibrating the threshold on in-sample scores, and with leave-one-out calibration the same model is silent at 10 transactions and near-nominal from 25. What survives is that PR-AUC alone does not tell you whether an operating point is usable, and that the operating point depends heavily on how the threshold is calibrated. Personalisation is actively harmful below a crossover where a population model built from strangers does better, on both datasets we tested, at different sizes. Isolation Forest was the worst detector at every history size on real Berka data, but was competitive-to-best on TabFormer — the "sophisticated models don't help under scarcity" claim, tested a second way, turned out to be dataset-dependent rather than universal, and we report the reversal rather than the more comfortable single-dataset story. Behavioural variability changes the requirement tenfold when manipulated, and a matching, weaker effect appears on the one real-structured proxy we could test it against (TabFormer's `state_entropy`), but we still could not detect it in real Berka accounts. We directly tested our own drift explanation for a puzzling negative result and it did not survive that test. A district-level disparate-impact audit found no Holm-significant geographic concentration of false-positive burden, though a sample of 300 cannot exclude weak effects. And no history size we evaluated made performance stable **across individuals**, which undermines any per-person guarantee derived from a population average.

The practical recommendation is not a number. It is that a system should use a hybrid that degrades to population behaviour when personal history is thin, should hold recent history rather than all history, should not assume that a detector which ranks well is a detector that can be deployed, and should not assume that a finding about which detector is "best under scarcity" transfers from one dataset to another without checking.

---

## 12. References

Verified sources. DOIs and URLs checked during the literature audit.

1. Bahnsen, A. C., Aouada, D., Stojanovic, A., and Ottersten, B. (2016). Feature engineering strategies for credit card fraud detection. *Expert Systems with Applications*, 51, 134-142. DOI: 10.1016/j.eswa.2015.12.030

2. Berka, P. and Sochorova, M. (1999). *PKDD'99 Discovery Challenge: Financial Dataset*. <https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm>

3. Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. (2000). LOF: Identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93-104. DOI: 10.1145/335191.335388

4. Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., and Bontempi, G. (2018). Credit card fraud detection: A realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), 3784-3797. DOI: 10.1109/TNNLS.2017.2736643

5. Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., and Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, 41(10), 4915-4928. DOI: 10.1016/j.eswa.2014.02.026

6. Jurgovsky, J., Granitzer, M., Ziegler, K., Calabretto, S., Portier, P.-E., He-Guelton, L., and Caelen, O. (2018). Sequence classification for credit-card fraud detection. *Expert Systems with Applications*, 100, 234-245. DOI: 10.1016/j.eswa.2018.01.037

7. Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation Forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM)*, 413-422. DOI: 10.1109/ICDM.2008.17

8. Lucas, Y. and Jurgovsky, J. (2020). Credit card fraud detection using machine learning: A survey. arXiv:2010.06479. <https://arxiv.org/abs/2010.06479>

9. Mohr, F. and van Rijn, J. N. (2022). Learning curves for decision making in supervised machine learning: A survey. arXiv:2201.12150. <https://arxiv.org/abs/2201.12150>

10. Padhi, I., Schiff, Y., Melnyk, I., Rigotti, M., Mroueh, Y., Dognin, P., Ross, J., Nair, R., and Altman, E. (2021). Tabular transformers for modeling multivariate time series. *ICASSP 2021*. Dataset: <https://github.com/IBM/TabFormer>

11. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

12. Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. DOI: 10.1371/journal.pone.0118432

13. Viering, T. and Loog, M. (2023). The shape of learning curves: A review. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(6), 7799-7819. DOI: 10.1109/TPAMI.2022.3220744

14. Zhang, Y. et al. SparseFraudNet: A graph-based approach for cold-start fraud detection with information aggregation. *ACM Transactions on Information Systems*. DOI: 10.1145/3748719

**Patents cited as evidence of deployed practice, not as scientific sources:**

15. US 11005864 B2 and US 11706234 B2. Feature-agnostic behavior profile based anomaly detection (per-user median/MAD profiles).
16. US 11025653 B2 and continuations (US 10270788, US 11743275, US 12244617). Machine learning based anomaly detection, including filtering of "unseasoned users" with insufficient history.

**Citation integrity note.** Every reference above was located during the audit and its bibliographic details verified against at least one primary or publisher source. Reference 14's full author list was not retrievable from an open source and is given incompletely rather than invented.

---

## Appendix A: Reproducibility

### A.1 Environment

Python 3.14.6, fresh virtual environment, this session. Verified versions: pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2 — resolved by `pip install` against this repository's `requirements.txt` lower bounds; not pinned to the exact point releases recorded by the session that produced the first draft of this paper (pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8), which are all older point releases of the same major/minor lines. No GPU. No PyTorch (the autoencoder is NumPy). Full detail in `docs/ENVIRONMENT.md`.

### A.2 Data integrity

**Berka.** The loader verifies row counts against the primary source and raises otherwise. This session additionally cross-checked every table's content, byte-for-byte after line-ending normalisation, against a second, independent GitHub mirror (`jlacko/berka-dataset`) — the original single-mirror design is documented in `docs/DECISION_LOG.md` §8. All eight tables matched on both mirrors. SHA-256 prefixes (primary mirror, LF-normalised):

| Table | Rows (verified) | SHA-256 prefix |
|---|---|---|
| trans | 1,056,320 | `111c70dc6626411d` |
| account | 4,500 | `215f4bfcb2520ab8` |
| client | 5,369 | `2b2f5253f21418c4` |
| disp | 5,369 | `77e08c0988473998` |
| order | 6,471 | `c1d909d5d8a56ce6` |
| loan | 682 | `0cf9fbe7ec2ebb7a` |
| card | 892 | `a4531f8d48f21ed5` |
| district | 77 | `d5422aa7326fde86` |

**TabFormer (E7, synthetic).** 24,386,900 transactions, 2,000 users, independently counted by a full chunked pass over the downloaded file this session, matching the source repository's stated figures. Single-source verification only (§5.4) — weaker than Berka's two-mirror check.

### A.3 Reproduction commands

```bash
pip install -r requirements.txt
python -m src.data_loader                # Berka: download + dual-mirror verify + print hashes
python -m experiments.pilot               # pipeline validation, ~4 min
python -m experiments.run_all             # E1-E4 real data
python -m experiments.e3_temporal_fixed   # corrected RQ5
python -m experiments.run_simulation      # Experiment B
python -m experiments.analyse             # real-data tables and figures
python -m experiments.analyse_simulation  # simulation tables and figures
python -m experiments.e5_multiseed        # E5: 5-seed replication (E1 + simulation)
python -m experiments.e6_drift            # E6: direct drift measurement (needs e1_impostor.csv)
python -m experiments.run_tabformer       # E7: download TabFormer first (see src/tabformer_loader.py)
python -m experiments.analyse_tabformer   # E7 tables and figures
python -m experiments.e8_disparate_impact # E8: disparate-impact audit (needs e1_impostor.csv)
python -m experiments.e9_loo_calibration  # E9: in-sample vs leave-one-out calibration (same cohort/labels as E1)
python -m experiments.e8b_disparate_impact_loo  # E8b: E8 repeated on LOO FPR (needs e9 output)
python -m experiments.e9_loo_full         # E9 full: all detectors + hybrid, 5 seeds, exact LOO (~4 h on 20 workers; resumable per seed)
python -m experiments.analyse_e9_full     # E9 full: summary and paired tables
```

Wall-clock times are intentionally not restated here: they are hardware- and load-dependent (this session ran three of these concurrently on a 22-logical-core machine, which is not representative of a quiet single run) and belong in a run log, not a reproducibility claim. Master seed 20260920 throughout, except E5, which sweeps `{20260920, 20260921, 20260922, 20260923, 20260924}` by design. Per-account seeds are derived deterministically as `seed * 100003 + account_id` (Berka) or a stable hash of the entity string (TabFormer), so a single account/entity can be re-run in isolation.

### A.4 Experimental parameters

| Parameter | Value |
|---|---|
| Evaluation window | 60 transactions (Berka, TabFormer), 100 (simulation) |
| Injection rate | 0.10 |
| History sizes | {1,5,10,25,50,100,250} Berka; + 500 simulation and TabFormer |
| Accounts/entities evaluated / donors | 300 / 400 (Berka, TabFormer) |
| Eligibility | Berka ≥ 310 transactions (1,319 qualify); TabFormer ≥ 560 (4,538 of 6,139 (User,Card) pairs qualify) |
| Threshold alpha | 0.05 |
| Hybrid κ | 25.0 |
| Bootstrap resamples | 2,000 |
| Isolation Forest | 200 trees, max_samples = min(256, n) |
| Autoencoder | d-4-d tanh (d=21 Berka, d=26 TabFormer), 300 epochs, Adam lr 0.01, weight decay 1e-4 |
| k-NN | k = 5 |
| E5 seeds | {20260920, 20260921, 20260922, 20260923, 20260924} |
| E6 drift blocks | recent_block = 100 txns, delta_block = 150 txns (positions 101-250 back) |
| E8 audit history sizes | n = 10, n = 50 |

### A.5 Not yet evaluated

Explicitly incomplete, not estimated. Items closed since the first draft are marked; everything else remains open.

- ~~Multi-seed repetition of cohort selection~~ — **closed for E1 and Experiment B (Mahalanobis/k-NN only)** by E5 (§6.8, §7.1). Result: the relative-90 threshold (25) and the global/personal crossover (n=10) are identical across all 5 seeds; across-seed SD of curve PR-AUC is 0.003-0.010, small relative to the curve's range. The headline numbers are not an artefact of one cohort draw. E2, Isolation Forest and the autoencoder in the simulation, and E3/E4/E7/E8 remain single-seed.
- History sizes above 250 on **real** Berka data (dataset depth ceiling is structural, not addressable without a deeper real dataset). TabFormer (synthetic) reaches 500.
- Any evaluation against **confirmed, adjudicated** fraud labels (no labelled real dataset with per-user histories was available). TabFormer's generator-produced `Is Fraud?` label was attempted as Protocol F (§7.10) but returned zero positive examples in the 300-entity evaluated cohort at 60-transaction windows — the base rate (0.122%) is too low for this sample size, not silently worked around.
- Adversarial anomalies designed to evade a personal baseline.
- ~~Replication on IBM TabFormer or Sparkov~~ — **closed for TabFormer** by E7 (§5.4, §7.10). Sparkov remains untried. TabFormer replication surfaced a genuine divergence (Isolation Forest's ranking reverses), not a clean confirmation.
- ~~Disparate-impact audit~~ — **closed at the ecological (district) level** by E8 (§7.11): no relationship found on the four district-economic axes tested. **Individual-level** demographic audit (age, tenure, immigration status, disability) remains impossible on Berka, which carries no such field.
- ~~Direct measurement of behavioural drift~~ — **closed** by E6 (§7.6): drift is real and lag-dependent, but does not predict the per-account 100→250 marginal-gain sign or size. The mechanism behind that marginal-gain result is now an open question again, not a settled interpretation.
- A targeted Protocol F follow-up on TabFormer that deliberately selects (User, Card) entities with historical generator-fraud, to get enough positive examples for a paired analysis (this session's attempt used an unselected random cohort and found none).
- Cross-seed replication of E3, E4, E7 and E8.
- **E9 follow-ups.** ~~Other four detectors, hybrid arm, five E5 seeds~~ — **closed for Berka E1** by the E9 full run (§7.2c, 2026-10-24): all six personal detectors and the shrinkage-hybrid arm, five seeds, exact leave-one-out. Still not evaluated: the global arm and the score-fusion k-NN hybrid under leave-one-out; TabFormer (E7) under leave-one-out; E2 (the parametric protocol) and the simulation under leave-one-out; alternative out-of-sample threshold rules (split-half, population-derived); and a check of whether Isolation Forest's behaviour at *n* ≤ 10 under leave-one-out is caused by the leave-one-out forest's different subsample size.
- A finer history-size grid between 10 and 25 (and 25 and 50), so that the relative-90 threshold is localised better than "the first grid point reached".
- A model-form sensitivity check on the saturation fit (e.g. power-law versus exponential); the relative-90 threshold is read from the empirical grid, not the fit, but the fitted τ values quoted in §7.1 are not cross-validated.
- Confirmation of the TabFormer download against its secondary IBM Box mirror (only the primary GitHub LFS endpoint was used this session).
