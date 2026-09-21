# How Much Personal History Is Enough? Measuring the Data Requirement for Individualised Financial Behavioural Anomaly Detection

**Status: first draft. All reported numbers were produced by the code in this repository. Nothing is estimated, extrapolated or illustrative.**

---

## Abstract

Systems that detect unusual financial behaviour by comparing a transaction against an individual's own history must decide when a person has transacted enough for their personal baseline to be worth using. Deployed systems and the patent literature assume such a sufficiency point exists and filter out users below it, but we could not find a study that measures where it lies, or whether it exists. We measure it.

We construct data-efficiency curves for unsupervised behavioural anomaly detection as a function of the number of transactions in one person's own history, on 300 accounts from the PKDD'99 Financial ("Berka") dataset of real Czech bank transactions, and on a controlled simulation of 240 individuals with deliberately varied behavioural parameters. Ground truth on real data is defined by provenance rather than by any statistical rule: a held-out transaction is replaced by a real transaction from a different real account, so labels cannot be circular with respect to any detector. Every history size is evaluated on an identical held-out window per account, making comparisons across sizes paired.

Five findings. First, a threshold exists and is earlier than expected: shrinkage-regularised Mahalanobis distance reaches 90% of its best observed ranking performance at 25 personal transactions (saturating-exponential fit R² = 0.970, τ ≈ 11.5), and gains nothing measurable beyond 100. Second, ranking quality and deployability diverge sharply. At 10 transactions the same model attains PR-AUC 0.441 against a 0.10 prevalence baseline while its history-calibrated operating point flags 96% of all transactions, making it useless in practice; usable false-positive rates appear only from 50 transactions. Third, model sophistication is counterproductive under scarcity: Isolation Forest is the weakest detector at every history size and never converges (τ ≈ 71), while univariate z-score baselines on transaction amount never learn at all, staying flat near 0.16 from 1 to 250 transactions. Fourth, a population model outperforms personalisation below roughly 10 transactions for Mahalanobis and below 50 for k-NN, and a shrinkage hybrid is never worse than either parent at any size, making it the dominant choice. Fifth, and against our own expectation, behavioural variability predicts the data requirement strongly under controlled manipulation (Spearman ρ = 0.48 to 0.71, all p < 0.001 after Holm correction; per-archetype thresholds range from 50 to 500 transactions) but none of eight behavioural descriptors measured on real accounts predicted that account's threshold after correction. We report the null.

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
| IBM TabFormer credit card | github.com/IBM/TabFormer | **No** | 2,000 users | Yes | Fraud | Rejected as the real-data layer: synthetic. Strong candidate for future extension. |
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
3. **No merchant, category or geography fields.** Four of the anomaly classes in the original brief cannot be studied on this dataset. They are moved to the simulation.
4. **1990s Czech retail banking.** Standing orders, cash withdrawals and pension credits dominate. Generalisation to modern card spending is an open question, not an assumption.
5. **Depth ceiling.** Discussed in §6.3.

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

**Clustering replaced by k-NN distance.** With *n* as small as 1 to 25 in a 23-dimensional, one-hot-heavy space, K-Means cluster counts and DBSCAN density parameters cannot be estimated without tuning on data we do not have. Distance to the k-th nearest training point is the representation-appropriate member of that family, requires no cluster count and degrades gracefully. Using all three regardless would have been variety for its own sake.

**No composite complexity index.** We could not justify a weighting across quantities on different scales. Descriptors are reported and tested individually with Holm correction.

### 6.4 Features

Twenty-three dimensions, all causal (depending only on the transaction and strictly earlier transactions of the same account): log amount; credit indicator; day-of-month and day-of-week as sine/cosine pairs; log inter-arrival gap; one-hot `operation` (5 + missing); one-hot `k_symbol` (7 + missing).

Standardisation is **fitted on training history only**. At *n* = 1 dispersion is undefined and scales collapse to unity, which is recorded as a degenerate regime rather than concealed.

### 6.5 Models

| Model | Form | Role |
|---|---|---|
| Gaussian z | \|z\| on log amount, mean/SD | Simple statistical baseline |
| Robust z | \|z\| on log amount, median/MAD | Robust statistical baseline |
| Mahalanobis | Shrinkage-regularised, ridge-stabilised | Simple multivariate |
| k-NN | Distance to 5th nearest training point | Non-parametric |
| Isolation Forest | 200 trees, seeded | Standard ML detector |
| Autoencoder | 23-4-23 tanh, full-batch Adam, NumPy | Nonlinear representation |

**Hybrid mechanism (RQ8).** Personal location and scatter are shrunk toward population estimates with a weight determined solely by history size:

    μ_hybrid = (n·μ_personal + κ·μ_global) / (n + κ)
    Σ_hybrid = (n·S_personal + κ·S_global) / (n + κ) + εI

κ = 25, fixed in advance, not tuned on test data. At *n* = 0 this is exactly the global model; as *n* grows it converges to the personal model. A score-fusion hybrid with weight n/(n+κ) over rank-normalised scores is also evaluated.

### 6.6 Evaluation

**Why not accuracy.** At 10% prevalence, a detector flagging nothing scores 90% accuracy while detecting nothing. Accuracy cannot separate a useful detector from an inert one here, and changes with prevalence without any change in the detector. We report **PR-AUC as primary** (both precision and recall condition on the minority class) alongside ROC-AUC.

**Threshold without leakage.** Operating-point metrics need a threshold, and choosing one on test labels would leak. We use the 95th percentile of the scores the fitted model assigns to **its own training data** — a rule a deployed system could actually apply, which degrades visibly when history is too short to estimate a quantile. That degradation is part of the measurement.

**Statistics.** Resampling is over **accounts**, the independent unit; resampling transactions would understate uncertainty. Paired bootstrap (2,000 resamples) and Wilcoxon signed-rank for arm comparisons, with rank-biserial effect size reported alongside every p value, and Holm correction within each family. Saturation curves are fitted **with R² reported**, so poor fits are visible.

### 6.7 Experiment B: controlled simulation

The real data cannot supply merchant, category or geographic channels, and cannot supply *controlled* variability. We simulate 240 individuals across six archetypes (40 each), each defined by a measurable parameter vector rather than a label: log-amount dispersion, merchant count and Zipf exponent, category count, location count and home-location weight, inter-arrival Gamma shape, hour-of-day von Mises concentration, and a drift rate that rotates merchant and category distributions over time.

Archetypes: **regular** (σ = 0.30, 5 merchants, 1 location), **moderate**, **variable** (σ = 1.10, 60 merchants, 8 locations), **traveller** (15 locations, home weight 0.35), **seasonal** (drift 0.004), **burst** (Gamma shape 0.35, producing inactivity punctuated by clusters).

Six anomaly classes generated from deliberately shifted distributions, independent of any detector: amount, merchant, category, geographic, temporal, and multivariate. Ground truth is the generating branch, which no detector observes.

**Nothing from the simulation is a real-world observation, and simulation results are never pooled with real-data results.**

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

**Answer to RQ2.** Yes, performance plateaus, and early. For Mahalanobis, 98% of the total observed gain occurs by *n* = 50. The 100→250 step is negative (−0.016), which is the "additional history can hurt" outcome; §7.5 gives a mechanism.

### 7.2 The gap between ranking quality and deployability

PR-AUC measures ranking. A deployed detector needs an operating point. Mahalanobis, personal arm, threshold at the 95th percentile of training scores:

| n | Precision | Recall | F1 | FPR | Flag rate |
|---|---|---|---|---|---|
| 1 | 0.100 | 1.000 | 0.182 | 0.999 | **0.999** |
| 5 | 0.101 | 1.000 | 0.184 | 0.988 | **0.989** |
| 10 | 0.104 | 0.997 | 0.189 | 0.956 | **0.960** |
| 25 | 0.256 | 0.899 | 0.391 | 0.335 | 0.391 |
| 50 | 0.357 | 0.811 | 0.483 | 0.189 | 0.251 |
| 100 | 0.417 | 0.753 | 0.523 | 0.134 | 0.196 |
| 250 | 0.460 | 0.687 | 0.536 | 0.104 | 0.162 |

**This is the most practically important result in the paper.** At *n* = 10, PR-AUC is 0.441, more than four times the prevalence baseline, which looks like a working detector. The same model at its own history-calibrated threshold flags 96% of all transactions. It achieves near-perfect recall by alerting on almost everything.

**Interpretation.** With a short history, the model cannot estimate the *scale* of its own score distribution even when it ranks tolerably well. The 95th percentile of 10 training scores is a poor estimate of the 95th percentile of the score distribution, and the resulting threshold sits far too low. Ranking ability and calibration have different data requirements, and calibration is the more expensive of the two.

**Consequence for RQ3.** A threshold defined on PR-AUC and a threshold defined on false-positive rate give different answers on the same curves. Any single-number "N transactions is enough" claim conceals which one is meant.

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

**Answer to RQ3.** A threshold can be identified, but only conditional on the definition. On the best real-data model: 25 transactions for ranking performance, 50 for a tolerable false-positive rate, and **no size at all** for across-individual stability.

### 7.4 RQ6 and RQ7: model complexity and global versus personal

**Model complexity.** Isolation Forest is the **worst** detector at every history size, from 0.100 at *n* = 1 (exactly the prevalence baseline, i.e. no signal) to 0.343 at 250, never approaching Mahalanobis. Its τ = 71.2 means it is still far from saturation at the largest size we can evaluate. The autoencoder is competitive but saturates below Mahalanobis (0.518 vs 0.658).

Computation matches: at *n* = 250 the median fit cost is 0.46 ms for Mahalanobis, 5.0 ms for k-NN, 22.9 ms for the autoencoder, and 156.3 ms for Isolation Forest, a 340-fold difference for the worst-performing model.

**Answer to RQ6.** No. Under scarcity the sophisticated models are strictly worse and far more expensive. Isolation Forest's subsampling-based isolation mechanism has nothing to work with when the training set is smaller than its own default subsample. The best detector in this study is a shrinkage-regularised covariance estimate.

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

**Interpretation, offered as one possible explanation.** This is consistent with behavioural drift: transactions from three years ago describe a person who no longer exists, and including them widens the estimated personal distribution so that genuinely unusual transactions no longer fall in its tail. This also offers a mechanism for the negative 100→250 marginal gain in §7.1. We did not measure drift directly, so this is an interpretation, not a finding.

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

---

## 8. Discussion

### 8.1 What we take to be established by this work

Within the stated scope, the following are experimental observations:

1. Personalised behavioural anomaly detection on real transaction histories improves steeply between 5 and 25 personal transactions and is essentially saturated by 50.
2. Ranking quality saturates well before calibration does. A model that ranks acceptably at 10 transactions is unusable at any threshold it can set for itself.
3. Isolation Forest, the most widely deployed off-the-shelf anomaly detector, is the worst performer at every history size in this setting and the most expensive by two orders of magnitude.
4. Personalisation is worse than a population model below a model-dependent crossover.
5. A history-weighted shrinkage hybrid dominates both parents when personal signal lives in the parameters of a shared representation, and fails when it lives in the representation.
6. Behavioural variability governs the data requirement under controlled manipulation; we could not detect that relationship in real accounts.

### 8.2 What remains interpretation

That the negative 100→250 marginal gain and the cost of reaching further back both arise from behavioural drift is an interpretation consistent with our results, not something we measured. Testing it requires an explicit drift measurement we did not perform.

That the real-data null on complexity reflects thin descriptors rather than a genuinely absent effect is also interpretation. The alternative — that observed variability genuinely does not predict the requirement in real retail banking — is equally consistent with what we measured.

### 8.3 Practical implication, stated carefully

If a practitioner wanted one number, this study's answer for its best real-data model would be: **about 25 transactions for ranking, about 50 before the false-positive rate is tolerable, and no number at all that guarantees a given individual is well served.** The third clause matters most. The across-account spread never stabilised at any history size we evaluated, so a system meeting a target on average will still fail badly for a substantial minority of users. Population averages are the wrong unit for a decision that is applied per person.

---

## 9. Limitations

**Data.**
1. One dataset, one bank, one country, 1993-1998 retail banking. Generalisation to modern card payments is untested.
2. Day-resolution timestamps preclude hour-of-day behaviour on real data.
3. No merchant, category or geographic channels on real data. The three anomaly classes that showed the steepest data requirements in simulation could not be tested on real behaviour.
4. Real-data history sizes stop at 250 because of dataset depth. We cannot rule out changes beyond that point.
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
13. Single seed for cohort selection. Repeating across seeds would strengthen the curves and is the first item of future work.
14. Wilcoxon p values over 300 paired accounts are extremely small for effects of any size, which is why effect sizes are reported alongside. Several differences that are statistically unambiguous (the hybrid at *n* = 10, −0.033) are practically negligible.

---

## 10. Ethical considerations

**An anomaly detector identifies deviations from a behavioural baseline; it does not establish fraud or wrongdoing.** Every ethical concern below follows from how easily that sentence is forgotten.

**Treating unusual behaviour as suspicious behaviour.** The transactions our detectors flag are, by construction, transactions that differ from a person's past. In real life those are holidays, house moves, medical costs, a new job, a death in the family, leaving an abusive relationship. A system that escalates deviation to suspicion punishes people for changing.

**False positives are not symmetric with false negatives.** A missed anomaly costs an institution money. A false positive can mean a declined card in a foreign country, a frozen account during an emergency, or an account-holder treated as a suspect. Our best real-data model at 50 transactions has a 19% false-positive rate, which at realistic transaction volumes means a very large number of wrongly flagged legitimate transactions.

**Sparse-history users bear the cost.** The people with the least history are new customers, young people, recent immigrants, people re-entering the formal banking system. §7.2 shows these are exactly the users for whom a personal model flags almost everything. Deploying personalisation without a history floor concentrates false positives on populations already more likely to face financial exclusion. The finding that a global or hybrid model outperforms personalisation in the sparse regime is directly relevant: for these users, personalisation is not merely useless, it is harmful.

**Discrimination.** Behavioural features are proxies for life circumstances. Irregular income, cash reliance and geographic mobility are correlated with class, occupation, immigration status and disability. A detector calibrated on regular salaried behaviour will treat irregular lives as anomalous by default. We did not audit for disparate impact, and no system of this kind should be deployed without such an audit.

**Privacy and surveillance.** Personal transaction histories are among the most revealing datasets about a person. Per-user behavioural modelling requires retaining and processing them continuously. That §7.6 finds recent history to be sufficient and older history mildly harmful is a rare case of a privacy interest and a performance interest aligning: shorter retention appears to cost nothing here.

**Explainability.** Mahalanobis distance yields a scalar with no account of which behaviour was unusual. Any deployment affecting a person's access to their money should be able to state the reason, and should support contest and correction.

**Our own data handling.** The Berka dataset was anonymised by its publisher before release. We redistribute no raw records, work only at account-identifier level, and make no attempt at re-identification.

**Misuse.** This work should not be cited as evidence that N transactions make a detector safe to deploy, nor that flagged transactions warrant adverse action against a customer.

---

## 11. Conclusion

We measured how much of a person's own transaction history an unsupervised behavioural anomaly detector needs, on real bank histories and under controlled simulation, using a ground-truth protocol defined by provenance rather than by any statistical rule.

A sufficiency point exists and is earlier than the framing of the question suggests: roughly 25 transactions for ranking quality on our best real-data model, saturating by 50. But the single number is the least useful part of the answer. Ranking quality and calibration have different requirements, and a model that ranks well at 10 transactions flags 96% of everything. Personalisation is actively harmful below a crossover where a population model built from strangers does better. The most sophisticated detector we evaluated was the worst at every size. Behavioural variability changes the requirement tenfold when manipulated, yet we could not detect that relationship in real accounts, and we report the null. And no history size we evaluated made performance stable **across individuals**, which undermines any per-person guarantee derived from a population average.

The practical recommendation is not a number. It is that a system should use a hybrid that degrades to population behaviour when personal history is thin, should hold recent history rather than all history, and should not assume that a detector which ranks well is a detector that can be deployed.

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

Python 3.11+. Verified versions: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8. No GPU. No PyTorch (the autoencoder is NumPy).

### A.2 Data integrity

The loader verifies row counts against the primary source and raises otherwise. SHA-256 prefixes of retrieved files:

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

### A.3 Reproduction commands

```bash
pip install -r requirements.txt
python -m src.data_loader              # download + verify + print hashes
python -m experiments.pilot            # pipeline validation, ~4 min
python -m experiments.run_all          # E1-E4 real data, ~25 min
python -m experiments.e3_temporal_fixed  # corrected RQ5, ~4 min
python -m experiments.run_simulation   # Experiment B, ~7 min
python -m experiments.analyse          # real-data tables and figures
python -m experiments.analyse_simulation # simulation tables and figures
```

Master seed 20260920 throughout. Per-account seeds are derived deterministically as `seed * 100003 + account_id`, so a single account can be re-run in isolation.

### A.4 Experimental parameters

| Parameter | Value |
|---|---|
| Evaluation window | 60 transactions (real), 100 (simulation) |
| Injection rate | 0.10 |
| History sizes | {1,5,10,25,50,100,250} real; + 500 simulation |
| Accounts evaluated / donors | 300 / 400 |
| Eligibility | ≥ 310 transactions (1,319 accounts qualify) |
| Threshold alpha | 0.05 |
| Hybrid κ | 25.0 |
| Bootstrap resamples | 2,000 |
| Isolation Forest | 200 trees, max_samples = min(256, n) |
| Autoencoder | 23-4-23 tanh, 300 epochs, Adam lr 0.01, weight decay 1e-4 |
| k-NN | k = 5 |

### A.5 Not yet evaluated

Explicitly incomplete, not estimated:

- Multi-seed repetition of cohort selection (single seed used).
- History sizes above 250 on real data (dataset depth).
- Any evaluation against confirmed fraud labels (no labelled dataset with per-user histories was available).
- Adversarial anomalies designed to evade a personal baseline.
- Replication on IBM TabFormer or Sparkov to test whether curve shapes transfer to card-payment structure.
- Disparate-impact audit across account or demographic segments.
- Direct measurement of behavioural drift to test the §8.2 interpretation.
