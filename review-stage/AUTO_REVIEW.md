# Auto review log: How Much Personal History Is Enough?

## Round 1 (2026-10-23)

### Assessment (Summary)
- Score: 8/10 (reviewer assumed ICML tier)
- Verdict: almost -> meets the stop rule (score >= 6 AND verdict in {ready, almost})
- Reviewer: `llm-chat` MCP, which here is a local **qwen3:8b**, not a frontier model.
  Treat the score as weak evidence. Several of its points are wrong or out of scope (see triage below).
- Key criticisms (as stated by the reviewer):
  1. No real fraud labels
  2. Data heterogeneity / no cross-dataset generalisation
  3. Simulation personal-novelty features "not rigorously grounded"
  4. Threshold stability undefined
  5. Ecological audit underpowered
  6. Single seed for E2/E3/E4/E7/E8
  7. Saturation fit not validated
  8. No adversarial-imitation anomalies

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**SCORE**: 8/10 (ICML tier, but would struggle for NeurIPS due to limited novelty).

**VERDICT**: **Almost** ready.

**WEAKNESSES**
1. **No real fraud labels**: All "anomalies" are synthetic (Protocol I/P) or simulated. The real dataset (PKDD) has no fraud labels, so the paper cannot evaluate true anomaly detection. This invalidates the "personalization threshold" claim as it is trained on a clean dataset, not real-world fraud.
   **FIX**: Use real fraud labels (e.g., from Kaggle/IBM datasets) or reframe the analysis to focus on synthetic anomaly detection performance, clearly stating the limitations.
2. **Data heterogeneity**: The real dataset (PKDD) has no merchant/geo fields, no fraud labels, and is from a single bank/era. Simulations are isolated from real data, and TabFormer is a synthetic dataset. No evidence of cross-dataset generalization.
   **FIX**: Add cross-validation across datasets (e.g., train on TabFormer, test on Berka) or clarify that results are dataset-specific.
3. **Simulation "personal-novelty features" are not rigorously grounded**: The paper claims personal-novelty features are computed against a "population reference," but the source of this reference (open-web search) is not systematic. This undermines the validity of the simulation.
   **FIX**: Define the population reference rigorously (e.g., via a systematic review of existing behavioral models) or abandon the personal-novelty features.
4. **Threshold stability is undefined**: All models have undefined thresholds (SD > median across accounts), making the "personalization threshold" claim uninterpretable.
   **FIX**: Re-define the threshold as a quantile (e.g., 95th percentile of training scores) and validate its stability across accounts.
5. **Ecological audit underpowered**: The audit on disparate impact (salary/unemployment/urbanization) uses 300 accounts and reports p >= 0.75, which is trivially non-significant.
   **FIX**: Use a larger, more diverse cohort or abandon the claim.
6. **Single seed for experiments (E2/E3/E4/E7/E8)**: Lacks reproducibility. Results are not confirmed across seeds, which is critical for claiming "relative-90 threshold" or "marginal gain" conclusions.
   **FIX**: Report results across multiple seeds or use bootstrapping to validate claims.
7. **"Personalization threshold" is a post-hoc fit**: The saturating exponential fit (tau=11.5) is not justified. No validation of the fit's robustness or cross-validation.
   **FIX**: Validate the fit with cross-validation or alternative models (e.g., linear regression).
8. **No adversarial-imitation anomalies**: The paper claims to measure "data requirements," but adversarial-imitation (e.g., fraudsters mimicking users) is a key real-world challenge.
   **FIX**: Add adversarial-imitation simulations or explicitly note the absence of this case.

**Ready?** **Almost**. The paper has strong methodology and real-world data but is fatally flawed by the lack of real fraud labels and underpowered ecological analysis. Fixing the first two weaknesses would make it submission-ready.

</details>

### Author triage of the review
- (1) Out of scope by design: the paper is explicitly a behavioural-anomaly study, not fraud detection (sections 1.2 and 2). Already a stated limitation (section 9 item 6, A.5). No change.
- (2) Partly valid and already stated (section 9 item 1). "Train on TabFormer, test on Berka" is not meaningful for per-user history curves. No change.
- (3) Wrong: the "population reference" is the donor-fit reference history inside the simulation's feature code, not a literature review. No change.
- (4) Misread: stability is a separate, deliberately reported threshold definition (section 7.3), not the main one.
- (5) Partly valid: added a power note (|rho| about 0.21 at n=183, about 0.16 at n=300) and softened the E8 wording. A larger cohort is not possible (Berka depth).
- (6) Already disclosed (section 9 item 13, A.5); E5 covers E1 and the simulation. No new seeds run this round.
- (7) Partly valid: the relative-90 threshold is read from the empirical grid, not the fit. Added the grid-resolution caveat and a not-yet-evaluated item for model-form sensitivity.
- (8) Already stated (section 9 item 6).

### Issue found by the author, not by the reviewer (the important one)
- `src/experiment_engine.py` computes the operating threshold from **in-sample** training scores (`s_train = model.score(X_hist)` after `fit(X_hist)`). For personal Mahalanobis at n <= d = 23 this is biased low by construction, and a k-NN training point counts itself as a neighbour. The paper's headline "flags 96% at n=10" and its explanation ("noisy quantile estimate") were therefore wrong, or at best untested.
- New experiment `experiments/e9_loo_calibration.py` (same cohort, seed and labels as E1; its in-sample rows reproduce the E1 table exactly) compares in-sample with leave-one-out calibration:

| Mahalanobis n | in-sample flag rate | LOO flag rate | LOO FPR | LOO recall |
|---|---|---|---|---|
| 5 | 0.989 | 0.152 | 0.151 | 0.163 |
| 10 | 0.960 | 0.012 | 0.007 | 0.058 |
| 25 | 0.391 | 0.110 | 0.066 | 0.503 |
| 50 | 0.251 | 0.126 | 0.073 | 0.612 |
| 250 | 0.162 | 0.136 | 0.080 | 0.647 |

- E8 (disparate impact) used the in-sample FPR, which sits at a ceiling at n=10, so that half of the audit was uninformative. Repeated as `experiments/e8b_disparate_impact_loo.py`: still no Holm-significant result; a weak negative salary trend (uncorrected p 0.024 to 0.046 at two sizes) is reported, not claimed.

### Actions Taken
- PAPER.md: rewrote section 7.2 (both rules, withdrawal of the 96% claim); section 7.3 RQ3 answer plus grid-resolution caveat and power note; section 7.11 (E8 caveat, E8b table); sections 6.6 and 6.8 (E9, E8b); section 8.1 items 2, 9 and 10; section 8.3; section 10 (FPR 19% to 7.3% under LOO, discrimination paragraph); section 11; abstract; status banner; section 9 limitation 15; A.3 commands; A.5.
- README.md headline row corrected.
- Added `experiments/e9_loo_calibration.py` and `experiments/e8b_disparate_impact_loo.py`; outputs `results/raw/e9_loo_calibration.csv`, `results/tables/e9_loo_calibration_summary.csv`, `results/tables/e8b_fpr_district_correlation_loo.csv`.

### Not done (also listed in the paper's A.5)
- E9 covers only Mahalanobis and k-NN on the primary seed. Not run for the other detectors, the hybrid arm, the E5 seeds or TabFormer.
- No second review round was run.

### Follow-up (same day): site and docs brought in line with the corrected paper
- `web/build_data.py` now emits `op_loo` (E9) and `e8b_corr` (E8b); `web/index_template.html` has the corrected section 2 (both tables, correction note), new stat cards, E8b table, revised ethics/limitations text and commands; `web/data.json` and `web/index.html` rebuilt (the embedded paper is the revised PAPER.md).
- Checked with a stubbed-DOM Node run (0 script errors; rendered tables match the CSVs). **Not checked in a real browser** (the Chrome extension was not connected), so layout is unverified.
- README, `docs/QC_CHECKLIST.md`, `docs/VERIFICATION.md` and `docs/DECISION_LOG.md` (new section 10) updated; earlier answers kept, with dated addenda.
- `results/_prior_run/` (archived earlier output, including an old copy of the site) deliberately left untouched.

### Status
- Stopping after Round 1 per the stop rule (8/10, "almost"). The stop was triggered by a low-reliability reviewer, so it should not be read as an independent endorsement. A second round with a stronger reviewer on the revised paper is advisable before submission.

### Follow-up 2 (2026-10-24): full E9 run
- Ran E9 for all six detectors plus the shrinkage hybrid at all five E5 seeds (exact leave-one-out; 1,500 accounts; ~3.8 h). Validation: in-sample rows match E5 on 14,700/14,700 rows per seed to 1.1e-16; first-E9 Mahalanobis/k-NN rows reproduced exactly.
- Headline: the Mahalanobis correction is stable across seeds; the artefact also hits the autoencoder and hybrid; it is not uniform in direction (Isolation Forest, robust z); usable operating point at about n=25 holds for Mahalanobis, autoencoder and hybrid only (k-NN about 50; Isolation Forest not within 250); hybrid has the best small-n operating behaviour (observation only).
- PAPER.md (abstract, banner, 6.8, 7.2c, RQ3 answer, 8.1 item 10, limitation 15, A.3, A.5), README, web/ (new all-detector table, rebuilt; checked under a stubbed DOM, not a real browser) and docs updated.
- Still uncovered: global arm, score-fusion hybrid, E2, simulation, TabFormer under leave-one-out; alternative out-of-sample rules; Isolation Forest scale-mismatch confound untested. No second review round has been run on the corrected paper.
