# Final quality control

Walking the 15-point checklist from `instruction.txt`'s "FINAL QUALITY CONTROL"
section explicitly, as instructed, before treating the paper as finished.
This is done at the end of the independent re-verification + extension
session (2026-09-21), on top of the checks the prior session already
performed and recorded (visible in `PAPER.md.reference`'s own rigor).

---

**1. Is the research question actually novel enough?**

Yes, with the narrowed claim the paper itself states (§1.1, §3.2): the
contribution is the *measurement and protocol*, not a new algorithm. Every
detector used is standard. Re-audited this session (see "Novelty" in this
project's planning phase) — no published work was found that measures
per-user data-efficiency curves for unsupervised transaction anomaly
detection with an operational sufficiency criterion and a matched
global/personal/hybrid comparison. The audit is stated as a documented
search, not a proof of absence (§3.3).

**2. Is the experiment capable of answering it?**

Yes. The split design (§6.1) directly measures performance as a function of
*n*, the paired structure isolates the effect of history size from
across-account difficulty, and three independent threshold definitions
(§7.3) give the sufficiency question multiple, disagreeing-where-they-should
-disagree answers rather than one number dressed up as certainty.

**3. Is there data leakage?**

No, by construction (§6.1): the evaluation window is fixed and always
strictly after the training history at every *n*; the global model is
fitted only on donor accounts' *earliest* transactions, so it cannot see
anything from the period being evaluated. Verified this session: reran the
pilot sanity checks (global-arm invariance across *n*, near-perfect-PR-AUC
rate near zero) and they held.

**4. Is the train/test methodology correct?**

Yes — paired at the account level, threshold set from training-data
quantiles only (never test labels, §6.6), donor/evaluated account sets
disjoint. E7 (TabFormer) and E6 (drift) both reuse this same design against
their own data, not a weaker ad hoc version.

**5. Are synthetic anomalies being treated honestly?**

Yes. Protocol I's ground truth is provenance (a real transaction from a
different account), not a statistical rule, so it cannot be circular with
any detector (§6.2). Protocol P is explicitly labelled parametric and
secondary. TabFormer's `Is Fraud?` (Protocol F) is explicitly labelled
generator-produced, never "confirmed fraud" (§7.10), and its null result
(zero positives in the evaluated cohort) is reported as a null, not
worked around.

**6. Is anomaly being incorrectly equated with fraud?**

No. §2 (Terminology) draws the anomaly/outlier/suspicious/fraud distinction
explicitly and the paper repeats "an anomaly detector identifies deviations
from a behavioural baseline; it does not establish fraud or wrongdoing" at
multiple points (abstract, §2, §10, conclusion). TabFormer's fraud column is
never described as confirmed fraud anywhere in this paper or the site.

**7. Are the metrics appropriate for imbalanced anomaly detection?**

Yes. PR-AUC is primary, with an explicit stated reason to reject accuracy
(§6.6), and the paper additionally distinguishes ranking quality from
deployability (§7.2) — arguably a stronger standard than the brief asked
for, since it shows a metric that looks fine (PR-AUC 0.441 at *n*=10) while
the actual deployed system would be useless (96% flag rate).

**8. Are conclusions supported by the actual results?**

Re-checked this session specifically for the *new* claims: the Isolation
Forest reversal (§7.10) is read directly from `e7_tabformer_impostor.csv`;
the drift null (§7.6/E6) is read directly from the Spearman test in
`e6_drift.py`'s output; the disparate-impact null (§7.11/E8) is read
directly from `e8_disparate_impact.py`'s output. No number in the revised
sections was estimated or backfilled from intuition.

**9. Are statistical tests appropriate?**

Yes — Wilcoxon signed-rank with rank-biserial effect size for paired
comparisons, Holm correction within every family of tests (RQ4 descriptors,
E6's two models, E8's three district variables), bootstrap resampling at
the account level (not the transaction level), consistent with §6.6 and
`src/stats.py`.

**10. Are the datasets properly cited?**

Yes. Berka: Berka & Sochorova, PKDD'99 Discovery Challenge, primary URL
given (§5.2, reference 2). TabFormer: Padhi et al. (2021), ICASSP,
repository URL given (§5.4, reference 10) — this citation already existed
in the prior session's reference list and was cross-checked against it
rather than re-invented.

**11. Are all citations real?**

The prior session's citation-integrity note (§12) states every reference
was located and verified against a primary or publisher source during its
own audit, with one exception (ref. 14's incomplete author list, stated as
incomplete rather than invented). This session did not add any new
citations beyond the TabFormer one, which was already present and verified.

**12. Can another researcher reproduce the experiment?**

Yes, and this session is itself evidence: we independently re-executed the
entire documented pipeline from a *different* environment (fresh Python
3.14.6 venv, newer dependency versions, different machine) using nothing
but the committed code and `README.md`'s reproduction commands, and got
**bit-exact matches** on every number checked (`docs/VERIFICATION.md`).
This is a stronger reproducibility demonstration than a written claim.

**13. Are limitations honestly stated?**

Yes, and expanded this session (§9 now has 15 items, up from 14, with
several narrowed rather than silently removed once E5-E8 partially
addressed them — e.g. limitation 1 is not deleted when E7 exists, it is
revised to say what E7 does and does not close).

**14. Does the experiment test the hypothesis rather than merely illustrate it?**

Yes — this is the standard this session applied most aggressively. The
prior session's own first draft *offered* the drift explanation for a
puzzling result without testing it (§8.2 as originally written explicitly
flagged this gap). Rather than leave a plausible-sounding illustration
standing, E6 was built specifically to test it, and the paper's conclusion
changed as a direct result: the drift explanation was tested and did not
survive. This is the clearest evidence in the whole project that findings
here are falsifiable claims, not illustrations chosen to look right.

**15. Is there a meaningful contribution beyond comparing ML algorithms?**

Yes. The paper's contribution is explicitly the *measurement protocol and
the resulting quantities* (§1.1), and this session's extensions strengthen
that framing rather than turning it into an algorithm comparison: E5 tests
whether the measurement is stable across sampling noise, E6 tests whether a
proposed mechanism is real, E7 tests whether the measurement generalises
across dataset structure, E8 tests a consequence of the measurement for
who bears its cost. None of these are "which algorithm wins."

---

## Verdict

All 15 checks pass. Two answers changed materially from what the prior
session's first draft could have claimed if it had stopped there: the
drift interpretation (§7.6) did not survive direct testing, and the
"sophisticated models can't help under scarcity" claim (§7.4) turned out to
be dataset-dependent rather than universal once a second dataset was
tried. Both are reported as what they are — a retraction of an untested
interpretation, and a genuine, not-fully-understood divergence — rather
than smoothed into the existing narrative.

---

## Addendum (2026-10-23): revisions after the automated review loop

This checklist was filled in before Experiment E9. The original answers above are kept as written; the items below were found to need qualification. See `review-stage/AUTO_REVIEW.md`.

- **Item 3 (data leakage) and item 4 (train/test methodology): answers stand for leakage, but incomplete.** There is no train/test leakage. However the operating threshold was computed from *in-sample* training scores (`s_train = model.score(X_hist)` after fitting on `X_hist`), which is biased low at small n. This is a calibration bias, not leakage, and the checklist did not look for it.
- **Item 7 (metrics appropriate): PR-AUC answer stands; the operating-point answer does not.** The "PR-AUC 0.441 vs 96% flag rate" example cited above as a strength is an in-sample-calibration artefact. With leave-one-out calibration (E9) Mahalanobis flags 1.2% at n=10 and has FPR 0.066 at n=25.
- **Item 8 (conclusions supported by results): the 96% claim and its 'noisy quantile' explanation were not supported.** They were untested interpretation. Withdrawn in PAPER.md section 7.2.
- **Item 12 (reproducibility): still true, but reproduction does not validate the method.** The bit-exact reproduction in VERIFICATION.md confirmed the code produced the reported numbers, including the biased ones.
- **Item 13 (limitations): extended** with limitation 15 (in-sample operating-point metrics outside E9) and the grid-resolution caveat.

Remaining open: E9 covers Mahalanobis and k-NN on the primary seed only; the other detectors, hybrid arm, E5 seeds and TabFormer still use the in-sample rule.

---

### Update (2026-10-24): E9 extended to all detectors and five seeds

The full run (`experiments/e9_loo_full.py`; exact leave-one-out for all six personal detectors and the shrinkage-hybrid arm on the five E5 cohorts) is complete. Validation: in-sample rows reproduce `e5_multiseed_e1.csv` on all 14,700 rows per seed (max abs diff 1.1e-16); primary-seed Mahalanobis and k-NN leave-one-out rows equal the first E9 exactly (8,400/8,400).

- Items above that said "Mahalanobis and k-NN, primary seed only" are superseded for Berka E1. Still not run under leave-one-out: global arm, score-fusion hybrid, E2, simulation, TabFormer.
- **QC item 7 refinement:** the in-sample rule is biased low for most detectors at small n but is not an upper bound for all (Isolation Forest at n=5 and robust z at n>=50 go the other way), so the wording 'upper bound on false-positive burden' used earlier in the paper was itself too strong and has been removed (limitation 15).
- Untested confound recorded: for Isolation Forest at n<=10 the leave-one-out forest has a different subsample size from the n-point forest, so its scores may not share a scale with the evaluation-window scores.
