# Claim-to-evidence audit (self-audit)

**Not a cross-model audit.** The skill calls for a fresh Codex thread; none was available (no `.aris/reviewer.json`, no Codex MCP), so this is the author-side check of every poster claim against the paper source in `paper/sections/` and the result tables. It is `REVIEW_UNAVAILABLE` for the independent pass.

| # | Poster claim | Paper source | Match |
|---|---|---|---|
| 1 | Banner: 25 transactions reach 90% of best PR-AUC (Berka) | 5_results (Table II; relative-90 = 25) | OK |
| 2 | Banner: personal model starts to beat the crowd at n = 10 | 5_results, Personal/Population/Hybrid (Mahalanobis) | OK (Mahalanobis; k-NN is 50, stated in card 8) |
| 3 | Banner/card 9: 96% to 1.2% flagged at n = 10 | 5_results Table IV (in-sample 0.960, LOO 0.012, primary cohort) | OK |
| 4 | Banner: 5 of 5 cohort draws give the same 25 threshold | 5_results, "threshold is not an artefact of one cohort" | OK (Mahalanobis) |
| 5 | Card 1: "unseasoned users" filtered by deployed systems | 1_introduction, one patent family | OK; "deployed systems" is the paper's "production systems ... one patent family" |
| 6 | Card 1: thin histories are new customers, young people, recent immigrants | 6_discussion, Ethical Considerations | OK |
| 7 | Card 2: 60-transaction window, 10% injected, six per window | 4_protocol | OK |
| 8 | Card 2: population model fitted on 400 donor accounts | 4_protocol, Split Design | OK |
| 9 | Card 3: 300 accounts / 240 people / 300 entities; real / controlled / synthetic | 3_setting_data | OK |
| 10 | Card 5: 0.657 best PR-AUC; 98% of gain by n = 50 | 5_results, Data-Efficiency Curves | OK |
| 11 | Card 6: y_max, tau, Rel-90, Rel-95 for six detectors | 5_results Table II | OK; a dash = never met on the grid |
| 12 | Card 6: stability criterion never met by a detector that learns | 5_results, "Three definitions" | OK |
| 13 | Card 7: 0.647 to 0.642 (n = 50 to 250), -0.036 (stride 4), rho = -0.08, Holm p = 0.32 | 5_results, Recency, Volume and Drift | OK (rho = -0.081, Mahalanobis) |
| 14 | Card 8: personal below global until n = 10; hybrid tracks or beats both | 5_results Table III | OK; paper's wording is "never detectably worse than the personal model and never meaningfully below the global model" |
| 15 | Card 8: hybrid equation for mu_h, kappa = 25 fixed in advance | 4_protocol, Shrinkage hybrid | OK (location equation only; scatter equation omitted) |
| 16 | Card 8: crossover n = 10 (Mahalanobis), n = 50 (k-NN) | 5_results | OK |
| 17 | Card 9: calibration table values for n = 5, 10, 25, 50, 100, 250 | 5_results Table IV | OK |
| 18 | Card 9: in-sample scores biased low; LOO silent at n = 10, near-nominal from n = 25 | 5_results, Calibration | OK; the paper adds that FPR still misses 0.05 by 1.6 to 3 points |
| 19 | Card 10: Mahalanobis threshold 250; Isolation Forest best at n = 5 to 50; crossover 25 to 100 | 5_results, Replication on a Second Dataset | OK; "stronger global model" is the paper's expected reason, not a tested cause |
| 20 | Card 11: archetype thresholds 50 vs 500; rho 0.48 to 0.71; real |rho| <= 0.12 | 5_results, Behavioural Variability | OK |
| 21 | Card 4: district audit finds no Holm-significant gap; age, tenure, immigration untested | 5_results Disparate-Impact; 6_discussion | OK |
| 22 | Takeaway "calibrate out-of-sample" | 6_discussion, Practical implication | OK as a summary of the LOO finding |

Omissions that a reader should know about: the n = 10 vs 250 flat-curve caveat for Rel-90, the in-sample vs LOO direction reversal for Isolation Forest, and the simulation's hybrid failure are in the paper but not on the poster.
