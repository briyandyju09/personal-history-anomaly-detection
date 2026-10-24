# Slide outline: How Much Personal History Is Enough?

**Talk type:** oral, 15 minutes, 16 slides, 16:9, speaker notes on.
**Colours:** the poster's pair (slate blue `#2D5F8B`, gold `#C9A24A`) on white, so the talk matches the paper and poster. The skill's venue palettes don't apply: the paper targets an IEEE journal, not an ML conference.
**Figures:** the paper's four vector PDFs (`paper/figures/`). Every number comes from `paper/sections/*.tex`.

| # | Slide title | Bullets (fragments) | Figure | Time |
|:-:|---|---|:-:|:-:|
| 1 | How Much Personal History Is Enough? | Briyan George Dyju; Independent Researcher, Dubai, UAE | none | 0:15 |
| 2 | A new customer opens an account | Personal detector: compare with your own past / Day one: your past is empty / One population rule serves people badly | none | 1:15 |
| 3 | Where does the baseline start to work? | Deployed systems filter "unseasoned users" / The cutoff is assumed, not measured / We found no published measurement / **How many transactions are enough?** | none | 0:45 |
| 4 | Measure it with one fixed test window | Fixed window: last 60 transactions / Train on the n just before it / n from 1 to 250, always paired / Anomaly: a stranger's real transaction swapped in / No detector's score defines the labels | none | 1:15 |
| 5 | Three data layers, six detectors | Berka: 300 real bank accounts / Simulation: 240 people, variability set / TabFormer: 300 entities, synthetic / Mahalanobis, k-NN, forest, autoencoder, z-scores | none | 0:45 |
| 6 | 25 transactions reach 90% of best | Mahalanobis PR-AUC **0.657** / 98% of the gain by n = 50 / z-scores never learn | Fig. 1 (Berka curves) | 1:30 |
| 7 | The threshold depends on the definition | Table: Rel-90 / Rel-95 per detector / Same 25 at all **5** cohort seeds / Stability criterion never met by a learner | table | 0:45 |
| 8 | Below n = 10, trust the crowd | Population model: 400 other accounts / Personal worse until n = 10 / k-NN: until n = 50 | Fig. 2 (arms) | 1:30 |
| 9 | Shrink toward the crowd while history is thin | Equation for the hybrid mean / kappa = 25, fixed in advance / n = 0 gives exactly the global model / Never meaningfully worse than either parent / Fails if signal lives in the features | equation | 0:45 |
| 10 | How you set the alert threshold matters | Table: in-sample vs leave-one-out, n = 5 to 50 / n = 10: **96%** flagged in-sample / Leave-one-out: **1.2%** flagged / FPR 0.066 at n = 25 (nominal 0.05) | table | 1:30 |
| 11 | A second dataset reverses a ranking | Same shape, later threshold (250) / Isolation forest best at n = 5 to 50 / It is worst on Berka | Fig. 4 (TabFormer) | 1:15 |
| 12 | Variability sets how fast it learns | Regular: 50 transactions; variable: 500 / Simulation: rho 0.48 to 0.71 / Real Berka accounts: no descriptor predicts it | Fig. 3 (archetypes) | 1:00 |
| 13 | Recent history beats more history | 50 to 250 transactions: 0.647 to 0.642 / Same 50 from 4x span: **-0.036** / Drift is real but does not predict loss | none | 0:45 |
| 14 | Limits and ethics | One bank, 1993 to 1998; second dataset synthetic / 10% injection far above fraud prevalence / A flag is a deviation, not fraud / Audit is district-level only | none | 0:45 |
| 15 | Takeaways | Baseline ranks well from about 25 transactions / Hybrid removes the cold-start penalty / Calibration moves the operating point / Keep recent history | none | 0:45 |
| 16 | Questions? | Contact: briyandyju@gmail.com / Paper link: to be added | none | 0:15 |

**Time budget:** 0:15 + 1:15 + 0:45 + 1:15 + 0:45 + 1:30 + 0:45 + 1:30 + 0:45 + 1:30 + 1:15 + 1:00 + 0:45 + 0:45 + 0:45 + 0:15 = **15:00**.

**Left out on purpose** (all in the paper): the stride-sampling control, Protocol P, the anomaly-class breakdown, the district audit tables, the other six-detector calibration tables, and related work (no Related Work slide in a 15-minute talk). They are good Q&A material and will appear in the script's anticipated questions.

**Opening hook:** slide 2 starts from a customer's first day, not "In this paper, we...".

**Not yet decided (need your input):** a QR code on slide 16 (no paper URL exists yet), and a venue name for the title slide footer (none, so it will read "Independent research, 2026").
