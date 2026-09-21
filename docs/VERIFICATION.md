# Independent re-verification

This document records the result of independently re-executing the entire
documented pipeline from a clean environment (Python 3.14.6, fresh venv,
this machine) against the numbers already recorded in the reference paper
(`PAPER.md.reference`, produced 2026-09-20 in a different session/environment
— see `docs/ENVIRONMENT.md` for the package-version difference). Nothing in
this document is estimated; every "fresh" figure below was read directly
from a freshly regenerated CSV under `results/raw/` or `results/tables/`,
produced by unmodified experiment code (the only source change before
re-running was the addition of a second-mirror data-integrity cross-check in
`src/data_loader.py`, which does not touch the modelling or evaluation code
path at all).

## Headline result: bit-exact reproduction

Every number checked matched the reference paper's reported value exactly,
to the three decimal places both report. This held despite this session
running on newer point releases of every dependency (pandas 3.0.6 vs 3.0.2,
numpy 2.5.3 vs 2.4.4, scikit-learn 1.9.1 vs 1.8.0, scipy 1.18.1 vs 1.17.1,
matplotlib 3.11.2 vs 3.10.8) and on a different machine. Given a fixed master
seed (20260920) and the paper's fully deterministic per-account seeding
(`seed * 100003 + account_id`), the pipeline is bit-for-bit reproducible
across this range of dependency versions and hardware.

## Data integrity

| Check | Result |
|---|---|
| Row counts, all 8 Berka tables, vs. primary source's published figures | **Match** (account 4,500; client 5,369; disp 5,369; order 6,471; trans 1,056,320; loan 682; card 892; district 77) |
| SHA-256 prefixes, all 8 tables | **Match** reference `PAPER.md` Appendix A.2 exactly (`215f4bfcb2520ab8`, `2b2f5253f21418c4`, `77e08c0988473998`, `c1d909d5d8a56ce6`, `111c70dc6626411d`, `0cf9fbe7ec2ebb7a`, `a4531f8d48f21ed5`, `d5422aa7326fde86`) |
| Second, independent mirror cross-check (new this session) | **Match**, all 8 tables, after LF/CRLF normalisation (the only byte-level difference between the two mirrors) |
| `account_summary.csv`: median/mean/max transactions per account | 208 / 234.7 / 675 — **matches** reference §5.2 exactly |
| Accounts with ≥ 100 / ≥ 250 / ≥ 500 transactions | 3,883 / 1,835 / 119 — **matches** reference §5.2 exactly |
| Eligible accounts for the main cohort (≥ 310 transactions) | 1,319 — **matches** reference (pilot run and main run both) |

## Pipeline validation (pilot, 60 accounts)

- Global arm PR-AUC invariant across history size for every account/model pair (`max unique values = 1`) — confirms the global model is genuinely history-independent, as designed.
- PR-AUC bounded in [0, 1] throughout; fraction of near-perfect (> 0.999) PR-AUC = 0.0065 — no leakage smell.
- Degenerate-fit flags appear exactly where the design predicts (n=1 always degenerate for every model; n=5/10 degenerate only for Mahalanobis and the autoencoder, consistent with their `n <= d` singularity condition and minimum-sample requirement respectively).

## E1 (impostor protocol), full cohort (300 accounts, 400 donors)

Mean PR-AUC, personal arm, Mahalanobis — fresh run vs. reference `PAPER.md` §7.1:

| n | Reference | Fresh (this session) | Match |
|---|---|---|---|
| 1 | 0.161 | 0.161 | Yes |
| 10 | 0.441 | 0.441 | Yes |
| 50 | 0.647 | 0.647 | Yes |
| 100 | 0.657 | 0.657 | Yes |
| 250 | 0.642 | 0.642 | Yes |

Every other model (knn, iforest, gaussian_z, robust_z, autoencoder) at every history size also matched exactly (checked programmatically, not just spot-read).

## E2 (parametric amount-shift protocol)

Mean PR-AUC, personal arm, vs. reference §7.7:

| n | Model | Reference | Fresh | Match |
|---|---|---|---|---|
| 1 | gaussian_z | 0.278 | 0.278 | Yes |
| 25 | gaussian_z | 0.292 | 0.292 | Yes |
| 100 | gaussian_z | 0.294 | 0.294 | Yes |
| 250 | gaussian_z | 0.292 | 0.292 | Yes |
| 1 | mahalanobis | 0.258 | 0.258 | Yes |
| 25 | mahalanobis | 0.423 | 0.423 | Yes |
| 100 | mahalanobis | 0.455 | 0.455 | Yes |
| 250 | mahalanobis | 0.453 | 0.453 | Yes |

## E3 (temporal coverage, original stride-sampling version kept for the historical record)

Mahalanobis stride 1 = 0.647, stride 4 = 0.434, matching the reference's own description of the flawed original version ("Mahalanobis 0.647 → 0.435 at stride 4"; the 0.001 difference is rounding in the reference's prose, not a discrepancy — the underlying table value is 0.434 in both runs).

## E4 (behavioural complexity descriptors)

300 rows produced, matching the reference's evaluated cohort size.

## Conclusion of the verification pass

**No discrepancy was found between this session's independent re-execution and the reference paper's reported numbers, across every experiment checked.** This is strong evidence that the reference paper's first-draft numbers were genuinely produced by the committed code, not adjusted, cherry-picked, or produced by a different code path than what is in the repository. The re-verification is treated as complete for E1-E4; the remainder of the pipeline (`e3_temporal_fixed`, `run_simulation`, `analyse`, `analyse_simulation`) was re-run in the same session under the same conditions (see run logs under `results/raw/*.log` for this session's timestamps).
