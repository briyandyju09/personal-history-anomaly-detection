# Environment actually used for this reproduction run

This supersedes Appendix A.1 of the reference `PAPER.md.reference`, which
recorded a different (unverifiable, prior-session) environment. Everything
below is what this machine actually has installed, checked directly.

- OS: Windows 11 Home 10.0.26100
- Python: 3.14.6 (`tags/v3.14.6:c63aec6, Jun 10 2026`), from `C:\Python314`
- Virtual environment: `research/.venv` (created fresh for this run)

Installed package versions (`pip install` against `requirements.txt`
constraints, resolved by pip, no version pinning forced):

| Package | Version |
|---|---|
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |

No GPU. No PyTorch (the autoencoder in `src/models.py` is pure NumPy, so no
deep learning framework is required).

**Note on the prior run's recorded environment.** The reference paper
(`PAPER.md.reference`, extracted from `files.zip`) states pandas 3.0.2, numpy
2.4.4, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8 — all older point
releases of the same major/minor lines used here. We did not have access to
that exact environment and did not attempt to reproduce it bit-for-bit; we
installed the same `requirements.txt` lower-bound constraints fresh and
report what pip actually resolved. Where re-verification differs from the
reference paper's numbers, `docs/VERIFICATION.md` records whether the
package-version difference is a plausible explanation.
