# Data

This directory holds raw data. **No raw records are redistributed in this repository.**

## Acquiring the data

    python -m src.data_loader

This downloads the eight PKDD'99 Financial ("Berka") tables and verifies each
one against the record counts published on the primary source page. If any
table does not match, the loader raises `DataIntegrityError` and refuses to
proceed.

## Source

- Primary description: https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm
- Prepared by Petr Berka and Marta Sochorova for the PKDD'99 Discovery Challenge.
- Real, anonymised Czech bank transactions, 1993-01-01 to 1998-12-31.
- 1,056,320 transactions, 4,500 accounts.

The primary host was not resolvable from our execution environment, so the
loader retrieves a GitHub mirror and verifies it. If the primary host is
reachable for you, prefer it and adjust `MIRROR_BASE` in `src/data_loader.py`.
The loader also cross-checks every table's content, byte-for-byte after
line-ending normalisation, against a second independent mirror
(`jlacko/berka-dataset`); a mismatch on either check raises and refuses to
proceed.

## Licence

Released for an open research challenge; the primary page carries no
machine-readable licence. Treat as research-use data. Do not redistribute raw
records. See docs/DECISION_LOG.md section 2.

---

## TabFormer (Experiment E7, SYNTHETIC — a separate layer)

    python -m experiments.run_tabformer   # downloads on first run if data/raw_tabformer/ is empty

Not downloaded by `src.data_loader` — it belongs to a different experiment
(E7, §5.4/§7.10 of `PAPER.md`) and is fetched by `src/tabformer_loader.py`
into `data/raw_tabformer/`, also not redistributed here.

- Source: <https://github.com/IBM/TabFormer> (Apache-2.0), file
  `data/credit_card/transactions.tgz`, fetched via the repository's Git-LFS
  media endpoint. No authentication required.
- **This dataset is synthetic** — generator-produced, not real financial
  behaviour, including its `Is Fraud?` column. Never presented as a
  real-world observation anywhere in this project.
- 24,386,900 transactions, 2,000 users, independently counted by a full
  chunked pass over the downloaded file (this session).
- Single-source verification only — no second independent mirror was
  fetched for this dataset (the README also lists an IBM Box mirror of the
  same archive, not separately downloaded here). Weaker provenance than the
  Berka dual-mirror check above; stated as a limitation in `PAPER.md` §5.4.
