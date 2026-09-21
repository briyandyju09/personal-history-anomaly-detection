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

## Licence

Released for an open research challenge; the primary page carries no
machine-readable licence. Treat as research-use data. Do not redistribute raw
records. See docs/DECISION_LOG.md section 2.
