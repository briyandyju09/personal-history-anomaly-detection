"""
Stitch web/index_template.html + web/data.json into web/index.html.

Run web.build_data first to (re)produce data.json from the current
results/ files, then this script to produce the page that actually gets
served/opened. Two steps, not one, so data.json stays inspectable on its
own (e.g. to spot-check a number) without re-parsing the whole page.

Usage:
    python -m web.build_data
    python -m web.build_site
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "index_template.html")
DATA = os.path.join(HERE, "data.json")
OUT = os.path.join(HERE, "index.html")


def main():
    with open(TEMPLATE, encoding="utf-8") as fh:
        template = fh.read()
    with open(DATA, encoding="utf-8") as fh:
        data_json = fh.read()

    # Defensive: a "</script>" substring anywhere in the JSON text would
    # prematurely close the <script type="application/json"> block under
    # HTML's parsing rules (which do not know about JSON syntax). None of
    # our data should ever contain this, but guard against it rather than
    # silently producing a broken page if it ever does.
    data_json_safe = data_json.replace("</script>", "<\\/script>")

    if "__DATA_JSON__" not in template:
        raise RuntimeError("index_template.html is missing the __DATA_JSON__ placeholder")

    out = template.replace("__DATA_JSON__", data_json_safe)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(out)

    print(f"wrote {OUT}, {os.path.getsize(OUT):,} bytes")


if __name__ == "__main__":
    main()
