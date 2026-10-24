"""
Minimal Markdown -> HTML converter, purpose-built for PAPER.md's syntax only
(headers, bold/italic, inline code, links, pipe tables, ordered/unordered
lists, horizontal rules, paragraphs). Not a general CommonMark implementation
-- it does not need to be, since the one input file it ever converts is
controlled by this repository.

Used by web/build_data.py to embed the full paper in the interactive site
(interactive.txt's "Paper" section) without adding a markdown dependency to
requirements.txt, which lists only the scientific-computing stack the
experiments themselves need.
"""

from __future__ import annotations

import html
import re


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    # autolinks <https://...> (escaped to &lt;...&gt; above)
    text = re.sub(r"&lt;(https?://[^\s&]+)&gt;", r'<a href="\1">\1</a>', text)
    # links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # bold **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # italic *text* (after bold, avoid eating ** pairs)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    # inline code `code`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _parse_table(lines: list[str]) -> str:
    rows = [l.strip().strip("|").split("|") for l in lines if l.strip()]
    rows = [[c.strip() for c in r] for r in rows]
    header, sep, *body = rows
    out = ["<div class='tblwrap'><table><thead><tr>"]
    for c in header:
        out.append(f"<th>{_inline(c)}</th>")
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(md: str) -> str:
    lines = md.split("\n")
    out = []
    i = 0
    in_ul = in_ol = False
    para: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "---":
            flush_para()
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith("```"):
            flush_para()
            close_lists()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            close_lists()
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|"):
            flush_para()
            close_lists()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(_parse_table(table_lines))
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            flush_para()
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(m.group(1))}</li>")
            i += 1
            continue

        m = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m:
            flush_para()
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_inline(m.group(1))}</li>")
            i += 1
            continue

        if stripped == "":
            flush_para()
            close_lists()
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    close_lists()
    return "\n".join(out)
