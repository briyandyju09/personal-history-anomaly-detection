# Slides review

**Status: REVIEW_UNAVAILABLE.** The skill's Phase 5 sends the outline and frames to an external reviewer (Codex MCP). None is configured here (no `.aris/reviewer.json`, no Codex connection), so no independent review happened. The table below is the author-side check against the same seven criteria. It is not a substitute.

| Criterion | Self-check |
|---|---|
| Story arc | Hook (empty history) to gap (unmeasured cutoff) to protocol to result to crossover to fix to calibration to transfer to limits to takeaway. One idea per slide. |
| Slide density | Enforced in the generator: at most 6 lines per slide and 8 words per line (assertion fails the build). |
| Time budget | 16 slides, 900 s exactly (15:00). Slides 6, 8 and 10 (the core results) get 90 s each. Not rehearsed aloud; a dry run is the real test. |
| Figure visibility | Slide figures are regenerated from the same result tables at larger type (axis text about 10 pt in a 4.7 in figure) so they survive projection. Figures take roughly 35 to 40% of the slide area, below the skill's 60% target: the 16:9 frame is wide and the plots are 1.47:1, so a 60% figure would leave no room for annotations. |
| Opening hook | Slide 2 opens on a new customer's first day. |
| Takeaway | Four fragments on slide 15; slide 16 is the QR to briyan.xyz/research (checked: HTTP 200). |
| Progressive build | Slides 4, 9 and 10 reveal bullets one at a time in the PDF (25 PDF pages for 16 slides). The PPTX shows full slides; add animations in PowerPoint if wanted. |

Known gaps:
- The PPTX could not be rendered here (no PowerPoint or LibreOffice). Geometry (no shape outside the slide) and an estimated text-fit check pass, but open it once and look.
- Speaker notes are in the Beamer source (`\note{}`) and in the PPTX notes pane, not typeset in the PDF.
