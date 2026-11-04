# SEQ 371 hybrid ABM+SD — poster

A0 landscape `tikzposter` on the full hybrid system: ABM pipeline, 8-mode
PT-extended stage-5 choice, unified logsum accessibility (incl. the
propagation fix), GA calibration status, and the SD feedback layer.

## Build

```bash
python make_figures.py        # regenerate figs/ (PDF + PNG) from saved data
pdflatex poster.tex           # run twice (tikz node positions)
pdflatex poster.tex
```

or `latexmk -pdf poster.tex`. Requires any TeX distribution with
`tikzposter`, `booktabs`, `amsmath` (TeX Live / MiKTeX both fine).
No internet needed. A prebuilt `poster.pdf` is committed; it was also
verified to compile with [Tectonic](https://tectonic-typesetting.github.io/)
(`tectonic -X compile poster.tex`).

## Files

- `make_figures.py` — regenerates all poster figures locally (needs only
  pandas + matplotlib):
  - `figs/fig_sd_stocks.*` — SD stocks 2025–2045 by scenario (speed factor,
    EV share, cost multiplier) + provenance CSVs (`sd_stocks.csv`,
    `sd_interfaces.csv`)
  - `figs/fig_sd_flow.png` — hybrid ABM↔SD loop with interfaces I1–I4
    (same source as `saved_data/luti/sd_flow_diagram.png`)
  - `figs/fig_ga_status.*` — GA search status from `<code>/_final_matrix.csv`
  - `figs/fig_acc_propagation.*` — behavioural vs gravity accessibility,
    371 zones (`saved_data/accessibility_371.csv`)
- `poster.tex` — the poster (TikZ diagrams for the pipeline, nests,
  accessibility chain; SD loop from `fig_sd_flow.png`).

## Notes

- Figures are read-only snapshots; re-run `make_figures.py` after data
  updates (e.g. the PT refit merge) and rebuild the PDF.
- The GA-status panel intentionally shows the pre-PT runs; refresh after
  the fix cycle (`runs/ga_parallel/*/merged`) completes.
