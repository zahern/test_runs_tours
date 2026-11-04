# -*- coding: utf-8 -*-
"""Embed ECF process PNGs into the crash-frequency slide."""
import base64, os, re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
FIGS = r"Z:\test_runs_tours\abm_safety\Figures"

# Encode the two workflow diagrams
images = {}
for fname, label in [("Article_Diagram_1.png", "ECF workflow"), ("Article_Diagram_2.png", "Segment-to-OD transfer")]:
    fp = os.path.join(FIGS, fname)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            images[label] = base64.b64encode(f.read()).decode()
        print(f"  {fname}: {os.path.getsize(fp):,} -> {len(images[label]):,} b64 chars")

with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find the crash slide SVG diagram section and replace with PNGs + brief SVG
svg_start = html.find('<svg viewBox="0 0 820 120"')
if svg_start < 0:
    print("SVG not found")
else:
    svg_end = html.find("</svg>", svg_start) + len("</svg>")

    IMAGE_BLOCK = f"""\
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw;">
            <figure class="evidence" style="margin:0;">
              <img src="data:image/png;base64,{images['ECF workflow']}" style="max-width:100%;max-height:26vh;object-fit:contain;border:1px solid var(--border);border-radius:6px;background:white;" alt="Expected crash frequency estimation workflow">
              <figcaption style="font-size:0.6rem;">Five-step ECF workflow: road segments &rarr; stereotype &rarr; SA-fitted SPFs &rarr; Empirical Bayes &rarr; expected crashes per segment</figcaption>
            </figure>
            <figure class="evidence" style="margin:0;">
              <img src="data:image/png;base64,{images['Segment-to-OD transfer']}" style="max-width:100%;max-height:26vh;object-fit:contain;border:1px solid var(--border);border-radius:6px;background:white;" alt="Segment crash-to-OD matrix transfer">
              <figcaption style="font-size:0.6rem;">Segment-level expected crashes accumulated along inter-zonal shortest paths into the origin&ndash;destination CRASH matrix (371&times;371, stored in OMX)</figcaption>
            </figure>
          </div>
          <div style="font-size:0.58rem; color:var(--text-muted); text-align:center;">From the paper (Appendices A1&ndash;A2): Article Diagrams 1 and 2. The fitted SPFs produce segment-level expected crashes; shortest-path accumulation converts them into the OD skim that Stage 5 reads.</div>"""

    html = html[:svg_start] + IMAGE_BLOCK + html[svg_end:]
    print(f"  Replaced SVG with ECF PNGs")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
