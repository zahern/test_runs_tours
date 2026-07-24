# -*- coding: utf-8 -*-
"""Extract and embed Cairns presentation images into the estimation slides."""
import zipfile, base64, io, os
from PIL import Image

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
ZIP = r"Z:\test_runs_tours\abm_safety\CairnsPresentation.zip"

# Extract and downscale images
imgs = {}
with zipfile.ZipFile(ZIP) as z:
    for fname, max_w in [("pictures/analyst.png", 800), ("pictures/gear.png", 600), 
                          ("pictures/bias_d.png", 800), ("pictures/canal.png", 400)]:
        data = z.read(fname)
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w > max_w:
            ratio = max_w / w
            img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        imgs[fname] = base64.b64encode(buf.getvalue()).decode()
        print(f"  {fname}: {len(data):,} -> {len(buf.getvalue()):,} -> {len(imgs[fname]):,} b64")

with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find the estimation slide (slide 10) and the paper slide (slide 8)
est_idx = html.find("The core problem: for any model")
paper_idx = html.find("MetaCountRegressor: a major motivation")

# ── 1. Add imagery to the estimation slide ──
if est_idx > 0:
    s = html.rfind("<section ", 0, est_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    slide = html[s:e]
    
    # Add analyst image to the left card and gear to the right card
    # Find the left card's closing </div> before the table
    left_table = slide.find("person attributes (~40 candidates)")
    if left_table > 0:
        # Find end of that table row
        row_end = slide.find("</tr>", slide.find("</tr>", left_table) + 5) + 5
        # Insert analyst image after the table
        img_tag = f'<div style="text-align:center;margin-top:0.3em;"><img src="data:image/png;base64,{imgs["pictures/analyst.png"]}" style="max-width:100%;max-height:12vh;object-fit:contain;border-radius:4px;opacity:0.9;" alt="Analyst-driven approach"></div>'
        slide = slide[:row_end] + img_tag + slide[row_end:]
    
    # Add gear image to the right card
    right_card = slide.find("Define the search space")
    if right_card > 0:
        # Find the closing </div> of the right card
        card_end = slide.find("</div>", slide.find("</div>", right_card + 100) + 6) + 6
        img_tag2 = f'<div style="text-align:center;margin-top:0.3em;"><img src="data:image/png;base64,{imgs["pictures/gear.png"]}" style="max-width:70%;max-height:8vh;object-fit:contain;border-radius:4px;opacity:0.85;" alt="Automated search"></div>'
        slide = slide[:card_end] + img_tag2 + slide[card_end:]
    
    html = html[:s] + slide + html[e:]
    print("Added Cairns images to estimation slide")

# ── 2. Add a new slide comparing analyst vs SA-driven specification ──
if paper_idx > 0:
    # Find end of paper slide
    s_p = html.rfind("<section ", 0, paper_idx)
    depth = 1; i = s_p + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    paper_end = i - 1
    
    COMPARE_SLIDE = f"""\
      <section class="slide" style="padding:2% 3.2%;">
        <h2 class="slide-kicker">Method &middot; Comparison</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.7rem);">Analyst-driven vs SA-driven specification: what changes</h1>
        <div class="body-area" style="gap:0.5vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div style="display:flex;flex-direction:column;gap:0.3em;">
              <div class="card" style="padding:0.4em 0.6em; border-left:4px solid #e74c3c;">
                <h3 style="margin:0 0 0.1em; font-size:0.8rem; color:#c0392b;">Analyst-driven specification</h3>
                <div style="font-size:0.62rem;line-height:1.5;">
                  <p style="margin:0 0 0.2em;">The analyst reviews literature, uses intuition, and tests a handful of specifications. The result is <b>whatever happened to be tested</b> &mdash; not necessarily the best model.</p>
                  <p style="margin:0;"><b>Risks:</b> omitted-variable bias (the untested variable is the one that matters), over-fitting (too many correlated terms), non-transferable forms.</p>
                </div>
              </div>
              <div style="text-align:center;"><img src="data:image/png;base64,{imgs['pictures/analysttest.jpg'] if 'pictures/analysttest.jpg' in imgs else imgs['pictures/analyst.png']}" style="max-width:95%;max-height:16vh;object-fit:contain;border-radius:6px;border:1px solid var(--border);" alt="Analyst testing"></div>
            </div>
            <div style="display:flex;flex-direction:column;gap:0.3em;">
              <div class="card" style="padding:0.4em 0.6em; border-left:4px solid var(--good);">
                <h3 style="margin:0 0 0.1em; font-size:0.8rem;">SA-driven specification</h3>
                <div style="font-size:0.62rem;line-height:1.5;">
                  <p style="margin:0 0 0.2em;">The SA metaheuristic searches <b>thousands</b> of candidate specifications, guided by BIC. The analyst only declares <b>what is forbidden</b>; the search decides <b>what is included</b>.</p>
                  <p style="margin:0;"><b>Benefits:</b> reproducible, data-driven, constraint-guided. Same objective across all stages. Specifications selected by information criteria, not analyst preference.</p>
                </div>
              </div>
              <div style="text-align:center;"><img src="data:image/png;base64,{imgs['pictures/bias_d.png']}" style="max-width:95%;max-height:16vh;object-fit:contain;border-radius:6px;border:1px solid var(--border);" alt="SA-driven approach"></div>
            </div>
          </div>
          <div class="card" style="padding:0.4em 0.7em; font-size:0.62rem; background:var(--orange-soft);">
            <b style="color:var(--orange);">What this means for the ABM:</b> Every stage's specification (CDAP, activity frequency, budget, duration, schedule, mode, destination) is selected by the <b>same SA engine</b> with the <b>same BIC objective</b>. The analyst writes ga_constraints.py once; the search evaluates 2<sup>k</sup> candidate specifications per stage. The result is a parsimonious, reproducible, data-driven ABM where every coefficient earned its place.
          </div>
        </div>
      </section>"""
    
    html = html[:paper_end] + "\n\n" + COMPARE_SLIDE + html[paper_end:]
    print("Added analyst-vs-SA comparison slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
