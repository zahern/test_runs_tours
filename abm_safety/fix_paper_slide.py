# -*- coding: utf-8 -*-
"""Embed paper first page, split ABM needs into two slides."""
import base64, os

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
PAPER_PNG = r"Z:\test_runs_tours\abm_safety\paper_first_page.png"

with open(PAPER_PNG, "rb") as f:
    paper_b64 = base64.b64encode(f.read()).decode()
print(f"Paper PNG: {os.path.getsize(PAPER_PNG):,} -> {len(paper_b64):,} b64")

with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Add paper first page image to the MetaCount paper slide ──
paper_idx = html.find("MetaCountRegressor: a major motivation")
if paper_idx > 0:
    s = html.rfind("<section ", 0, paper_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    slide = html[s:e]
    
    # Insert the paper image after the left card
    img_html = f'''<div style="flex:1; min-height:0; display:flex; align-items:center; justify-content:center; margin-top:0.5em;">
              <img src="data:image/png;base64,{paper_b64}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;box-shadow:0 4px 20px rgba(0,0,0,0.3);" alt="MetaCountRegressor paper - first page">
            </div>'''
    
    # Insert after the left card closing </div>
    left_card_end = slide.find('Key contributions:')
    if left_card_end > 0:
        # Find the closing of that div
        end_div = slide.find('</div>', slide.find('</div>', left_card_end) + 6)
        if end_div > 0:
            end_div += 6
            slide = slide[:end_div] + img_html + slide[end_div:]
    
    html = html[:s] + slide + html[e:]
    print(f"Added paper first page to MetaCount slide")

# ── 2. Split ABM needs into title slide + 6 ingredients slide ──
abm_idx = html.find("Six ingredients every ABM must bring together")
if abm_idx > 0:
    s = html.rfind("<section ", 0, abm_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    slide = html[s:e]
    
    # Split: first slide = title only, second = 6 cards
    TITLE_SLIDE = """\
      <section class="slide dark-slide" style="padding:0; display:flex; align-items:center; justify-content:center;">
        <div style="text-align:center; max-width:70ch;">
          <div class="eyebrow">What an activity-based model requires</div>
          <h1 class="slide-title" style="font-size:clamp(2rem,4vw,3.2rem); color:#FFFFFF; line-height:1.15;">Six ingredients every ABM must bring together</h1>
          <p style="font-size:1rem; color:#C7D0F5; margin-top:1.5em;">Disaggregate persons &middot; spatial zone system &middot; network skims<br>activity generation &middot; scheduling &amp; timing &middot; spatial choice + trip chaining</p>
          <p style="font-family:var(--font-mono); font-size:0.72rem; color:#9FADE0; margin-top:2em;">This framework adds a seventh: crash-frequency skims as a demand input</p>
        </div>
      </section>"""
    
    # The 6-cards slide stays as the original
    CARDS_SLIDE = slide  # keep the existing dark slide with 6 cards
    
    html = html[:s] + TITLE_SLIDE + CARDS_SLIDE + html[e:]
    print(f"Split ABM needs into title + cards")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
