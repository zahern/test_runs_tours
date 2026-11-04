# -*- coding: utf-8 -*-
"""Fix ABM needs slide (make it own dedicated slide), fix safety slide spacing, fix broken tag."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 0. Fix broken tag ──
html = html.replace("\n      /section>\n", "\n      </section>\n")
print("Fixed broken </section> tag")

# ── 1. Fix "What an ABM requires" - ensure it fills the slide ──
idx = html.find("What an activity-based model requires")
if idx > 0:
    s = html.rfind("<section ", 0, idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    NEW_ABM = """\
      <section class="slide dark-slide" style="padding:2.5% 3.5%;">
        <div class="eyebrow">What an activity-based model requires</div>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.3vw,1.8rem); color:#FFFFFF;">Six ingredients every ABM must bring together</h1>
        <div class="body-area" style="gap:0.8vh; margin-top:0.8vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.6vw; flex:1; min-height:0;">
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">1</div>
              <h3 style="margin-top:0.4em;">Disaggregate persons</h3>
              <p style="font-size:0.76rem;">Every individual carries their own attributes: age, income, employment status, study level, household composition, vehicle ownership, licence. Aggregate zone-level averages cannot capture the joint distribution that drives heterogeneous travel behaviour.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">2</div>
              <h3 style="margin-top:0.4em;">Spatial zone system</h3>
              <p style="font-size:0.76rem;">A consistent geography (371 SA2 zones across SEQ) with land-use data per zone: employment by industry, education places, retail, residential acreage, accessibility. Every destination choice needs an attractor.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">3</div>
              <h3 style="margin-top:0.4em;">Network skims</h3>
              <p style="font-size:0.76rem;">Origin&ndash;destination matrices for every mode: auto travel time, walk/bike distance, transit IVT. These impedance terms make every OD pair comparable. Without skims, destinations are just points on a map.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">4</div>
              <h3 style="margin-top:0.4em;">Activity generation</h3>
              <p style="font-size:0.76rem;">How many tours of each purpose? Ordered logit and count models predict work, education, social, escort, shopping, maintenance, discretionary, and eating-out frequencies from person attributes.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">5</div>
              <h3 style="margin-top:0.4em;">Scheduling &amp; timing</h3>
              <p style="font-size:0.76rem;">When do activities happen? A daily time budget constrains total out-of-home time. Log-normal models predict per-activity duration. MNL over 6 periods places each activity in the day.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">6</div>
              <h3 style="margin-top:0.4em;">Spatial choice + chaining</h3>
              <p style="font-size:0.76rem;">Where do trips go, and by which mode? Nested logit over 371 zones and 5 modes with ACDC agglomeration. Full trip chains from home to activities and back, ready for DTA.</p>
            </div>
          </div>
          <p style="font-size:0.7rem; color:#9FADE0; text-align:center;">This framework adds a seventh ingredient: <b style="color:#FFFFFF;">crash-frequency skims</b> as an explicit demand input across all decision layers.</p>
        </div>
      </section>"""
    
    html = html[:s] + NEW_ABM + html[e:]
    print(f"Fixed ABM needs slide ({e-s:,} -> {len(NEW_ABM):,} bytes)")

# ── 2. Fix safety slide - fill out properly with better spacing ──
idx = html.find("Safety shapes travel demand")
if idx > 0:
    s = html.rfind("<section ", 0, idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    # Make the reveal items take full height and the literature column take full height too
    slide = html[s:e]
    
    # Fix the grid to use all available space
    slide = slide.replace(
        'gap:0.8vw; flex:1; min-height:0;">',
        'gap:0.8vw; flex:1; min-height:0; align-items:stretch;">'
    )
    
    # Make reveal items fill more space
    slide = slide.replace(
        'font-size:0.82rem; line-height:1.6; overflow-y:auto;">',
        'font-size:0.88rem; line-height:1.65; overflow-y:auto; display:flex; flex-direction:column; gap:0.3em;">'
    )
    
    # Make literature cards fill the right column
    slide = slide.replace(
        'overflow-y:auto;font-size:0.58rem;">',
        'overflow-y:auto;font-size:0.58rem; flex:1;">'
    )
    
    html = html[:s] + slide + html[e:]
    print(f"Fixed safety slide spacing")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
