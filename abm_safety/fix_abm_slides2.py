# -*- coding: utf-8 -*-
"""Fix slide structure - re-insert ABM needs slides."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find the gap between title slide end and "Why This Matters"
title_end = html.find("</section>", 10418) + len("</section>")
wm = html.find("<!-- 2. WHY THIS MATTERS -->")

# The ABM needs slides should go between title and why-matters
ABM_DETAIL = """\

      <!-- 1b. WHAT AN ABM NEEDS - DETAIL -->
      <section class="slide dark-slide" style="padding:3% 4%; display:flex; align-items:center; justify-content:center;">
        <div style="max-width:80ch;">
          <div class="eyebrow" style="font-size:0.9rem;">What an activity-based model requires</div>
          <h1 class="slide-title" style="font-size:clamp(1.8rem,3.5vw,2.8rem); color:#FFFFFF; line-height:1.2; margin-bottom:0.7em;">Six ingredients every ABM must bring together</h1>
          <div style="display:flex;flex-direction:column;gap:0.55em;">
            <div style="display:flex;align-items:baseline;gap:0.6em;"><span style="font-family:var(--font-mono);font-size:0.9rem;color:#9FD9F0;min-width:2em;">1.</span><span style="font-size:1.05rem;color:#E4E9FA;line-height:1.45;"><b style="color:#FFFFFF;">Disaggregate persons</b> &mdash; every individual carries their own age, income, employment, study level, household composition, vehicle ownership, and licence. Aggregated zone averages cannot capture the joint distribution that drives heterogeneous travel.</span></div>
            <div style="display:flex;align-items:baseline;gap:0.6em;"><span style="font-family:var(--font-mono);font-size:0.9rem;color:#9FD9F0;min-width:2em;">2.</span><span style="font-size:1.05rem;color:#E4E9FA;line-height:1.45;"><b style="color:#FFFFFF;">Spatial zone system</b> &mdash; a consistent geography (371 SA2 zones across SEQ) with land-use: employment by industry, education places, retail floorspace, residential acreage, and accessibility measures.</span></div>
            <div style="display:flex;align-items:baseline;gap:0.6em;"><span style="font-family:var(--font-mono);font-size:0.9rem;color:#9FD9F0;min-width:2em;">3.</span><span style="font-size:1.05rem;color:#E4E9FA;line-height:1.45;"><b style="color:#FFFFFF;">Network skims</b> &mdash; origin&ndash;destination matrices for every mode: auto travel time, walk distance, bike distance, transit IVT. These impedance terms make every OD pair comparable.</span></div>
            <div style="display:flex;align-items:baseline;gap:0.6em;"><span style="font-family:var(--font-mono);font-size:0.9rem;color:#9FD9F0;min-width:2em;">4.</span><span style="font-size:1.05rem;color:#E4E9FA;line-height:1.45;"><b style="color:#FFFFFF;">Activity generation</b> &mdash; how many tours of each purpose? Ordered logit and count models predict work, education, social, escort, shopping, maintenance, discretionary, and eating-out frequencies from person attributes.</span></div>
            <div style="display:flex;align-items:baseline;gap:0.6em;"><span style="font-family:var(--font-mono);font-size:0.9rem;color:#9FD9F0;min-width:2em;">5.</span><span style="font-size:1.05rem;color:#E4E9FA;line-height:1.45;"><b style="color:#FFFFFF;">Scheduling &amp; timing</b> &mdash; a daily time budget constrains total out-of-home hours. Log-normal models predict per-activity duration. MNL over 6 time periods places each activity in the day.</span></div>
            <div style="display:flex;align-items:baseline;gap:0.6em;"><span style="font-family:var(--font-mono);font-size:0.9rem;color:#9FD9F0;min-width:2em;">6.</span><span style="font-size:1.05rem;color:#E4E9FA;line-height:1.45;"><b style="color:#FFFFFF;">Spatial choice + trip chaining</b> &mdash; nested logit over 371 zones and 5 modes with ACDC agglomeration and crash-frequency skims. Full trip chains from home to activities and back, ready for DTA.</span></div>
          </div>
          <p style="font-family:var(--font-mono);font-size:0.82rem;color:#9FD9F0;margin-top:1em;text-align:center;">This framework adds a seventh: <b style="color:#FFFFFF;">crash-frequency skims</b> as an explicit demand input across all decision layers</p>
        </div>
      </section>

      <!-- 1c. WHAT AN ABM NEEDS - CARDS -->
      <section class="slide dark-slide" style="padding:2.2% 3.5%;">
        <div class="eyebrow">What an activity-based model requires</div>
        <h1 class="slide-title" style="font-size:clamp(1.4rem,2.5vw,1.9rem); color:#FFFFFF;">Six ingredients every ABM must bring together</h1>
        <div class="body-area" style="gap:0.6vh; margin-top:0.6vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.5vw; flex:1; min-height:0; align-content:stretch;">
            <div class="card on-primary" style="padding:0.6em 0.8em; display:flex; flex-direction:column;">
              <div class="num-badge">1</div>
              <h3 style="margin-top:0.35em; font-size:0.95rem;">Disaggregate persons</h3>
              <p style="font-size:0.82rem; line-height:1.5; flex:1;">Every individual carries their own attributes: age, income, employment status, study level, household composition, vehicle ownership, licence. Aggregate zone-level averages cannot capture the joint distribution that drives heterogeneous travel behaviour.</p>
            </div>
            <div class="card on-primary" style="padding:0.6em 0.8em; display:flex; flex-direction:column;">
              <div class="num-badge">2</div>
              <h3 style="margin-top:0.35em; font-size:0.95rem;">Spatial zone system</h3>
              <p style="font-size:0.82rem; line-height:1.5; flex:1;">A consistent geography (371 SA2 zones across SEQ) with land-use data per zone: employment by industry, education places, retail floorspace, residential acreage, accessibility measures. Every destination choice needs an attractor.</p>
            </div>
            <div class="card on-primary" style="padding:0.6em 0.8em; display:flex; flex-direction:column;">
              <div class="num-badge">3</div>
              <h3 style="margin-top:0.35em; font-size:0.95rem;">Network skims</h3>
              <p style="font-size:0.82rem; line-height:1.5; flex:1;">Origin&ndash;destination matrices for every mode: auto travel time, walk distance, bike distance, transit in-vehicle time. These are the impedance terms that make every OD pair comparable. Without skims, destinations are just points on a map.</p>
            </div>
            <div class="card on-primary" style="padding:0.6em 0.8em; display:flex; flex-direction:column;">
              <div class="num-badge">4</div>
              <h3 style="margin-top:0.35em; font-size:0.95rem;">Activity generation</h3>
              <p style="font-size:0.82rem; line-height:1.5; flex:1;">How many tours of each purpose does each person make? Ordered logit and count models predict work, education, social, escort, shopping, maintenance, discretionary, and eating-out frequencies from person attributes and accessibility.</p>
            </div>
            <div class="card on-primary" style="padding:0.6em 0.8em; display:flex; flex-direction:column;">
              <div class="num-badge">5</div>
              <h3 style="margin-top:0.35em; font-size:0.95rem;">Scheduling &amp; timing</h3>
              <p style="font-size:0.82rem; line-height:1.5; flex:1;">When do activities happen? A daily time budget constrains total out-of-home time. Log-normal models predict per-activity duration. An MNL over 6 time periods (early, AM peak, midday, PM peak, evening, late) places each activity in the day, respecting the budget.</p>
            </div>
            <div class="card on-primary" style="padding:0.6em 0.8em; display:flex; flex-direction:column;">
              <div class="num-badge">6</div>
              <h3 style="margin-top:0.35em; font-size:0.95rem;">Spatial choice + trip chaining</h3>
              <p style="font-size:0.82rem; line-height:1.5; flex:1;">Where do trips go, and by which mode? Nested logit over 371 SA2 zones and 5 modes with ACDC agglomeration and competition. Full trip chains from home to activities and back, ready for dynamic traffic assignment.</p>
            </div>
          </div>
          <p style="font-size:0.72rem; color:#9FADE0; text-align:center;">This framework adds a seventh ingredient: <b style="color:#FFFFFF;">crash-frequency skims</b> as an explicit demand input across all decision layers &mdash; safety enters the model the same way travel time does.</p>
        </div>
      </section>

"""

# Insert between title end and why matters
html = html[:title_end] + ABM_DETAIL + html[title_end:]

# Also fix the broken </section></section> between slide 6 and slide 8
html = html.replace("</section></section>\n\n      <!-- 4. POPULATION + ZONES -->",
                     "</section>\n\n      <!-- 4. POPULATION + ZONES -->")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Re-inserted ABM needs slides. Wrote {len(html):,} bytes.")
