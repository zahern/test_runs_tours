# -*- coding: utf-8 -*-
"""Add ABM needs slide and crash workflow slide with link-to-OD explanation."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Add "What an ABM needs" slide right after the title slide ──
# Find slide 1 end (dark-slide title)
title_comment = html.find("<!-- 1. TITLE -->")
if title_comment >= 0:
    # Find the first </section> after title
    depth = 1; i = html.find("<section ", title_comment) + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    title_end = i - 1

    WHAT_ABM_NEEDS = """\
      <section class="slide dark-slide" style="padding:3% 4.5%;">
        <div class="eyebrow">What an activity-based model requires</div>
        <h1 class="slide-title" style="font-size:clamp(1.4rem,2.5vw,2rem); color:#FFFFFF;">Six ingredients every ABM must bring together</h1>
        <div class="body-area" style="gap:1.2vh; margin-top:1vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.8vw;">
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">1</div>
              <h3 style="margin-top:0.4em;">Disaggregate persons</h3>
              <p>Every individual carries their own attributes: age, income, employment status, study level, household composition, vehicle ownership, licence. Aggregate zone-level averages cannot capture the joint distribution that drives heterogeneous travel behaviour.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">2</div>
              <h3 style="margin-top:0.4em;">Spatial zone system</h3>
              <p>A consistent geography (here: 371 SA2 zones across SEQ) with land-use data per zone: employment by industry, education places, retail floorspace, residential acreage, accessibility measures. Every destination choice needs an attractor.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">3</div>
              <h3 style="margin-top:0.4em;">Network skims</h3>
              <p>Origin&ndash;destination matrices for every mode: auto travel time, walk distance, bike distance, transit in-vehicle time. These are the impedance terms that make every OD pair comparable. Without skims, destinations are just points on a map.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">4</div>
              <h3 style="margin-top:0.4em;">Activity generation</h3>
              <p>How many tours of each purpose does each person make? Ordered logit and count models predict work, education, social, escort, shopping, maintenance, discretionary, and eating-out frequencies from person attributes.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">5</div>
              <h3 style="margin-top:0.4em;">Scheduling &amp; timing</h3>
              <p>When do activities happen? A daily time budget constrains total out-of-home duration. Log-normal models predict per-activity duration. An MNL over 6 time periods places each activity in the day, respecting the budget.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <div class="num-badge">6</div>
              <h3 style="margin-top:0.4em;">Spatial choice + chaining</h3>
              <p>Where do trips go, and by which mode? Nested logit over 371 zones and 5 modes with ACDC agglomeration. The output: full trip chains from home to activities and back, ready for dynamic traffic assignment.</p>
            </div>
          </div>
          <p style="font-family:var(--font-mono); font-size:0.64rem; color:#9FADE0; text-align:center; margin-top:0.5em;">This framework adds a seventh ingredient: <b style="color:#FFFFFF;">crash-frequency skims</b> as an explicit demand input across all decision layers.</p>
        </div>
      </section>"""

    html = html[:title_end] + WHAT_ABM_NEEDS + html[title_end:]
    print(f"  Added 'What an ABM needs' slide ({len(WHAT_ABM_NEEDS):,} bytes)")

# ── 2. Enhance the crash workflow slide (5b) with link-to-OD explanation ──
crash_comment = html.find("<!-- 5b. CRASH FREQUENCY LAYER -->")
if crash_comment >= 0:
    crash_sec = html.find("<section ", crash_comment)
    depth = 1; i = crash_sec + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    crash_end = i - 1

    CRASH_SLIDE = """\
      <section class="slide" style="padding:2.5% 3.5%;">
        <h2 class="slide-kicker">Safety Input &middot; Crash-Frequency Layer</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem);">From road segments to OD crash risk: how safety enters the model</h1>
        <div class="body-area" style="gap:1vh;">
          <div style="font-size:0.74rem; line-height:1.55; max-width:90ch; color:var(--text);">
            <b>Bottom line:</b> Safety is not computed after the model runs. Expected crash frequency is produced at the <b>100&thinsp;m road-segment level</b>, then <b>accumulated along inter-zonal shortest paths</b> into an origin&ndash;destination matrix (CRASH). This matrix sits in the same OMX file as travel times. Every tour's mode and destination utility reads from it. The SA search decides whether the coefficient is large or small &mdash; the analyst only decides that it must be present.
          </div>

          <div class="pipeline-row" style="font-size:0.66rem;">
            <div class="pipe-box" style="padding:0.5em;"><b>1. SEGMENT DATA</b>QLD crash records + TMR road assets &rarr; 100&thinsp;m homogeneous segments. Each segment: AADT, width, speed, curvature, lane count, road stereotype (6 types).</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box" style="padding:0.5em;"><b>2. FIT SPFs</b>Per stereotype &times; crash type (6&times;5 = 30 cells). SA search over count families, transforms, mixing distributions. Best-BIC GCF per cell.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box hot" style="padding:0.5em;"><b>3. PREDICT</b>Apply fitted GCFs to 2020&ndash;23 road conditions. Expected crashes per segment. Empirical-Bayes: blend model &harr; observed via overdispersion weight.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box" style="padding:0.5em;"><b>4. ACCUMULATE</b>Expected crashes per segment summed along inter-zonal shortest paths (all-or-nothing assignment). Produces one OD matrix: CRASH.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box" style="padding:0.5em;"><b>5. DEMAND INPUT</b>CRASH skim stored alongside travel times in OMX. Stage 5 utilities read from it. SA search decides coefficient sign, magnitude, and interactions.</div>
          </div>

          <div style="display:flex; gap:0.8vw; align-items:center; justify-content:center; padding:0.5em 0;">
            <svg viewBox="0 0 820 120" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="arr2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5 Z" fill="#1C7293"/></marker></defs>
              <!-- Link level -->
              <rect x="10" y="10" width="160" height="40" rx="5" fill="#1E2761" opacity="0.9"/>
              <text x="90" y="28" fill="white" font-size="8" font-weight="bold" text-anchor="middle">Road segment i</text>
              <text x="90" y="42" fill="#C9D3F5" font-size="6" text-anchor="middle">AADT, width, speed, curvature</text>

              <rect x="10" y="58" width="160" height="40" rx="5" fill="#C85A14" opacity="0.85"/>
              <text x="90" y="76" fill="white" font-size="8" font-weight="bold" text-anchor="middle">E[crash<sub>i</sub>] = SPF(x<sub>i</sub>)</text>
              <text x="90" y="90" fill="#FFE3CE" font-size="6" text-anchor="middle">Fitted GCF per stereotype &times; type</text>

              <line x1="170" y1="78" x2="195" y2="78" stroke="#1C7293" stroke-width="1.2" marker-end="url(#arr2)"/>

              <!-- Route level -->
              <rect x="200" y="20" width="200" height="32" rx="5" fill="#2E7D6B" opacity="0.85"/>
              <text x="300" y="32" fill="white" font-size="8" font-weight="bold" text-anchor="middle">Shortest path from zone A to zone B</text>
              <text x="300" y="46" fill="#D5F5E3" font-size="6" text-anchor="middle">Sequence of links: i<sub>1</sub>, i<sub>2</sub>, ..., i<sub>n</sub></text>

              <rect x="200" y="58" width="200" height="40" rx="5" fill="#1E2761" opacity="0.9"/>
              <text x="300" y="74" fill="white" font-size="8" font-weight="bold" text-anchor="middle">CRASH<sub>AB</sub> = &Sigma;<sub>k</sub> E[crash<sub>ik</sub>]</text>
              <text x="300" y="90" fill="#C9D3F5" font-size="6" text-anchor="middle">Sum of expected crashes along path</text>

              <line x1="400" y1="78" x2="425" y2="78" stroke="#1C7293" stroke-width="1.2" marker-end="url(#arr2)"/>

              <!-- OD Matrix -->
              <rect x="430" y="20" width="180" height="32" rx="5" fill="#1E2761" opacity="0.9"/>
              <text x="520" y="34" fill="white" font-size="8" font-weight="bold" text-anchor="middle">OD Crash Matrix (371 &times; 371)</text>
              <text x="520" y="47" fill="#C9D3F5" font-size="6" text-anchor="middle">Stored as CRASH skim in OMX</text>

              <rect x="430" y="58" width="180" height="40" rx="5" fill="#C85A14" opacity="0.85"/>
              <text x="520" y="74" fill="white" font-size="8" font-weight="bold" text-anchor="middle">V<sub>mode,j</sub> = ... + &beta;<sub>crash</sub> &middot; CRASH<sub>ij</sub></text>
              <text x="520" y="90" fill="#FFE3CE" font-size="6" text-anchor="middle">Enters mode &amp; destination utility</text>

              <line x1="610" y1="78" x2="635" y2="78" stroke="#1C7293" stroke-width="1.2" marker-end="url(#arr2)"/>

              <!-- Demand output -->
              <rect x="640" y="20" width="170" height="78" rx="6" fill="url(#bg2)"/>
              <text x="725" y="42" fill="white" font-size="8" font-weight="bold" text-anchor="middle">Travel demand shifts</text>
              <text x="725" y="58" fill="#D5F5E3" font-size="6" text-anchor="middle">Mode shares, destination</text>
              <text x="725" y="72" fill="#D5F5E3" font-size="6" text-anchor="middle">patterns, trip chains</text>
              <text x="725" y="86" fill="#D5F5E3" font-size="6" text-anchor="middle">respond to crash risk</text>

              <defs><linearGradient id="bg2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1E2761"/><stop offset="100%" stop-color="#2E3B8E"/></linearGradient></defs>

              <!-- Annotation -->
              <text x="410" y="112" fill="#7f8c8d" font-size="6" text-anchor="middle">Link-level prediction &rarr; route accumulation &rarr; OD matrix &rarr; utility input &rarr; demand response</text>
            </svg>
          </div>

          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">100&thinsp;m</div><div class="label">homogeneous road segments (ARMIS + QLD crash data)</div></div>
            <div class="stat"><div class="value">30</div><div class="label">SPF cells (6 stereotypes &times; 5 crash types)</div></div>
            <div class="stat"><div class="value">~780&ndash;5,011</div><div class="label">candidate models evaluated per SPF cell</div></div>
            <div class="stat hero"><div class="value">371&times;371</div><div class="label">OD CRASH matrix fed directly into Stage 5 utilities</div></div>
          </div>

          <div class="card" style="background:var(--orange-soft); padding:0.6em 0.9em; font-size:0.68rem;">
            <b style="color:var(--orange);">Key insight:</b> The CRASH skim is produced <em>once</em> from historical crash data and road attributes, then used as a <em>static input</em> to the demand model. This is the same pattern as travel-time skims: compute from the network, store in OMX, read into utilities. The difference is that travel times come from network assignment and crash frequencies come from SPFs &mdash; but from the demand model's perspective, both are just impedance terms with searchable coefficients.
          </div>
        </div>
      </section>"""

    html = html[:crash_sec] + CRASH_SLIDE + html[crash_end:]
    print(f"  Enhanced crash slide: {crash_end-crash_sec:,} -> {len(CRASH_SLIDE):,} bytes")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
