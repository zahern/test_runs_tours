# -*- coding: utf-8 -*-
"""Redesign crash workflow slide with network crash map + link-to-OD diagram."""
import base64, os

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
NETWORK_PNG = r"Z:\test_runs_tours\abm_safety\Figures\ExpCrashes_DTM_SEQ_Network.png"

with open(NETWORK_PNG, "rb") as f:
    net_b64 = base64.b64encode(f.read()).decode()
print(f"  Network crash map: {os.path.getsize(NETWORK_PNG):,} -> {len(net_b64):,} b64")

with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find the crash slide
crash_comment = html.find("<!-- 5b. CRASH FREQUENCY LAYER -->")
crash_sec = html.find("<section ", crash_comment)
depth = 1; i = crash_sec + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
crash_end = i - 1

NEW_CRASH = f"""\
      <section class="slide" style="padding:2% 3.2%;">
        <h2 class="slide-kicker">Safety Input &middot; Crash-Frequency Layer</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.2vw,1.8rem);">From road segments to OD crash risk: how safety enters the model</h1>
        <div class="body-area" style="gap:0.6vh;">
          <div class="grid" style="grid-template-columns: 0.55fr 0.45fr; gap:0.5vw; flex:1; min-height:0;">
            <div style="display:flex; flex-direction:column; gap:0.4vh;">
              <div class="card" style="padding:0.5em 0.7em; flex:1;">
                <h3 style="margin:0 0 0.3em; font-size:0.76rem;">Step 1: Predict crashes on every road segment</h3>
                <div style="font-size:0.62rem;line-height:1.5;">
                  <b>Input:</b> 100&thinsp;m homogeneous road segments with AADT, width, speed, curvature, lane count. 6 road stereotypes &times; 5 crash types = 30 SPF cells.<br>
                  <b>Method:</b> metacountregressor SA search fits a Generalised Crash Function per cell (~780&ndash;5,011 candidates each). Best-BIC model selected. Empirical-Bayes blending via overdispersion weight &omega;.<br>
                  <b>Output:</b> Expected crash frequency <b>per segment</b> (2020&ndash;23 period). These are the coloured lines on the map &rarr;
                </div>
              </div>
              <div class="card" style="padding:0.5em 0.7em; flex:1; background:var(--orange-soft);">
                <h3 style="margin:0 0 0.3em; font-size:0.76rem; color:var(--orange);">Step 2: Convert link-level crashes to OD skims</h3>
                <div style="font-size:0.62rem;line-height:1.5;">
                  <b>For every OD pair (371&times;371):</b> find the shortest path through the network.<br>
                  <b>Accumulate:</b> sum the expected crashes of every road segment along that path.<br>
                  <b>Result:</b> CRASH<sub>AB</sub> = expected crashes a traveller would encounter going from zone A to zone B.<br>
                  <b>Store:</b> as the CRASH matrix in the same OMX file as travel times.<br>
                  <b>Consume:</b> Stage 5 reads CRASH<sub>ij</sub> for every (person, destination) pair. Enters mode and destination utility alongside travel time, distance, and cost.
                </div>
              </div>
            </div>
            <figure class="evidence" style="margin:0; flex:1; min-height:0;">
              <img src="data:image/png;base64,{net_b64}" style="max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--border);border-radius:6px;background:white;" alt="Expected crashes on SEQ DTA network">
              <figcaption style="font-size:0.54rem;">Expected crashes per road segment on the SEQ DTA network. Each coloured line = one road segment's predicted crash frequency (2020&ndash;23, EB-adjusted). These segment-level values are summed along shortest paths to produce the OD CRASH skim.</figcaption>
            </figure>
          </div>

          <div style="display:flex; align-items:center; justify-content:center; gap:1vw; padding:0.3em 0;">
            <svg viewBox="0 0 780 95" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="a3" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto"><path d="M0,0 L6,2.5 L0,5 Z" fill="#1C7293"/></marker></defs>
              <!-- Link segment -->
              <rect x="5" y="15" width="110" height="28" rx="4" fill="#C85A14" opacity="0.85"/>
              <text x="60" y="26" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Segment A</text>
              <text x="60" y="37" fill="#FFE3CE" font-size="5.5" text-anchor="middle">E[crash] = 0.0032</text>
              <text x="60" y="53" fill="#888" font-size="6" text-anchor="middle">&times; 100m</text>

              <rect x="125" y="15" width="110" height="28" rx="4" fill="#C85A14" opacity="0.6"/>
              <text x="180" y="26" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Segment B</text>
              <text x="180" y="37" fill="#FFE3CE" font-size="5.5" text-anchor="middle">E[crash] = 0.0018</text>
              <text x="180" y="53" fill="#888" font-size="6" text-anchor="middle">&times; 100m</text>

              <rect x="245" y="15" width="110" height="28" rx="4" fill="#C85A14" opacity="0.4"/>
              <text x="300" y="26" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Segment C</text>
              <text x="300" y="37" fill="#FFE3CE" font-size="5.5" text-anchor="middle">E[crash] = 0.0041</text>
              <text x="300" y="53" fill="#888" font-size="6" text-anchor="middle">&times; 100m</text>

              <line x1="355" y1="29" x2="380" y2="29" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>

              <!-- Shortest path -->
              <rect x="385" y="10" width="135" height="40" rx="5" fill="#1E2761" opacity="0.9"/>
              <text x="452" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Shortest path</text>
              <text x="452" y="38" fill="#C9D3F5" font-size="5.5" text-anchor="middle">A &rarr; B &rarr; C</text>
              <text x="452" y="58" fill="#888" font-size="6" text-anchor="middle">0.0032+0.0018+0.0041 =</text>

              <line x1="520" y1="30" x2="542" y2="30" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>

              <!-- OD cell -->
              <rect x="546" y="10" width="105" height="40" rx="5" fill="#2E7D6B" opacity="0.9"/>
              <text x="598" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">CRASH<sub>ij</sub></text>
              <text x="598" y="38" fill="#D5F5E3" font-size="5.5" text-anchor="middle">= 0.0091</text>
              <text x="598" y="53" fill="#aaa" font-size="5.5" text-anchor="middle">expected on OD pair</text>

              <line x1="651" y1="30" x2="673" y2="30" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>

              <!-- Utility -->
              <rect x="677" y="10" width="98" height="40" rx="5" fill="url(#bg3)"/>
              <text x="726" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">V<sub>mode,j</sub></text>
              <text x="726" y="38" fill="#D5F5E3" font-size="5.5" text-anchor="middle">&beta;<sub>crash</sub>&middot;0.0091</text>

              <defs><linearGradient id="bg3" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1E2761"/><stop offset="100%" stop-color="#2E3B8E"/></linearGradient></defs>

              <text x="390" y="85" fill="#888" font-size="6" text-anchor="middle">Segment-level prediction &rarr; route accumulation &rarr; OD cell &rarr; utility input</text>
            </svg>
          </div>

          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">100&thinsp;m</div><div class="label">homogeneous road segments with road attributes</div></div>
            <div class="stat"><div class="value">30</div><div class="label">SPF cells (6 stereotypes &times; 5 crash types)</div></div>
            <div class="stat hero"><div class="value">371&times;371</div><div class="label">OD CRASH matrix fed into Stage 5 utilities</div></div>
            <div class="stat"><div class="value">&omega;=0.63&ndash;1.70</div><div class="label">EB overdispersion: noisy models lean on observed data</div></div>
          </div>
        </div>
      </section>"""

html = html[:crash_sec] + NEW_CRASH + html[crash_end:]
print(f"  Crash slide: {crash_end-crash_sec:,} -> {len(NEW_CRASH):,} bytes")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
