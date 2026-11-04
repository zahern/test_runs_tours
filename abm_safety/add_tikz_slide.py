# -*- coding: utf-8 -*-
"""Add Cairns TikZ flowchart as SVG, fix slide 8."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── Check slide 8 boundary and fix if needed ──
s8_start = html.find("MetaCountRegressor: a major motivation")
s8 = html.rfind("<section ", 0, s8_start)
depth = 1; i = s8 + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
s8_end = i - 1
print(f"Slide 8: {s8}-{s8_end} ({s8_end-s8:,} bytes)")

# Check if there are nested sections
nested = html.count("<section ", s8, s8_end)
if nested > 1:
    print(f"  WARNING: {nested} nested sections found!")
else:
    print("  Clean - no nesting")

# ── Add TikZ flowchart as a new comparison slide ──
# The TikZ shows: select characteristics -> functional form -> estimation -> evaluate -> validate -> (loop back)
# I'll add this as an SVG in a new slide before the estimation motivation slide

est_idx = html.find("The core problem: for any model")
if est_idx > 0:
    est_s = html.rfind("<section ", 0, est_idx)
    
    ANALYST_SLIDE = """\
      <section class="slide" style="padding:2% 3.2%;">
        <h2 class="slide-kicker">Method &middot; The Analyst's Challenge</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.7rem);">The iterative specification process &mdash; and why it doesn't scale</h1>
        <div class="body-area" style="gap:0.5vh;">
          <div class="grid" style="grid-template-columns: 0.45fr 0.55fr; gap:0.6vw; flex:1; min-height:0;">
            <div style="display:flex;align-items:center;justify-content:center;">
              <svg viewBox="0 0 280 420" style="max-width:100%;max-height:100%;" xmlns="http://www.w3.org/2000/svg">
                <defs><marker id="a7" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto"><path d="M0,0 L6,2.5 L0,5 Z" fill="#1C7293"/></marker></defs>
                <!-- Box 1: Select -->
                <rect x="40" y="15" width="200" height="50" rx="6" fill="#e74c3c" opacity="0.25"/>
                <rect x="40" y="15" width="200" height="50" rx="6" fill="none" stroke="#e74c3c" stroke-width="2"/>
                <text x="140" y="38" fill="#c0392b" font-size="9" font-weight="bold" text-anchor="middle">1. Select characteristics</text>
                <text x="140" y="54" fill="#c0392b" font-size="7" text-anchor="middle">Choose candidate variables X</text>
                <!-- Arrow -->
                <line x1="140" y1="65" x2="140" y2="85" stroke="#1C7293" stroke-width="1.5" marker-end="url(#a7)"/>
                <!-- Box 2: Functional -->
                <rect x="40" y="87" width="200" height="50" rx="6" fill="#00BCD4" opacity="0.2"/>
                <rect x="40" y="87" width="200" height="50" rx="6" fill="none" stroke="#00BCD4" stroke-width="2"/>
                <text x="140" y="110" fill="#00838F" font-size="9" font-weight="bold" text-anchor="middle">2. Functional form</text>
                <text x="140" y="126" fill="#00838F" font-size="7" text-anchor="middle">Transform, encode, interact</text>
                <line x1="140" y1="137" x2="140" y2="157" stroke="#1C7293" stroke-width="1.5" marker-end="url(#a7)"/>
                <!-- Box 3: Estimate -->
                <rect x="40" y="159" width="200" height="50" rx="6" fill="#9C27B0" opacity="0.2"/>
                <rect x="40" y="159" width="200" height="50" rx="6" fill="none" stroke="#9C27B0" stroke-width="2"/>
                <text x="140" y="182" fill="#7B1FA2" font-size="9" font-weight="bold" text-anchor="middle">3. Estimation</text>
                <text x="140" y="198" fill="#7B1FA2" font-size="7" text-anchor="middle">Fit model, compute LL</text>
                <line x1="140" y1="209" x2="140" y2="229" stroke="#1C7293" stroke-width="1.5" marker-end="url(#a7)"/>
                <!-- Box 4: Evaluate -->
                <rect x="40" y="231" width="200" height="50" rx="6" fill="#4CAF50" opacity="0.2"/>
                <rect x="40" y="231" width="200" height="50" rx="6" fill="none" stroke="#4CAF50" stroke-width="2"/>
                <text x="140" y="254" fill="#2E7D32" font-size="9" font-weight="bold" text-anchor="middle">4. Evaluate</text>
                <text x="140" y="270" fill="#2E7D32" font-size="7" text-anchor="middle">Goodness-of-fit, hypothesis tests</text>
                <line x1="140" y1="281" x2="140" y2="301" stroke="#1C7293" stroke-width="1.5" marker-end="url(#a7)"/>
                <!-- Box 5: Validate -->
                <rect x="40" y="303" width="200" height="50" rx="6" fill="#FF9800" opacity="0.2"/>
                <rect x="40" y="303" width="200" height="50" rx="6" fill="none" stroke="#FF9800" stroke-width="2"/>
                <text x="140" y="326" fill="#E65100" font-size="9" font-weight="bold" text-anchor="middle">5. Validate</text>
                <text x="140" y="342" fill="#E65100" font-size="7" text-anchor="middle">Check predictions, iterate</text>
                <!-- Feedback loop -->
                <path d="M 240 328 L 260 328 L 260 40 L 240 40" fill="none" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#a7)"/>
                <text x="268" y="185" fill="#c0392b" font-size="7" text-anchor="middle" transform="rotate(90,268,185)">Not good enough? Start over</text>
                <!-- Time annotation -->
                <text x="140" y="375" fill="#888" font-size="6" text-anchor="middle">Iterative, time-consuming, subjective</text>
                <text x="140" y="390" fill="#888" font-size="6" text-anchor="middle">Risk: bias, over-fitting, non-transferable</text>
              </svg>
            </div>
            <div style="display:flex;flex-direction:column;gap:0.5em;justify-content:center;">
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid #e74c3c;">
                <h3 style="margin:0 0 0.2em; font-size:0.84rem; color:#c0392b;">The analyst's workflow (from Cairns, 2025)</h3>
                <p style="font-size:0.68rem;margin:0;">For <b>every</b> model in the ABM (CDAP, activity frequency, budget, duration, schedule, mode, destination, and crash SPFs), the analyst must cycle through this 5-step loop: choose variables &rarr; transform them &rarr; estimate &rarr; evaluate fit &rarr; validate. If the model isn't good enough, start over with different variables. This is <b>subjective</b> (analyst decides what to test), <b>time-consuming</b> (each iteration requires re-estimation), and <b>biased</b> (you only find what you test for).</p>
              </div>
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--good);">
                <h3 style="margin:0 0 0.2em; font-size:0.84rem;">The SA alternative: automate the loop</h3>
                <p style="font-size:0.68rem;margin:0;">Instead of the analyst manually iterating, the SA search <b>automates the entire loop</b>. The 5-step cycle becomes a single command: define the search space + constraints, then let the metaheuristic explore thousands of specifications. BIC guides selection toward parsimonious, well-fitting models. The analyst's role shifts from <b>choosing variables</b> to <b>declaring what is forbidden</b>.</p>
              </div>
              <div class="card" style="padding:0.5em 0.7em; background:var(--orange-soft);">
                <h3 style="margin:0 0 0.2em; font-size:0.84rem; color:var(--orange);">Applied across the ABM</h3>
                <p style="font-size:0.68rem;margin:0;">This is not just for crash models. The identical SA engine selects specifications for <b>every ABM stage</b>: CDAP pattern classification, activity frequency generation, day budget, activity duration, schedule placement, tour mode choice, and destination choice. Uniform BIC objective. Uniform constraint architecture. One metaheuristic, seven estimation problems.</p>
              </div>
            </div>
          </div>
        </div>
      </section>"""
    
    html = html[:est_s] + ANALYST_SLIDE + "\n\n" + html[est_s:]
    print("Added analyst workflow slide with TikZ flowchart SVG")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
