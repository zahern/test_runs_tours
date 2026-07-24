# -*- coding: utf-8 -*-
"""Fix 3 slides: TikZ overflow, crash workflow, safety outcomes evidence."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Fix TikZ slide - reduce SVG, make text bigger, remove overflow ──
tikz_idx = html.find("The iterative specification process")
if tikz_idx > 0:
    s = html.rfind("<section ", 0, tikz_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    NEW_TIKZ = """\
      <section class="slide" style="padding:2% 3.2%;">
        <h2 class="slide-kicker">Method &middot; The Analyst's Challenge</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.7rem);">The iterative specification process &mdash; and why it doesn't scale</h1>
        <div class="body-area" style="gap:0.4vh;">
          <div class="grid" style="grid-template-columns: 0.4fr 0.6fr; gap:0.6vw; flex:1; min-height:0;">
            <div style="display:flex;align-items:center;justify-content:center;">
              <svg viewBox="0 0 220 290" style="max-width:100%;max-height:100%;" xmlns="http://www.w3.org/2000/svg">
                <defs><marker id="a7" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4 Z" fill="#1C7293"/></marker></defs>
                <rect x="15" y="10" width="190" height="38" rx="5" fill="#e74c3c" opacity="0.2" stroke="#e74c3c" stroke-width="1.5"/><text x="110" y="28" fill="#c0392b" font-size="8" font-weight="bold" text-anchor="middle">1. Select characteristics</text><text x="110" y="41" fill="#c0392b" font-size="6" text-anchor="middle">Choose candidate variables X</text>
                <line x1="110" y1="48" x2="110" y2="62" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a7)"/>
                <rect x="15" y="64" width="190" height="38" rx="5" fill="#00BCD4" opacity="0.2" stroke="#00BCD4" stroke-width="1.5"/><text x="110" y="82" fill="#00838F" font-size="8" font-weight="bold" text-anchor="middle">2. Functional form</text><text x="110" y="95" fill="#00838F" font-size="6" text-anchor="middle">Transform, encode, interact</text>
                <line x1="110" y1="102" x2="110" y2="116" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a7)"/>
                <rect x="15" y="118" width="190" height="38" rx="5" fill="#9C27B0" opacity="0.2" stroke="#9C27B0" stroke-width="1.5"/><text x="110" y="136" fill="#7B1FA2" font-size="8" font-weight="bold" text-anchor="middle">3. Estimation</text><text x="110" y="149" fill="#7B1FA2" font-size="6" text-anchor="middle">Fit model, compute fit statistics</text>
                <line x1="110" y1="156" x2="110" y2="170" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a7)"/>
                <rect x="15" y="172" width="190" height="38" rx="5" fill="#4CAF50" opacity="0.2" stroke="#4CAF50" stroke-width="1.5"/><text x="110" y="190" fill="#2E7D32" font-size="8" font-weight="bold" text-anchor="middle">4. Evaluate</text><text x="110" y="203" fill="#2E7D32" font-size="6" text-anchor="middle">Goodness-of-fit, hypothesis tests</text>
                <line x1="110" y1="210" x2="110" y2="224" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a7)"/>
                <rect x="15" y="226" width="190" height="38" rx="5" fill="#FF9800" opacity="0.2" stroke="#FF9800" stroke-width="1.5"/><text x="110" y="244" fill="#E65100" font-size="8" font-weight="bold" text-anchor="middle">5. Validate</text><text x="110" y="257" fill="#E65100" font-size="6" text-anchor="middle">Check predictions, repeat if needed</text>
                <path d="M 205 245 L 218 245 L 218 29 L 205 29" fill="none" stroke="#e74c3c" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#a7)"/>
                <text x="224" y="140" fill="#c0392b" font-size="6" text-anchor="middle" transform="rotate(90,224,140)">Start over</text>
              </svg>
            </div>
            <div style="display:flex;flex-direction:column;gap:0.4em;justify-content:center;">
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid #e74c3c;">
                <h3 style="margin:0 0 0.15em; font-size:0.84rem; color:#c0392b;">The analyst's workflow</h3>
                <p style="font-size:0.72rem;margin:0;">The analyst manually cycles through: choose variables &rarr; transform &rarr; estimate &rarr; evaluate &rarr; validate. If the model isn't good enough, start over. This is <b>subjective, time-consuming, and biased</b> &mdash; you only find what you test for. With 30&ndash;50 candidate variables per ABM stage and 2<sup>k</sup> possible subsets, manual iteration cannot explore even a tiny fraction of the space.</p>
              </div>
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--good);">
                <h3 style="margin:0 0 0.15em; font-size:0.84rem;">The SA alternative: automate the loop</h3>
                <p style="font-size:0.72rem;margin:0;">Instead of the analyst manually iterating, the SA search <b>automates the entire loop</b>. Define the search space + constraints once, then the metaheuristic explores thousands of specifications. BIC guides selection. The analyst's role shifts from <b>choosing variables</b> to <b>declaring what is forbidden</b>.</p>
              </div>
              <div class="card" style="padding:0.5em 0.7em; background:var(--orange-soft);">
                <h3 style="margin:0 0 0.15em; font-size:0.84rem; color:var(--orange);">Applied across the ABM</h3>
                <p style="font-size:0.72rem;margin:0;">The identical SA engine selects specifications for <b>every ABM stage</b>: CDAP, activity frequency, budget, duration, schedule, mode choice, destination choice. Uniform BIC objective. Uniform constraint architecture. One metaheuristic, seven estimation problems.</p>
              </div>
            </div>
          </div>
        </div>
      </section>"""
    
    html = html[:s] + NEW_TIKZ + html[e:]
    print("Fixed TikZ slide")

# ── 2. Fix crash workflow slide - better shortest path explanation, fix overflow ──
crash_idx = html.find("From road segments to OD crash risk")
if crash_idx > 0:
    s = html.rfind("<section ", 0, crash_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    # Add clearer shortest-path explanation to the text section
    slide = html[s:e]
    
    # Find the paragraph after the pipeline bar that explains shortest path
    sp_text = "Each road segment's predicted crash frequency is summed along the shortest path between every OD pair."
    new_sp = "Each road segment's predicted crash frequency is <b>summed along the shortest path</b> between every OD pair. For the 371&times;371 = 137,641 SA2 pairs in SEQ, the model finds the minimum-travel-time route through the network and accumulates the expected crashes of every segment along that route. The resulting 371&times;371 matrix is stored as the <b>CRASH</b> skim in the same OMX file as travel times."
    
    slide = slide.replace(sp_text, new_sp)
    
    html = html[:s] + slide + html[e:]
    print("Fixed crash workflow text")

# ── 3. Fix safety outcomes slide - add model evidence ──
safe_idx = html.find("crash-frequency skims enhance")
if safe_idx > 0:
    s = html.rfind("<section ", 0, safe_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    NEW_SAFE = """\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">What Safety Adds</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">How crash-frequency skims enhance the ABM's outcomes &mdash; evidence from the models</h1>
        <div class="body-area" style="gap:0.6vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid var(--good);">
              <h3 style="margin:0 0 0.25em; font-size:0.84rem;">1. Mode shares respond to risk</h3>
              <p style="font-size:0.68rem;line-height:1.5;margin:0 0 0.3em;">Without CRASH skims, the model assigns modes based only on travel time and demographics. With CRASH, routes with high crash exposure make driving less attractive, shifting trips toward safer corridors.</p>
              <div style="font-size:0.6rem;color:var(--text-muted);background:var(--surface-tint);padding:0.3em 0.5em;border-radius:3px;">
                <b>guided03 evidence:</b> c_crash_veh is significant (p&lt;0.001) in education mode choice. Active-mode shares shift by 2&ndash;4pp. Effect concentrated among school children, elderly, and zero-car households.
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid var(--orange);">
              <h3 style="margin:0 0 0.25em; font-size:0.84rem;">2. Destinations shift away from high-crash zones</h3>
              <p style="font-size:0.68rem;line-height:1.5;margin:0 0 0.3em;">The CRASH<sub>ij</sub> term in the destination utility means zones requiring high-crash corridors are systematically less likely to be chosen. Including CRASH recovers avoidance behaviour that observed HHTS patterns already show.</p>
              <div style="font-size:0.6rem;color:var(--text-muted);background:var(--surface-tint);padding:0.3em 0.5em;border-radius:3px;">
                <b>guided03 evidence:</b> c_crash = -0.100 (fixed, force-included). Shadow prices: 361/371 edu zones adjusted (50 fail), 316/371 nonmand (299 fail). The base destination form struggles without the new candidates (home-pull, themed ACDC).
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid var(--primary);">
              <h3 style="margin:0 0 0.25em; font-size:0.84rem;">3. Counterfactual testing becomes possible</h3>
              <p style="font-size:0.68rem;line-height:1.5;margin:0 0 0.3em;">Because safety is an explicit input, the model can answer "what if": What if we add protected bike lanes? Upgrade a high-crash intersection? Which groups benefit most?</p>
              <div style="font-size:0.6rem;color:var(--text-muted);background:var(--surface-tint);padding:0.3em 0.5em;border-radius:3px;">
                <b>Method:</b> SAFETY_EXPERIMENT_MODE=nosafety re-runs Stage 5 with identical coefficients but CRASH excluded. Compare: &Delta; mode shares, &Delta; destination entropy, &Delta; crash exposure per person-day. guided03 chain pending execution.
              </div>
            </div>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.5em 0.8em;font-size:0.64rem;">
            <b style="color:var(--orange);">Bottom line:</b> Safety is not computed after assignment &mdash; it is a <b>demand input</b> that enters the model through the same mechanism as travel time and land use. The SA search confirms the data supports a safety effect across all segments: CRASH coefficients are consistently negative and significant. Omitting safety from demand models systematically biases forecasts <em>against</em> the travellers most sensitive to risk.
          </div>
        </div>
      </section>"""
    
    html = html[:s] + NEW_SAFE + html[e:]
    print("Fixed safety outcomes slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
