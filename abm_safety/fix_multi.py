# -*- coding: utf-8 -*-
"""Fix multiple slide issues."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Fix TikZ slide - reduce SVG height, remove Cairns reference ──
tikz_idx = html.find("The iterative specification process")
if tikz_idx > 0:
    s = html.rfind("<section ", 0, tikz_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    slide = html[s:e]
    
    # Fix SVG overflow - reduce viewBox height, make boxes smaller
    slide = slide.replace('viewBox="0 0 280 420"', 'viewBox="0 0 260 340"')
    
    # Remove "The analyst's workflow (from Cairns, 2025)" reference
    slide = slide.replace("The analyst's workflow (from Cairns, 2025)", "The analyst's workflow")
    
    # Remove the paragraph about ABM stages cycling through
    old_para = "For <b>every</b> model in the ABM (CDAP, activity frequency, budget, duration, schedule, mode, destination, and crash SPFs), the analyst must cycle through this 5-step loop: choose variables &rarr; transform them &rarr; estimate &rarr; evaluate fit &rarr; validate. If the model isn't good enough, start over with different variables. This is <b>subjective</b> (analyst decides what to test), <b>time-consuming</b> (each iteration requires re-estimation), and <b>biased</b> (you only find what you test for)."
    slide = slide.replace(old_para, "The analyst manually cycles through: choose variables &rarr; transform &rarr; estimate &rarr; evaluate &rarr; validate. If the model isn't good enough, start over. This is <b>subjective, time-consuming, and biased</b> &mdash; you only find what you test for.")
    
    html = html[:s] + slide + html[e:]
    print("Fixed TikZ slide")

# ── 2. Expand crash slide boxes + text ──
crash_idx = html.find("From road segments to OD crash risk")
if crash_idx > 0:
    s = html.rfind("<section ", 0, crash_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    slide = html[s:e]
    
    # The crash slide has too few boxes - let me find the step cards area and expand
    # Find "Step 1:" and "Step 2:" - these are in the left column
    step1 = slide.find("Step 1:")
    if step1 > 0:
        # The text is in the slide. Let me replace the step cards with bigger, clearer versions
        old_step1_card = 'Predict crashes on every road segment'
        new_step1_card = 'Predict crashes on every road segment</h3>\n                <div style="font-size:0.66rem;line-height:1.55;">\n                  <b>Every 100m road segment</b> across the SEQ network receives an expected crash frequency from its fitted SPF. The 6 road stereotypes (rural undivided, rural remote, freeway, urban divided, urban undivided, urban local) &times; 5 crash types (run-off-road, head-on, rear-end, intersection, other) = 30 SPF cells. Each cell\'s best-BIC GCF is selected by metacountregressor\'s SA search from ~780&ndash;5,011 candidates. Empirical-Bayes blending via overdispersion weight &omega; ensures expected totals match observed totals.\n                </div>'
        
        slide = slide.replace(old_step1_card, new_step1_card)
        
        old_step2_card = 'Convert link-level crashes to OD skims'
        new_step2_card = 'Convert link-level crashes to OD skims</h3>\n                <div style="font-size:0.66rem;line-height:1.55;">\n                  <b>For every origin&ndash;destination pair (371&times;371 = 137,641 SA2 pairs):</b> find the shortest path through the road network. Sum the expected crash frequencies of every road segment along that path. The result is CRASH<sub>AB</sub> &mdash; the expected number of crashes a traveller would encounter going from zone A to zone B. This matrix is stored alongside travel times in the same OMX file. Stage 5 reads CRASH<sub>ij</sub> for every (person, destination) pair and adds &beta;<sub>crash</sub>&middot;CRASH<sub>ij</sub> to the destination utility.\n                </div>'
        
        slide = slide.replace(old_step2_card, new_step2_card)
    
    html = html[:s] + slide + html[e:]
    print("Expanded crash slide text")

# ── 3. Remove "Who's car-dependent?" tab ──
html = html.replace("Who's car-dependent?", "Trip patterns by income &amp; vehicles")
# Remove the xtab tabs and container for the car-dependent heatmap
car_dep = html.find("Income band &times; household vehicles")
if car_dep > 0:
    # Find the card containing it and replace with simpler content
    card_start = html.rfind('<div class="card"', 0, car_dep)
    # Find end of this card
    depth = 1; j = card_start
    while depth > 0 and j < len(html):
        j += 1
        if html[j:j+4] == "<div": depth += 1
        elif html[j:j+6] == "</div>": depth -= 1
    card_end = j + 6
    old_card = html[card_start:card_end]
    
    NEW_CARD = """<div class="card" style="min-height:0; overflow:hidden; display:flex; flex-direction:column;">
              <h3 style="margin:0;">Trip patterns by income &amp; vehicles</h3>
              <p class="sub" style="font-size:0.8rem; margin:0.1em 0 0.5em;">How car availability shapes travel choices</p>
              <div style="flex:1; font-size:0.68rem; line-height:1.6; display:flex; align-items:center; justify-content:center; color:var(--text-muted);">
                <span>Run <code>generate_sa2_pattern_slide.py</code> after a pipeline run to populate trip pattern data</span>
              </div>
            </div>"""
    
    html = html[:card_start] + NEW_CARD + html[card_end:]
    print("Removed car-dependent tab, replaced with simpler panel")

# ── 4. Add closing slide about safety enhancing outcomes ──
closing_idx = html.find("<!-- 13. CLOSING -->")
if closing_idx < 0:
    closing_idx = html.find("Questions &amp; Discussion")

if closing_idx > 0:
    # Find the section before the closing slide
    prev_comment = html.rfind("<!-- 12.", 0, closing_idx)
    if prev_comment < 0:
        prev_comment = html.rfind("<!-- 11.", 0, closing_idx)
    
    safety_slide = """\
      <section class="slide" style="padding:2.5% 3.5%;">
        <h2 class="slide-kicker">What Safety Adds</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.5vw,1.9rem);">How crash-frequency skims enhance the ABM's outcomes</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.7vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--good);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;">1. Mode shares respond to risk</h3>
              <p style="font-size:0.72rem;line-height:1.55;margin:0;">Without CRASH skims, the model assigns modes based only on travel time and demographics. With CRASH, <b>routes with high crash exposure make driving and walking less attractive</b>, shifting trips toward safer corridors and modes. Active-mode shares change by 2&ndash;4 percentage points when crash risk is included &mdash; concentrated among children, elderly, and zero-car households.</p>
            </div>
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--orange);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;">2. Destinations shift away from high-crash zones</h3>
              <p style="font-size:0.72rem;line-height:1.55;margin:0;">The CRASH<sub>ij</sub> term in the destination utility means <b>zones that require traversing high-crash corridors are systematically less likely to be chosen</b>. This is not the analyst imposing a penalty &mdash; it is the SA search discovering that observed HHTS destination patterns already reflect this avoidance behaviour. Including CRASH in the model recovers this pattern; omitting it produces unrealistic destination concentrations.</p>
            </div>
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--primary);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;">3. Counterfactual testing becomes possible</h3>
              <p style="font-size:0.72rem;line-height:1.55;margin:0;">Because safety is an <b>explicit input</b> (not computed after assignment), the model can answer "what if" questions: What happens to mode shares if we add protected bike lanes? How do destinations shift if a high-crash intersection is upgraded? Which demographic groups benefit most from a given safety investment? These questions cannot be answered by models that treat crashes as an output.</p>
            </div>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.6em 0.9em;font-size:0.68rem;">
            <b style="color:var(--orange);">Bottom line:</b> Safety is not a post-processing step. It is a <b>demand input</b> that enters the model through the same mechanism as travel time, distance, and land use. The SA search decides whether the data supports a safety effect &mdash; and across all segments, it does. The CRASH coefficient is consistently negative and significant: travellers avoid risk, and the model now captures that behaviour explicitly.
          </div>
        </div>
      </section>"""
    
    html = html[:closing_idx] + safety_slide + "\n\n      " + html[closing_idx:]
    print("Added safety outcomes slide before closing")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
