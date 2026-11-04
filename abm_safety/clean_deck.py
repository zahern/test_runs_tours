# -*- coding: utf-8 -*-
"""Clean up the deck: remove issue slides, Tweed mentions, improve MetaCount slide."""
import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

changes = 0

# ── 1. Remove issue/bug slides (6-10b) ─────────────────────────────────
# The paradox through post-mortem slides
slides_to_remove = [
    "6. THE PARADOX",
    "7. ROOT CAUSE 1",
    "8. ROOT CAUSES 2-4",
    "9. FIXES",
    "10. INFRA",
    "10b. JULY POST-MORTEM",
]

for slide_name in slides_to_remove:
    pattern = f"<!-- {slide_name} -->"
    idx = html.find(pattern)
    if idx < 0:
        print(f"  Not found: {slide_name}")
        continue

    # Find the <section> after the comment
    section_start = html.find("<section ", idx)
    if section_start < 0:
        print(f"  No section after: {slide_name}")
        continue

    # Find matching </section> by counting depth
    depth = 1
    i = section_start + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ":
            depth += 1
        elif html[i:i+10] == "</section>":
            depth -= 1
        i += 1

    section_end = i - 1
    html = html[:idx] + html[section_end:]
    changes += 1
    print(f"  Removed: {slide_name} ({section_end - idx:,} bytes)")

# ── 2. Remove Tweed mentions ────────────────────────────────────────────
tweed_count = len(re.findall(r"Tweed", html))
html = html.replace("Tweed Shire + border zones", "Border zones")
html = html.replace("Tweed Shire + border", "Border zones")
html = html.replace("Tweed Shire", "Border")
html = html.replace("tweed", "border")
html = html.replace(" + Tweed Shire", "")
html = html.replace("SEQ + Tweed Shire zone system re-derived", "SEQ zone system re-derived")
html = html.replace("357</div><div class=\"label\">SA2 zones, SEQ + Tweed Shire</div>", "371</div><div class=\"label\">SA2 zones (SEQ)</div>")
html = html.replace("Tweed Valley", "Border Valley")
tweed_after = len(re.findall(r"Tweed", html))
print(f"  Tweed mentions: {tweed_count} -> {tweed_after}")

# ── 3. Fix population/zone stats slide (update 357->371, remove Tweed) ──
html = html.replace("324→357", "324→371")
html = html.replace("357</div><div class=\"label\">SA2 zones, SEQ</div>", "371</div><div class=\"label\">SA2 zones</div>")

# ── 4. IMPROVE METACOUNT SLIDE (5c) ────────────────────────────────────
old_meta_start = html.find("<!-- 5c. METACOUNTREGRESSOR FLOWCHART -->")
if old_meta_start >= 0:
    meta_section = html.find("<section ", old_meta_start)
    depth = 1
    i = meta_section + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    meta_end = i - 1

    NEW_META = """\
      <section class="slide">
        <h2 class="slide-kicker">Method &middot; Extensive Hypothesis Testing</h2>
        <h1 class="slide-title">metacountregressor: metaheuristic count-model selection</h1>
        <div class="body-area" style="gap:1vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw;">
            <div class="card" style="border-left:3px solid #e74c3c; padding:0.7em 0.9em;">
              <h3 style="margin:0 0 0.3em; font-size:0.84rem; color:#c0392b;">The problem: count-model selection is combinatorial</h3>
              <ul class="plain" style="font-size:0.7rem; gap:0.3em; margin:0;">
                <li>Crash data is zero-inflated, overdispersed &mdash; Poisson is wrong, NB is better, but which NB? Random parameters? Which covariates? Which transformations?</li>
                <li>The analyst <b>cannot</b> test every combination. With 4 count families &times; 5 covariate transforms &times; 5 mixing distributions = 100 base specs, each with k covariates (2<sup>k</sup> subsets), the space is ~10<sup>4</sup>&ndash;10<sup>5</sup> models per cell.</li>
                <li>Manual specification risks: omitted-variable bias, over-fitting, non-transferable forms &mdash; &ldquo;best practice&rdquo; is just what the last paper tested.</li>
              </ul>
            </div>
            <div class="card" style="border-left:3px solid var(--good); padding:0.7em 0.9em;">
              <h3 style="margin:0 0 0.3em; font-size:0.84rem;">The solution: simulated-annealing search over the specification space</h3>
              <ul class="plain" style="font-size:0.7rem; gap:0.3em; margin:0;">
                <li><b>SA inner loop:</b> 5 random initial solutions &rarr; propose neighbouring specification (add/drop a covariate, swap a transform, toggle random parameters) &rarr; fit by MLE &rarr; accept if BIC improves, or with temperature-dependent probability if worse</li>
                <li><b>Cooling schedule:</b> &alpha; = 0.975 per iteration, up to 5,000 iterations / 48 hours per stereotype&times;crash-type cell</li>
                <li><b>Output:</b> best-BIC count model for each of the 6&times;5 = 30 cells (rural remote with >99% zeros gets a restricted search without random parameters)</li>
                <li><b>Same engine</b> scores Stages 1&ndash;3 ABM fitness evaluations (activity frequency, day budget, duration) with identical penalised-BIC objective</li>
              </ul>
            </div>
          </div>
          <div style="display:flex; gap:0.5vw; align-items:stretch;">
            <div class="pipe-box"><b>INPUT</b>Crash counts per 100&thinsp;m segment + road covariates (AADT, width, speed, curvature&hellip;)</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box hot" style="flex:1.4;"><b>SA SEARCH SPACE</b>Count family: Poisson &middot; NB &middot; RP-Poisson &middot; RP-NB &nbsp;|&nbsp; Transforms: none &middot; log &middot; &radic;x &middot; normal-score &middot; arcsine &nbsp;|&nbsp; Mixing: Normal &middot; Uniform &middot; LogN &middot; Triangular &middot; TruncNormal</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box"><b>OUTPUT</b>Best-BIC GCF per cell &rarr; EB-adjusted expected crashes &rarr; OD CRASH skim</div>
          </div>
          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">6 &times; 5</div><div class="label">road stereotypes &times; crash types, each with its own fitted crash function</div></div>
            <div class="stat"><div class="value">~780&ndash;5,011</div><div class="label">candidate models evaluated per stereotype/type cell</div></div>
            <div class="stat"><div class="value">BIC</div><div class="label">fitness objective: balances likelihood against parameter-count penalty</div></div>
            <div class="stat hero"><div class="value">&alpha;=0.975</div><div class="label">SA cooling rate &mdash; slower cooling = more thorough search, 48h per cell</div></div>
          </div>
          <div class="card" style="background:var(--orange-soft); padding:0.6em 0.9em;">
            <h3 style="margin:0 0 0.2em; color:var(--orange); font-size:0.78rem;">Where it fits: the same constrained-search engine runs the ABM's variable selection</h3>
            <p style="font-size:0.66rem; margin:0;">The metacountregressor SA loop is the <b>same code</b> that scores candidate chromosomes in the SA-Bandit search for Stages 1&ndash;3 (activity generation, day-duration budget, activity duration). The objective function (&minus;BIC/n + significance penalty) is identical across crash-function fitting and ABM variable selection. This means selection pressure and shrinkage are <b>uniform</b> throughout the entire pipeline &mdash; the metaheuristic doesn't know whether it's fitting a crash function or an activity model; it just minimises BIC.</p>
          </div>
        </div>
      </section>"""

    html = html[:meta_section] + NEW_META + html[meta_end:]
    old_size = meta_end - meta_section
    print(f"  Replaced MetaCount slide ({old_size:,} -> {len(NEW_META):,} bytes)")

# ── 5. Also clean up the CRASH FREQUENCY LAYER slide (5b) - remove "Tweed" ──
html = html.replace("SEQ + Tweed Shire", "SEQ")
html = html.replace("SEQ&nbsp;+&nbsp;Tweed Shire", "SEQ")

# ── Write back ──
with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nWrote {len(html):,} bytes. {changes} slides removed.")
