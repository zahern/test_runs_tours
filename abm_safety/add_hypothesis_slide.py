# -*- coding: utf-8 -*-
"""Add extensive hypothesis testing slide from PPTX content to the deck."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"

SLIDE = """\
      <section class="slide">
        <h2 class="slide-kicker">Method &middot; Extensive Hypothesis Testing</h2>
        <h1 class="slide-title">Why the SA-Bandit search replaces analyst-driven specification</h1>
        <div class="body-area" style="gap:0.9vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw;">
            <div class="card" style="border-left:3px solid #e74c3c; padding:0.6em 0.9em;">
              <h3 style="margin:0 0 0.3em; font-size:0.82rem; color:#c0392b;">The old way: analyst picks variables</h3>
              <ul class="plain" style="font-size:0.66rem; gap:0.35em; margin:0;">
                <li>Each sub-model's utility function hand-crafted by the analyst</li>
                <li>Risk of omitted-variable bias &mdash; the variable you never test is the one that matters</li>
                <li>Risk of over-fitting &mdash; too many correlated variables inflate k without new information</li>
                <li>Specifications don't transfer between regions or populations</li>
                <li>&ldquo;Best practice&rdquo; is just the specification the last paper used</li>
              </ul>
            </div>
            <div class="card" style="border-left:3px solid var(--good); padding:0.6em 0.9em;">
              <h3 style="margin:0 0 0.3em; font-size:0.82rem;">SA-Bandit: the metaheuristic decides</h3>
              <ul class="plain" style="font-size:0.66rem; gap:0.35em; margin:0;">
                <li><b>Mathematical programming formulation:</b> minimise BIC subject to variable constraints, transformation constraints, and pre-specification constraints (Beeramoole, Kelly, Haque, Pinz, &amp; Paz, 2024)</li>
                <li><b>Bandit outer loop</b> selects feature masks per segment; <b>simulated-annealing inner loop</b> fits candidate models; <b>BIC penalises</b> complexity</li>
                <li>Constraint-guided: analyst declares what's <i>forbidden</i> (mutually exclusive encodings, must-include structural terms, never-random parameters), not what's <i>included</i></li>
                <li>Same objective across all 6 stages &mdash; uniform penalised-BIC selection pressure</li>
              </ul>
            </div>
          </div>

          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.6vw;">
            <div class="card" style="border-top:3px solid var(--teal); padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.25em; font-size:0.78rem;">metacountregressor</h3>
              <p style="font-size:0.62rem; margin:0 0 0.4em;">Stages 1&ndash;3 fitness evaluator: each candidate chromosome is scored by an SA-tuned count-model fit (Poisson / NB / random-param variants with automatic covariate-transform selection). <b>Also</b> fits the 6&times;5 segment-level crash functions that become the CRASH skim.</p>
              <p style="font-size:0.56rem; color:var(--text-muted); margin:0;">~780&ndash;5,011 candidates per cell. BIC objective. 48h search with &alpha;=0.975 cooling.</p>
            </div>
            <div class="card" style="border-top:3px solid var(--orange); padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.25em; font-size:0.78rem;">SearchLibrium</h3>
              <p style="font-size:0.62rem; margin:0 0 0.4em;">Stage 4 fitness: schedule-placement MNL/MXL. ConstraintBuilder format (0.0.138) is the single source of truth &mdash; one declaration feeds every searcher and every estimator. Random-coefficient policy: GA_RAND_PARAM_SEARCH toggles which selected variables get random coefficients.</p>
              <p style="font-size:0.56rem; color:var(--text-muted); margin:0;">SearchLogit heritage (Beeramoole et al., Transp. Res. C, 2024).</p>
            </div>
            <div class="card" style="border-top:3px solid var(--primary-2); padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.25em; font-size:0.78rem;">Larch (Stage 5)</h3>
              <p style="font-size:0.62rem; margin:0 0 0.4em;">Destination + mode choice (NL/MNL over 371 zones). CRASH forced into every utility by the constraint registry. Elastic-net penalty (L2) on every optimiser invocation. Zero-gradient engine guard: jax &harr; numba retry.</p>
              <p style="font-size:0.56rem; color:var(--text-muted); margin:0;">Flat CA dataset pattern (ActivitySim pre-joined).</p>
            </div>
          </div>

          <div style="display:flex; gap:0.5vw; align-items:stretch;">
            <div class="pipe-box" style="flex:1; font-size:0.6rem; padding:0.5em;"><b>BANDIT OUTER LOOP</b>Exponential-weight arm selection across feature masks, per segment, 8-way parallel. Significance penalty (GA_SIG_PENALTY_WEIGHT) pushes toward parsimony.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box" style="flex:1; font-size:0.6rem; padding:0.5em;"><b>PENALISED FITNESS</b>&minus;BIC/n + significance penalty + elastic-net shrinkage. Same objective across stages 1&ndash;5 &mdash; uniform selection pressure throughout the pipeline.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box hot" style="flex:1; font-size:0.6rem; padding:0.5em;"><b>SELECTED SPECS</b>ga_results.pkl per stage &rarr; merged by BIC &rarr; staged full-population application. Coefficients estimated once on the best feature set, not averaged across runs.</div>
          </div>

          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">30&ndash;50</div><div class="label">candidate features per stage segment</div></div>
            <div class="stat"><div class="value">2<sup>k</sup></div><div class="label">possible models per segment (intractable to enumerate)</div></div>
            <div class="stat"><div class="value">5&ndash;8</div><div class="label">selected features per guided03 segment (parsimonious)</div></div>
            <div class="stat hero"><div class="value">BIC</div><div class="label">fitness objective &mdash; balances fit (+LL) against complexity (+k&middot;ln&nbsp;n)</div></div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.62rem;">Constraint declarations in ga_constraints.py: force_include (CRASH, size_term, logsum, Mu_Root), mutually_exclusive (income encodings, age encodings, ACDC variants, home-pull variants), min_behavioral (at least one industry attractor for work), never_random (ASCs, nest params). The search decides the enrichment &mdash; the analyst declares the boundaries.</p>
        </div>
      </section>"""

# Inject right before the existing "Where metacountregressor and SearchLibrium sit" slide (5d)
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find the two-engines slide and replace it with this enhanced version
target = '<!-- 5d. TWO ENGINES IN THE ABM -->'
idx = html.find(target)
if idx < 0:
    print("Two engines slide not found")
else:
    # Find the end of that slide
    slide_start = html.find('<section class="slide">', idx)
    # Find closing </section> by counting nested <section> tags
    depth = 0
    end = slide_start
    for i in range(slide_start, len(html)):
        if html[i:i+9] == '<section ':
            depth += 1
        elif html[i:i+10] == '</section>':
            depth -= 1
            if depth == 0:
                end = i + 10
                break
    old_len = end - slide_start
    html = html[:slide_start] + SLIDE + html[end:]
    print(f"Replaced two-engines slide ({old_len:,} -> {len(SLIDE):,} bytes)")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes. Done.")
