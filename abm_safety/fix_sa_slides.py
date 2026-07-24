# -*- coding: utf-8 -*-
"""Fix both SA search slides - bigger text, highlighting."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── Fix slide 1: "How the SA search replaces" (MetaCount/SearchLibrium visual) ──
idx1 = html.find("How the SA search replaces analyst-driven specification")
s1 = html.rfind("<section ", 0, idx1)
depth = 1; i1 = s1 + 1
while depth > 0 and i1 < len(html):
    if html[i1:i1+9] == "<section ": depth += 1
    elif html[i1:i1+10] == "</section>": depth -= 1
    i1 += 1
e1 = i1 - 1

NEW_HOW = """\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Method &middot; SA Search Engine</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.5vw,1.9rem);">How the SA search replaces analyst-driven specification</h1>
        <div class="body-area" style="gap:0.7vh;">

          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw;">
            <div class="card" style="border-left:4px solid #e74c3c; padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.2em; font-size:0.9rem; color:#c0392b;">The problem</h3>
              <p style="font-size:0.78rem; margin:0;">Crash data is zero-inflated and overdispersed. With 4 count families &times; 5 transforms &times; 5 mixing distributions &times; 2<sup>k</sup> covariate subsets = <b>~10<sup>4</sup>&ndash;10<sup>5</sup> models per cell</b>. The analyst cannot test every combination.</p>
            </div>
            <div class="card" style="border-left:4px solid var(--good); padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.2em; font-size:0.9rem;">The solution</h3>
              <p style="font-size:0.78rem; margin:0;"><b>Simulated-annealing search</b> over the specification space, guided by BIC. 5 random starts &rarr; propose neighbour &rarr; fit MLE &rarr; accept if BIC better (or probabilistically if worse). <b>&alpha;=0.975</b> cooling. Up to <b>5,000 iterations / 48h</b> per cell.</p>
            </div>
          </div>

          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--teal);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;"><span style="color:var(--teal);">metacountregressor</span> <span style="font-size:0.66rem;color:var(--text-muted);">crash functions + stages 1&ndash;3</span></h3>
              <div style="display:flex;flex-direction:column;gap:0.25em;">
                <div class="pipe-box" style="font-size:0.7rem;padding:0.4em;"><b>INPUT</b> Crash counts + road covariates per 100&thinsp;m segment</div>
                <div style="text-align:center;font-size:0.8rem;color:var(--text-faint);">&darr;</div>
                <div class="pipe-box hot" style="font-size:0.7rem;padding:0.4em;"><b>SA SEARCH</b> 4 families &times; 5 transforms &times; 5 mixing &times; 2<sup>k</sup> covariates</div>
                <div style="text-align:center;font-size:0.8rem;color:var(--text-faint);">&darr;</div>
                <div class="pipe-box" style="font-size:0.7rem;padding:0.4em;"><b>OUTPUT</b> Best-BIC GCF per cell &rarr; EB-adjusted &rarr; OD CRASH skim</div>
              </div>
              <p style="font-size:0.7rem;color:var(--text-muted);margin-top:0.4em;">30 SPF cells. ~780&ndash;5,011 candidates each. Same SA loop scores ABM Stages 1&ndash;3 fitness.</p>
            </div>

            <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--orange);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;"><span style="color:var(--orange);">SearchLibrium</span> <span style="font-size:0.66rem;color:var(--text-muted);">stage 4 + constraints</span></h3>
              <div style="display:flex;flex-direction:column;gap:0.25em;">
                <div class="pipe-box" style="font-size:0.7rem;padding:0.4em;"><b>INPUT</b> Schedule-placement MNL over 6 time periods</div>
                <div style="text-align:center;font-size:0.8rem;color:var(--text-faint);">&darr;</div>
                <div class="pipe-box hot" style="font-size:0.7rem;padding:0.4em;"><b>SA SEARCH</b> MNL/MXL with random-coefficient toggling (BIC-greedy)</div>
                <div style="text-align:center;font-size:0.8rem;color:var(--text-faint);">&darr;</div>
                <div class="pipe-box" style="font-size:0.7rem;padding:0.4em;"><b>OUTPUT</b> Best-BIC schedule model &rarr; 6-period activity placement</div>
              </div>
              <p style="font-size:0.7rem;color:var(--text-muted);margin-top:0.4em;">ConstraintBuilder (0.0.138): one declaration feeds every searcher. force_include, mutually_exclusive, min_behavioral, never_random.</p>
            </div>
          </div>

          <div style="flex:1; min-height:8vh; display:flex; align-items:center; justify-content:center;">
            <svg viewBox="0 0 700 85" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="a4b" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4 Z" fill="#888"/></marker></defs>
              <rect x="5" y="10" width="55" height="20" rx="3" fill="#EBEFFA" stroke="#DCE1F0"/><text x="32" y="24" fill="#1E2761" font-size="7" font-weight="bold" text-anchor="middle">5 random</text>
              <line x1="60" y1="20" x2="76" y2="20" stroke="#888" stroke-width="0.8" marker-end="url(#a4b)"/>
              <rect x="78" y="6" width="65" height="28" rx="3" fill="#1E2761" opacity="0.8"/><text x="110" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Propose</text><text x="110" y="29" fill="#C9D3F5" font-size="5.5" text-anchor="middle">&plusmn;1 covar, &plusmn;transform</text>
              <line x1="143" y1="20" x2="159" y2="20" stroke="#888" stroke-width="0.8" marker-end="url(#a4b)"/>
              <rect x="161" y="6" width="55" height="28" rx="3" fill="#1E2761" opacity="0.8"/><text x="188" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Fit MLE</text><text x="188" y="29" fill="#C9D3F5" font-size="5.5" text-anchor="middle">compute BIC</text>
              <line x1="216" y1="20" x2="232" y2="20" stroke="#888" stroke-width="0.8" marker-end="url(#a4b)"/>
              <rect x="234" y="6" width="70" height="28" rx="3" fill="#2E7D6B" opacity="0.85"/><text x="269" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Accept?</text><text x="269" y="29" fill="#D5F5E3" font-size="5.5" text-anchor="middle">P=exp(-&Delta;BIC/T)</text>
              <line x1="304" y1="20" x2="320" y2="20" stroke="#888" stroke-width="0.8" marker-end="url(#a4b)"/>
              <rect x="322" y="6" width="45" height="28" rx="3" fill="#1E2761" opacity="0.8"/><text x="344" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Cool</text><text x="344" y="29" fill="#C9D3F5" font-size="5.5" text-anchor="middle">T&times;0.975</text>
              <path d="M 344 48 L 344 58 L 110 58 L 110 34" fill="none" stroke="#888" stroke-width="0.7" stroke-dasharray="3,2" marker-end="url(#a4b)"/>
              <text x="227" y="70" fill="#888" font-size="6" text-anchor="middle">repeat 5,000&times; or 48h per cell</text>
              <line x1="367" y1="20" x2="383" y2="20" stroke="#888" stroke-width="0.8" marker-end="url(#a4b)"/>
              <rect x="385" y="6" width="65" height="28" rx="3" fill="#C85A14" opacity="0.85"/><text x="417" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Best BIC</text><text x="417" y="29" fill="#FFE3CE" font-size="5.5" text-anchor="middle">model wins</text>
              <rect x="465" y="2" width="230" height="48" rx="3" fill="white" stroke="#DCE1F0"/><text x="580" y="10" fill="#888" font-size="6" text-anchor="middle">BIC trace (lower = better)</text>
              <polyline points="470,46 485,38 500,33 515,40 530,26 545,23 560,28 575,20 590,16 605,18 620,13 635,12 650,10 662,8" fill="none" stroke="#C85A14" stroke-width="1.2"/><text x="680" y="12" fill="#C85A14" font-size="6" text-anchor="middle">min</text>
            </svg>
          </div>

          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">6&times;5</div><div class="label">SPF cells (stereotypes &times; crash types)</div></div>
            <div class="stat"><div class="value">~780&ndash;5,011</div><div class="label">candidates per cell evaluated</div></div>
            <div class="stat hero"><div class="value">&alpha;=0.975</div><div class="label">SA cooling &mdash; 5,000 iter / 48h per cell</div></div>
            <div class="stat"><div class="value">BIC</div><div class="label">penalised objective &mdash; uniform across pipeline</div></div>
          </div>
        </div>
      </section>"""

html = html[:s1] + NEW_HOW + html[e1:]
print(f"  How slide: {e1-s1:,} -> {len(NEW_HOW):,} bytes")

# ── Fix slide 2: "Why the SA search replaces" (extensive hypothesis testing) ──
idx2 = html.find("Why the SA search replaces analyst-driven specification")
s2 = html.rfind("<section ", 0, idx2)
depth = 1; i2 = s2 + 1
while depth > 0 and i2 < len(html):
    if html[i2:i2+9] == "<section ": depth += 1
    elif html[i2:i2+10] == "</section>": depth -= 1
    i2 += 1
e2 = i2 - 1

NEW_WHY = """\
      <section class="slide" style="padding:2.5% 3.8%;">
        <h2 class="slide-kicker">Method &middot; Extensive Hypothesis Testing</h2>
        <h1 class="slide-title" style="font-size:clamp(1.4rem,2.6vw,2rem);">Why the SA search replaces analyst-driven specification</h1>
        <div class="body-area" style="gap:1.2vh;">

          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.8vw; flex:1;">
            <div class="card" style="border-left:4px solid #e74c3c; padding:0.8em 1em;">
              <h3 style="margin:0 0 0.4em; font-size:1rem; color:#c0392b;">The old way: analyst picks variables</h3>
              <ul class="plain" style="font-size:0.84rem; gap:0.5em;">
                <li>Each sub-model's utility function is <b>hand-crafted by the analyst</b> based on literature, intuition, and trial-and-error</li>
                <li><b>Omitted-variable bias:</b> the variable you never test is the one that matters most</li>
                <li><b>Over-fitting risk:</b> too many correlated variables inflate <em>k</em> without new information &mdash; BIC penalises you either way</li>
                <li>Specifications don't transfer between regions or populations &mdash; "best practice" is just what the last paper used</li>
              </ul>
            </div>
            <div class="card" style="border-left:4px solid var(--good); padding:0.8em 1em;">
              <h3 style="margin:0 0 0.4em; font-size:1rem;">The SA search: the data decides</h3>
              <ul class="plain" style="font-size:0.84rem; gap:0.5em;">
                <li><b>Mathematical programming:</b> minimise BIC subject to variable, transformation, and pre-specification constraints (Beeramoole, Kelly, Haque, Pinz, &amp; Paz, 2024, <em>Transp. Res. C</em>)</li>
                <li><b>SA outer loop</b> selects feature masks per segment; <b>SA inner loop</b> fits candidate models; <b>BIC</b> penalises complexity</li>
                <li>Constraint-guided: the analyst declares what is <b>forbidden</b> (mutually exclusive encodings, must-include structural terms), not what is included</li>
                <li>Same penalised-BIC objective across all 6 stages &mdash; uniform selection pressure throughout the pipeline</li>
              </ul>
            </div>
          </div>

          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.6vw;">
            <div class="card" style="border-top:4px solid var(--teal); padding:0.6em 0.8em; background:var(--surface-tint);">
              <h3 style="margin:0 0 0.3em; font-size:0.95rem;">metacountregressor</h3>
              <p style="font-size:0.8rem; margin:0 0 0.4em;">Stages 1&ndash;3 fitness evaluator: each candidate chromosome scored by an SA-tuned count-model fit (Poisson / NB / random-param variants with automatic covariate-transform selection). Also fits the 6&times;5 crash functions that become the CRASH skim.</p>
              <p style="font-size:0.72rem; color:var(--text-muted); margin:0;">~780&ndash;5,011 candidates per cell. BIC. &alpha;=0.975. 48h search.</p>
            </div>
            <div class="card" style="border-top:4px solid var(--orange); padding:0.6em 0.8em; background:var(--orange-soft);">
              <h3 style="margin:0 0 0.3em; font-size:0.95rem; color:var(--orange);">SearchLibrium</h3>
              <p style="font-size:0.8rem; margin:0 0 0.4em;">Stage 4 fitness: schedule-placement MNL/MXL. ConstraintBuilder (0.0.138) is the single source of truth &mdash; one declaration feeds every searcher. Random-coefficient toggling via BIC-greedy search.</p>
              <p style="font-size:0.72rem; color:var(--text-muted); margin:0;">SearchLogit heritage (Beeramoole et al., 2024).</p>
            </div>
            <div class="card" style="border-top:4px solid var(--primary-2); padding:0.6em 0.8em; background:var(--surface-tint);">
              <h3 style="margin:0 0 0.3em; font-size:0.95rem;">Larch (Stage 5)</h3>
              <p style="font-size:0.8rem; margin:0 0 0.4em;">Destination + mode choice (NL/MNL over 371 zones). CRASH force-included by constraint registry. Elastic-net penalty on every optimiser invocation. JAX/numba engine guard.</p>
              <p style="font-size:0.72rem; color:var(--text-muted); margin:0;">Flat CA dataset (ActivitySim pre-joined pattern).</p>
            </div>
          </div>

          <div style="display:flex; gap:0.6vw; align-items:stretch;">
            <div class="pipe-box" style="flex:1; font-size:0.76rem; padding:0.6em;"><b>SA OUTER LOOP</b> Exponential-weight feature-mask selection, per segment, 8-way parallel. Significance penalty toward parsimony.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box" style="flex:1; font-size:0.76rem; padding:0.6em;"><b>PENALISED FITNESS</b> &minus;BIC/n + significance penalty + elastic-net. Same objective across stages 1&ndash;5.</div>
            <div class="pipe-arrow"></div>
            <div class="pipe-box hot" style="flex:1; font-size:0.76rem; padding:0.6em;"><b>SELECTED SPECS</b> ga_results.pkl per stage &rarr; merged by BIC &rarr; staged full-population application.</div>
          </div>

          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">30&ndash;50</div><div class="label">candidate features per segment</div></div>
            <div class="stat"><div class="value">2<sup>k</sup></div><div class="label">possible models per segment (intractable)</div></div>
            <div class="stat"><div class="value">5&ndash;8</div><div class="label">selected features in guided03 (parsimonious)</div></div>
            <div class="stat hero"><div class="value">BIC</div><div class="label">fitness &mdash; balances LL against k&middot;ln n</div></div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.74rem;">Constraint declarations: force_include (CRASH, size_term, logsum, Mu_Root), mutually_exclusive (income/age/ACDC/home-pull encodings), min_behavioral, never_random (ASCs, nest params). The analyst declares boundaries; the search decides enrichment.</p>
        </div>
      </section>"""

html = html[:s2] + NEW_WHY + html[e2:]
print(f"  Why slide: {e2-s2:,} -> {len(NEW_WHY):,} bytes")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
