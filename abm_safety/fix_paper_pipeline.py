# -*- coding: utf-8 -*-
"""Fix pipeline DTA text, fix paper slide layout, add Cairns estimation slides."""
import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Fix pipeline slide - replace vehicle.dat with DTA input ──
html = html.replace("vehicle.dat", "DTA input")
print("Replaced vehicle.dat with DTA input")

# ── 2. Fix paper slide - move related publication to RHS ──
paper_idx = html.find("MetaCountRegressor: a major motivation")
if paper_idx > 0:
    s = html.rfind("<section ", 0, paper_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    slide = html[s:e]
    
    # Find related pub card
    rel_idx = slide.find('Related publication')
    if rel_idx > 0:
        # Find the card div
        card_start = slide.rfind('<div class="card on-primary"', 0, rel_idx)
        # Find matching closing </div>
        depth2 = 1; j = card_start
        while depth2 > 0 and j < len(slide):
            j += 1
            if slide[j:j+4] == "<div": depth2 += 1
            elif slide[j:j+6] == "</div>": depth2 -= 1
        card_end = j + 6
        related_card = slide[card_start:card_end]
        
        # Remove from current position
        slide = slide[:card_start] + slide[card_end:]
        
        # Find searchlibrium card end to insert after
        sl_idx = slide.find("searchlibrium")
        if sl_idx > 0:
            # Find end of that card
            sl_card_start = slide.rfind('<div class="card', 0, sl_idx)
            depth2 = 1; j = sl_card_start
            while depth2 > 0 and j < len(slide):
                j += 1
                if slide[j:j+4] == "<div": depth2 += 1
                elif slide[j:j+6] == "</div>": depth2 -= 1
            sl_card_end = j + 6
            slide = slide[:sl_card_end] + related_card + slide[sl_card_end:]
            print("Moved related pub card to RHS")
    
    # ── 3. Also add estimation process slides after the paper slide ──
    EST_SLIDE = """\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Method &middot; Estimation Process</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.5vw,1.9rem);">How the SA search applies to any discrete outcome model</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.7vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid #e74c3c;">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem; color:#c0392b;">The problem: estimation is combinatorial</h3>
              <p style="font-size:0.74rem;margin:0 0 0.5em;">Every sub-model in an ABM (CDAP, activity frequency, duration, schedule, mode, destination) requires the analyst to choose which variables enter the utility function. With k candidate variables, there are 2<sup>k</sup> possible specifications. For k=40, that's ~10<sup>12</sup> models &mdash; impossible to enumerate.</p>
              <p style="font-size:0.7rem;margin:0;"><b>The traditional approach:</b> analyst uses literature, intuition, and trial-and-error to select variables. This introduces omitted-variable bias, over-fitting risk, and produces specifications that don't transfer between regions or populations.</p>
            </div>
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid var(--good);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;">The solution: SA-guided specification search</h3>
              <p style="font-size:0.74rem;margin:0 0 0.5em;">A simulated-annealing metaheuristic searches the specification space, guided by a penalised information criterion (BIC). The analyst declares <b>constraints</b> (what is forbidden); the search decides <b>enrichment</b> (what is included).</p>
              <div style="font-size:0.68rem;line-height:1.5;color:var(--text-muted);">
                <b>Works across model types:</b><br>
                &bull; <b>Count models:</b> Poisson, NB, random-param variants (stages 1&ndash;3, crash SPFs)<br>
                &bull; <b>Discrete choice:</b> MNL, NL, MXL (stages 4&ndash;5)<br>
                &bull; <b>Regression:</b> lognormal, OLS (stage 3 duration, stage 2 budget)<br>
                &bull; <b>Mixed objectives:</b> multi-objective search for Pareto-optimal trade-offs
              </div>
            </div>
          </div>
          <div style="display:flex; align-items:center; justify-content:center; gap:0.5vw; padding:0.3em 0;">
            <svg viewBox="0 0 750 70" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="a6" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4 Z" fill="#888"/></marker></defs>
              <rect x="5" y="10" width="100" height="22" rx="3" fill="#1E2761"/><text x="55" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Define constraints</text>
              <line x1="105" y1="21" x2="125" y2="21" stroke="#888" stroke-width="0.8" marker-end="url(#a6)"/>
              <rect x="128" y="6" width="95" height="30" rx="3" fill="#C85A14"/><text x="175" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">SA search</text><text x="175" y="30" fill="#FFE3CE" font-size="5.5" text-anchor="middle">propose &rarr; fit &rarr; accept/reject</text>
              <line x1="223" y1="21" x2="243" y2="21" stroke="#888" stroke-width="0.8" marker-end="url(#a6)"/>
              <rect x="246" y="6" width="85" height="30" rx="3" fill="#2E7D6B"/><text x="288" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Best BIC wins</text><text x="288" y="30" fill="#D5F5E3" font-size="5.5" text-anchor="middle">parsimonious spec</text>
              <line x1="331" y1="21" x2="351" y2="21" stroke="#888" stroke-width="0.8" marker-end="url(#a6)"/>
              <rect x="354" y="6" width="95" height="30" rx="3" fill="#1E2761"/><text x="401" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Apply to ABM</text><text x="401" y="30" fill="#C9D3F5" font-size="5.5" text-anchor="middle">estimate on HHTS &rarr; predict synth pop</text>
              <line x1="449" y1="21" x2="469" y2="21" stroke="#888" stroke-width="0.8" marker-end="url(#a6)"/>
              <rect x="472" y="6" width="100" height="30" rx="3" fill="#1E2761"/><text x="522" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Validate</text><text x="522" y="30" fill="#C9D3F5" font-size="5.5" text-anchor="middle">in-sample + aggregate</text>
              <line x1="572" y1="21" x2="592" y2="21" stroke="#888" stroke-width="0.8" marker-end="url(#a6)"/>
              <rect x="595" y="6" width="150" height="30" rx="3" fill="url(#bg4)"/><text x="670" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Hypothesis testing complete</text><text x="670" y="30" fill="#D5F5E3" font-size="5.5" text-anchor="middle">specifications saved &rarr; reproducible</text>
              <defs><linearGradient id="bg4" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1E2761"/><stop offset="100%" stop-color="#2E3B8E"/></linearGradient></defs>
              <text x="375" y="58" fill="#888" font-size="6" text-anchor="middle">Same metaheuristic across all ABM stages and crash-function fitting &mdash; uniform BIC objective throughout</text>
            </svg>
          </div>
          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat"><div class="value">5</div><div class="label">ABM stages with SA variable selection</div></div>
            <div class="stat"><div class="value">30</div><div class="label">SPF cells with SA specification search</div></div>
            <div class="stat"><div class="value">BIC</div><div class="label">uniform objective across all models</div></div>
            <div class="stat hero"><div class="value">2<sup>k</sup></div><div class="label">possible models per stage &mdash; intractable to enumerate</div></div>
          </div>
        </div>
      </section>"""
    
    html = html[:e] + "\n\n" + EST_SLIDE + html[e:]
    print("Added estimation process slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
